"""按交易日流式预测和验证五分钟收益率回归模型。

从 Qlib 和股票日元数据逐时段构造候选，将特征放入固定大小的推理批次。
每天只保留候选位置、预测值和事后标签，可逐条导出预测，也可统计归一化 Huber 损失和两个方向的 Spearman 秩相关。
"""

import sys  # 将验证进度写入终端的标准错误流。
from collections.abc import Iterable, Iterator  # 声明验证日期来源和时段迭代接口。
from itertools import islice  # 限制试跑范围内每个交易日读取的合格股票数。
from math import isfinite, sqrt  # 核对收益率尺度并计算秩相关分母中的平方根。

import numpy as np  # 组织批量特征、日期内的预测和标签。
import torch  # 执行 PyTorch 推理并管理设备和模型模式。
from scipy.stats import rankdata  # 复用 SciPy 的排序及并列名次平均规则。
from torch.nn.functional import huber_loss  # 复用训练使用的 PyTorch Huber 损失公式。

from src.data.metadata import Metadata  # 声明股票日元数据读取接口。
from src.data.minutes import MinuteLibrary  # 声明 Qlib 分钟行情读取接口。
from src.features.samples import (  # 复用已有样本和时间切分规则。
    FEATURES,  # 取得模型输入特征的顺序和数量。
    TRADING_MINUTES,  # 取得一天中全部连续竞价分钟。
    WINDOW,  # 取得历史窗口长度。
    SessionSamples,  # 声明单时段构造的样本数组及标签接口。
    sample_session,  # 由正式样本规则构造候选和五分钟标签。
    sampling_conditions,  # 将股票日元数据转成限价和停牌条件。
    split_for,  # 检查预测日期在研究期内以及验证日期属于验证集。
    stock_day_reason,  # 确定各股票日在研究范围内。
    stock_day_sessions,  # 复用正式的连续竞价时段切片。
)  # 完成数据和样本接口导入。
from src.models.training_data import (  # 使用统一的特征输入接口。
    prepare_model_inputs,  # 复用训练入口的模型通道组装规则。
)  # 完成特征接口导入。

_MIN_PAIRS = 20  # 按确认的口径，单组至少有二十对有效数值。
_MINUTES = np.asarray(TRADING_MINUTES)  # 将全日研究分钟固定为按时间排序的数组。


def _eligible_stock_days(  # 按代码升序交付当天可进入采样的股票。
    library: MinuteLibrary,  # 由调用方打开的 Qlib 分钟库。
    metadata: Metadata,  # 由调用方打开的股票日元数据。
    day: str,  # 北京时间下的 ISO 交易日。
) -> Iterator[tuple[str, tuple[int, int], list[tuple[float, float]]]]:  # 返回股票代码、限价和停复牌区间。
    """输入行情库、元数据读取器和北京时间日期。每次返回代码、以分计的涨跌停价及以秒计的停牌区间。

    行情和元数据访问异常由其读取接口继续报告。
    """
    codes = {  # 汇总当天存在行情覆盖的证券对应的 Qlib 代码。
        metadata.canonical_code(code, day)  # 按当天代码变更口径统一别名。
        for code in library.spans  # 遍历 Qlib 证券目录。
        if library.has_coverage(code, day)  # 跳过该日没有行情覆盖的目录项。
    }  # 合并指向同一个股票日的别名。
    for code in sorted(codes):  # 按统一后的股票代码升序筛选。
        reason, daily = stock_day_reason(library, metadata, code, day)  # 应用现有股票日资格规则。
        if reason == "eligible":  # 只将合格股票日交给分钟窗口计算。
            bounds, halts = sampling_conditions(daily)  # 取得时段共同使用的限价和停牌边界。
            yield code, bounds, halts  # 交付已通过日级筛选的采样条件。


def _date_sessions(  # 逐股、逐时段读取当日的候选。
    library: MinuteLibrary,  # 由调用方打开的 Qlib 分钟库。
    metadata: Metadata,  # 由调用方打开的股票日元数据。
    day: str,  # 北京时间下的 ISO 交易日。
    max_stock_days: int | None,  # 试跑时限制进入分钟读取的合格股票数。
) -> Iterator[tuple[int, str, np.ndarray, SessionSamples]]:  # 返回股票行与代码、分钟列及同序样本数组。
    """每次返回一段行情的日期矩阵行、规范代码、候选分钟列和样本数组。读取异常原样传出。"""
    stock_days = _eligible_stock_days(library, metadata, day)  # 取得稳定的合格股票顺序。
    for row, (code, bounds, halts) in enumerate(islice(stock_days, max_stock_days)):  # 给每个入选证券分配唯一矩阵行。
        historical_code = metadata.historical_code(code, day)  # 取得交易所当日使用的代码。
        for _, minutes, bars, excluded in stock_day_sessions(library, code, day, historical_code):  # 读取时段并沿用例外规则。
            if excluded:  # 已排除的时段无法产生候选。
                continue  # 避免不必要的特征和标签计算。
            session = sample_session(bars, minutes, bounds, halts)  # 应用正式采样口径。
            columns = np.searchsorted(_MINUTES, np.asarray(minutes)[session.positions])  # 将候选时钟分钟映射到全天矩阵列。
            yield row, code, columns, session  # 交付与股票代码及原预测时点对齐的时段结果。


def _rank_ic(labels: np.ndarray, predictions: np.ndarray) -> float | None:  # 计算单组的 Spearman 秩相关。
    """输入形状相同的一维标签和预测数组。有效对不足或秩为常量时返回 None。"""
    usable = np.isfinite(labels) & np.isfinite(predictions)  # 只比较收益率和预测都有限的样本。
    size = int(np.count_nonzero(usable))  # 计算同组可用于比较的样本数。
    if size < _MIN_PAIRS:  # 样本数不足时不把该组纳入日均值。
        return None  # 表示该组无可报告的秩相关。
    pairs = np.column_stack((labels[usable], predictions[usable]))  # 并排整理真实和预测收益率。
    ranks = rankdata(pairs, method="average", axis=0)  # 分别排序两列，平值取并列平均秩。
    ranks -= (size + 1) / 2  # 平移两列到各自秩的均值零。
    actual, predicted = ranks.T  # 取出对齐的实际秩和预测秩。
    denominator = sqrt(np.dot(actual, actual) * np.dot(predicted, predicted))  # 计算两列秩的标准化分母。
    if denominator == 0:  # 任意一列的秩恒定时相关系数无定义。
        return None  # 将常量组从日均值中排除。
    return float(np.clip(np.dot(actual, predicted) / denominator, -1, 1))  # 返回秩相关并消除浮点边界误差。


def _average(values: list[float]) -> float | None:  # 计算有效组或有效日期的等权均值。
    """values 装有同一级别的有限秩相关。没有可用数值时返回 None。"""
    return float(np.mean(values)) if values else None  # 确保输出可用标准 JSON 表示。


def _huber_total(labels: np.ndarray, predictions: np.ndarray, scale: float) -> tuple[float, int]:  # 计算当天的损失总和和样本数。
    """输入同形状的实际和预测收益率矩阵及正有限训练标签尺度，空有效对返回零损失和零计数。"""
    usable = np.isfinite(labels) & np.isfinite(predictions)  # 损失只使用实际和预测收益率都有限的样本。
    count = int(np.count_nonzero(usable))  # 记录分母，便于跨日期按样本加权。
    if not count:  # 当天没有可计算损失的配对时直接返回。
        return 0.0, 0  # 使外层正确识别整段验证都缺少损失配对的情况。
    actual = torch.from_numpy(labels[usable]).div_(scale)  # 复制选中的标签，再除以训练标签尺度。
    predicted = torch.from_numpy(predictions[usable]).div_(scale)  # 让模型预测与实际收益使用同一个无量纲尺度。
    total = huber_loss(predicted, actual, reduction="sum", delta=1.0)  # 在 CPU 上一次求出当天有效样本的 Huber 总损失。
    return total.item(), count  # 返回标量损失和有效样本数，释放当天的临时向量。


def _predict_date(  # 生成指定交易日的全部合格候选预测。
    model: torch.nn.Module,  # 已放到 device 且处于推理模式的收益率模型。
    library: MinuteLibrary,  # 读取 Qlib 分钟行情。
    metadata: Metadata,  # 提供当日范围和交易事件。
    day: str,  # 北京时间下的 ISO 交易日。
    batch_size: int,  # 单次推理允许的最大样本条数。
    device: torch.device | str,  # 接收输入批次的 PyTorch 设备。
    max_stock_days: int | None,  # 试跑时每个交易日的合格股票上限。
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # 返回有序代码、标签、预测和候选掩码。
    """复用固定大小的特征缓冲区。只以预测时点的样本资格确定候选，模型及行情错误原样报告。

    代码为按股票排列的一维数组，标签、预测及候选掩码均为与全天时钟对齐的二维矩阵。
    """
    rows = len(library.spans) if max_stock_days is None else min(len(library.spans), max_stock_days)  # 确定日期矩阵行数上限。
    labels = np.full((rows, len(_MINUTES)), np.nan)  # 未出现的股票与分钟组合保持缺失。
    predictions = np.full_like(labels, np.nan)  # 按同样的股票行和分钟列保存模型输出。
    codes = np.empty(rows, dtype="U8")  # 记录各股票行对应的八位规范代码。
    candidate_mask = np.zeros(labels.shape, dtype=bool)  # 单独记录候选，以便保留标签缺失的窗口。
    windows = np.empty((batch_size, WINDOW, len(FEATURES)), dtype=np.float32)  # 只分配一批原始特征的内存。
    batch_rows = np.empty(batch_size, dtype=np.intp)  # 记录待推理样本对应的股票行。
    batch_columns = np.empty(batch_size, dtype=np.intp)  # 记录待推理样本对应的分钟列。
    input_channels = getattr(model, "input_channels", len(FEATURES))  # 未声明规格的模型使用已有六个量价通道。
    requires_minutes = getattr(model, "requires_prediction_minutes", getattr(model, "minute_return_scales", None) is not None)  # 识别新旧模型对预测分钟的要求。
    size = used_rows = 0  # 初始化待推理样本和使用行的计数。

    def flush(count: int) -> None:  # 用当前缓冲区中已填充的样本进行一次推理。
        """count 为待推理窗口数。结果写回预测矩阵，模型输出形状错误时抛出 ValueError。"""
        market_inputs = torch.from_numpy(windows[:count]).transpose(1, 2)  # 将固定批次内的量价窗口转换为卷积布局。
        minutes = _MINUTES[batch_columns[:count]]  # 还原各候选窗口实际的北京时间预测分钟。
        inputs = prepare_model_inputs(market_inputs, minutes, input_channels).to(device)  # 与训练使用同样的通道组装规则。
        if requires_minutes:  # 时钟预测头或分钟尺度需要每条窗口的北京时间预测分钟。
            minute_tensor = torch.from_numpy(minutes).to(device=device, dtype=torch.long)  # 组装与输入同设备的时刻索引。
            output = model(inputs, minute_tensor)  # 按预测分钟换回物理收益率。
        else:  # 使用全局尺度的模型只接收窗口输入。
            output = model(inputs)  # 得到每个窗口对应的物理收益率预测。
        if output.shape != (count,):  # 检查模型对每条样本是否返回一个标量。
            raise ValueError("Model must emit one return prediction per sample")  # 明确报告输出形状不符。
        predictions[batch_rows[:count], batch_columns[:count]] = output.cpu().numpy()  # 按原股票和分钟保存预测。

    for row, code, columns, session in _date_sessions(library, metadata, day, max_stock_days):  # 依次获取同日各股票的候选。
        count = len(columns)  # 候选数包括随后无法观测到标签的窗口。
        used_rows = row + 1  # 更新当天有时段数据的股票行范围。
        codes[row] = code  # 把该股票行与规范证券代码对齐。
        candidate_mask[row, columns] = True  # 不以事后标签是否存在作为预测条件。
        labels[row, columns] = session.y  # 收益率按候选原时点保存，缺失标签保留为 NaN。
        start = 0  # 开始复制当前时段的窗口。
        while start < count:  # 若当前批空间不足，则分段取完该时段的窗口。
            copied = min(batch_size - size, count - start)  # 计算当前批剩余容量可接收的条数。
            stop = start + copied  # 定位当前分段在时段数组中的末尾。
            pending = slice(size, size + copied)  # 定位当前分段在推理缓冲区中的位置。
            windows[pending] = session.x[start:stop]  # 复制这一段特征，不保留全日窗口。
            batch_rows[pending] = row  # 保留该段窗口对应的股票行。
            batch_columns[pending] = columns[start:stop]  # 保留该段窗口对应的分钟列。
            start, size = stop, size + copied  # 同时推进时段遍历和已填充条数。
            if size == batch_size:  # 当前批装满后立刻推理。
                flush(size)  # 把当前批预测写回当天的预测矩阵。
                size = 0  # 释放缓冲区位置给后续窗口。
    if size:  # 当天结束后处理不足整批的剩余窗口。
        flush(size)  # 保留尾批中的每一条预测。

    return (  # 只保留读取范围实际用到的股票行。
        codes[:used_rows],  # 与以下矩阵行对齐的股票代码。
        labels[:used_rows],  # 可观测的实际五分钟收益率。
        predictions[:used_rows],  # 候选窗口模型给出的五分钟收益率预测。
        candidate_mask[:used_rows],  # 表示每个时点是否通过预测资格筛选。
    )  # 完成当日结果的交付。


def _evaluate_date(  # 使用共享预测结果计算一天的秩相关和验证损失。
    model: torch.nn.Module,  # 已放到 device 且处于验证模式的收益率模型。
    library: MinuteLibrary,  # 读取 Qlib 分钟行情。
    metadata: Metadata,  # 提供当日范围和交易事件。
    day: str,  # 北京时间下的验证集交易日。
    batch_size: int,  # 单次推理允许的最大样本条数。
    device: torch.device | str,  # 接收验证批次的 PyTorch 设备。
    max_stock_days: int | None,  # 试跑时每个验证日的合格股票上限。
) -> tuple[int, int, list[float], list[float], float, int]:  # 返回候选和有效标签数、两类有效组 IC、损失总和及配对数。
    """计算指定交易日的统计量，验证 Huber 始终以全局训练标签尺度归一化。"""
    _, labels, predictions, candidate_mask = _predict_date(  # 只推理一次，再汇总当天的所有指标。
        model, library, metadata, day, batch_size, device, max_stock_days,  # 使用调用方确认的行情和批量配置。
    )  # 取得原始物理收益率，不在这里选择是否使用分钟尺度模型。
    candidates = int(np.count_nonzero(candidate_mask))  # 计入事后标签缺失的全部候选。
    labels_available = int(np.count_nonzero(np.isfinite(labels)))  # 有限实际收益率构成覆盖率分子。
    ts = [_rank_ic(y, p) for y, p in zip(labels, predictions, strict=True)]  # 每一行即一个股票日。
    cs = [_rank_ic(y, p) for y, p in zip(labels.T, predictions.T, strict=True)]  # 每一列即同日同分钟的截面。
    scale = float(getattr(model, "return_scale", 1.0))  # 普通模型没有训练尺度时按物理收益率直接计算。
    if not isfinite(scale) or scale <= 0:  # 只有有限的正尺度才能用于归一化。
        raise ValueError("Validation return scale must be positive and finite")  # 明确报告模型的收益率尺度错误。
    loss_total, loss_count = _huber_total(labels, predictions, scale)  # 复用当天的预测与标签，计算损失及其分母。
    ts_valid = [ic for ic in ts if ic is not None]  # 只交付有定义的日内时序 IC 组。
    cs_valid = [ic for ic in cs if ic is not None]  # 只交付有定义的截面 IC 组。
    return candidates, labels_available, ts_valid, cs_valid, loss_total, loss_count  # 统一交付当天的统计量。


def predict_day(  # 导出一个北京时间交易日内的所有合格候选预测。
    model: torch.nn.Module,  # 已放到 device 的收益率模型。
    library: MinuteLibrary,  # 已打开的 Qlib 分钟行情读取器。
    metadata: Metadata,  # 已打开的股票日元数据读取器。
    day: str,  # 研究期内与 Qlib 日历匹配的 ISO 交易日。
    *,  # 推理设置通过参数名传入。
    batch_size: int,  # 最大并行推理的窗口数，尾批也会处理。
    device: torch.device | str,  # 模型和批次所在的 PyTorch 设备。
    max_stock_days_per_date: int | None = None,  # 默认遍历所有合格股票，试跑可以限制数量。
) -> dict[str, np.ndarray]:  # 返回按规范代码和预测分钟排序的同长一维数组。
    """交付 code、t、prediction、label 和 label_valid，t 为北京时间的 datetime64[m]。

    预测和实际标签均为对数收益率。不具备事后标签的候选以 NaN 保留。
    日期或批量参数无效时抛出 ValueError，预测出现非有限值时抛出 FloatingPointError。
    """
    split_for(day)  # 检查日期属于项目已定义的历史范围。
    if day not in library.dates:  # 检查实际行情库存在该交易日。
        raise ValueError(f"{day} is not a Qlib trading date")  # 避免为不可用的日期生成空预测。
    if batch_size <= 0 or (max_stock_days_per_date is not None and max_stock_days_per_date <= 0):  # 检查推理和试跑上限。
        raise ValueError("Prediction batch size and stock-day limit must be positive")  # 在读取前报告无效配置。

    was_training = model.training  # 保留调用方原先的模型模式。
    model.eval()  # 使用并行批次进行推理。
    try:  # 即使模型或数据报错也恢复调用者的模型模式。
        with torch.inference_mode():  # 仅推理，不记录梯度和反向图。
            codes, labels, predictions, candidate_mask = _predict_date(  # 与验证使用相同的当日预测流程。
                model, library, metadata, day, batch_size, device, max_stock_days_per_date,  # 交付现有样本和设备设置。
            )  # 收集矩阵形式的日内结果。
    finally:  # 推理执行结束后恢复原来的模式。
        model.train(was_training)  # 让调用方继续使用同一个模型。

    rows, columns = np.nonzero(candidate_mask)  # 按代码和分钟提取全部候选，标签可用性不影响选行。
    prediction = predictions[rows, columns]  # 取得候选时点的物理收益率预测。
    if not np.isfinite(prediction).all():  # 任何已进入候选的预测无效时停止交付。
        raise FloatingPointError(f"{day} has a non-finite return prediction")  # 明确报告无法导出的日期。
    label = labels[rows, columns]  # 取得同一行的事后观测收益率。
    label_valid = np.isfinite(label)  # 区分可观测和不可观测标签。
    return {  # 交付能够直接构造时序结果表的等长数组。
        "code": codes[rows],  # 规范化的股票代码。
        "t": np.datetime64(day, "m") + _MINUTES[columns].astype("timedelta64[m]"),  # 以北京时间表示预测时点。
        "prediction": prediction,  # 该时点已知信息给出的预测。
        "label": np.where(label_valid, label, np.nan),  # 填入事后标签，不可用时用 NaN 表示。
        "label_valid": label_valid,  # 提供明确的标签可用性标记。
    }  # 完成当日历史预测的交付。


def evaluate(  # 在指定验证日期上计算覆盖率、秩相关和归一化 Huber 损失。
    model: torch.nn.Module,  # 接收 float32 [批大小, 通道数, 30] 并输出 [批大小] 的收益率预测。
    library: MinuteLibrary,  # 已打开的 Qlib 分钟行情读取器。
    metadata: Metadata,  # 调用方负责在整个验证过程中保持元数据读取器打开。
    dates: Iterable[str],  # 北京时间的 ISO 验证交易日，可只传入一部分做试跑。
    *,  # 剩余设置通过参数名传入。
    batch_size: int,  # 推理批次的样本数上限，尾批也会处理。
    device: torch.device | str,  # 模型已放到的设备，输入移动到同一设备。
    max_stock_days_per_date: int | None = None,  # 默认遍历全部合格股票，试跑可限制每天股票数。
) -> dict[str, int | float | None]:  # 输出可序列化为 JSON 的值。
    """计算先组内排名再日内汇总的指标，各日期等权。

    空名单、重复或非验证交易日和无效的批量参数抛出 ValueError。
    无有效组时相应指标为 None，没有候选时覆盖率也为 None。
    Huber 对全部预测与标签均有限的样本等权。预测和标签都除以模型的 return_scale，
    默认尺度为 1，损失阈值为 1。没有可用配对时损失均值为 None。
    """
    days = sorted(dates)  # 按北京时间日期升序处理验证数据。
    if not days or len(set(days)) != len(days):  # 拒绝缺少日期或重复加权同一个日期。
        raise ValueError("Validation dates must be nonempty and unique")  # 指明日期列表的基本约定。
    for day in days:  # 开始读取行情之前先核对整个验证范围。
        if split_for(day) != "validation" or day not in library.dates:  # 只允许验证期内 Qlib 日历中的交易日。
            raise ValueError(f"{day} is not a Qlib validation trading date")  # 防止训练集或测试集进入验证。
    if batch_size <= 0 or (max_stock_days_per_date is not None and max_stock_days_per_date <= 0):  # 检查推理与试跑上限。
        raise ValueError("Validation batch size and stock-day limit must be positive")  # 无效参数时在读取前退出。

    ts_daily, cs_daily = [], []  # 记录每个有有效组的日期均值。
    candidates = labels_available = ts_groups = cs_groups = 0  # 初始化全期候选、标签和有效组计数。
    huber_sum = 0.0  # 累积全期有效配对的 Huber 损失。
    huber_samples = 0  # 累积计算 Huber 损失的有效样本数。
    was_training = model.training  # 保留调用者原来的模型训练模式。
    model.eval()  # 验证期间禁用 Dropout 等训练期行为。
    try:  # 推理或行情读取失败时也要恢复模型模式。
        with torch.inference_mode():  # 在验证时关闭梯度记录。
            for index, day in enumerate(days, 1):  # 每次只处理一个验证交易日并记录轮次内进度。
                day_candidates, day_available, ts, cs, loss_total, loss_count = _evaluate_date(  # 取得当天计数、组 IC 和损失。
                    model, library, metadata, day,  # 使用同一套模型和数据口径。
                    batch_size, device, max_stock_days_per_date,  # 按调用方指定的设备和批量推理。
                )  # 当天数组在函数返回后释放。
                candidates += day_candidates  # 累加各日合格候选。
                labels_available += day_available  # 累加各日有效标签。
                huber_sum += loss_total  # 各日以损失总和参与汇总。
                huber_samples += loss_count  # 保留样本数，避免样本量不同的日期被等权平均。
                ts_groups += len(ts)  # 统计全期能计算时序 IC 的股票日数。
                cs_groups += len(cs)  # 统计全期能计算截面 IC 的分钟数。
                if ts:  # 有至少一个有效时序分组的日期才纳入日期均值。
                    ts_daily.append(float(np.mean(ts)))  # 同日有效股票等权。
                if cs:  # 有至少一个有效截面分组的日期才纳入日期均值。
                    cs_daily.append(float(np.mean(cs)))  # 同日有效预测分钟等权。
                if index % 10 == 0:  # 每完成十个验证交易日汇报一次进度。
                    print(  # 进度输出独立于验证指标结果。
                        f"验证已完成 {index}/{len(days)} 个交易日，最近完成 {day}",  # 报告日期和本轮已完成数。
                        file=sys.stderr,  # 保留标准输出供训练指标和结构化结果使用。
                        flush=True,  # 实时写入终端和运行日志。
                    )  # 完成当前验证进度输出。
    finally:  # 在验证完成或抛出异常时一律恢复调用前模式。
        model.train(was_training)  # 允许训练循环直接继续下一轮。

    return {  # 交付覆盖率、Huber 损失、两个秩相关和相应计数。
        "ts_rank_ic": _average(ts_daily),  # 有效日期等权的时序秩相关。
        "cross_section_rank_ic": _average(cs_daily),  # 有效日期等权的截面秩相关。
        "candidates": candidates,  # 各日通过预测输入筛选的候选总数。
        "labels_available": labels_available,  # 各日有有限收益率的候选总数。
        "label_coverage": labels_available / candidates if candidates else None,  # 有效事后标签占全部候选的比例。
        "mean_huber": huber_sum / huber_samples if huber_samples else None,  # 跨日期按有效样本数汇总。
        "huber_samples": huber_samples,  # 报告标签和预测都有限的样本数。
        "ts_group_count": ts_groups,  # 有效时序分组数。
        "cross_section_group_count": cs_groups,  # 有效截面分组数。
        "ts_day_count": len(ts_daily),  # 时序指标覆盖的日期数。
        "cross_section_day_count": len(cs_daily),  # 截面指标覆盖的日期数。
    }  # 完成验证结果字典。
