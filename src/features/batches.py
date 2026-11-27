"""直接从 Qlib 构造一轮训练批次。

调用方提供行情及元数据读取器、股票日列表和随机数生成器。
先打乱股票日，再按组构造并打乱有效训练窗口。
逐批交付特征、标签和样本键，整组处理后释放窗口。
"""

from collections.abc import Iterable, Iterator  # 引入股票日输入和批次迭代接口的类型。
from dataclasses import dataclass  # 引入数据类，用于组织同批的数组与审计计数。

import numpy as np  # 引入数组库，用于排列窗口、组织批次和检查标签。

from src.data.metadata import Metadata  # 引入只读的日线和事件元数据接口。
from src.data.minutes import MinuteLibrary  # 读取 Qlib 的分钟行情。
from src.features.samples import (  # 复用正式样本规则。
    sample_session,  # 在连续竞价时段内构造样本。
    sampling_conditions,  # 从股票日元数据取得限价和停牌边界。
    split_for,  # 按交易日判定所属数据集。
    stock_day_reason,  # 判定股票日的研究资格。
    stock_day_sessions,  # 读取一个股票日并拆为连续竞价时段。
)  # 完成共享样本接口的导入。

DEFAULT_TRAIN_BATCH_SIZE = 2_048  # 第一版每个训练批次的样本数上限。
DEFAULT_STOCK_DAYS_PER_GROUP = 128  # 第一版每组读入的股票日数。

@dataclass(frozen=True)  # 将一批结果固定在同一个对象中。
class TrainingBatch:  # 定义交给训练入口的一批样本。
    """记录一批等长排列的量价输入、收益率和样本键。"""

    x: np.ndarray  # 保存形状为 [B, 30, 6] 的 float32 特征。
    y: np.ndarray  # 保存形状为 [B] 的有限 float64 对数收益率。
    code: np.ndarray  # 保存每行对应的 Qlib 股票代码。
    t: np.ndarray  # 保存每行对应的北京时间分钟结束时刻。


@dataclass  # 允许读取器逐组累加审计数字。
class TrainingSampleCounts:  # 保存训练数据读取中的候选计数。
    """记录已构造的合格候选总数和其中有效标签的数量。"""

    candidates: int = 0  # 记录通过预测时点筛选的候选数。
    labels_available: int = 0  # 记录这些候选中可用于训练的标签数。


def _labeled_stock_day_samples(  # 构造一个股票日的有效训练样本并更新计数。
    library: MinuteLibrary,  # 接收已加载交易日历的 Qlib 分钟行情读取器。
    metadata: Metadata,  # 接收已打开的股票日元数据读取器。
    code: str,  # 接收转换为 Qlib 口径后的股票代码。
    day: str,  # 接收北京时间下的 YYYY-MM-DD 交易日。
    counts: TrainingSampleCounts | None,  # 接收可选的训练候选计数。
) -> Iterator[TrainingBatch]:  # 每次返回同一时段内带有效标签的样本数组。
    """按时段交付该股票日的有效训练输入、标签和预测键。

    股票日不合格时结束迭代，行情或元数据读取错误由调用方处理。
    """
    reason, daily = stock_day_reason(library, metadata, code, day)  # 按正式规则筛选股票日。
    if reason != "eligible":  # 检查股票日是否被排除。
        return  # 跳过不合格股票日。
    bounds, halts = sampling_conditions(daily)  # 取得供分钟筛选使用的限价和停牌区间。
    historical_code = metadata.historical_code(code, day)  # 取得当天证券代码用于判定市场异常时段。
    for _, minutes, bars, excluded in stock_day_sessions(library, code, day, historical_code):  # 依次处理两个时段。
        if excluded:  # 市场异常时段没有训练候选。
            continue  # 进入下一时段。
        session = sample_session(bars, minutes, bounds, halts)  # 直接读取按时段排列的候选数组。
        label_count = int(np.count_nonzero(session.label_valid))  # 统计可用于回归的收益率标签。
        if counts is not None:  # 调用方需要标签覆盖率时才累计。
            counts.candidates += len(session.y)  # 覆盖率分母包括没有有效事后标签的候选。
            counts.labels_available += label_count  # 分子只统计有效收益率。
        if label_count:  # 只有含有效标签的时段才进入组批。
            selected = slice(None) if label_count == len(session.y) else session.label_valid  # 全有效时直接复用数组。
            offsets = np.asarray(minutes)[session.positions[selected]].astype("timedelta64[m]")  # 取得预测时刻距午夜的分钟数。
            yield TrainingBatch(  # 交付一整个时段的有效样本数组。
                x=session.x[selected],  # 取出通过标签筛选的归一化窗口。
                y=session.y[selected],  # 按同一选择保留收益率。
                code=np.full(label_count, code, dtype="U8"),  # 为每个预测时点记入相同的股票代码。
                t=np.datetime64(day, "m") + offsets,  # 将交易日和日内偏移合成为北京时间预测时刻。
            )  # 完成时段内样本的组装。


def iter_training_batches(  # 为一轮训练按两层随机顺序生成批次。
    library: MinuteLibrary,  # 接收 Qlib 分钟行情读取器。
    metadata: Metadata,  # 接收股票日元数据读取器。
    requests: Iterable[tuple[str, str]],  # 接收股票代码与北京时间交易日的成对列表。
    *,  # 后续参数由调用方通过名称传入。
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,  # 每个训练批次的样本数上限，默认 2,048 条。
    rng: np.random.Generator,  # 接收调用方创建的随机数生成器。
    stock_days_per_group: int = DEFAULT_STOCK_DAYS_PER_GROUP,  # 指定每组读入的股票日数，默认 128。
    counts: TrainingSampleCounts | None = None,  # 可传入新的计数对象，用完一轮后计算覆盖率。
) -> Iterator[TrainingBatch]:  # 逐批产出有效标签样本及其样本键。
    """按训练日期选取请求，每轮打乱股票日并在组内重新打乱样本。

    batch_size 默认为 2,048 条样本，组尾不足一批的样本也会交付。
    计数随每组构造而更新，完整消耗迭代器后得到全轮统计。
    批次或组大小非正、训练交易日不存在或股票日重复时抛出 ValueError。
    """
    if batch_size <= 0 or stock_days_per_group <= 0:  # 防止无效大小导致切片或循环出错。
        raise ValueError("Batch size and stock-days per group must be positive")  # 提示调用方传入正数。

    stock_days = []  # 收集训练集股票日，供第一层打乱。
    for code, day in requests:  # 逐项检查调用方请求的股票日。
        if split_for(day) != "train":  # 保留时间切分指定的训练日。
            continue  # 跳过验证和测试日期。
        if day not in library.dates:  # 检查训练请求是否对应实际交易日。
            raise ValueError(f"{day} is absent from the Qlib trading calendar")  # 指出不存在的交易日。
        stock_days.append((metadata.canonical_code(code, day), day))  # 统一代码口径后再分组。
    if len(set(stock_days)) != len(stock_days):  # 检查代码统一后是否产生重复股票日。
        raise ValueError("Multiple requested codes refer to the same stock-day")  # 提示重复请求。
    rng.shuffle(stock_days)  # 在读取分钟数据之前随机排列训练股票日。

    for start in range(0, len(stock_days), stock_days_per_group):  # 沿已打乱的股票日列表依次取组。
        sessions = [  # 收集当前组含有效标签的时段数组。
            session  # 每个时段已经返回对齐的特征、标签和预测键。
            for code, day in stock_days[start:start + stock_days_per_group]  # 按已打乱的次序读取股票日。
            for session in _labeled_stock_day_samples(library, metadata, code, day, counts)  # 沿时段和分钟次序收集样本。
        ]  # 完成本组候选构造。
        if not sessions:  # 整组没有带标签的样本时没有批次可交付。
            continue  # 直接读取下一组。
        x = np.concatenate([session.x for session in sessions])  # 合并本组的特征数组。
        y = np.concatenate([session.y for session in sessions])  # 按相同顺序合并标签。
        codes = np.concatenate([session.code for session in sessions])  # 按相同顺序合并股票代码。
        times = np.concatenate([session.t for session in sessions])  # 按相同顺序合并预测时刻。
        del sessions  # 组装完成即释放各时段的原始数组。
        order = rng.permutation(len(y))  # 将当前组的有效样本做第二层打乱。
        for offset in range(0, len(order), batch_size):  # 按批次取样，最后的不足一批也交付。
            indices = order[offset:offset + batch_size]  # 取得本批在组内的随机索引。
            yield TrainingBatch(x[indices], y[indices], codes[indices], times[indices])  # 用 NumPy 索引生成独立的批次数组。
        del x, y, codes, times  # 在读取下一组前释放本组数据。
