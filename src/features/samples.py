"""将选定股票日的 Qlib 分钟行情转换成预测候选和收益率标签。

先按股票日和连续竞价时段组织数据，再用截至预测时点的行情筛选和归一化。
通过筛选的候选保留输入，之后根据目标分钟行情和停牌记录计算标签或标记缺失。
"""

import csv  # 读取需要处理的股票日清单。
import re  # 校验证券代码的格式。
from dataclasses import dataclass  # 将同一时段的候选数组和审计原因放在一起。
from datetime import date  # 检查日期是否真实存在且格式一致。
from decimal import ROUND_HALF_UP, Decimal  # 按证券报价精度计算涨跌停价。

import numpy as np  # 处理分钟行情数组和窗口特征。

from src.data.minutes import (  # 读取分钟行情字段的固定顺序。
    AMOUNT,  # 成交额列的索引。
    CLOSE,  # 收盘价列的索引。
    FACTOR,  # Qlib 缩放因子列的索引。
    FIELDS,  # 分钟行情的全部字段名。
    HIGH,  # 最高价列的索引。
    LOW,  # 最低价列的索引。
    OPEN,  # 开盘价列的索引。
    VOLUME,  # 成交量列的索引。
)  # 完成分钟行情字段的导入。

FEATURES = FIELDS[:6]  # 样本构造交付开高低收、成交量和成交额六列基础量价。
TIME_FEATURES = ("is_afternoon", "session_progress")  # 八通道模型依次追加下午标记和时段进度。
WINDOW = 30  # 每个输入窗口使用连续 30 根分钟线。
ENTRY_OFFSET = 1  # 使用预测时点下一根分钟线的开盘价。
HORIZON = 5  # 使用预测时点之后第五根分钟线的收盘价。
RETURN_LABEL = "next_open_to_t_plus_5_close"  # 用固定名称记录该对数收益率口径。
SESSIONS = {  # 列出北京时间下参加采样的连续竞价时段。
    "morning": range(9 * 60 + 31, 11 * 60 + 31),  # 上午含 09:31 和 11:30。
    "afternoon": range(13 * 60 + 1, 14 * 60 + 58),  # 下午含 13:01 和 14:57。
}  # 结束交易时段定义。
# 拼接两个时段的分钟位置，供读取器一次读取一个股票日。
TRADING_MINUTES = tuple(minute for minutes in SESSIONS.values() for minute in minutes)
TICK = Decimal("0.01")  # 普通 A 股的最小报价单位为 0.01 元。


def _clock_feature_table() -> tuple[np.ndarray, np.ndarray]:  # 预先计算每个可用预测分钟的窗口时间特征。
    """无需参数，返回从北京时间午夜起按分钟索引的特征表及窗口有效掩码。"""
    values = np.empty((24 * 60, len(TIME_FEATURES), WINDOW), dtype=np.float32)  # 为一天的所有钟表分钟预留位置。
    supported = np.zeros(24 * 60, dtype=bool)  # 默认每个预测分钟都不具备有效历史窗口。
    history = np.arange(WINDOW - 1, -1, -1)  # 依次记录从窗口首根到末根行情距离预测的分钟数。
    for name, minutes in SESSIONS.items():  # 上午、下午分别按各自的时段长度计算。
        ends = np.asarray(minutes)[WINDOW - 1:]  # 选取一个完整历史窗口能够结束的位置。
        ordinal = np.arange(WINDOW, len(minutes) + 1)  # 取得每个窗口末根行情在时段内从一开始的序号。
        values[ends, 0, :] = name == "afternoon"  # 同一窗口的每根行情共用所属时段的下午标记。
        values[ends, 1, :] = (ordinal[:, None] - history) / len(minutes)  # 逐根计算本时段序号占总分钟数的比例。
        supported[ends] = True  # 只开放可以在单个连续竞价时段内取得全部历史的预测分钟。
    return values, supported  # 交付预计算的时间特征及配套掩码。


_CLOCK_FEATURES, _SUPPORTED_CLOCK = _clock_feature_table()  # 训练和验证都可按预测分钟直接查询。


def time_features(prediction_minutes: np.ndarray) -> np.ndarray:  # 构造按卷积布局排列的两个时间通道。
    """prediction_minutes 为形状 [批大小] 的北京时间预测分钟整数，自午夜起算。

    返回 float32 [批大小, 2, 30]，每个位置依次为下午标记和时段内进度。输入形状、
    类型、时钟分钟或连续历史窗口不符合规格时抛出 ValueError。
    """
    minutes = np.asarray(prediction_minutes)  # 保留调用方提供的预测分钟顺序。
    if minutes.ndim != 1 or not np.issubdtype(minutes.dtype, np.integer):  # 只接受一维整数分钟。
        raise ValueError("Prediction minutes must be a one-dimensional integer array")  # 提示调用方先转换原始时间戳。
    if np.any(minutes < 0) or np.any(minutes >= len(_SUPPORTED_CLOCK)):  # 防止负索引和跨出北京时间日期。
        raise ValueError("Prediction minutes must be within a Beijing calendar day")  # 拒绝超出当天的时钟分钟。
    if not _SUPPORTED_CLOCK[minutes].all():  # 确认预测分钟处于连续竞价且窗口未跨越时段边界。
        raise ValueError("Prediction minutes must have 30 bars in one continuous trading session")  # 说明缺少完整的同一时段窗口。
    return _CLOCK_FEATURES[minutes]  # 批量查询窗口时间特征，并保留调用方提供的行顺序。


def split_for(day):  # 根据交易日确定数据集。
    """接收 YYYY-MM-DD 格式的北京时间交易日 day，返回数据集名称。

    起止日期均含端点，超出研究日期范围时抛出 ValueError。
    """
    if "2023-01-03" <= day <= "2024-12-31":  # 最早两年的数据用于训练。
        return "train"  # 返回训练集标识。
    if "2025-01-02" <= day <= "2025-06-30":  # 2025 年上半年用于验证。
        return "validation"  # 返回验证集标识。
    if "2025-07-01" <= day <= "2025-12-31":  # 2025 年下半年用于测试。
        return "test"  # 返回测试集标识。
    raise ValueError(f"{day} is outside the configured study dates")  # 拒绝范围外日期。


def limit_cents(preclose):  # 用前收参考价推算当日限价。
    """接收单位为元的 Decimal 前收参考价 preclose，返回跌停价和涨停价的整数分值。

    参考价必须有限、大于或等于一分且符合报价精度，否则抛出 ValueError。
    """
    # 参考价须能直接用当前股票的最小报价单位表示。
    if not preclose.is_finite() or preclose < TICK or preclose != preclose.quantize(TICK):
        raise ValueError(f"Invalid reference price {preclose} for ¥0.01 tick")  # 报出无效参考价。
    # 前收参考价乘以 90%，按分四舍五入。
    lower = (preclose * Decimal("0.90")).quantize(TICK, rounding=ROUND_HALF_UP)
    # 前收参考价乘以 110%，按分四舍五入。
    upper = (preclose * Decimal("1.10")).quantize(TICK, rounding=ROUND_HALF_UP)
    return (  # 依次返回跌停、涨停的整数分值。
        int(max(TICK, min(lower, preclose - TICK)) / TICK),  # 跌停价至少一分。
        int(max(upper, preclose + TICK) / TICK),  # 涨停价比参考价至少高一分。
    )  # 返回两侧限价。


def input_quality(bars):  # 逐根分钟线检查缺失和基础有效性。
    """接收形状为（分钟数，7）的 Qlib 分钟行情数组 bars。

    返回两个等长布尔数组，依次标记字段缺失和全部字段满足基础规则的分钟。
    """
    missing = np.isnan(bars).any(axis=1)  # 七列中任一字段为 NaN 即为缺失。
    valid = (  # 同时检查有限性、价格区间和量价字段的符号。
        np.isfinite(bars).all(axis=1)  # 所需字段均为有限值。
        & (bars[:, :4] > 0).all(axis=1)  # 开高低收均须为正。
        & (bars[:, [OPEN, CLOSE]] >= bars[:, LOW, None]).all(axis=1)  # 开收不低于最低价。
        & (bars[:, [OPEN, CLOSE]] <= bars[:, HIGH, None]).all(axis=1)  # 开收不高于最高价。
        & (bars[:, [VOLUME, AMOUNT]] >= 0).all(axis=1)  # 成交量、成交额允许为零。
        & (bars[:, FACTOR] > 0)  # 缩放因子用于换算，必须为正。
    )  # 合并得到每根分钟线的有效性。
    return missing, valid  # 分开返回缺失标记和有效标记。


def elapsed_seconds(value):  # 将停复牌时刻换算成从当日午夜起算的秒数。
    """接收北京时间的 datetime.time 对象 value，返回可含小数的秒数。"""
    # 小数部分保留公告时刻中可能出现的微秒。
    return 3600 * value.hour + 60 * value.minute + value.second + value.microsecond / 1_000_000


def sampling_conditions(daily):  # 准备同一股票日各时段共用的采样条件。
    """接收元数据读取器返回的合格股票日 daily，返回限价和盘中停牌区间。

    限价依次为跌停和涨停的整数分值，停牌区间使用北京时间从午夜起算的秒数。
    前收参考价无效时保留限价计算抛出的 ValueError。
    """
    bounds = limit_cents(daily.preclose)  # 将前收参考价换算成整数分限价。
    halts = [  # 将每次停复牌都转换为秒数区间。
        (elapsed_seconds(start), elapsed_seconds(stop))  # 保留开始和恢复时刻。
        for start, stop in daily.intraday_halts  # 遍历该股票日的已知盘中停牌。
    ]  # 完成停复牌记录的时间单位换算。
    return bounds, halts  # 返回构造样本时共用的价格和时间约束。


def _positive_medians(values):  # 批量计算每个窗口的正值中位数。
    """接收形状为（窗口数，分钟数）的成交数据，按行返回中位数。

    仅使用大于零的数值。缺少正值的行返回 NaN。
    """
    positive = values > 0  # 标记原口径下参与计算的非零成交。
    counts = positive.sum(axis=1)  # 取得每个窗口的正值数。
    ordered = np.sort(np.where(positive, values, np.inf), axis=1)  # 将被排除的值移到右侧。
    right = counts // 2  # 取得每行右中间值的位置。
    rows = np.arange(len(values))  # 建立每个窗口的行号。
    medians = ordered[rows, right]  # 奇数个正值时直接取唯一的中间值。
    even = (counts > 0) & (counts % 2 == 0)  # 找到需要平均两个中间值的窗口。
    middle_pair = np.column_stack((ordered[rows[even], right[even] - 1], medians[even]))  # 取得两个中间值。
    medians[even] = middle_pair.mean(axis=1)  # 沿用 NumPy 中位数的平均与舍入行为。
    medians[counts == 0] = np.nan  # 标记缺少成交基准的窗口。
    return medians  # 每个窗口返回自己的正值中位数。


def _normalize_windows(windows):  # 一次归一化同一时段的多个窗口。
    """接收形状为（窗口数，30，7）的窗口，原输入保持只读。

    返回形状为（窗口数，30，6）的 float32 特征和逐窗口有效掩码。
    成交股数或成交额缺少正值中位数时，相应特征行填充 NaN，掩码为 False。
    """
    shares = windows[:, :, VOLUME] * windows[:, :, FACTOR]  # 将每个分钟的 Qlib 成交量还原为股数。
    amounts = windows[:, :, AMOUNT]  # 取得单位为元的成交额。
    share_medians = _positive_medians(shares)  # 每个窗口只用自身正股数计算基准。
    amount_medians = _positive_medians(amounts)  # 每个窗口只用自身正成交额计算基准。
    valid = (share_medians > 0) & (amount_medians > 0)  # 只有两个成交基准都可用时才生成特征。
    all_valid = valid.all()  # 检查本批能否直接使用所有窗口。
    selection = slice(None) if all_valid else valid  # 仅有缺失时才拷贝可用窗口子集。
    selected = windows[selection]  # 取出要归一化的窗口。
    close_medians = np.median(selected[:, :, CLOSE], axis=1)  # 独立计算每个窗口的收盘价基准。
    prices = np.log(selected[:, :, :4] / close_medians[:, None, None])  # 一次计算四列价格特征。
    volumes = np.log1p(shares[selection] / share_medians[selection, None])  # 计算成交股数特征。
    turnovers = np.log1p(amounts[selection] / amount_medians[selection, None])  # 计算成交额特征。
    normalized = np.concatenate(  # 合并为模型输入顺序。
        (prices, volumes[..., None], turnovers[..., None]),  # 依次放入开高低收、股数和成交额。
        axis=2,  # 在字段方向合并，保留窗口和分钟位置。
        dtype=np.float32,  # 直接写入模型输入约定的单精度数组。
    )  # 完成特征数组的拼接。
    if all_valid:  # 当每行都合格时，可直接使用完整输出。
        return normalized, valid  # 交付按输入顺序排列的特征和掩码。
    result = np.full((*windows.shape[:2], len(FEATURES)), np.nan, dtype=np.float32)  # 为无效窗口保留空位。
    result[valid] = normalized  # 将合格特征写回原有窗口位置。
    return result, valid  # 返回完整位置与对应掩码。


def normalize(window):  # 保留单个窗口的特征计算接口。
    """接收形状为（30，7）的有效 Qlib 分钟行情窗口 window。

    返回形状为（30，6）的 float32 特征。成交股数或成交额没有正值时返回 None。
    """
    features, valid = _normalize_windows(window[None, ...])  # 将单窗口交给共用的归一化逻辑。
    return features[0] if valid[0] else None  # 保留原接口的单窗结果和缺失表示。


def _endpoint_reasons(prices, volumes, endpoint):  # 批量判断买入或退出端的价格、成交量可用性。
    """prices 和 volumes 是对齐的一维数组，endpoint 为 entry 或 target。

    依次判断缺失、无效和零成交，返回各分钟的首个不可用原因或 accepted。
    """
    missing = np.isnan(prices) | np.isnan(volumes)  # 价格或成交量缺失时优先记录缺失。
    invalid = ~np.isfinite(prices) | ~np.isfinite(volumes) | (prices <= 0) | (volumes < 0)  # 价格必须正且有限，成交量必须非负且有限。
    return np.select(  # 按先匹配先采用的顺序选择各分钟的状态。
        (missing, invalid, volumes == 0),  # 依次检查缺失、无效和零成交。
        (f"missing_{endpoint}", f"invalid_{endpoint}", f"zero_volume_at_{endpoint}"),  # 用端点区分事后审计原因。
        default="accepted",  # 价格有效且有成交时允许使用该端点。
    )  # 返回与输入行对齐的端点状态。


def _window_labels(bars, minutes, predictions, targets, halts):  # 批量判断标签状态并计算收益率。
    """predictions 和 targets 为对齐的预测、退出行索引，bars、minutes、halts 与 sample_session 相同。

    返回标签原因和按原行情精度计算的收益率。任一端点不可用或发生停牌时填 NaN。
    """
    entries = predictions + ENTRY_OFFSET  # 定位每个候选预测分钟之后的买入端。
    entry_prices = bars[entries, OPEN]  # 取买入端的 Qlib 开盘价。
    target_prices = bars[targets, CLOSE]  # 取退出端的 Qlib 收盘价。
    entry_reasons = _endpoint_reasons(entry_prices, bars[entries, VOLUME], "entry")  # 检查买入开盘价和该分钟成交量。
    target_reasons = _endpoint_reasons(target_prices, bars[targets, VOLUME], "target")  # 检查退出收盘价和该分钟成交量。
    reasons = np.where(entry_reasons == "accepted", target_reasons, entry_reasons)  # 两端异常时优先记录买入端原因。
    clock = np.asarray(minutes)  # 将分钟末时刻转为可索引的数组。
    halted = _overlaps_halt(clock[predictions] * 60, clock[targets] * 60, halts)  # 检查预测结束到退出结束之间的停牌。
    reasons[halted] = "halt_before_target"  # 停牌优先于买入和退出端的异常。
    accepted = reasons == "accepted"  # 仅在两端行情可用且无停牌时计算收益率。
    observed = np.log(target_prices[accepted] / entry_prices[accepted])  # 用退出价除以买入价得到对数收益。
    y = np.full(len(predictions), np.nan, dtype=observed.dtype)  # 默认将标签标记为不可用。
    y[accepted] = observed  # 将算出的收益率写到对应候选。
    return reasons, y  # 交付按预测索引排列的标签状态与收益率。


def label_for_target(bars, minutes, t, target, halts):  # 计算指定预测时点的事后收益率。
    """根据下一根分钟线的开盘价和退出分钟的收盘价计算对数收益率。

    bars 是形状为（分钟数，7）的 Qlib 分钟行情数组。
    minutes 是同一连续竞价时段的分钟末时刻，单位为从北京时间午夜起算的分钟。
    t 和 target 分别是预测分钟和退出分钟在数组中的行号。
    halts 是同一股票日已知的盘中停复牌区间，单位为从北京时间午夜起算的秒。
    返回原因和收益率。发生未来停牌或任一端点行情不可用时返回 NaN。
    """
    reasons, y = _window_labels(bars, minutes, np.array([t]), np.array([target]), halts)  # 复用两端行情检查和批量标签公式。
    reason = str(reasons[0])  # 取出这一个预测时点的标签状态。
    return reason, y[0] if reason == "accepted" else np.nan  # 保留原单条接口的收益率精度和缺失值。


def _window_has_flags(flags, starts, ends):  # 批量判断每个输入窗口是否含有指定标记。
    """接收逐分钟布尔标记、窗口起点和不含右端的窗口终点，返回逐窗口布尔值。"""
    cumulative = np.empty(len(flags) + 1, dtype=np.intp)  # 为每个分钟位置保留累计计数。
    cumulative[0] = 0  # 用零初始化第一根分钟线之前的计数。
    np.cumsum(flags, out=cumulative[1:])  # 一次累计全时段的缺失或无效分钟数。
    return cumulative[ends] != cumulative[starts]  # 两端计数不同表示窗口内存在该标记。


def _overlaps_halt(starts, ends, halts):  # 检查给定的时间窗口是否与盘中停牌重合。
    """starts、ends 为按秒配对的窗口边界，halts 为按秒计的停复牌区间。"""
    intervals = tuple(halts)  # 收集已知的停复牌区间。
    if not intervals:  # 没有已知盘中停牌时窗口都通过此项检查。
        return np.zeros(len(starts), dtype=bool)  # 返回与时间窗口等长的全假掩码。
    halt_starts, halt_ends = np.asarray(intervals).T  # 取得各次停牌开始和结束的秒数。
    return ((halt_starts[:, None] < ends) & (halt_ends[:, None] > starts)).any(axis=0)  # 严格相交才视为重合。


def _input_reasons(bars, minutes, bounds, halts, excluded):  # 批量检查各预测分钟的输入资格。
    """输入与 sample_session 相同，返回按时间排列的首个输入拒绝原因。

    输入资格合格时返回 None，表示该窗口可继续检查归一化基准。
    """
    lower_cents, upper_cents = bounds  # 取得涨跌停边界。
    missing, valid = input_quality(bars)  # 一次检查时段内每根分钟线的输入质量。
    first_prediction = WINDOW - 1  # 第一个具备完整历史的分钟行号。
    end_prediction = max(first_prediction, len(minutes) - HORIZON)  # 只保留目标仍在时段内的预测分钟。
    reasons = np.full(len(minutes), None, dtype=object)  # 预留按分钟排列的判断结果。
    reasons[:first_prediction] = "insufficient_history"  # 先标记无法累积完整窗口的分钟。
    reasons[end_prediction:] = "target_outside_session"  # 标记目标超出时段且历史已完整的分钟。
    if first_prediction == end_prediction:  # 时段过短，找不到满足两个时间边界的预测分钟。
        return reasons  # 直接返回时间条件对应的原因。
    eligible = slice(first_prediction, end_prediction)  # 定位满足时间约束的预测分钟。
    if excluded:  # 已知市场异常时，剔除该时段通过时间检查的所有窗口。
        reasons[eligible] = "market_trading_anomaly"  # 保留开头和结尾的时间原因。
        return reasons  # 排除时段不参与后续行情检查。

    predictions = np.arange(first_prediction, end_prediction)  # 得到同一时段的全部候选行号。
    starts = predictions - WINDOW + 1  # 计算每个输入窗口的首行。
    ends = predictions + 1  # 用不含右端的位置计算窗口内的累计计数。
    clock = np.asarray(minutes)  # 将分钟末时刻转为可索引的数组。
    halted = _overlaps_halt((clock[starts] - 1) * 60, clock[predictions] * 60, halts)  # 从首根分钟线开端检查到预测结束。
    missing_windows = _window_has_flags(missing, starts, ends)  # 检查窗口内是否存在缺失。
    invalid_windows = _window_has_flags(~valid, starts, ends)  # 检查窗口内是否存在无效行情。
    check_limit = ~(halted | missing_windows | invalid_windows)  # 对通过前置检查的输入做限价换算。
    price_rows = predictions[check_limit]  # 选出能够安全读取价格和换算因子的预测行。
    at_limit = np.zeros(len(predictions), dtype=bool)  # 给未到达限价检查的行保留未标记状态。
    if price_rows.size:  # 至少有一个候选需要比较限价。
        close_cents = np.floor(bars[price_rows, CLOSE] / bars[price_rows, FACTOR] * 100 + 0.5)  # 批量还原价格到分。
        at_limit[check_limit] = ~((lower_cents < close_cents) & (close_cents < upper_cents))  # 标记触及或越过限价。
    reasons[eligible] = np.select(  # 采用第一个成立的条件，保留原筛选优先级。
        (halted, missing_windows, invalid_windows, at_limit, bars[eligible, VOLUME] == 0),  # 依次检查各项规则。
        (  # 将每项条件映射为原审计原因。
            "input_overlaps_halt",  # 输入窗口碰到停牌。
            "missing_input",  # 输入窗口有缺失行情。
            "invalid_input",  # 输入窗口有无效行情。
            "close_at_price_limit",  # 预测分钟收盘价触及或越过限价。
            "zero_volume_at_prediction",  # 预测分钟没有成交。
        ),  # 完成原因定义。
        default=None,  # 通过初筛的行可以继续检查归一化基准。
    )  # 将批量结果写回其原有的分钟位置。
    return reasons  # 交付时段内全部分钟的输入判断。


@dataclass(frozen=True)  # 保存一个时段内按预测行对齐的候选与审计结果。
class SessionSamples:  # 定义时段样本的数组接口。
    """候选数组只含合格预测时点，审计原因覆盖时段内所有分钟。"""

    positions: np.ndarray  # 形状为（候选数，），按时间排列，记录候选在原时段中的分钟行索引。
    x: np.ndarray  # 形状为（候选数，30，6），存储 float32 归一化特征。
    y: np.ndarray  # 形状为（候选数，），存储 float64 对数收益率，标签不可用时为 NaN。
    label_valid: np.ndarray  # 形状为（候选数，），标记收益率是否有限。
    reasons: np.ndarray  # 形状为（分钟数，），记录各分钟的首个剔除原因或后续标签状态。


def sample_session(bars, minutes, bounds, halts, excluded=False) -> SessionSamples:  # 构造同一时段的候选数组。
    """bars 为（分钟数，7）的 Qlib 分钟行情，minutes 为从北京时间午夜起算的分钟末时刻。

    bounds 为以分计的跌停价和涨停价，halts 为从北京时间午夜起算的停复牌秒数区间。
    excluded 默认 False，表示不因市场异常排除时段。返回按时点排列的输入和事后标签，
    收益率从预测时点后第一根的开盘到第五根的收盘计算，另附逐分钟原因。
    预测输入合格而事后标签不可用时，保留输入并将标签记为 NaN。
    """
    reasons = _input_reasons(bars, minutes, bounds, halts, excluded)  # 批量判断输入窗口的资格。
    positions = np.flatnonzero(np.equal(reasons, None))  # 按时间提取通过初筛的预测行位置。
    if not positions.size:  # 没有候选时无需计算窗口和目标行情。
        return SessionSamples(  # 交付带固定维度的空候选和实际审计原因。
            positions,  # 保留空的候选行索引。
            np.empty((0, WINDOW, len(FEATURES)), dtype=np.float32),  # 保留特征数组的时间和通道维。
            np.empty(0, dtype=np.float64),  # 表示没有候选收益率。
            np.empty(0, dtype=bool),  # 表示没有待核对的候选标签。
            reasons,  # 保留被剔除分钟的原始原因。
        )  # 完成空结果的构造。

    windows = np.lib.stride_tricks.sliding_window_view(bars, WINDOW, axis=0)  # 使用 NumPy 从时段行情切出滑动窗口。
    x, valid = _normalize_windows(windows.swapaxes(1, 2)[positions - WINDOW + 1])  # 批量计算候选窗口的特征。
    if not valid.all():  # 缺少正成交股数或成交额基准时，剔除相应候选。
        reasons[positions[~valid]] = "missing_positive_median"  # 记录归一化阶段的首个剔除原因。
        positions, x = positions[valid], x[valid]  # 同步筛选时点与特征，保持逐行对齐。
    targets = positions + HORIZON  # 找到各候选之后五分钟的退出行。
    label_reasons, y = _window_labels(bars, minutes, positions, targets, halts)  # 检查候选的未来买入与退出端并计算标签。
    reasons[positions] = label_reasons  # 将事后标签状态写到对应的预测分钟。
    y = y.astype(np.float64, copy=False)  # 统一时段数组的标签精度，符合训练和审计的交付约定。
    return SessionSamples(positions, x, y, np.isfinite(y), reasons)  # 将特征、标签和审计状态交付调用方。


def read_requests(path):  # 读取并校验调用方指定的股票日。
    """接收 pathlib.Path 类型的 CSV 路径 path，返回按日期和代码排序的股票日列表。

    文件必须有 code 和 date 两列。列表为空、行格式无效或股票日重复时抛出 ValueError。
    """
    with path.open(newline="", encoding="utf-8-sig") as source:  # 兼容有或没有 BOM 的 UTF-8。
        rows = csv.DictReader(source)  # 以表头列名读取每条记录。
        if rows.fieldnames is None or not {"code", "date"}.issubset(rows.fieldnames):  # 校验表头。
            raise ValueError("Stock-day CSV requires code and date columns")  # 报出缺少必需列。
        requests = []  # 存放校验通过的（证券代码，交易日）。
        for line, row in enumerate(rows, 2):  # 行号从表头后的第二行开始。
            code, day = row["code"], row["date"]  # 取出需要处理的代码和日期。
            if code is None or not re.fullmatch(r"(?:SH|SZ|BJ)\d{6}", code):  # 校验证券代码。
                raise ValueError(f"Invalid Qlib security code on stock-day CSV line {line}")  # 报出行号。
            if day is None or date.fromisoformat(day).isoformat() != day:  # 要求规范日期格式。
                raise ValueError(f"Invalid ISO date on stock-day CSV line {line}")  # 报出错误日期所在行。
            requests.append((code, day))  # 将已校验的股票日放入清单。
    if not requests or len(set(requests)) != len(requests):  # 清单必须非空，且股票日唯一。
        raise ValueError("Provide a nonempty list of unique stock-days")  # 拒绝空清单或重复项。
    return sorted(requests, key=lambda item: (item[1], item[0]))  # 按交易日和代码稳定排序。


def stock_day_reason(library, metadata, code, day):  # 在读取分钟行情前筛选股票日。
    """返回股票日的筛选原因及用于后续构造的日级元数据。

    library 提供 Qlib 证券覆盖期，metadata 提供板块、日线和历史交易事件。
    code 是 Qlib 证券代码，day 是 YYYY-MM-DD 格式的北京时间交易日。
    超出分钟覆盖或主板范围时，元数据返回 None。必需来源缺失或冲突时传递读取错误。
    """
    if not library.has_coverage(code, day):  # 确认交易日在证券的分钟覆盖期内。
        return "outside_qlib_coverage", None  # 无需为覆盖期外请求读取日级元数据。
    if not metadata.is_mainboard(code, day):  # 按当天生效的代码和板块判断范围。
        return "outside_mainboard", None  # 研究范围只纳入符合条件的沪深主板。
    daily = metadata.day(code, day)  # 读取日线参考价、交易状态和停牌记录。
    if daily.delist:  # 退市整理期整体排除。
        return "delisting_period", daily  # 记录退市整理期状态。
    if daily.ipo:  # 新股上市初期可能适用特殊限幅。
        return "listing_rules_exclusion", daily  # 记录上市规则导致的排除。
    if daily.st:  # 风险警示股票的限幅不属于本轮口径。
        return "risk_warning_limit", daily  # 记录风险警示状态。
    if daily.full_halt:  # 官方披露该股票全天停牌。
        return "official_full_day_halt", daily  # 全天不构造样本。
    if not daily.trading and not daily.intraday_halts:  # 日线停牌且没有盘中区间可解释。
        return "unresolved_daily_status", daily  # 排除交易状态无法核实的股票日。
    return "eligible", daily  # 保留合格股票日及后续所需的元数据。


def stock_day_sessions(library, code, day, historical_code):  # 一次读取并切分两个时段。
    """按时间顺序返回某个股票日的时段名、分钟列表、行情切片和异常排除标记。

    library 是 Qlib 分钟行情读取器。code 是 Qlib 数据中的证券代码。
    day 是 YYYY-MM-DD 格式的北京时间交易日，historical_code 是该日实际使用的代码。
    返回的分钟时刻从北京时间午夜起算，行情切片形状为（时段分钟数，7）。
    读取器发现行情文件结构不符合预期时，传递其读取错误。
    """
    bars = library.session_bars(code, day, TRADING_MINUTES)  # 一次读取全天的连续竞价分钟。
    start = 0  # 当前时段首行相对于当天行情数组的位置。
    for session, minutes in SESSIONS.items():  # 先处理上午，再处理下午。
        end = start + len(minutes)  # 得到当前时段的切片结束位置。
        # 以当天实际代码判断是否属于沪市，按既定日期排除异常早盘。
        excluded = historical_code.startswith("SH") and day == "2024-09-27" and session == "morning"
        yield session, minutes, bars[start:end], excluded  # 分时段返回，不跨午休构造窗口。
        start = end  # 下一时段从当前切片结束位置开始。
