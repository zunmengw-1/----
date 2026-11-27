"""从固定的训练股票日抽样估计各预测分钟的收益率尺度。

使用正式训练批次的有效标签，按北京时间的预测分钟流式汇总样本量和总体矩。
返回单精度尺度表及可写入训练报告的原始收益率统计。
"""

from collections.abc import Iterator  # 声明缓存股票日抽样的惰性迭代接口。
from itertools import islice  # 按随机索引从缓存中读取指定股票日。
from math import isfinite  # 检查备用收益率尺度的数值有效性。

import numpy as np  # 用固定种子抽样并按日内分钟累计标签数量和矩。

from src.data.metadata import Metadata  # 声明读取股票日元数据的接口。
from src.data.minutes import MinuteLibrary  # 声明训练分钟库和股票日缓存接口。
from src.features.batches import (  # 导入训练样本的现有入口。
    iter_training_batches,  # 使用正式训练路径构造相同口径的有效标签。
)  # 完成训练取批接口导入。
from src.features.price_scale import (
    PriceScaleAccumulator,  # 在同一次训练抽样中累计价格尺度。
)
from src.features.samples import split_for  # 检查抽中的股票日都属于训练集。

_MINUTES_PER_DAY = 24 * 60  # 尺度表以北京时间午夜起算的每一分钟为索引。


def _sample_stock_days(  # 按需读取缓存中的随机股票日。
    library: MinuteLibrary,  # 已配置训练股票日缓存的 Qlib 分钟库。
    positions: np.ndarray,  # 按升序排列且不重复的缓存索引。
) -> Iterator[tuple[str, str]]:  # 每次交付所选股票代码和北京时间交易日。
    """流式跳过未抽中的索引。缓存长度不足或抽中非训练日期时抛出 ValueError。"""
    requests = iter(library.cached_stock_days())  # 从缓存重新开始遍历股票日索引。
    previous = -1  # 第一项的上一索引置为负一，便于计算要跳过的距离。
    for position in positions:  # 在单次扫描中按顺序读取所选股票日。
        request = next(islice(requests, int(position) - previous - 1, None), None)  # 跳过间隔内未抽中的股票日。
        if request is None:  # 缓存索引不足以访问所选位置。
            raise ValueError("训练股票日缓存短于声明的股票日数量")  # 避免对不完整样本计算尺度。
        if split_for(request[1]) != "train":  # 不允许验证或测试标签用于尺度估计。
            raise ValueError(f"分钟尺度的抽样日期不属于训练集：{request[1]}")  # 明确报告源范围有误。
        yield request  # 只把抽中的股票日交给正式训练取批。
        previous = int(position)  # 记录当前索引，下一次继续向后读取。


def estimate_minute_scales(  # 按预测分钟估计标签的总体标准差。
    library: MinuteLibrary,  # 提供分钟行情及训练股票日索引。
    metadata: Metadata,  # 调用方在估计期间保持打开的元数据读取器。
    stock_day_count: int,  # 已核对的缓存索引股票日数。
    *,  # 抽样及备用尺度设置按名称传入。
    seed: int,  # 固定抽样索引和统计读取顺序用的独立随机种子。
    sample_size: int = 12_288,  # 默认不放回均匀抽取的训练股票日数量。
    fallback_scale: float,  # 未能估计有效标准差时使用的全局原始对数收益率尺度。
    price_accumulator: PriceScaleAccumulator | None = None,  # 选用相同的训练窗口估计价格输入倍率。
) -> tuple[np.ndarray, list[dict[str, int | float]]]:  # 返回 float32 [1440] 尺度表和按分钟排序的原始统计。
    """仅使用正式训练标签计算总体标准差，分钟为北京时间从午夜起算的整数。

    每个有观测的分钟返回样本数、算术均值和总体标准差。无标签、只有一条标签、
    全部标签相同或标准差无法用正有限 float32 表示时使用全局尺度。
    提供 price_accumulator 时，同步更新训练窗口的一二阶矩供估计价格输入倍率。
    抽样参数或尺度无效、缓存范围与训练集冲突时抛出 ValueError。
    """
    if stock_day_count <= 0 or sample_size <= 0:  # 至少需要一个可抽样的训练股票日。
        raise ValueError("分钟尺度的缓存股票日数量和抽样数量必须为正")  # 明确指定无效的范围。
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):  # 单独核对全局尺度转换后的有效性。
        fallback = np.float32(fallback_scale)  # 以模型实际使用的精度保存备用尺度。
    if not isfinite(fallback) or fallback <= 0:  # 全局尺度必须能表示成正有限单精度数。
        raise ValueError("分钟尺度的备用值必须为单精度可表示的正有限数")  # 防止归一化除以无效数值。

    rng = np.random.default_rng(seed)  # 使用独立随机状态，避免影响正式训练顺序。
    positions = np.sort(rng.choice(stock_day_count, size=min(sample_size, stock_day_count), replace=False))  # 不放回抽样并按缓存索引排序。
    requests = _sample_stock_days(library, positions)  # 保留惰性读取，只向训练取批交付抽样请求。
    counts = np.zeros(_MINUTES_PER_DAY, dtype=np.int64)  # 记录每个预测分钟收到的标签条数。
    sums = np.zeros(_MINUTES_PER_DAY, dtype=np.float64)  # 双精度累计标签的一阶矩。
    squares = np.zeros_like(sums)  # 双精度累计标签平方和。
    minima = np.full_like(sums, np.inf)  # 识别全部标签都相同的分钟，避免舍入误差引入假波动。
    maxima = np.full_like(sums, -np.inf)  # 对每个预测分钟同步维护标签最大值。
    for batch in iter_training_batches(library, metadata, requests, rng=rng):  # 复用正式训练样本口径，只处理抽样股票日。
        minutes = batch.t.astype(np.int64) % _MINUTES_PER_DAY  # 将北京时间分钟戳换算成日期内索引。
        if price_accumulator is not None:  # 抽样窗口已有所需的窗内归一化量价。
            price_accumulator.update(batch.x, minutes)  # 使用相同批次拟合输入尺度，无须重新建样。
        counts += np.bincount(minutes, minlength=_MINUTES_PER_DAY)  # 统计该批标签分布在各分钟的数量。
        sums += np.bincount(minutes, weights=batch.y, minlength=_MINUTES_PER_DAY)  # 将每条有效收益归入对应分钟。
        squares += np.bincount(minutes, weights=np.square(batch.y), minlength=_MINUTES_PER_DAY)  # 将每条收益的平方归入对应分钟。
        np.minimum.at(minima, minutes, batch.y)  # 更新每个分钟的最小有效收益率。
        np.maximum.at(maxima, minutes, batch.y)  # 更新每个分钟的最大有效收益率。

    observed = np.flatnonzero(counts)  # 只报告至少有一个有效标签的分钟。
    means = sums[observed] / counts[observed]  # 以每个分钟自身样本数计算算术均值。
    variances = squares[observed] / counts[observed] - np.square(means)  # 根据二阶矩与均值计算总体方差。
    variances[minima[observed] == maxima[observed]] = 0.0  # 全部标签同值时总体方差为零。
    stds = np.sqrt(np.maximum(variances, 0.0))  # 忽略舍入可能引入的微小负方差。
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):  # 检查标准差转换成模型尺度后的表示。
        observed_scales = stds.astype(np.float32)  # 将观测标准差转换到实际训练精度。
    usable = (counts[observed] >= 2) & np.isfinite(observed_scales) & (observed_scales > 0)  # 仅使用可表示的正标准差。
    scales = np.full(_MINUTES_PER_DAY, fallback, dtype=np.float32)  # 覆盖未观测或样本太少的预测分钟。
    scales[observed[usable]] = observed_scales[usable]  # 为有有效标准差的分钟写入对应尺度。
    stats = [  # 保存可序列化的原始统计，便于复验训练使用的尺度。
        {  # 一个观测到的预测分钟对应一条统计记录。
            "minute": int(minute),  # 北京时间从午夜起算的分钟数。
            "samples": int(counts[minute]),  # 对应分钟的有限标签总数。
            "mean": float(mean),  # 原始对数收益率均值。
            "std": float(std),  # 原始对数收益率总体标准差。
        }  # 完成一个预测分钟的统计记录。
        for minute, mean, std in zip(observed, means, stds, strict=True)  # 按分钟升序对应均值与标准差。
    ]  # 完成尺度估计的统计报告。
    return scales, stats  # 将模型使用的尺度表与原始统计分别交给训练入口。
