"""按预测分钟汇总训练窗口的量价数值尺度，拟合价格通道的固定倍率。

输入为已经逐窗归一化的六列量价。倍率以窗口末端的北京时间分钟为索引，
每个窗口的四个价格通道共用同一个倍率。启用逐位置去均值时，在同一遍训练
取样中拟合收盘价、量、额的均值表，并根据逐位置去均值后的离散程度拟合倍率。
"""

from dataclasses import dataclass  # 保存拟合出来的倍率表和统计记录。

import numpy as np  # 以双精度流式累计按分钟分组的总体矩。

from src.features.samples import FEATURES, WINDOW  # 复用量价列顺序和输入窗口长度。

PRICE_CHANNELS = FEATURES.index("volume")  # 价格通道排在成交股数及成交额之前。
_MINUTES_PER_DAY = 24 * 60  # 倍率表按自北京时间午夜起的分钟索引。
_CLOSE = FEATURES.index("close")  # 位置均值表从收盘价列开始，后续两列依次是量和额。


@dataclass(frozen=True)  # 将固定倍率及核查信息作为一次拟合的结果交付。
class PriceScaleEstimate:  # 表示用训练窗口拟合出来的价格通道倍率。
    """倍率表以北京时间预测分钟索引，无可用分时估计的位置使用全局倍率。"""

    multipliers: np.ndarray  # 结构为 [1440] 的正有限 float32 倍率。
    global_multiplier: float  # 全训练样本合并计算的倍率。
    statistics: list[dict[str, int | float | bool | None]]  # 实际观测分钟的可序列化尺度统计。
    position_means: np.ndarray | None = None  # 去均值时提供 float32 [1440,30,3] 的收盘价、量、额均值表。


def _channel_scales(variance):  # 按通道分组，把各列总体方差汇总为量价两组尺度。
    """variance 最后一维按 FEATURES 顺序存储各通道的总体方差。"""
    price = np.sqrt(variance[..., :PRICE_CHANNELS].mean(axis=-1))  # 对四个价格方差先平均再开根号。
    turnover = np.sqrt(variance[..., PRICE_CHANNELS:].mean(axis=-1))  # 对成交量和成交额的方差做同样汇总。
    return price, turnover  # 交付两组尺度供计算倍率及记录原始统计。


def _group_scales(count, sums, squares, constant):  # 由每列的总体矩求价格组和量额组的尺度。
    """count 为窗口数，sums 和 squares 按最后一维存量价列，constant 标记各列是否恒定。"""
    bars = np.asarray(count)[..., None] * WINDOW  # 每列在每个分组中有窗口数乘以 30 个观测值。
    with np.errstate(over="ignore", invalid="ignore"):  # 后续统一拒绝浮点溢出的尺度。
        variance = squares / bars - np.square(sums / bars)  # 每列以全分组均值为基准计算总体方差。
        variance[constant] = 0.0  # 全部值相同的列显式置零，避免舍入制造假波动。
        np.maximum(variance, 0.0, out=variance)  # 将普通舍入导致的负方差截为零。
        return _channel_scales(variance)  # 两种特征口径使用相同的价量分组和汇总方式。


class _PositionMoments:  # 保存拟合位置均值和去均值后尺度所需的统计量。
    """每个分钟和位置累计六列均值，每个分钟另外累计六列在各位置组内的离均差平方和。"""

    def __init__(self) -> None:  # 分配固定大小的统计表，无须保留历史窗口。
        self.means = np.zeros((_MINUTES_PER_DAY, WINDOW, len(FEATURES)), dtype=np.float64)  # 各预测分钟、各位置的量价样本均值。
        self.within_ss = np.zeros((_MINUTES_PER_DAY, len(FEATURES)), dtype=np.float64)  # 六列在 30 个位置各自均值周围的离差平方和之和。

    def update(self, windows, minutes, previous_counts):  # 把一批训练窗口的分组均值和离差合并进历史记录。
        """windows 为非空的 [B,30,6]，minutes 为其日内分钟，previous_counts 为本批之前各分钟的样本数。"""
        order = np.argsort(minutes, kind="stable")  # 将相同分钟排成连续片段以便分组求和。
        observed, starts, n = np.unique(minutes[order], return_index=True, return_counts=True)  # 找出每个非空分钟的起点和窗口数。
        values = windows[order].astype(np.float64, copy=False)  # 独立的批次副本可原地改写，双精度保留细小的价差。
        anchors = values[starts]  # 每组第一条窗口仅作为数值求和的参考点。
        values -= np.repeat(anchors, n, axis=0)  # 将常量组精确变为零，并避免对很大的原始值直接求平方。
        offsets = np.add.reduceat(values, starts, axis=0) / n[:, None, None]  # 得到各位置的批内均值相对参考点的偏差。
        batch_means = anchors + offsets  # 还原每个位置的实际批内均值。
        values -= np.repeat(offsets, n, axis=0)  # 各窗口减去同分钟、同位置、同列的批内均值。
        batch_ss = np.add.reduceat(np.einsum("bjc,bjc->bc", values, values), starts, axis=0)  # 累加各位置围绕其自身均值的平方差。
        old_n = previous_counts[observed]  # 取得相同分钟在已处理批次中的样本数。
        share = n / (old_n + n)  # 新批次在合并后各分钟窗口数中的占比。
        delta = batch_means - self.means[observed]  # 两批样本的逐位置均值差。
        self.within_ss[observed] += batch_ss + (old_n * share)[:, None] * np.einsum("mjc,mjc->mc", delta, delta)  # 合并平方离差时补入批间均值差的贡献。
        self.means[observed] += share[:, None, None] * delta  # 更新每个分钟和位置六列的双精度均值。

    def fit(self, observed, counts):  # 生成均值表并从各位置统计量算出减去该表后的尺度。
        """observed 是非空分钟，counts 与其对齐；返回 float32 均值表、全天两组尺度和逐分钟两组尺度。"""
        n = counts[:, None]  # 各分钟的训练窗口数，也等于该分钟任一位置的观测数。
        means = self.means[observed]  # 按非空分钟取出 30 个位置各列的均值。
        centers = means[:, :, _CLOSE:]  # 实际部署只需收盘价、成交量、成交额三列均值。
        fallback = np.average(centers, axis=0, weights=counts)  # 未观测分钟按所有训练窗口加权使用全天各位置的均值。
        table = np.empty((_MINUTES_PER_DAY, WINDOW, centers.shape[-1]), dtype=np.float32)  # 表结构和精度与模型缓冲区相同。
        table[:] = fallback  # 先将全天各位置的三列均值填入每日全部分钟。
        table[observed] = centers  # 已观测分钟优先使用各自的均值，并转为预测时使用的精度。
        if not np.isfinite(table).all():  # 均值表必须能表示为可直接用于训练和预测的单精度值。
            raise ValueError("训练窗口的位置均值无法生成有限的单精度表")  # 明确指出无效的是中心而非价格尺度。
        applied = table[observed]  # 以模型实际保存的单精度中心计算残差的均值。
        means[:, :, :PRICE_CHANNELS] -= applied[:, :, :1]  # 四个价格通道共用相对收盘价的中心。
        means[:, :, PRICE_CHANNELS:] -= applied[:, :, 1:]  # 成交量与成交额分别减去自己的中心。
        residual_means = means.mean(axis=1)  # 每个分钟合并 30 个位置后每列的残差均值。
        means -= residual_means[:, None, :]  # 取得各位置残差均值相对于本分钟均值的偏差。
        minute_ss = self.within_ss[observed] + n * np.einsum("mjc,mjc->mc", means, means)  # 每分钟总体离差包含位置内波动和位置间均值差异。
        bar_counts = n * WINDOW  # 每分钟每列共有窗口数乘以窗口长度个观测。
        minute_scales = _channel_scales(minute_ss / bar_counts)  # 从去均值后各列的分钟总体方差求两组尺度。
        day_mean = np.average(residual_means, axis=0, weights=counts)  # 跨分钟汇总残差均值时按实际样本数加权。
        day_ss = minute_ss.sum(axis=0) + (bar_counts * np.square(residual_means - day_mean)).sum(axis=0)  # 合并各分钟离差并计入分钟间均值差。
        day_scales = _channel_scales(day_ss / bar_counts.sum())  # 全天回退倍率也从已按各自分钟去均值的残差估计。
        return table, day_scales, minute_scales  # 所有返回值都来自同一组训练窗口。


class PriceScaleAccumulator:  # 跨批次累加量价尺度所需的统计量。
    """只保存每个预测分钟的计数和矩，构造时无需读取训练行情。"""

    def __init__(self, *, center_positions: bool = False) -> None:  # 根据特征口径分配相应的训练统计量。
        """center_positions 为真时拟合逐位置均值并重估残差倍率，默认保留窗内归一化直接估倍率的口径。"""
        self._positions = _PositionMoments() if center_positions else None  # 按需要分配固定大小的位置统计表。
        self._counts = np.zeros(_MINUTES_PER_DAY, dtype=np.int64)  # 记录每个预测时钟分钟的窗口数。
        self._sums = np.zeros((_MINUTES_PER_DAY, len(FEATURES)), dtype=np.float64)  # 存每个时点每列的总和。
        self._squares = np.zeros_like(self._sums)  # 存每个时点每列的平方和。
        self._minimums = np.full_like(self._sums, np.inf)  # 记录每列最小值，帮助判断是否恒定。
        self._maximums = np.full_like(self._sums, -np.inf)  # 记录每列最大值，帮助判断是否恒定。

    def update(self, windows: np.ndarray, minutes: np.ndarray) -> None:  # 汇总一批已按窗归一化的训练输入。
        """windows 为有限实数的 [B, 30, 6]，minutes 为与窗口逐行对齐的北京时间预测分钟整数。

        遇到不符形状、非有限特征、越界或非整数分钟时抛出 ValueError。允许空批次。
        """
        if windows.ndim != 3 or windows.shape[1:] != (WINDOW, len(FEATURES)):  # 检查窗口的时间和特征维。
            raise ValueError("价格倍率拟合需要形状为 [B, 30, 6] 的量价窗口")  # 报告不能汇总的输入布局。
        if minutes.shape != (len(windows),) or not np.issubdtype(minutes.dtype, np.integer):  # 确保索引与窗口逐条对应。
            raise ValueError("价格倍率拟合的预测分钟须为与窗口对应的一维整数数组")  # 拒绝形状或类型错误。
        if ((minutes < 0) | (minutes >= _MINUTES_PER_DAY)).any():  # 防止错误分钟进入查找表。
            raise ValueError("价格倍率拟合的预测分钟须在 0 至 1439 内")  # 报告有效的分钟范围。
        if windows.dtype.kind not in "fiu" or not np.isfinite(windows).all():  # 拒绝无法计算总体矩的特征值。
            raise ValueError("价格倍率拟合只接受有限实数量价窗口")  # 告诉调用方需先校验训练输入。
        indices = minutes.astype(np.intp, copy=False)  # 将索引转换成 bincount 支持的整数类型。
        if not len(indices):  # 空批次不会更新任一统计量。
            return  # 与未调用本次更新等效。
        if self._positions is not None:  # 逐位置去均值需要先保留各位置组内的离差。
            self._positions.update(windows, indices, self._counts)  # 根据本批之前的样本数合并训练统计。
            self._counts += np.bincount(indices, minlength=_MINUTES_PER_DAY)  # 增加本批窗口数供后续批次使用。
            return  # 已经取得重新估倍率所需的全部统计。
        with np.errstate(over="ignore", invalid="ignore"):  # 若输入数值过大，在拟合完成时拒绝坏的总体矩。
            window_sums = windows.sum(axis=1, dtype=np.float64)  # 在每个窗口内独立累加各列的 30 个值。
            window_squares = np.einsum("btc,btc->bc", windows, windows, dtype=np.float64, casting="unsafe")  # 在双精度下计算每个窗口各列的平方和。
            self._counts += np.bincount(indices, minlength=_MINUTES_PER_DAY)  # 按预测分钟累计窗口数。
            for channel in range(len(FEATURES)):  # 每次给一个量价通道累计按分钟分组的矩。
                self._sums[:, channel] += np.bincount(indices, weights=window_sums[:, channel], minlength=_MINUTES_PER_DAY)  # 累加该列的一阶矩。
                self._squares[:, channel] += np.bincount(indices, weights=window_squares[:, channel], minlength=_MINUTES_PER_DAY)  # 累加该列的二阶矩。
        np.minimum.at(self._minimums, indices, windows.min(axis=1))  # 对重复分钟逐次更新每列最小值。
        np.maximum.at(self._maximums, indices, windows.max(axis=1))  # 对重复分钟逐次更新每列最大值。

    def _raw_scales(self, observed):  # 从原始训练矩生成未启用去均值时的两组尺度。
        """observed 为有有效训练窗口的分钟，返回全天和逐分钟尺度。"""
        day_scales = _group_scales(  # 合并所有预测分钟得到一份总体尺度。
            self._counts.sum(),  # 使用实际累计的窗口数。
            self._sums.sum(axis=0),  # 跨所有分钟汇总每列的总和。
            self._squares.sum(axis=0),  # 跨所有分钟汇总每列的平方和。
            self._minimums.min(axis=0) == self._maximums.max(axis=0),  # 识别跨所有分钟恒定的列。
        )  # 完成全局两组尺度的估计。
        minute_scales = _group_scales(  # 对每一个实际观测分钟估计两组尺度。
            self._counts[observed],  # 各分钟按自己的窗口数折算总 bar 数。
            self._sums[observed],  # 取各分钟各列的一阶矩。
            self._squares[observed],  # 取各分钟各列的二阶矩。
            self._minimums[observed] == self._maximums[observed],  # 显式识别各分钟内保持恒定的列。
        )  # 完成分时尺度估计。
        return day_scales, minute_scales  # 保持原有倍率的尺度口径。

    def finish(self) -> PriceScaleEstimate:  # 生成固定的分时倍率和总体回退值。
        """各组尺度为组内总体方差的平均值开根，倍率为量额组尺度除以价格组尺度。

        没有训练窗口或无法拟合全局倍率时抛出 ValueError。未观测分钟以及倍率不能
        表示为正有限 float32 的分钟使用全局倍率。
        """
        observed = np.flatnonzero(self._counts)  # 只报告真正收到训练窗口的预测分钟。
        if not len(observed):  # 无样本时没有可计算的标准差。
            raise ValueError("价格倍率拟合未收到训练窗口")  # 避免空统计产生无效倍率。
        table = None  # 旧口径保持只输出倍率。
        if self._positions is None:  # 未启用去均值时复用原始窗口方差。
            day_scales, minute_scales = self._raw_scales(observed)  # 得到原口径的总体与逐分钟尺度。
        else:  # 新口径使用实际写入单精度表的中心来拟合倍率。
            table, day_scales, minute_scales = self._positions.fit(observed, self._counts[observed])  # 在训练统计完成后一次性生成配套均值与尺度。
        price, turnover = day_scales  # 全天尺度用于计算没有可用分时统计时的回退倍率。
        price_stds, turnover_stds = minute_scales  # 逐分钟尺度用于生成有效倍率及报告。
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):  # 统一处理零分母或单精度超界。
            global_multiplier = float(turnover / price)  # 保留全局倍率的双精度估计。
            fallback = np.float32(global_multiplier)  # 核对回退值能按实际模型精度使用。
        if not np.isfinite(fallback) or fallback <= 0:  # 全局回退必须严格为正且有限。
            raise ValueError("训练窗口的量价总体尺度无法生成正有限的单精度倍率")  # 拒绝拟合失败的总体。
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):  # 拟合失败的单个分钟走全局回退。
            ratios = turnover_stds / price_stds  # 先在双精度计算量额组与价格组的尺度比。
            scale_values = ratios.astype(np.float32)  # 转成实际模型查表使用的单精度。
        usable = np.isfinite(scale_values) & (scale_values > 0)  # 识别可以保存并应用的时点倍率。
        multipliers = np.full(_MINUTES_PER_DAY, fallback, dtype=np.float32)  # 缺失或不可用的时点默认使用全局倍率。
        multipliers[observed[usable]] = scale_values[usable]  # 将各时点自己的可用倍率填入查找表。
        statistics = [  # 每行保存训练窗口数及一组可复核的尺度统计。
            {  # 保存分钟索引、两个尺度、实际采用的倍率及回退状态。
                "minute": int(minute),  # 使用与推理查表一致的北京时间分钟索引。
                "windows": int(self._counts[minute]),  # 记录该分钟实际收到的窗口数。
                "price_rms_std": float(price_std) if np.isfinite(price_std) else None,  # 无法表示的尺度在 JSON 里写为缺失。
                "volume_amount_rms_std": float(turnover_std) if np.isfinite(turnover_std) else None,  # 记录量额组的未缩放尺度。
                "price_multiplier": float(ratio) if valid else global_multiplier,  # 保留有效倍率或回退值的双精度估计。
                "uses_global_fallback": not bool(valid),  # 标出由于本分钟尺度不可用而使用全局值。
            }  # 完成本分钟的统计。
            for minute, price_std, turnover_std, ratio, valid in zip(observed, price_stds, turnover_stds, ratios, usable, strict=True)  # 将各数组按分钟对齐。
        ]  # 完成可直接写入训练报告的统计表。
        return PriceScaleEstimate(multipliers, global_multiplier, statistics, table)  # 交付模型使用的倍率及可选位置均值表。
