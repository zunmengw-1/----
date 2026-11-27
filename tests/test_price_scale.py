"""检验按窗口末端分钟拟合的固定价格倍率。"""

import json  # 验证统计可写入标准 JSON。
from math import sqrt  # 用手算方差得到预期尺度。

import numpy as np  # 构造值与索引已知的分钟窗口。
import pytest  # 检查无效输入和无法拟合的情况。

from src.features.price_scale import (  # 导入正式价格倍率拟合入口。
    PRICE_CHANNELS,  # 取出价格通道与成交通道的分界。
    PriceScaleAccumulator,  # 使用逐批拟合器构造倍率表。
)  # 完成价格倍率接口导入。
from src.features.samples import FEATURES, WINDOW  # 使用项目固定的量价顺序和窗口长度。


def test_price_scale_matches_pooled_channel_variance_by_prediction_minute():  # 检查分钟分组和组内方差汇总。
    """以不同价量尺度的上午、下午窗口核对倍率、分组统计和无观测分钟的回退。"""
    alternating = np.tile([-1.0, 1.0], WINDOW // 2)[:, None]  # 让每列每个窗口内的总体方差便于手算。
    amplitudes = np.array([1, 2, 3, 4, 3, 4], dtype=np.float32)  # 六列对应标准差分别为给定振幅。
    afternoon_amplitudes = amplitudes * np.array([2, 2, 2, 2, 1, 1])  # 下午四列价格的尺度加倍。
    offsets = np.array([2, 3, 4, 5, 10, 20])  # 让每列均值不同，检验先逐列计算方差。
    morning = (alternating * amplitudes + offsets).astype(np.float32)  # 构造上午归一化后的六列量价。
    afternoon = (alternating * afternoon_amplitudes + offsets).astype(np.float32)  # 构造下午归一化后的六列量价。
    accumulator = PriceScaleAccumulator()  # 初始化跨批次的分钟统计。
    accumulator.update(np.stack([afternoon, morning]), np.array([810, 600]))  # 先打乱分钟顺序并交付两条样本。
    accumulator.update(morning[None, :, :], np.array([600]))  # 让同一预测分钟在另一批次继续累加。
    estimate = accumulator.finish()  # 生成实际模型可用的倍率表。
    assert PRICE_CHANNELS == 4 and FEATURES[PRICE_CHANNELS] == "volume"  # 价格四列在量额之前。
    assert estimate.multipliers.shape == (24 * 60,) and estimate.multipliers.dtype == np.float32  # 核对输出布局与精度。
    np.testing.assert_allclose(estimate.multipliers[[600, 810, 0]], [sqrt(5 / 3), sqrt(5 / 3) / 2, sqrt(5 / 6)])  # 上午、下午和回退的倍率均可手算。
    assert estimate.global_multiplier == pytest.approx(sqrt(5 / 6))  # 全局倍率按全部窗口汇总。
    assert [(row["minute"], row["windows"]) for row in estimate.statistics] == [(600, 2), (810, 1)]  # 只报告有训练窗口的分钟。
    assert estimate.statistics[0]["price_rms_std"] == pytest.approx(sqrt(7.5))  # 先逐列算方差再求组尺度。
    assert estimate.statistics[0]["volume_amount_rms_std"] == pytest.approx(sqrt(12.5))  # 独立核对量额尺度。
    json.dumps(estimate.statistics, allow_nan=False)  # 确认报告字段可交给标准 JSON 写入。


def test_price_scale_falls_back_for_observed_minute_with_constant_prices():  # 检查有窗口但价量比不可估计时的回退。
    """下午的四列价格保持不变时，使用全部训练窗口的有效倍率。"""
    alternating = np.tile([-1.0, 1.0], WINDOW // 2)[:, None]  # 每列共有正负相等的两种偏移。
    offsets = np.array([0.125, -0.25, 0.5, 0.0625, 2, 2], dtype=np.float32)  # 价格可以围绕不同的基准偏移。
    active = (alternating * np.array([1, 1, 1, 1, 2, 2]) + offsets).astype(np.float32)  # 上午两个组的尺度都为正。
    flat = active.copy()  # 从上午窗口构造只有价格不变的下午窗口。
    flat[:, :PRICE_CHANNELS] = offsets[:PRICE_CHANNELS]  # 下午各价格通道分别恒定。
    accumulator = PriceScaleAccumulator()  # 汇总两个时点的训练窗口。
    accumulator.update(np.stack([flat, active]), np.array([810, 600]))  # 保留两个时点实际计数。
    estimate = accumulator.finish()  # 从有效总体计算回退值。
    assert estimate.multipliers[600] == pytest.approx(2.0)  # 上午量额尺度是价格的两倍。
    assert estimate.multipliers[810] == pytest.approx(2 * sqrt(2))  # 下午使用两组合并后的方差比。
    assert estimate.statistics[1]["price_rms_std"] == 0  # 观察到的常量价格组尺度明确写为零。
    assert estimate.statistics[1]["price_multiplier"] == pytest.approx(estimate.global_multiplier)  # 报告与回退表一致。
    json.dumps(estimate.statistics, allow_nan=False)  # 零价差仍应生成标准 JSON。


@pytest.mark.parametrize("minute", [-1, 1440, 600.5])  # 检查时点超界和非整数三种情况。
def test_price_scale_rejects_invalid_minute(minute):  # 避免错误地写入日内倍率表。
    """以一条合规窗口验证非法预测分钟会被拒绝。"""
    accumulator = PriceScaleAccumulator()  # 初始化未接收窗口的统计器。
    window = np.zeros((1, WINDOW, len(FEATURES)), dtype=np.float32)  # 构造形状正确的输入。
    with pytest.raises(ValueError, match="预测分钟"):  # 报错指出问题属于时钟分钟。
        accumulator.update(window, np.array([minute]))  # 请求非法分钟。


def test_price_scale_rejects_unusable_global_scale():  # 检查缺失全局回退时是否明确失败。
    """没有训练窗口或所有训练特征恒定时，均无法生成有效倍率。"""
    accumulator = PriceScaleAccumulator()  # 还未交付窗口。
    with pytest.raises(ValueError, match="训练"):  # 空总体无法估计。
        accumulator.finish()  # 请求空数据的倍率表。
    window = np.full((1, WINDOW, len(FEATURES)), np.float32(0.00337))  # 模拟浮点值恒定的六列特征。
    accumulator.update(window, np.array([600]))  # 整个训练样本只有一个常量窗口。
    with pytest.raises(ValueError, match="总体"):  # 总体量价都没有可匹配的波动尺度。
        accumulator.finish()  # 拒绝生成看似可用的倍率。


def _group_reference(values):  # 直接从实际变换后的数组求两组通道的标准差。
    """values 包含同一预测分钟或整个训练样本的 [窗口, 位置, 通道]。"""
    variances = np.var(values, axis=(0, 1), dtype=np.float64, ddof=0)  # 每列合并窗口及位置计算总体方差。
    return np.sqrt(variances[:PRICE_CHANNELS].mean()), np.sqrt(variances[PRICE_CHANNELS:].mean())  # 返回价格尺度与量额尺度。


def test_centered_price_scale_matches_explicit_training_residuals():  # 用直接减均值的参考算法核对分时和全天统计。
    """不平衡的窗口数、打散的分钟及不同的六列走势共同检验按 t 和 j 取数的口径。"""
    minutes = np.array([810, 600, 600, 810, 600])  # 同一预测分钟可跨越多个取样批次。
    rng = np.random.default_rng(281)  # 固定少量样本的扰动以便复验。
    slopes = np.array([1, 3, 0.5, 1, 8, 13])  # 不同位置上各价格与成交列的均值曲线刻意不同。
    position = np.arange(WINDOW)[None, :, None] / WINDOW  # 给窗口内每个位置不同的确定性走势。
    windows = (rng.normal(size=(len(minutes), WINDOW, len(FEATURES))) + position * slopes + (minutes - 600)[:, None, None] * 0.01 * slopes).astype(np.float32)  # 形成两组均值和波动均可分辨的窗口。
    centered_fit = PriceScaleAccumulator(center_positions=True)  # 开启正式逐位置去均值口径。
    raw_fit = PriceScaleAccumulator()  # 同批原输入同时按原有基线估计。
    for batch in (slice(0, 2), slice(2, None)):  # 将相同分钟拆到不同批次。
        centered_fit.update(windows[batch], minutes[batch])  # 用拟合器累计中心和随后计算倍率所需的矩。
        raw_fit.update(windows[batch], minutes[batch])  # 基线继续在窗口输入上估计。
    result, raw = centered_fit.finish(), raw_fit.finish()  # 获得训练入口将接收的两套估计。
    assert raw.position_means is None  # 不去均值的基线没有位置均值表。
    centers = result.position_means  # 检查均值表可按分钟及窗口位置直接索引。
    assert centers.shape == (1440, WINDOW, 3) and centers.dtype == np.float32  # 实际模型只需要收盘价、股数和成交额三列。
    np.testing.assert_array_equal(centers[601], windows[:, :, 3:].mean(axis=0, dtype=np.float64).astype(np.float32))  # 未观测分钟的位置均值按全部窗口的数量加权。
    residuals = np.empty_like(windows)  # 直接构造所有实际去均值后的模型输入作为参考。
    for row in result.statistics:  # 用拟合器报告的每个分钟独立复核计算。
        minute = row["minute"]  # 以预测分钟区分两组不同曲线。
        selected = minutes == minute  # 选择该预测时刻的所有窗口。
        means = windows[selected, :, 3:].mean(axis=0, dtype=np.float64).astype(np.float32)  # 跨样本求固定每个位置三列的均值。
        np.testing.assert_array_equal(centers[minute], means)  # 核对入模时使用的单精度均值。
        residuals[selected] = windows[selected] - means[:, [0, 0, 0, 0, 1, 2]]  # 四价共减相对收盘均值，量额分别减各自均值。
        price, turnover = _group_reference(residuals[selected])  # 直接对变换后输入计算两组总体尺度。
        np.testing.assert_allclose([row["price_rms_std"], row["volume_amount_rms_std"], result.multipliers[minute]], [price, turnover, turnover / price], rtol=1e-6)  # 检查报告原始尺度和模型查表倍率。
        raw_price, raw_turnover = _group_reference(windows[selected])  # 在原输入上独立求基线倍率。
        np.testing.assert_allclose(raw.multipliers[minute], raw_turnover / raw_price, rtol=1e-6)  # 保持原口径与变更前一致。
    price, turnover = _group_reference(residuals)  # 合并已经按各自分钟去均值后的所有窗口。
    np.testing.assert_allclose([result.global_multiplier, result.multipliers[601]], turnover / price, rtol=1e-6)  # 全天倍率及未观测分钟的回退来自同一组残差。


def test_centered_scale_fallback_after_deterministic_position_curve_removed():  # 检查去掉均值曲线后恰好零波动的时点。
    """分钟内行情完全由位置决定时，共同价格走势抵消，量额各自固定曲线也抵消。"""
    positions = np.arange(WINDOW, dtype=np.float32)[:, None]  # 为全部列构造随位置变化的均值走势。
    curve = positions * np.array([1, 1, 1, 1, 2, 3], dtype=np.float32)  # 四列价格共享曲线，成交量额用各自曲线。
    fit = PriceScaleAccumulator(center_positions=True)  # 使用去均值后的口径。
    fit.update(curve[None], np.array([810]))  # 下午只有这一组确定性的窗口位置曲线。
    with pytest.raises(ValueError):  # 去均值后全天价格及量额标准差均为零。
        fit.finish()  # 此时还无法计算有效的全局倍率。
    offset = np.array([1, 1, 1, 1, 2, 2], dtype=np.float32)  # 让上午价格及成交相对均值分别出现一倍和两倍偏差。
    fit.update(np.stack([curve + offset, curve - offset]), np.array([600, 600]))  # 增加能拟合有效倍率的第二个分钟。
    estimate = fit.finish()  # 此时全天拥有正有限的回退尺度。
    afternoon = estimate.statistics[1]  # 取出按分钟排序的下午分组。
    assert afternoon["minute"] == 810 and afternoon["uses_global_fallback"]  # 确定性曲线所在时点使用了全天倍率。
    assert afternoon["price_rms_std"] == afternoon["volume_amount_rms_std"] == 0  # 均值曲线去除后两组确实都无剩余方差。
    np.testing.assert_array_equal(estimate.multipliers[[600, 810, 601]], np.full(3, 2.0))  # 上午、下午和无观测分钟均可手算得两倍。
