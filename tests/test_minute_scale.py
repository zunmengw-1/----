"""验证按训练股票日抽样得到的分钟级收益率尺度及备用值。"""

import json  # 检查原始统计可以写入标准 JSON。
from types import SimpleNamespace  # 模拟缓存索引与训练批次所需的少量字段。

import numpy as np  # 构造固定的标签、时刻和预期抽样索引。
import pytest  # 比较浮点统计并核对无效数据源的异常。

from src.models import minute_scale  # 调用正式分钟尺度估计入口。


def test_estimate_minute_scales_uses_seeded_cache_sample_and_population_std(monkeypatch):  # 同时检查抽样和统计口径。
    """以已知的缓存顺序和跨日期标签替换读取接口，验证抽样、总体标准差及备用尺度。"""
    cache = [(f"SH{600000 + index}", "2024-06-03") for index in range(12)]  # 准备一个有序的训练股票日索引。
    library = SimpleNamespace(cached_stock_days=lambda: iter(cache))  # 提供每次重新遍历的只读缓存接口。
    recorded = []  # 保存实际交给正式取批入口的抽样股票日。

    def training_batches(_library, _metadata, requests, *, rng):  # 暴露送入训练批次的请求并返回可手算的标签。
        """接收原读取器与抽样请求，为两个日期构造原始物理收益率和预测时点。"""
        recorded.extend(requests)  # 按实际发送顺序收集抽中的股票日。
        yield SimpleNamespace(  # 第一批含有四个预测分钟。
            y=np.array((0.001, 0.002, 0.006, 0.001)),  # 提供四个原始对数收益率。
            t=np.array(("2024-06-03T10:00", "2024-06-03T10:01", "2024-06-03T10:02", "2024-06-03T13:30"), dtype="datetime64[m]"),  # 记录上午与下午预测时点。
        )  # 完成首个合成训练批次。
        yield SimpleNamespace(  # 第二批位于另一交易日。
            y=np.array((0.003, 0.006, 0.005)),  # 对应三个第一批出现过的预测分钟。
            t=np.array(("2024-06-04T10:00", "2024-06-04T10:02", "2024-06-04T13:30"), dtype="datetime64[m]"),  # 使用同一日内时刻检查跨日汇总。
        )  # 完成第二个合成训练批次。

    monkeypatch.setattr(minute_scale, "iter_training_batches", training_batches)  # 观察正式取批入口而保留真实估计逻辑。
    scales, stats = minute_scale.estimate_minute_scales(  # 从该缓存固定抽取五个训练股票日。
        library, None, len(cache), seed=7, sample_size=5, fallback_scale=0.004,  # 传入实际缓存长度及备用尺度。
    )  # 完成分钟尺度估计。
    indices = sorted(np.random.default_rng(7).choice(len(cache), size=5, replace=False))  # 独立算出固定种子的不放回抽样索引。
    assert recorded == [cache[index] for index in indices]  # 只有抽中的股票日交给训练取批器。
    assert scales.shape == (1440,) and scales.dtype == np.float32  # 模型可用整数分钟索引单精度表。
    np.testing.assert_allclose(scales[[0, 600, 601, 602, 810, 1439]], [0.004, 0.001, 0.004, 0.004, 0.002, 0.004])  # 未观测、单样本和零方差使用备用尺度。
    assert [row["minute"] for row in stats] == [600, 601, 602, 810]  # 只交付有观测的分钟，并保持顺序。
    assert [row["samples"] for row in stats] == [2, 1, 2, 2]  # 每个分钟跨日期统计各自的样本量。
    assert [row["mean"] for row in stats] == pytest.approx([0.002, 0.002, 0.006, 0.003])  # 算术均值使用原始标签单位。
    assert [row["std"] for row in stats] == pytest.approx([0.001, 0.0, 0.0, 0.002])  # 使用总体标准差，不以样本数减一为分母。
    json.dumps(stats, allow_nan=False)  # 统计可直接写入训练报告。


@pytest.mark.parametrize("dates, stock_day_count", [(("2024-06-03",), 2), (("2025-01-02",), 1)])  # 覆盖缓存截断和缓存包含验证日。
def test_estimate_minute_scales_rejects_inconsistent_cache(monkeypatch, dates, stock_day_count):  # 防止未核对的源被用于估计。
    """dates 是模拟缓存中的日期，stock_day_count 是调用者声明的行数。无效时要求抛出 ValueError。"""
    library = SimpleNamespace(cached_stock_days=lambda: iter(("SH600001", day) for day in dates))  # 向估计器提供合成缓存索引。

    def training_batches(_library, _metadata, requests, *, rng):  # 通过实际抽样接口检查异常。
        """接收读取器和请求，先遍历完整的抽样列表。"""
        list(requests)  # 触发缓存长度或日期范围的检查。
        pytest.fail("抽样应拒绝不一致的训练股票日缓存")  # 不能继续计算任何训练统计。

    monkeypatch.setattr(minute_scale, "iter_training_batches", training_batches)  # 仅替换从请求读行情的环节。
    with pytest.raises(ValueError, match="训练"):  # 错误信息应描述训练缓存的冲突。
        minute_scale.estimate_minute_scales(  # 请求缓存中的全部股票日。
            library, None, stock_day_count, seed=7, fallback_scale=0.004,  # 使用实际失败场景的声明行数。
        )  # 开始估计并触发输入错误。
