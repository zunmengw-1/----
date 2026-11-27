"""检查训练入口的中心化开关与三种价格倍率设置。"""

from types import SimpleNamespace  # 以已拟合的统计量检验入口选项。

import numpy as np  # 构造尺寸与拟合结果一致的输入表。
import pytest  # 覆盖六种去均值和倍率组合。
import torch  # 核对实际交给模型的数值。

from scripts.train_tcn import _input_tables  # 检验正式入口的输入配置。


@pytest.mark.parametrize("center", [True, False])  # 覆盖新口径和原基线。
@pytest.mark.parametrize("mode", ["minute", "global", "none"])  # 倍率的选择独立于去均值。
def test_price_mode_keeps_requested_centering(center, mode):  # 核对两种开关的独立含义。
    """默认去均值路径在关闭倍率或使用全天倍率时继续交付原位置均值。"""
    factors = np.linspace(2.0, 8.0, 1440, dtype=np.float32)  # 准备逐分钟变化的倍率。
    means = np.ones((1440, 30, 3), dtype=np.float32) if center else None  # 只在启用去均值时拟合该表。
    estimate = SimpleNamespace(multipliers=factors, global_multiplier=5.0, position_means=means)  # 模拟训练统计接口。
    if not center and mode == "none":  # 没有输入变换的旧基线不会触发拟合。
        estimate = None  # 验证入口也接受空统计量。
    positions, multipliers, fallback = _input_tables(estimate, center, mode)  # 复用正式参数组装函数。
    if center:  # 检查所有倍率模式都保留训练位置均值。
        np.testing.assert_array_equal(positions.numpy(), means)  # 不放大或重算均值。
    else:  # 历史输入路径不减去位置均值。
        assert positions is None  # 不引入新的均值缓冲区。
    if mode == "none":  # 单位倍率不需要查表。
        assert multipliers is None and fallback == 1.0  # 仅价格乘数回到一。
    else:  # 剩余两种模式都保存倍率表。
        expected = torch.from_numpy(factors) if mode == "minute" else torch.full((1440,), 5.0)  # 构造预期实际输入乘数。
        torch.testing.assert_close(multipliers, expected, rtol=0, atol=0)  # 逐位核对冻结的倍率。
        assert fallback == 5.0  # 回退倍率应出自同一份拟合结果。
