"""检验按预测分钟恢复物理收益率以及新旧模型状态加载。"""

import pytest  # 核对异常行为。
import torch  # 创建模型输入和分钟尺度表。

from src.models.tcn import ReturnTCN  # 调用正式的收益率预测模型。


def test_return_tcn_selects_and_restores_minute_scales():  # 验证按样本选择尺度和检查点恢复。
    """固定三个归一化输出，核对打乱分钟顺序下的物理收益率及恢复后的状态。"""
    scales = torch.full((1_440,), 0.003, dtype=torch.float32)  # 为所有北京时间分钟提供正值。
    scales[600] = 0.002  # 设置 10:00 的尺度。
    scales[810] = 0.004  # 设置 13:30 的不同尺度。
    model = ReturnTCN(return_scale=0.003, input_channels=8, minute_return_scales=scales)  # 配置八通道模型。
    scales.zero_()  # 修改调用方的张量，已校验的模型缓冲区应保留原值。
    x = torch.zeros(3, 8, 30)  # 零输入下便于取得可手算的模型分数。
    minutes = torch.tensor([810, 600, 810], dtype=torch.int32)  # 分钟顺序可以打乱并重复。
    expected = torch.tensor([0.004, 0.002, 0.004])  # 按样本顺序列出预计选中的尺度。
    with torch.no_grad():  # 构造可手算的输出，并执行纯预测。
        model.head.bias.fill_(2)  # 令模型输出固定归一化分数 2。
        torch.testing.assert_close(model.scale_for(minutes), expected)  # 核对查询按预测分钟索引。
        torch.testing.assert_close(model.return_scale, torch.tensor(0.003))  # 全局尺度仍独立保留。
        predicted = model(x, minutes)  # 将归一化分数恢复为物理对数收益率。
        torch.testing.assert_close(predicted, 2 * expected)  # 每个样本乘对应的分钟尺度。

    weights = model.state_dict()  # 取得包含分钟尺度的新状态。
    restored = ReturnTCN(input_channels=8, minute_return_scales=torch.ones(1_440, dtype=torch.float32))  # 新建同样的尺度槽位。
    restored.load_state_dict(weights, strict=True)  # 用状态中保存的真实分钟尺度覆盖占位值。
    with torch.inference_mode():  # 检查恢复后的正式推理路径。
        torch.testing.assert_close(restored(x, minutes), predicted, rtol=0, atol=0)  # 权重和分钟尺度均恢复。


def test_global_return_tcn_uses_same_state_without_minute_table():  # 检查旧权重仍可严格加载。
    """在原有全局尺度模式下，检查状态字段和不传时间的调用方式。"""
    legacy = ReturnTCN(return_scale=0.003)  # 创建只含全局尺度的六通道模型。
    weights = legacy.state_dict()  # 取得旧格式的权重。
    assert "minute_return_scales" not in weights  # 空分钟表不应新增状态字段。
    restored = ReturnTCN()  # 默认仍采用全局尺度。
    restored.load_state_dict(weights, strict=True)  # 严格核对状态字段兼容。
    torch.testing.assert_close(restored.scale_for(), torch.tensor(0.003))  # 无时间参数时返回全局标量。
    x = torch.zeros(2, 6, 30)  # 使用旧模型的输入形状。
    with torch.inference_mode():  # 比较恢复前后的旧调用。
        torch.testing.assert_close(restored(x), legacy(x), rtol=0, atol=0)  # 输出保持一致。


@pytest.mark.parametrize(  # 检查构造时是否拦截无法安全使用的尺度表。
    "scales",  # 逐个传入待检查的分钟尺度表。
    [torch.ones(1_439), torch.ones(1_440, dtype=torch.float64), torch.zeros(1_440), torch.full((1_440,), float("inf"))],  # 覆盖长度、类型、正值和有限性。
)  # 完成尺度表参数化。
def test_return_tcn_rejects_invalid_minute_table(scales):  # 核对无效表的异常。
    """将长度错误、双精度、零值或非有限表交给构造函数，要求立即报错。"""
    with pytest.raises(ValueError, match="分钟尺度表"):  # 报错应明确指出出错的模型配置。
        ReturnTCN(minute_return_scales=scales)  # 无效表不得进入预测路径。


def test_return_tcn_requires_one_integer_minute_per_sample():  # 核对逐样本索引的输入边界。
    """在按分钟模式中拒绝缺时间、非整数、越界以及错开批次长度的调用。"""
    model = ReturnTCN(minute_return_scales=torch.ones(1_440))  # 使用有效的分钟尺度表。
    x = torch.zeros(2, 6, 30)  # 构造两条待预测样本。
    with pytest.raises(ValueError, match="预测分钟"):  # 此模式需要调用方传入时点。
        model(x)  # 故意省略预测分钟。
    with pytest.raises(ValueError, match="预测分钟"):  # 不接受浮点分钟。
        model(x, torch.tensor([600.0, 810.0]))  # 传入含浮点类型的分钟张量。
    with pytest.raises(ValueError, match="批次"):  # 分钟数必须等于特征样本数。
        model(x, torch.tensor([600]))  # 故意只提供一条分钟。
    with pytest.raises((IndexError, RuntimeError)):  # PyTorch 索引会拒绝越界位置。
        model(x, torch.tensor([-1, 810]))  # 防止负数回绕到同日最后一分钟。
