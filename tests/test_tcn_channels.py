"""检验时钟融合方式、模型输入边界和恢复后的预测结果。"""

import pytest  # 核对无效输入时抛出的异常。
import torch  # 创建分钟特征张量并比较预测。

from src.models.tcn import ReturnTCN  # 使用支持六通道或八通道的正式模型。


def test_return_tcn_reloads_eight_channel_model():  # 验证八通道预测、反向传播和模型恢复。
    """以八通道分钟窗口训练一次反向传播，并检查按保存配置恢复的模型输出。"""
    model = ReturnTCN(return_scale=0.003, input_channels=8)  # 配置八个特征通道和物理收益率尺度。
    batch = torch.randn(2, 30, 8).transpose(1, 2)  # 使用与训练加载器相同的通道前置布局。
    batch.requires_grad_()  # 观察最后两个通道是否参与反向传播。
    predictions = model(batch)  # 计算每条样本的五分钟对数收益率。
    assert model.input_channels == 8 and predictions.shape == (2,)  # 核对公开通道数和预测形状。
    predictions.square().sum().backward()  # 通过合成损失反向传播。
    assert batch.grad is not None and torch.isfinite(batch.grad).all()  # 确认前向和反向的数值正常。
    assert torch.count_nonzero(batch.grad[:, 6:]) > 0  # 检查时间特征确实进入网络。

    checkpoint = {"input_channels": model.input_channels, "model": model.state_dict()}  # 模拟保存结构配置和权重。
    restored = ReturnTCN(input_channels=checkpoint["input_channels"])  # 恢复前按配置建立对应的卷积层。
    restored.load_state_dict(checkpoint["model"])  # 将模型参数和收益率尺度恢复到新实例。
    with torch.inference_mode():  # 使用纯推理路径比较恢复前后。
        torch.testing.assert_close(restored(batch), predictions, rtol=0, atol=0)  # 预测应保持一致。


def test_return_tcn_rejects_channel_mismatch():  # 验证模型能在卷积之前报告输入通道错误。
    """以错误通道数构造模型或给八通道模型传六通道输入，两者均抛出 ValueError。"""
    with pytest.raises(ValueError, match="6 或 8"):  # 指明项目只支持两种特征布局。
        ReturnTCN(input_channels=7)  # 七通道没有对应的特征定义。
    model = ReturnTCN(input_channels=8)  # 按时间特征实验的通道数构建模型。
    with pytest.raises(ValueError, match=r"\[批大小, 8, 30\]"):  # 异常应列出模型实际需要的形状。
        model(torch.zeros(2, 6, 30))  # 六通道输入不适用于八通道配置。


def test_return_tcn_fuses_only_prediction_clock_at_the_head():  # 核对后融合的卷积边界、时点对齐及模型重载。
    """给同一量价窗口配上不同分钟，观测预测头的实际输入并比较检查点重载前后的输出。"""
    minutes = torch.tensor([829, 600, 810], dtype=torch.int32)  # 以非时序顺序混合上午和下午的样本。
    scales = torch.full((1_440,), 0.003)  # 准备与预测分钟区分开的物理收益率尺度。
    scales[829] = 0.004  # 令第一条样本采用单独的尺度。
    scales[600] = 0.002  # 令第二条样本采用另一不同尺度。
    model = ReturnTCN(input_channels=6, time_at_head=True, minute_return_scales=scales)  # 启用六列卷积和时钟后融合。
    assert model.requires_prediction_minutes  # 让正常训练和验证路径传入每条样本的时点。
    assert any(isinstance(layer, torch.nn.LeakyReLU) for layer in model.head.modules())  # 验证预测头可学习量价和时钟的非线性交互。
    window = torch.linspace(-0.5, 0.5, 6 * 30).reshape(1, 6, 30)  # 准备所有样本共用的合成量价窗口。
    batch = window.repeat(len(minutes), 1, 1)  # 同一窗口重复三次，各样本只在预测时点上不同。
    encoder_inputs = []  # 用于检查进入时间卷积的实际张量。
    head_calls = []  # 用于检查预测头前后的实际张量。

    encoder_hook = model.minute_block.register_forward_pre_hook(  # 观察模型首个卷积块的入口。
        lambda _layer, args: encoder_inputs.append(args[0].clone())  # 捕获卷积块的全部输入通道。
    )  # 保留句柄，以便独立验证编码结果。
    head_hook = model.head.register_forward_hook(  # 观察后融合的网络头。
        lambda _layer, args, output: head_calls.append((args[0].clone(), output.clone()))  # 保存预测头输入及无量纲分数。
    )  # 记录真实前向期间的时钟特征。
    with torch.inference_mode():  # 在与验证相同的推理模式下检查输出。
        predicted = model(batch, minutes)  # 生成三个以物理收益率表示的预测。
    encoder_hook.remove()  # 移除观测后独立调用量价编码器。
    head_hook.remove()  # 后续可自由重载并预测。
    (encoder_input,) = encoder_inputs  # 预期模型仅调用一次量价卷积块。
    ((head_input, scores),) = head_calls  # 预期整个批次仅调用一次预测头。
    torch.testing.assert_close(encoder_input, window.repeat(3, 1, 1), rtol=0, atol=0)  # 卷积仅观察三十根分钟线的六列量价。
    assert head_input.shape == (3, 18)  # 预测头读取十六个量价表示和两个时钟值。
    expected_clock = torch.tensor([[1, 49 / 117], [0, 30 / 120], [1, 30 / 117]])  # 手算三个分钟的下午标记及进度。
    torch.testing.assert_close(head_input[:, -2:], expected_clock, rtol=0, atol=0)  # 只拼接最终预测分钟的时钟值。
    with torch.inference_mode():  # 核对独立编码器和恢复的模型。
        torch.testing.assert_close(head_input[:, :16], model.encode(batch), rtol=0, atol=0)  # 前十六列是无时钟的量价表示。
        torch.testing.assert_close(predicted, scores.flatten() * scales[minutes.long()], rtol=0, atol=0)  # 每条预测按各自分钟的正尺度还原。
        restored = ReturnTCN(input_channels=6, time_at_head=True, minute_return_scales=torch.ones_like(scales))  # 按模型配置重建网络。
        restored.load_state_dict(model.state_dict(), strict=True)  # 从状态恢复卷积、预测头和标签尺度。
        torch.testing.assert_close(restored(batch, minutes), predicted, rtol=0, atol=0)  # 检查时钟位置和恢复后的预测完全一致。
