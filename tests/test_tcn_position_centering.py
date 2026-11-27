"""核对模型中统一的逐位置去均值、价格缩放和检查点口径。"""

from copy import deepcopy  # 创建使用相同参数和手工处理输入的对照模型。
from io import BytesIO  # 在内存中验证正式检查点的读写流程。

import pytest  # 对照带价格倍率及仅去均值的输入配置。
import torch  # 构造合成窗口和期望卷积输入。

from src.models.tcn import ReturnTCN  # 使用正式的模型输入和前向路径。


@pytest.mark.parametrize(("channels", "time_at_head"), [(6, False), (6, True), (8, False)])  # 覆盖三种已有通道配置。
@pytest.mark.parametrize("scale_prices", [True, False])  # 有无价格倍率都可以应用位置均值。
def test_model_preprocesses_windows_by_forecast_minute(channels, time_at_head, scale_prices):  # 核对时点对齐和变换顺序。
    """使用手算的不同分钟、窗口位置、特征均值及倍率，核对实际卷积输入和模型输出。"""
    means = torch.zeros(1440, 30, 3)  # 保存对应时刻和窗内位置的收盘、成交量、成交额均值。
    offsets = torch.arange(30, dtype=torch.float32) / 10  # 让同一均值表中各个窗内位置可区分。
    means[600] = offsets[:, None] + torch.tensor([0, 1, 2])  # 上午三列均值不同。
    means[810] = offsets[:, None] + torch.tensor([3, 4, 5])  # 下午使用另一行均值。
    prices = torch.ones(1440) if scale_prices else None  # 仅在当前用例启用价格倍率时提供表。
    if prices is not None:  # 通过不同乘数核对价格缩放也使用正确时刻。
        prices[[600, 810]] = torch.tensor([2, 7], dtype=torch.float32)  # 按测试时钟配置倍率。
    model = ReturnTCN(  # 同一模型同时接收两张表。
        input_channels=channels, time_at_head=time_at_head,  # 选择现有量价和时钟接入路径。
        minute_price_multipliers=prices, position_means=means,  # 加载固定倍率与位置均值。
    ).eval()  # 使用与正式预测相同的模型模式。
    control = deepcopy(model)  # 在参数完全相同的模型上对照手工处理后的输入。
    control.position_means = None  # 对照模型直接接收人工去均值的输入。
    control.minute_price_multipliers = None  # 对照模型不重复应用倍率。
    minutes = torch.tensor([810, 600, 810])  # 模拟交错抽样得到的批次。
    x = (torch.arange(3 * 30 * channels, dtype=torch.float32) / 100).reshape(3, 30, channels).transpose(1, 2)  # 复现加载器转置布局。
    original = x.clone()  # 观察正常前向是否会更改原输入。
    expected = x.clone()  # 用原始数据独立手算输入变换。
    expected[:, :4] -= means[minutes, :, 0][:, None, :]  # 开高低收共用每根位置的收盘均值。
    expected[:, 4] -= means[minutes, :, 1]  # 股数特征减去自己的均值。
    expected[:, 5] -= means[minutes, :, 2]  # 成交额特征减去自己的均值。
    if prices is not None:  # 手工对已中心化价格执行统一缩放。
        expected[:, :4] *= prices[minutes][:, None, None]  # 整窗按同一个预测分钟倍率。
    seen = []  # 存储真实模型进入首个残差块的张量。

    def capture_inputs(_model, inputs):  # 在卷积前直接取得实际处理结果。
        """inputs 中的唯一张量为 [批大小, 通道数, 30]。"""
        seen.append(inputs[0])  # 保留各输入通道以逐项比对。

    hook = model.minute_block.register_forward_pre_hook(capture_inputs)  # 通过正式网络观察变换结果。
    with torch.inference_mode():  # 仅检查变换和前向一致性。
        actual = model(x, minutes)  # 执行真实模型的完整预测路径。
        reference = control(expected, minutes)  # 通过人工等效变换得到参考结果。
    hook.remove()  # 恢复模型的常规前向设置。
    assert model.requires_prediction_minutes and len(seen) == 1  # 即使没有价格倍率，去均值仍需要时点。
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)  # 核对全部通道和先减后乘的顺序。
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)  # 核对新旧前向路径在等价输入上的输出。
    torch.testing.assert_close(x, original, rtol=0, atol=0)  # 确认原始输入没有就地修改。


def test_position_mean_buffer_is_defensive_and_reloads_strictly():  # 确认均值表随检查点保存并兼容旧模型。
    """验证模型与调用者互不共享均值数据，新状态严格恢复，旧状态没有额外字段。"""
    means = torch.zeros(1440, 30, 3, requires_grad=True)  # 输入表可带梯度，模型应保存独立的固定版本。
    with torch.no_grad():  # 为有效预测分钟提供非零均值。
        means[600].fill_(0.125)  # 使实际卷积输入在变换后发生变化。
    model = ReturnTCN(position_means=means).eval()  # 没有价格倍率时也可使用位置均值。
    with torch.no_grad():  # 修改调用者自己的数据以检测缓冲区共享。
        means.zero_()  # 已保存的均值仍应保持原值。
    assert model.position_means is not None and not model.position_means.requires_grad  # 缓冲区不作为可优化参数。
    assert torch.count_nonzero(model.position_means[600]) == 90  # 模型保存了独立的原值副本。
    file = BytesIO()  # 不在工作区写入合成检查点。
    torch.save(model.state_dict(), file)  # 使用正式保存方式将均值表连同模型权重序列化。
    file.seek(0)  # 将读取位置移回检查点起点。
    state = torch.load(file, map_location="cpu", weights_only=True)  # 读取受限制的权重对象。
    config = {"center_positions": True, "price_input_scale_mode": "none"}  # 仅去均值的检查点明确标记单位倍率。
    restored = ReturnTCN.from_checkpoint({"model": state, "config": config}).eval()  # 使用正式入口严格恢复。
    x, minutes = torch.zeros(1, 6, 30), torch.tensor([600])  # 使用明确预测分钟的单窗口。
    with torch.inference_mode():  # 对比两个检查点的实际前向。
        torch.testing.assert_close(restored(x, minutes), model(x, minutes), rtol=0, atol=0)  # 断言重载后逐位一致。
    baseline = ReturnTCN()  # 旧格式没有位置均值表。
    legacy_state = baseline.state_dict()  # 获取旧状态字段集合。
    assert "position_means" not in legacy_state  # 默认关闭时无需多出状态项。
    ReturnTCN.from_checkpoint({"model": legacy_state, "config": {}})  # 旧状态依然严格兼容。


@pytest.mark.parametrize("missing", ["position_means", "minute_price_multipliers", "minute_return_scales"])  # 新检查点不能静默丢表。
def test_checkpoint_requires_declared_tables(missing):  # 覆盖去均值、价格缩放和标签尺度的完整性。
    """声明分时倍率和标签尺度的中心化检查点缺少任一表时须终止重载。"""
    config = {"center_positions": True, "price_input_scale_mode": "minute", "label_scale_mode": "prediction_minute"}  # 声明默认方案。
    model = ReturnTCN(position_means=torch.zeros(1440, 30, 3), minute_price_multipliers=torch.ones(1440), minute_return_scales=torch.ones(1440))  # 准备完整状态。
    state = model.state_dict()  # 取得全部缓冲区和参数。
    del state[missing]  # 模拟遗失指定统计表的检查点。
    with pytest.raises(KeyError, match=missing):  # 直接指出缺少的表名。
        ReturnTCN.from_checkpoint({"model": state, "config": config})  # 使用正式预测的重载接口。


@pytest.mark.parametrize("means", [torch.zeros(1440, 30, 6), torch.zeros(1440, 30, 3, dtype=torch.float64), torch.full((1440, 30, 3), torch.nan)])  # 分别检测形状、类型和非有限数。
def test_model_rejects_invalid_position_mean_table(means):  # 检查训练前的固定表校验。
    """使用一种不合法均值表，要求构造模型时立即报错。"""
    with pytest.raises(ValueError, match="位置均值表"):  # 确认异常指出均值表。
        ReturnTCN(position_means=means)  # 尝试使用非法表构造模型。
