"""检查首块后乘性时钟门、可配置卷积宽度以及旧检查点恢复。"""

from io import BytesIO  # 在内存中检查模型状态完整重载。

import pytest  # 参数化多种通道宽度和无效配置。
import torch  # 创建模拟量价、检查点和模型断言。

from src.features.samples import time_features  # 取得已确认的逐分钟时钟。
from src.models.tcn import ReturnTCN  # 走正式模型接口。


@pytest.mark.parametrize("width", [16, 32, 24])  # 核对两个试验宽度和正整数配置不只限两者。
def test_post_residual_clock_gate_tracks_body_width_and_restores(width):  # 检查有效门控、卷积布局和重载。
    """同一批模拟窗口传入六列特征，预测头依旧使用十六维隐藏层。"""
    model = ReturnTCN(window_clock_mode="gate", hidden_channels=width).eval()  # 只启用首块输出后的时钟门。
    assert model.input_channels == 6 and model.hidden_channels == width  # 加载器仍交付六列量价。
    assert model.head_clock is None and model.requires_prediction_minutes  # 时钟只进入卷积门且需要预测分钟。
    assert model.minute_block.conv1[0].in_channels == 6 and model.minute_block.conv1[0].out_channels == width  # 首块只读取量价。
    assert model.two_minute_block.conv1[0].in_channels == model.two_minute_block.conv1[0].out_channels == width  # 第二块沿用相同宽度。
    downsample = (model.downsample_pairs[0], model.downsample_triples[0], model.collapse_time[0])  # 依次取得两级降采样和末级汇总。
    assert [layer.kernel_size for layer in downsample] == [(2,), (3,), (5,)]  # 保持三个时间轴卷积核。
    assert [layer.stride for layer in downsample] == [(2,), (3,), (1,)]  # 保持同样的采样位点。
    assert all(layer.in_channels == layer.out_channels == width for layer in downsample)  # 不改变通道宽度。
    assert model.head[0].in_features == width and model.head[0].out_features == model.head[2].in_features == 16  # MLP 隐藏层固定十六维。
    assert model.head[2].out_features == 1 and isinstance(model.head[1], torch.nn.LeakyReLU)  # 回归头结构和激活沿用原配置。
    assert model.clock_gate is not None  # 取得门控层以验证实际乘数。
    assert model.clock_gate.in_features == 2 and model.clock_gate.out_features == width  # 两个时钟给每个卷积通道一个门值。
    assert not torch.count_nonzero(model.clock_gate.weight) and not torch.count_nonzero(model.clock_gate.bias)  # 初始纯乘门等于一。

    with torch.no_grad():  # 手动设定一组可复验的非零门权重。
        model.clock_gate.weight[:, 0].fill_(0.2)  # 下午标记贡献。
        model.clock_gate.weight[:, 1].fill_(0.3)  # 逐根时段进度贡献。
        model.clock_gate.bias.fill_(0.1)  # 线性项的固定贡献。
    x = torch.randn(2, 6, 30)  # 创建模拟量价窗口。
    t = torch.tensor([811, 600])  # 混合上午下午并故意按降序交付。
    first, next_ = [], []  # 保存首块输出和门控后下采样的输入。
    first_hook = model.minute_block.register_forward_hook(lambda _, __, out: first.append(out))  # 收集真实的首块输出。
    next_hook = model.downsample_pairs.register_forward_pre_hook(lambda _, args: next_.append(args[0]))  # 收集门后实际输出。
    with torch.inference_mode():  # 走正式推理接口并取得回归值。
        prediction = model(x, t)  # 计算物理收益率预测。
    first_hook.remove()  # 清理首块观察回调。
    next_hook.remove()  # 清理下采样观察回调。
    clock = torch.from_numpy(time_features(t.numpy()))  # 按预测分钟独立重算窗口时钟。
    factor = 1 + 0.1 + 0.2 * clock[:, 0] + 0.3 * clock[:, 1]  # 手算所选样本的乘法门。
    torch.testing.assert_close(next_[0], first[0] * factor[:, None], rtol=1e-6, atol=1e-7)  # 首块后只发生乘法。

    config = {"window_clock_mode": "gate", "time_fusion": "gate_only", "head_features": []}  # 三项配置确定单位置模式。
    if width != 16:  # 模拟历史十六宽度门控检查点省略宽度字段。
        config["hidden_channels"] = width  # 仅新宽度需要额外字段。
    stream = BytesIO()  # 避免向项目目录保存测试权重。
    torch.save({"config": config, "model": model.state_dict()}, stream)  # 写入与正式检查点相同的字段。
    stream.seek(0)  # 转到文件开头读取。
    loaded = ReturnTCN.from_checkpoint(torch.load(stream, weights_only=True)).eval()  # 按同一接口严格重建。
    with torch.inference_mode():  # 比较重载前后模型的完整前向。
        torch.testing.assert_close(loaded(x, t), prediction, rtol=0, atol=0)  # 恢复和原预测须逐位一致。


def test_gate16_initial_shared_state_matches_legacy_head():  # 验证十六宽度沿用旧种子的公共参数。
    """新模型从同种子后融合模型初始化，并保留隐藏层前十六列和公共卷积状态。"""
    torch.manual_seed(20261126)  # 固定对照参数的随机版本。
    legacy = ReturnTCN(time_at_head=True)  # 原后融合模型有十八维输入的预测头。
    rng = torch.random.get_rng_state()  # 记录构造完整对照后随机状态。
    torch.manual_seed(20261126)  # 让门控模型拥有同一随机起点。
    model = ReturnTCN(window_clock_mode="gate")  # 不传宽度时，和历史门控一样使用十六个通道。
    assert torch.equal(torch.random.get_rng_state(), rng)  # 门控固定初始化无额外随机消耗。
    for name, expected in legacy.state_dict().items():  # 比较所有共享卷积和预测头权重。
        if name == "head.0.weight":  # 仅头部输入删除两列时钟。
            expected = expected[:, :16]  # 保留原来十六个量价输入对应的权重。
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)  # 状态逐位匹配。


@pytest.mark.parametrize("options", [{"hidden_channels": 32}, {"window_clock_mode": "gate", "hidden_channels": 0}, {"window_clock_mode": "direct"}, {"window_clock_mode": "gate", "time_at_head": True}])  # 旧架构不加宽，门不双接时钟。
def test_model_rejects_unavailable_gate_combinations(options):  # 错误的训练设置必须在开始前报出。
    """使用无效宽度、仅临时的模式或重复的时钟条件构建模型。"""
    with pytest.raises(ValueError):  # 检查配置错误立即触发。
        ReturnTCN(**options)  # 尝试构建项目没有采用的模型组合。
