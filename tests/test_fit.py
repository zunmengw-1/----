"""检验按批训练的收益率尺度、消耗上限与损失异常处理。"""

import numpy as np  # 创建训练批次的股票和时刻字段。
import pytest  # 核对数值和异常。
import torch  # 创建合成样本并调用 PyTorch 优化器。

from src.models.fit import fit_epoch  # 导入正式的单轮训练入口。
from src.models.tcn import ReturnTCN  # 导入预测物理收益率的 TCN。
from src.models.training_data import (  # 查看加载器的批次接口。
    TorchTrainingBatch,  # 使用正式加载器交付的批次类型。
    prepare_model_inputs,  # 取得合成批次的预期时间特征。
)  # 完成批次类型的导入。


def _batch(labels: tuple[float, ...]) -> TorchTrainingBatch:  # 生成一批输出可手算的合成训练数据。
    """labels 为物理对数收益率，返回同等样本数的零输入模型批次。"""
    size = len(labels)  # 按标签数确定批次大小。
    return TorchTrainingBatch(  # 构造与正式加载器字段一致的训练批次。
        x=torch.zeros(size, 6, 30),  # 零输入配合零偏置使初始预测为零。
        y=torch.tensor(labels, dtype=torch.float32),  # 保持实际标签的单精度表示。
        code=np.full(size, "SH600001", dtype="U8"),  # 为合成样本填入同一定位代码。
        t=np.full(size, np.datetime64("2024-06-03T10:00", "m")),  # 填入合成预测时刻。
    )  # 完成批次构造。


def test_fit_epoch_stops_without_fetching_a_third_batch():  # 检查组尾、归一化损失和读取上限。
    """使用三批合成样本的读取器，断言只消耗两批且损失按样本数加权。"""
    model = ReturnTCN(return_scale=0.002)  # 固定收益率尺度以手算归一化残差。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)  # 保持权重固定，仍执行实际优化器步骤。

    def loader():  # 提供两个可训练批次，并检测第三次读取。
        """交付大小为二和一的批次；多取一批时触发 pytest 失败。"""
        yield _batch((0.002, -0.001))  # 归一化残差绝对值为 1 和 0.5。
        yield _batch((0.004,))  # 归一化残差绝对值为 2。
        pytest.fail("训练超过指定批次数后仍向加载器取数")  # 第三个批次不应被请求。

    stats = fit_epoch(model, loader(), optimizer, device="cpu", max_batches=2)  # 只训练两个批次。
    assert stats["batches"] == 2 and stats["samples"] == 3  # 保留大小不足的批次。
    assert stats["mean_huber"] == pytest.approx((0.5 + 0.125 + 1.5) / 3)  # 检查逐样本 Huber 均值。
    assert optimizer.state[model.head.bias]["step"] == 2  # 每批实际调用一次优化器更新。


def test_return_tcn_restores_physical_return_scale():  # 检查前向尺度和权重恢复。
    """将带尺度的权重载入默认模型，断言编码一致且预测使用保存的尺度。"""
    model = ReturnTCN(return_scale=0.002)  # 设定物理尺度后再读取保存状态。
    restored = ReturnTCN()  # 无需预先知道保存时采用的尺度。
    restored.load_state_dict(model.state_dict())  # 训练尺度随模型权重一起恢复。
    x = torch.ones(2, 6, 30)  # 使用两个合成分钟窗口。
    with torch.inference_mode():  # 比较纯推理结果。
        scores = model.head(model.encode(x)).squeeze(-1)  # 直接取无量纲的回归头分数。
        torch.testing.assert_close(model(x), scores * 0.002)  # 模型输出物理收益率。
        torch.testing.assert_close(restored.encode(x), model.encode(x))  # 权重恢复不改变特征表示。
        torch.testing.assert_close(restored(x), model(x))  # 恢复后的模型按原尺度推理。


def test_fit_epoch_rejects_nonfinite_loss_before_optimizer_step():  # 避免不可用损失污染模型权重。
    """故意传入非有限标签，断言抛出 FloatingPointError 且优化器未执行更新。"""
    model = ReturnTCN()  # 生成待训练模型。
    optimizer = torch.optim.AdamW(model.parameters())  # 记录是否有参数经过优化器更新。
    with pytest.raises(FloatingPointError, match="非有限损失"):  # 检查异常类型和原因。
        fit_epoch(model, [_batch((float("nan"),))], optimizer, device="cpu")  # 触发损失非有限。
    assert not optimizer.state  # 优化器未构建任何参数更新状态。


def test_fit_epoch_reports_weighted_segments_and_tail():  # 核对分段覆盖、样本权重和回调顺序。
    """以三个不同规模的批次训练，验证第二批和第三批更新后的损失报告。"""
    model = ReturnTCN(return_scale=0.002)  # 固定尺度，使归一化残差可手算。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)  # 记录优化器步数，保持预测为零。
    reported = []  # 按调用顺序收集已报告的批次位置。

    def loader():  # 制造一个完整区间和一个单批尾区间。
        """依次交付两批组成的整段和一批尾段，读取多于三批时测试失败。"""
        yield _batch((0.002, -0.001))  # 损失为 0.5 和 0.125。
        yield _batch((0.004,))  # 损失为 1.5。
        assert reported == [2] and model.training  # 第一段报告在下一批读取前完成，模型已恢复训练模式。
        yield _batch((0.001, 0.001))  # 尾段两个样本的损失均为 0.125。
        pytest.fail("分段报告触发了上限之外的数据读取")  # 不得多取第四批。

    def on_report(end_batch):  # 模拟报告期间执行的验证。
        """接收已完成的批次位置，验证该批已经更新，再切换到评估模式。"""
        assert optimizer.state[model.head.bias]["step"] == end_batch  # 回调发生在当前批更新后。
        reported.append(end_batch)  # 记录此次报告对应的批次位置。
        model.eval()  # 模拟验证阶段切换模型模式。

    stats = fit_epoch(  # 恰好消耗三个批次，每两批报告一次。
        model, loader(), optimizer, device="cpu", max_batches=3, report_every=2, on_report=on_report,  # 提供试跑和回调设置。
    )  # 获取轮次统计。
    first, tail = stats["loss_history"]  # 获得一个完整段和一个尾段。
    assert first == {"start_batch": 1, "end_batch": 2, "samples": 3, "mean_huber": pytest.approx(2.125 / 3)}  # 整段按样本加权。
    assert tail == {"start_batch": 3, "end_batch": 3, "samples": 2, "mean_huber": pytest.approx(0.125)}  # 尾段独立计入。
    assert stats["mean_huber"] == pytest.approx((2.125 + 0.25) / 5)  # 全轮均值仍按五个样本计算。
    assert reported == [2, 3] and model.training  # 两段各回调一次，尾段过后仍保持训练模式。


def test_fit_epoch_reports_exact_segment_once():  # 整数个分段结束后无需再报告尾段。
    """一个批次恰好填满一段，验证报告和回调各执行一次。"""
    model = ReturnTCN()  # 使用单位尺度进行一批训练。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)  # 保持模型权重便于核对。
    reported = []  # 收集回调收到的批次位置。
    stats = fit_epoch(  # 令报告间隔和批次总数恰好一致。
        model, [_batch((0.002,))], optimizer, device="cpu", report_every=1, on_report=reported.append,  # 提供唯一批次和报告设置。
    )  # 获取训练和报告统计。
    assert reported == [1] and len(stats["loss_history"]) == 1  # 不应重复报告已经结束的区间。


def test_fit_epoch_appends_time_features_in_batch_order():  # 检查正式训练入口按实际时间补充输入通道。
    """同一批选取不同日期的上午和下午样本，核对模型实际接收的完整张量。"""
    batch = _batch((0.002, -0.001))  # 用已有夹具构造两条不同的收益标签。
    batch.t[:] = np.array(("2024-06-04T13:49", "2024-06-03T10:00"), dtype="datetime64[m]")  # 模拟跨股票日打乱后的预测时刻。
    expected = prepare_model_inputs(batch.x, np.array((829, 600)), 8)  # 手算北京时间距午夜的分钟数。
    model = ReturnTCN(input_channels=8)  # 使用包含两个时间通道的模型。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)  # 运行实际训练循环并保持参数不变。
    inputs: list[torch.Tensor] = []  # 保存传给模型的实际输入。

    def observe_inputs(_model, args):  # 在前向开始前捕获组装完成的特征。
        """args 为 PyTorch 提供的前向参数，首项即批次特征。"""
        inputs.append(args[0].detach())  # 保留与训练设备无关的检查张量。

    model.register_forward_pre_hook(observe_inputs)  # 观察实际模型入口的调用。
    stats = fit_epoch(model, [batch], optimizer, device="cpu")  # 通过正常训练路径处理该合成批次。
    assert stats["samples"] == 2 and len(inputs) == 1  # 两条样本在同一批传入。
    torch.testing.assert_close(inputs[0], expected, rtol=0, atol=0)  # 保留原特征和标签顺序，并使用真实预测时刻。


def test_fit_epoch_normalizes_each_sample_with_its_own_minute_scale():  # 通过实际模型核对物理预测及损失缩放。
    """构造两个日期和不同时刻的标签，检查输入索引顺序及每条样本归一化后的 Huber。"""
    batch = _batch((0.002, 0.002))  # 两个目标有相同的物理收益率。
    batch.t[:] = np.array(("2024-06-04T13:49", "2024-06-03T10:00"), dtype="datetime64[m]")  # 保留打乱后的日内分钟顺序。
    scales = torch.full((1440,), 0.003)  # 使用备用尺度填满所有分钟。
    scales[600] = 0.002  # 设置上午样本的尺度。
    scales[829] = 0.004  # 设置下午样本的尺度。
    model = ReturnTCN(return_scale=0.003, minute_return_scales=scales)  # 启用单独的分钟收益率尺度表。
    with torch.no_grad():  # 只设置可手算的无量纲预测值。
        model.head.bias.fill_(0.25)  # 零输入的模型头输出固定为 0.25。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)  # 执行优化器流程同时保持已设权重。
    actual_minutes = []  # 记录生产训练入口传给模型的分钟索引。

    def observe_minutes(_model, args):  # 检查分钟索引的布局和实际值。
        """args 为模型前向输入，第二项是模型读取尺度的分钟索引。"""
        actual_minutes.append(args[1])  # 保存前向收到的分钟张量。

    model.register_forward_pre_hook(observe_minutes)  # 在正常前向过程中捕获样本分钟。
    result = fit_epoch(model, [batch], optimizer, device="cpu")  # 使用正式训练入口更新一个批次。
    assert result["samples"] == 2 and len(actual_minutes) == 1  # 每个样本仅参与一次更新计算。
    torch.testing.assert_close(actual_minutes[0], torch.tensor((829, 600), dtype=torch.long))  # 验证北京时间索引类型和批次顺序。
    assert result["mean_huber"] == pytest.approx((0.25**2 + 0.75**2) / 4)  # 两条归一化残差为 0.25 和 0.75。


def test_fit_epoch_passes_minutes_for_late_fusion_with_global_scale():  # 核对时钟后融合独立于按分钟标签缩放。
    """通过正常训练路径给后融合模型传入乱序预测分钟，卷积仍只接收六列量价。"""
    batch = _batch((0.002, -0.001))  # 创建两个不同收益率的合成样本。
    batch.t[:] = np.array(("2024-06-04T13:49", "2024-06-03T10:00"), dtype="datetime64[m]")  # 模拟跨股票日乱序。
    model = ReturnTCN(input_channels=6, time_at_head=True)  # 时钟后融合仍可使用全局标签尺度。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)  # 执行正式优化路径同时保留初始参数。
    observed = []  # 捕获训练实际提交给前向接口的参数。
    model.register_forward_pre_hook(lambda _layer, args: observed.append(args))  # 记录量价窗口和预测分钟。
    result = fit_epoch(model, [batch], optimizer, device="cpu")  # 从正常训练入口消费一个批次。
    assert result["samples"] == 2 and len(observed) == 1  # 同一批一次提交全部样本。
    x, minutes = observed[0]  # 取得按实际送模顺序设置的输入。
    torch.testing.assert_close(x, batch.x, rtol=0, atol=0)  # 全组保持原始六列量价。
    torch.testing.assert_close(minutes, torch.tensor([829, 600]), rtol=0, atol=0)  # 时钟仍按样本顺序独立提交。
