"""使用正式训练批次更新 TCN，并汇总各训练区间和整轮的 Huber 损失。"""

from collections.abc import Callable, Iterable  # 声明报告回调和可逐批读取的训练数据。
from itertools import islice  # 在指定批次上限处停止读取。
from math import isfinite  # 验证损失可以用于参数更新。
from time import perf_counter  # 记录训练用时。

import numpy as np  # 将北京时间预测时刻换算为日内分钟。
import torch  # 使用 PyTorch 损失和设备类型。

from src.models.tcn import ReturnTCN  # 训练带收益率尺度的正式模型。
from src.models.training_data import (  # 导入训练与验证共用的特征组装入口。
    TorchTrainingBatch,  # 接收正式加载器的批次格式。
    prepare_model_inputs,  # 根据模型规格准备量价与时间通道。
)  # 完成模型输入相关的接口导入。


def fit_epoch(  # 用一轮有效样本拟合 TCN。
    model: ReturnTCN,  # 接收已移到训练设备上的 TCN。
    loader: Iterable[TorchTrainingBatch],  # 接收逐批交付特征和有效标签的迭代对象。
    optimizer: torch.optim.Optimizer,  # 使用调用方为当前模型配置的优化器。
    *,  # 以下可配项须通过名称传入。
    device: torch.device | str,  # 将批次张量移到模型所在设备。
    max_batches: int | None = None,  # 默认消耗整轮，可限制实际读取和训练的批次数。
    report_every: int = 1_000,  # 默认每一千批报告一次，轮次末尾也报告未满区间。
    on_report: Callable[[int], None] | None = None,  # 每次报告后可调用，接收刚完成更新的批次序号。
) -> dict[str, int | float | list[dict[str, int | float]]]:  # 返回轮次统计和分段损失记录。
    """以训练标签尺度归一化预测和标签，使用阈值为 1 的 Huber 损失。

    加载器交付 [B, 6, 30] 的量价张量和北京时间预测时刻，时钟按模型配置送到卷积或预测头。
    标签形状为 [B]，表示下一分钟开盘至第 5 分钟收盘的对数收益率。
    模型启用分钟尺度时，按每条样本的预测时刻选取尺度。整轮及区间损失均按样本数加权。
    训练时间不含报告回调。
    无训练批次、批次上限非正或报告间隔非正时抛出 ValueError。
    损失非有限时在更新模型前抛出 FloatingPointError，回调异常原样传递。
    """
    if max_batches is not None and max_batches < 1:  # 批次上限必须允许至少一次更新。
        raise ValueError("训练批次上限必须为正整数")  # 向调用方报告无效配置。
    if report_every < 1:  # 每个区间至少要有一批训练样本。
        raise ValueError("训练报告间隔必须为正整数")  # 向调用方报告无效间隔。

    start = perf_counter()  # 从准备进入本轮训练时开始计时。
    model.train()  # 让模型按 PyTorch 训练模式执行。
    criterion = torch.nn.HuberLoss(delta=1.0)  # 在归一化尺度上区分小误差和大误差。
    requires_minutes = model.requires_prediction_minutes  # 确认时钟预测头或标签尺度是否需要分钟。
    samples = batches = 0  # 累计本轮处理的样本数和批次数。
    total_loss = 0.0  # 累计所有已训练样本的归一化 Huber 损失。
    segment_start = 1  # 批次序号从一开始，首段也从第一批开始。
    segment_samples = 0  # 累计当前区间实际用于更新的样本数。
    segment_loss = 0.0  # 累计当前区间每个样本的归一化 Huber 损失。
    callback_seconds = 0.0  # 单列回调耗时，使验证计时不混进训练结果。
    loss_history: list[dict[str, int | float]] = []  # 按时间顺序保存完整区间和末尾区间。

    def report_segment(end_batch: int) -> None:  # 汇报当前区间，然后调用外部报告函数。
        """end_batch 为刚完成参数更新的最后一个批次序号，含该批。"""
        nonlocal callback_seconds  # 从整轮计时中扣除本次回调耗时。
        mean_huber = segment_loss / segment_samples  # 按区间内实际样本数计算均值。
        loss_history.append({  # 记录当前区间的边界和损失。
            "start_batch": segment_start,  # 第一个已纳入统计的批次。
            "end_batch": end_batch,  # 最后一个已纳入统计的批次。
            "samples": segment_samples,  # 区间内使用的全部有效标签样本数。
            "mean_huber": mean_huber,  # 本区间的逐样本 Huber 均值。
        })  # 将结果附加到按顺序排列的损失记录。
        print(  # 把区间指标立即报告到终端。
            f"第 {segment_start:,}–{end_batch:,} 批、{segment_samples:,} 样本，"  # 展示本区间的范围。
            f"分段 Huber {mean_huber:.6f}，训练用时 {perf_counter() - start - callback_seconds:.1f} 秒",  # 报告区间损失和累计训练耗时。
            flush=True,  # 立即写出，便于查看长时间运行的日志。
        )  # 结束进度输出。
        if on_report is not None:  # 调用方可在当前权重上开展验证。
            callback_start = perf_counter()  # 从验证等回调开始前计时。
            try:  # 回调即使失败，也恢复模型的训练状态。
                on_report(end_batch)  # 将刚完成参数更新的批次序号交给调用方。
            finally:  # 保证评估模式不会影响下一批训练。
                model.train()  # 恢复训练模式。
                callback_seconds += perf_counter() - callback_start  # 单独累计回调耗时。

    for batches, batch in enumerate(islice(loader, max_batches), start=1):  # 按上限停止，不预读下一批。
        minutes = batch.t.astype(np.int64) % (24 * 60)  # 从北京时间分钟精度的预测时刻提取当天分钟位置。
        x = prepare_model_inputs(batch.x, minutes, model.input_channels).to(device)  # 按模型规格组装通道后移到训练设备。
        y = batch.y.to(device)  # 将物理收益率标签移到相同设备。
        optimizer.zero_grad(set_to_none=True)  # 清除上一次更新留下的梯度。
        if requires_minutes:  # 预测头与损失按需使用相同的预测时刻。
            minute_indices = torch.as_tensor(minutes, dtype=torch.long, device=device)  # 按批次原顺序取得训练设备上的分钟索引。
            scale = model.scale_for(minute_indices)  # 按样本预测分钟从模型查找对应的正尺度。
            prediction = model(x, minute_indices)  # 用同一组分钟尺度还原物理对数收益率。
        else:  # 全局模式保留原有的单参数前向。
            scale = model.return_scale  # 读取模型中保存的全局标签标准差。
            prediction = model(x)  # 使用原来的全局尺度输出物理收益。
        loss = criterion(prediction / scale, y / scale)  # 将预测和标签除以同一个尺度再计算 Huber。
        batch_loss = float(loss.detach())  # 取出用于检查和统计的损失数值。
        if not isfinite(batch_loss):  # 发现异常损失时停止本轮。
            raise FloatingPointError(f"第 {batches} 个训练批次出现非有限损失")  # 带批次位置报告异常。
        loss.backward()  # 对本批损失反向传播。
        optimizer.step()  # 使用本批梯度更新模型参数。
        batch_size = len(y)  # 使用本批的实际大小，包括不足一批的组尾。
        loss_sum = batch_loss * batch_size  # 将批次平均损失还原为样本和。
        samples += batch_size  # 更新整轮样本数。
        total_loss += loss_sum  # 更新整轮的样本损失和。
        segment_samples += batch_size  # 同时累计当前区间的样本数。
        segment_loss += loss_sum  # 同时累计当前区间的样本损失和。
        if batches % report_every == 0:  # 一个完整区间始终在对应批次更新后报告。
            report_segment(batches)  # 输出当前区间的统计并执行回调。
            segment_start = batches + 1  # 下一区间从下一批开始。
            segment_samples = 0  # 重新累计下一区间的样本。
            segment_loss = 0.0  # 重新累计下一区间的损失。

    if not batches:  # 如果加载器没有交付任何训练数据，无法给出损失统计。
        raise ValueError("没有可用的训练批次")  # 让调用方处理空训练数据。
    if segment_samples:  # 最后一个区间未满时仍保留已完成的训练结果。
        report_segment(batches)  # 在返回前汇报尾段，并执行对应的回调。
    return {  # 汇总实际完成的训练量、用时和损失记录。
        "batches": batches,  # 记录实际完成的优化器更新次数。
        "samples": samples,  # 记录已参与参数更新的样本总数。
        "mean_huber": total_loss / samples,  # 计算全轮逐样本平均损失。
        "elapsed_seconds": perf_counter() - start - callback_seconds,  # 返回该轮耗时，排除报告回调。
        "loss_history": loss_history,  # 保存全部已完成区间的逐样本损失。
    }  # 交付训练统计。
