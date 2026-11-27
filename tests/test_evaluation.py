"""用可以手算秩相关的合成时段样本验证指标。

测试通过正式验证入口筛选股票日及读取分钟行情，并用受控的候选窗口与标签检查批量推理及汇总。
"""

import json  # 核对空指标和正常数值可用标准 JSON 表示。
from decimal import Decimal  # 构造分钟行情筛选所需的前收参考价。
from types import SimpleNamespace  # 提供测试所需的行情和元数据对象。

import numpy as np  # 构造带并列名次、缺失标签的候选样本。
import pytest  # 提供测试夹具及精度判断。
import torch  # 构造与正式训练入口兼容的试验模型。

from src.features.samples import (  # 使用正式的样本字段和窗口形状。
    FEATURES,  # 列出输入特征。
    WINDOW,  # 指定每个输入的历史分钟长度。
    SessionSamples,  # 按正式采样接口组织时段数组。
)  # 完成样本接口导入。
from src.models import evaluation  # 通过正式验证接口计算覆盖率和两个秩相关。
from src.models.training_data import (  # 使用统一的特征输入接口。
    prepare_model_inputs,  # 取得指定验证分钟的预期时间特征。
)  # 完成特征接口导入。

DAYS = ("2025-01-02", "2025-01-06")  # 使用两个验证日，以检验日期等权。
CODES = tuple(f"SH{600000 + i}" for i in range(23))  # 定义二十三个有序的合成沪市股票代码。


class _ReadReturn(torch.nn.Module):  # 创建从输入直接取得预设预测值的模型。
    """不学习参数，按实际输入批维交付每个窗口的预设收益预测。"""

    def __init__(self):  # 初始化用于观察批次大小的模型。
        """无需参数，创建记录各次推理批次长度的列表。"""
        super().__init__()  # 初始化 PyTorch 模型。
        self.batch_sizes = []  # 保存每个输入批的样本数。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 读取预先写入指定特征位置的预测值。
        """x 为 float32 [批大小, 6, 30]，返回模型预测的一维张量。模式或输入不符时触发断言。"""
        assert not self.training and torch.is_inference_mode_enabled()  # 验证调用发生在关闭梯度的评估模式。
        assert x.shape[1:] == (6, 30) and x.dtype == torch.float32  # 验证模型收到与训练一致的特征接口。
        self.batch_sizes.append(len(x))  # 记录实际推理批次大小。
        return x[:, 0, 0]  # 读取样本构造夹具为当前窗口写入的预测值。


@pytest.fixture  # 为每项测试提供全新的行情和已读取股票日列表。
def validation_data(monkeypatch):  # 构造两天合成分钟行情及相应元数据。
    """接收 pytest 替换工具，返回行情库、日线元数据读取器及按实际顺序读取的股票日。"""
    reads = []  # 记录正式验证流程实际读取的股票日。

    def session_bars(code, day, minutes):  # 将股票日标识交给受控采样函数。
        """根据股票代码、日期和所需的分钟序列返回对应的模拟分钟行情。"""
        reads.append((code, day))  # 观察是否按规范代码升序读取。
        identity = [DAYS.index(day), CODES.index(code)]  # 选取当前日期和股票的合成序号。
        return np.tile(identity, (len(minutes), 1))  # 将日期和股票序号沿时段传递。

    def candidates(bars, minutes, bounds, halts):  # 构造具有可控秩相关、并列值和标签缺失的样本。
        """按传入行情标记、分钟时点及采样条件返回正式 SessionSamples 结构。"""
        day, code = bars[0]  # 取出当前时段所属的日期和股票序号。
        morning = minutes[0] < 12 * 60  # 仅上午生成候选，保留正式连续竞价时段切分。
        count = (20 + 2 * day) if morning else 0  # 两天分别有二十和二十二个同分钟的截面组。
        positions = np.arange(WINDOW - 1, WINDOW - 1 + count, dtype=np.intp)  # 设置按分钟排序的候选索引。
        y = (np.arange(count) // 2 + code // 2).astype(np.float64)  # 制造时序和截面方向上的并列收益。
        if day == 0 and code == len(CODES) - 1:  # 第一天最后一只股票的日内收益恒定。
            y.fill(12.0)  # 此股时序秩相关无定义，仍保留同分钟截面的标签。
        x = np.zeros((count, WINDOW, len(FEATURES)), dtype=np.float32)  # 构造模型接口使用的特征窗口。
        x[:, 0, 0] = y if day == 0 else -y  # 令两天分组内的秩相关分别精准为正一与负一。
        if day == 0 and code == 0 and count:  # 第一天第一只股票只有十九对有效样本。
            y[0] = np.nan  # 保留候选窗口，去掉这一条的事后标签。
        valid = np.isfinite(y)  # 按样本接口提供标签有效掩码。
        reasons = np.full(len(minutes), "insufficient_history", dtype=object)  # 保留逐分钟前置筛选原因。
        reasons[positions] = np.where(valid, "accepted", "missing_target")  # 为原位置补充目标端标签状态。
        return SessionSamples(positions, x, y, valid, reasons)  # 返回合成的时段数组。

    library = SimpleNamespace(  # 提供验证流程所需的 Qlib 查询接口。
        dates=set(DAYS),  # 暴露允许枚举的验证交易日。
        spans=dict.fromkeys((*reversed(CODES), "SH600999")),  # 目录倒序并包含一个别名。
        has_coverage=lambda code, day: True,  # 所有目录代码在这两天均有行情覆盖。
        session_bars=session_bars,  # 让正式时段切分使用上述合成行情。
    )  # 完成合成行情库。
    metadata = SimpleNamespace(  # 提供正式股票日筛选使用的元数据字段。
        canonical_code=lambda code, day: CODES[0] if code == "SH600999" else code,  # 将别名映射到同一股票。
        historical_code=lambda code, day: code,  # 其余代码在交易所沿用 Qlib 代码。
        is_mainboard=lambda code, day: day == DAYS[0] or code in CODES[3:],  # 第二天排除三只股票，仅剩二十只。
        day=lambda code, day: SimpleNamespace(  # 返回普通可交易的股票日状态。
            preclose=Decimal("10.00"), st=False, trading=True, ipo=False, delist=False,  # 给出日线参考价和资格标志。
            full_halt=False, intraday_halts=(),  # 测试行情没有停牌。
        ),  # 完成单个股票日状态。
    )  # 完成股票日元数据对象。
    monkeypatch.setattr(evaluation, "sample_session", candidates)  # 仅替换行情转窗口步骤，保留正式的日级筛选和批量流程。
    return library, metadata, reads  # 交付夹具并暴露实际读取记录。


def test_evaluate_averages_dates_equally_and_reports_coverage(validation_data):  # 验证指标聚合、过滤及批量推理。
    """从夹具取两天不等分组数的预设收益，核对等权日均值、覆盖率和分组计数。"""
    library, metadata, reads = validation_data  # 取得测试用行情、状态和读取记录。
    model = _ReadReturn()  # 创建一个不需要训练就能读取预设预测的模型。
    result = evaluation.evaluate(  # 从调换顺序的验证日期计算两个秩相关。
        model, library, metadata, reversed(DAYS),  # 交给验证入口按日期和股票代码排序。
        batch_size=64, device="cpu",  # 一批覆盖多个股票日的窗口。
    )  # 完成两个交易日的验证。
    assert model.training  # 验证结束后恢复调用前的训练模式。
    assert reads == [(code, DAYS[0]) for code in CODES] + [(code, DAYS[1]) for code in CODES[3:]]  # 检查资格筛选、别名去重和顺序。
    assert model.batch_sizes == [64] * 7 + [12] + [64] * 6 + [56]  # 批次跨股票拼接，且每天输出完整尾批。
    assert result["candidates"] == 900 and result["labels_available"] == 899  # 未观测到标签的候选仍计入覆盖率分母。
    assert result["label_coverage"] == pytest.approx(899 / 900)  # 覆盖率按整个验证期候选总数计算。
    assert result["ts_group_count"] == 41 and result["cross_section_group_count"] == 42  # 排除十九对样本的组和常量时序组。
    assert result["ts_day_count"] == result["cross_section_day_count"] == 2  # 两个方向均有两个可汇总的日期。
    assert result["ts_rank_ic"] == pytest.approx(0) and result["cross_section_rank_ic"] == pytest.approx(0)  # 日期等权抵消正负相关。
    json.dumps(result, allow_nan=False)  # 确认结果可写入标准 JSON 日志。


def test_evaluate_limits_eligible_stocks_and_handles_undefined_groups(validation_data):  # 检查小规模验证的范围和空指标。
    """从有序股票中试跑两只，核对有限时序组、空截面组和尾批长度。"""
    library, metadata, reads = validation_data  # 使用与完整指标测试相同的正式市场筛选夹具。
    model = _ReadReturn().eval()  # 从调用前已经是验证模式的模型开始。
    result = evaluation.evaluate(  # 只取第二天按代码升序的两个合格股票日。
        model, library, metadata, DAYS[1:],  # 传入有三个低代码股票被排除的交易日。
        batch_size=16, device="cpu", max_stock_days_per_date=2,  # 验证上限只计入通过资格筛选的股票。
    )  # 完成受限范围的验证。
    assert reads == [(code, DAYS[1]) for code in CODES[3:5]]  # 排除低代码无资格股票后再计入试跑限制。
    assert model.batch_sizes == [16, 16, 12] and not model.training  # 检查尾批及调用前验证模式仍保留。
    assert result["cross_section_rank_ic"] is None and result["cross_section_group_count"] == 0  # 两只股票达不到截面门槛。
    assert result["ts_group_count"] == 2 and result["ts_rank_ic"] == pytest.approx(-1)  # 两只股票自己的时序组仍可计算。
    assert result["mean_huber"] == 12.5 and result["huber_samples"] == 44  # 默认尺度 1，四十四条有效配对总损失为 550。
    json.dumps(result, allow_nan=False)  # 无定义指标也应以标准 JSON 空值交付。


@pytest.mark.parametrize(  # 检查正常输出、少量非有限预测和全部预测无效三种情况。
    "invalid_predictions, pairs, total",  # 提供预测失效范围、预期配对数和手算总损失。
    [  # 汇总两个验证日的损失和配对数。
        ("none", 899, 2200.25),  # 第一日四百五十九对损失都为零，第二日四百四十对贡献全部损失。
        ("some", 893, 2197.25),  # 移除第二日六个归一化绝对误差为一的配对，每个损失 0.5。
        ("all", 0, 0.0),  # 无有限预测时没有可计算损失的配对。
    ],  # 完成场景和期望值定义。
)  # 用相同数据口径执行所有场景。
def test_evaluate_huber_uses_normalized_sample_weighted_loss(validation_data, invalid_predictions, pairs, total):  # 验证损失单位、分母和边界。
    """由 validation_data 提供标签，invalid_predictions 指定如何置换预测。pairs 和 total 分别是手算配对数和损失和。"""
    library, metadata, _ = validation_data  # 读取现有的两个合成验证日。
    model = _ReadReturn()  # 启用与现有验证回归相同的预测模型。
    model.register_buffer("return_scale", torch.tensor(4.0))  # 模拟 TCN 保存的训练标签尺度。

    def mask_predictions(_model, _inputs, outputs):  # 按测试用例选择需要置为非有限值的预测。
        """接收 PyTorch 提供的模型、输入批次和预测张量，返回保留或置换后的预测。"""
        if invalid_predictions == "all":  # 模拟没有任何有限预测的模型。
            return torch.full_like(outputs, float("inf"))  # 覆盖整个输出批次。
        if invalid_predictions == "some":  # 仅剔除第二日预测为负二的六个样本。
            return outputs.masked_fill(outputs == -2.0, float("inf"))  # 其他物理收益预测保留原值。
        return outputs  # 正常场景不干预模型输出。

    model.register_forward_hook(mask_predictions)  # 在正式模型推理流程中模拟非有限输出。
    result = evaluation.evaluate(  # 使用正式汇总入口计算损失，包含样本数不相同的两天。
        model, library, metadata, DAYS,  # 复用同一对行情和元数据来源。
        batch_size=64, device="cpu",  # 按同样的批次依次推理这两个验证日。
    )  # 完成验证统计。
    assert result["labels_available"] == 899 and result["candidates"] == 900  # 预测无效不改变标签覆盖率口径。
    assert result["huber_samples"] == pairs  # 损失分母只统计标签和预测都有限的窗口。
    expected = total / pairs if pairs else None  # 将两天损失总和除以两天配对总数。
    assert result["mean_huber"] == expected  # 验证阈值、尺度、跨日样本加权及空结果。
    assert sum(model.batch_sizes) == 900  # 汇总损失复用现有预测，每个候选只推理一次。
    json.dumps(result, allow_nan=False)  # 所有场景下输出均可写入标准 JSON 日志。


@pytest.mark.parametrize("dates", [[], [DAYS[0], DAYS[0]], [DAYS[0], "2025-07-01"], ["2025-01-03"]])  # 覆盖空、重复、测试期和非日历日期。
def test_evaluate_rejects_invalid_dates_before_reading(validation_data, dates):  # 防止无效日期范围触发验证读取。
    """传入无效日期序列，要求抛出 ValueError 并保持尚未读取任何行情。"""
    library, metadata, reads = validation_data  # 取得可观察读入过程的验证夹具。
    with pytest.raises(ValueError):  # 无效日期必须在预测或行情访问之前报错。
        evaluation.evaluate(_ReadReturn(), library, metadata, dates, batch_size=16, device="cpu")  # 按指定无效范围调用。
    assert not reads  # 输入检查应在全部行情读取之前完成。


def test_evaluate_appends_time_features_aligned_with_validation_slots(validation_data):  # 验证输入时间随窗口正确跨批拼接。
    """使用两个股票日的上午候选，逐批比较时间通道与真实预测分钟的对应关系。"""
    library, metadata, _ = validation_data  # 使用已有的真实验证编排夹具。

    class TimedReadReturn(_ReadReturn):  # 复用预设预测值，但要求八通道输入。
        """将收到的两个时间通道保存在列表中，用于核对窗口和时钟位置。"""

        input_channels = 8  # 声明模型所需的量价与时间通道数。

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # 保留实际输入，再由父类读取预设收益。
            """x 为 [批大小, 8, 30]，返回父类计算的 [批大小] 收益率。"""
            observed.append(x[:, len(FEATURES):].clone())  # 复制当前时间通道，保证尾批不会覆盖已检查的值。
            return super().forward(x[:, :len(FEATURES)])  # 复用不依赖时间变量的预设预测。

    observed: list[torch.Tensor] = []  # 保存固定容量缓冲区陆续提交的各个批次。
    result = evaluation.evaluate(  # 在第二个合成验证日处理两只股票。
        TimedReadReturn(), library, metadata, DAYS[1:],  # 使用接受八通道输入的验证模型。
        batch_size=16, device="cpu", max_stock_days_per_date=2,  # 同时覆盖完整批次、跨股票拼接和尾批。
    )  # 完成一次正式路径的验证。
    assert result["labels_available"] == 44 and result["ts_rank_ic"] == pytest.approx(-1)  # 验证排序和有效样本数保持原定义。
    minutes = np.tile(np.arange(600, 622), 2)  # 两只股票的首个完整窗口结束于 10:00，各有二十二条候选。
    x = torch.zeros(len(minutes), len(FEATURES), WINDOW)  # 创建只用于计算预期时间通道的原始特征占位。
    expected = prepare_model_inputs(x, minutes, 8)[:, len(FEATURES):]  # 按真实时钟分钟得到正确输入。
    torch.testing.assert_close(torch.cat(observed), expected, rtol=0, atol=0)  # 比对全部时间通道及其行顺序。


def test_predict_day_keeps_unlabeled_windows_and_passes_minutes(validation_data):  # 验证导出行及模型时刻入参。
    """使用首个验证日的两只股票，核对所有预测候选的代码、时刻、标签和收益率。"""
    library, metadata, reads = validation_data  # 复用包含股票别名和缺失标签的合成行情。

    class ScaledByMinute(_ReadReturn):  # 创建能检查分钟对齐的收益率模型。
        """把预设预测值乘以显式的预测时刻尺度，并记录收到的时刻。"""

        def __init__(self):  # 初始化可按日内分钟查询的各不相同的尺度。
            """不接收参数。"""
            super().__init__()  # 保留用于观察实际推理批量的基类。
            self.register_buffer("minute_return_scales", torch.arange(1, 1441) / 600)  # 设置正尺度且使各分钟有别。

        def forward(self, x: torch.Tensor, minutes: torch.Tensor) -> torch.Tensor:  # 使用显式分钟张量生成预测。
            """x 为量价窗口，minutes 为同设备的长整型预测分钟。"""
            assert minutes.dtype == torch.long and minutes.device == x.device  # 核对可直接用于模型索引。
            observed.append(minutes.clone())  # 保留本次推理各行的预测时钟分钟。
            return super().forward(x) * self.minute_return_scales[minutes]  # 返回逐分钟还原尺度的收益率。

    observed: list[torch.Tensor] = []  # 保存将被按顺序拼接的所有分钟。
    model = ScaledByMinute()  # 让导出流程识别需要时钟分钟的模型。
    result = evaluation.predict_day(  # 通过正式候选筛选和推理流程导出当日预测。
        model, library, metadata, DAYS[0],  # 第一个股票日包含一条事后收益不可观测的候选。
        batch_size=16, device="cpu", max_stock_days_per_date=2,  # 跨股票拼批并保留尾批。
    )  # 交付预测行。
    assert model.training and model.batch_sizes == [16, 16, 8]  # 恢复此前模式并核对尾批。
    assert reads == [(code, DAYS[0]) for code in CODES[:2]]  # 代码规范化去重后按顺序读取。
    minutes = np.tile(np.arange(600, 620), 2)  # 手算两个窗口序列的北京时间预测分钟。
    expected_times = np.datetime64(DAYS[0], "m") + minutes.astype("timedelta64[m]")  # 将分钟转成准确的时刻。
    np.testing.assert_array_equal(result["code"], np.repeat(CODES[:2], 20))  # 每只股票的候选行相邻。
    np.testing.assert_array_equal(result["t"], expected_times)  # 列顺序按每只股票的时刻递增。
    np.testing.assert_array_equal(torch.cat(observed), minutes)  # 实际传入尺度模型的分钟与导出时刻逐行一致。
    assert not result["label_valid"][0] and result["label_valid"].sum() == 39  # 缺失未来收益的样本仍在交付结果中。
    assert np.isnan(result["label"][0]) and len(result["prediction"]) == 40  # 缺失标签用 NaN 表示，预测不缺行。
    expected_predictions = np.tile(np.arange(20) // 2, 2) * (minutes + 1) / 600  # 手算按预测时刻变换后的收益率。
    np.testing.assert_allclose(result["prediction"], expected_predictions, rtol=1e-6)  # 核对各行模型输出。


def test_predict_day_rejects_nonfinite_predictions_and_restores_mode(validation_data):  # 避免向下游交付无效收益率。
    """将一批中一个候选预测置为非有限，检查错误行为和模型模式。"""
    library, metadata, _ = validation_data  # 取得包含真实候选流程的夹具。
    model = _ReadReturn()  # 复用与现有验证回归相同的预设预测模型。

    def invalidate(_model, _inputs, outputs):  # 保留除最后一条之外的预测。
        """将最后一个输出替换为无穷，代表不可接受的模型结果。"""
        predictions = outputs.clone()  # 复制输出避免修改原始特征缓冲区。
        predictions[-1] = torch.inf  # 至少一条无效预测即拒绝导出。
        return predictions  # 继续推理以验证导出前的有限值检查。

    model.register_forward_hook(invalidate)  # 注入异常模型输出。
    with pytest.raises(FloatingPointError, match="non-finite"):  # 核对导出在发现异常时明确报错。
        evaluation.predict_day(  # 通过正常入口处理候选窗口。
            model, library, metadata, DAYS[0], batch_size=16, device="cpu", max_stock_days_per_date=1,  # 只读一个股票日。
        )  # 不交付包含无穷值的结果。
    assert model.training  # 即使导出失败，仍恢复调用前的训练模式。
