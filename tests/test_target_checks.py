"""核对按时段批量检查目标行情时的样本行为。

输入是一段故意混合缺失、无效和零成交的合成行情。先检查逐分钟目标
状态，再对照正式样本接口与单条标签接口，验证收益率和候选窗口保持不变。
"""

from decimal import Decimal  # 以精确到分的前收盘价构造测试限价。

import numpy as np  # 构造分钟矩阵并比较候选窗口。
import pytest  # 参数化测试覆盖不同目标状态和停牌时间边界。

from src.features import samples  # 使用正式的目标状态、样本窗口和单条标签接口。


@pytest.mark.parametrize("endpoint", ("entry", "target"))  # 两端都沿用缺失、无效、零成交的优先级。
def test_endpoint_reasons_preserve_priority(endpoint):  # 对买入和退出行情执行同一组端点状态测试。
    """传入端点名 endpoint，核对逐分钟价格和成交量的原因优先级。

    如果状态顺序、冲突优先级或原行情数据被改变，抛出断言异常。
    """
    rows = (  # 按分钟构造端点价格、成交量和期望状态。
        (10.0, 100.0, "accepted"),  # 收盘价和成交量有效时可计算收益率。
        (np.nan, -1.0, "missing_target"),  # 收盘价缺失优先于无效成交量。
        (np.nan, 0.0, "missing_target"),  # 收盘价缺失优先于零成交。
        (0.0, np.nan, "missing_target"),  # 成交量缺失优先于无效收盘价。
        (np.inf, 0.0, "invalid_target"),  # 非有限收盘价优先于零成交。
        (0.0, 0.0, "invalid_target"),  # 收盘价为零也是价格无效。
        (-1.0, 3.0, "invalid_target"),  # 收盘价为负时标签无效。
        (10.0, -1.0, "invalid_target"),  # 成交量不能小于零。
        (10.0, np.inf, "invalid_target"),  # 成交量不能为无穷大。
        (10.0, 0.0, "zero_volume_at_target"),  # 价格有效、无成交时单独记为目标零成交。
        (10.0, 1.0, "accepted"),  # 末行仍应独立判断为有效。
    )  # 完成目标分钟及其预期结果的定义。
    prices = np.array([price for price, _, _ in rows])  # 仅取该端点需要的开盘价或收盘价。
    volumes = np.array([volume for _, volume, _ in rows])  # 取每分钟对应的成交量。
    original = np.stack((prices, volumes))  # 保存两列行情副本以检测原地修改。
    result = samples._endpoint_reasons(prices, volumes, endpoint)  # 检查批量端点规则。
    expected = [reason.replace("target", endpoint) for _, _, reason in rows]  # 将测试预期中的原因改为对应端点。
    np.testing.assert_array_equal(result, expected)  # 核对各分钟的首个无效原因。
    np.testing.assert_array_equal(np.stack((prices, volumes)), original)  # 批量端点检查不应修改输入行情。


def test_session_checks_both_endpoints_and_keeps_candidate_windows(monkeypatch):  # 验证每个端点各一次批量检查且保留合格预测。
    """输入 pytest 的临时替换工具，验证批量调用与候选标签；无返回值。

    若有多次检查、目标无效导致窗口丢失、未来信息改变输入或收益率错误，抛出断言异常。
    """
    minutes = samples.SESSIONS["morning"]  # 采用上午连续竞价的实际分钟顺序。
    healthy = np.tile([10.0, 10.0, 10.0, 10.0, 100.0, 1000.0, 1.0], (len(minutes), 1))  # 创建输入和目标都正常的基准行情。
    altered = healthy.copy()  # 将异常修改限制在使基准输入仍可手算的未来分钟。
    t = samples.WINDOW - 1  # 选择第一个历史窗口完整的预测时点。
    target = t + samples.HORIZON  # 定位该采样的首个目标分钟。
    altered[target, :4] = 10.2  # 令第一个目标时点有正收益率。
    altered[target + 1, samples.CLOSE] = np.nan  # 制造只有未来收盘价缺失的目标。
    altered[target + 1, samples.VOLUME] = -1  # 给同一目标设无效量，检查缺失优先。
    altered[target + 2, [samples.CLOSE, samples.VOLUME]] = 0  # 目标价无效且零成交时应先记录无效。
    altered[target + 3, samples.VOLUME] = 0  # 目标价格正常时保留零成交原因。
    bounds = samples.limit_cents(Decimal("10.00"))  # 计算普通股票预测时点适用的限价。
    expected = samples.sample_session(healthy, minutes, bounds, ())  # 保留未改动未来行情时的对照候选。
    endpoint_check = samples._endpoint_reasons  # 保存原批量检查实现，以便在记录后正常调用。
    calls = []  # 记录正式时段检查过的端点及输入行数。

    def observe_endpoint_rows(prices, volumes, endpoint):  # 记录端点检查收到的候选行数。
        """接收对齐的价格、成交量数组及端点名，返回真实的端点状态。

        检查失败时由原批量实现报告异常。
        """
        calls.append((endpoint, len(prices), len(volumes)))  # 确认价格与成交量均按候选批量提供。
        return endpoint_check(prices, volumes, endpoint)  # 委托真实端点规则进行分类。

    monkeypatch.setattr(samples, "_endpoint_reasons", observe_endpoint_rows)  # 从正式样本入口观测批量实现。
    actual = samples.sample_session(altered, minutes, bounds, ())  # 在同一时段处理混合的退出状态。
    count = len(actual.positions)  # 读取正式样本返回的候选数。
    expected_calls = [("entry", count, count), ("target", count, count)]  # 每个端点的全部候选应在一次检查中完成。
    assert calls == expected_calls  # 核对单次批量检查和入参与候选索引的对齐。
    excluded = samples.sample_session(altered, minutes, bounds, (), excluded=True)  # 构造整个时段排除的空候选结果。
    assert calls == expected_calls  # 没有候选时不应检查任何端点。
    assert excluded.positions.shape == excluded.y.shape == excluded.label_valid.shape == (0,)  # 空结果保留一维索引、标签和掩码。
    assert excluded.x.shape == (0, samples.WINDOW, len(samples.FEATURES))  # 空特征保留时间和通道维。
    assert actual.positions.dtype == np.dtype(np.intp) and np.all(np.diff(actual.positions) > 0)  # 核对候选行索引的类型和时间顺序。
    assert actual.x.dtype == np.float32 and actual.y.dtype == np.float64  # 核对特征和标签保留各自的数值精度。
    assert actual.label_valid.dtype == np.dtype(bool)  # 标签可用性应使用布尔掩码表示。
    np.testing.assert_array_equal(actual.positions[:4], np.arange(t, t + 4))  # 目标缺失、无效或零成交均保留原候选。
    expected_reasons = ("accepted", "missing_target", "invalid_target", "zero_volume_at_target")  # 记录前四条候选的预期标签状态。
    np.testing.assert_array_equal(actual.reasons[t:t + 4], expected_reasons)  # 按预测分钟核对四种结果。
    np.testing.assert_array_equal(actual.x[:4], expected.x[:4])  # 未来价格和成交不能改变四个历史输入窗口。
    np.testing.assert_array_equal(actual.label_valid[:4], [True, False, False, False])  # 只认可第一个标签可用。
    assert actual.y[0] == np.log(np.float64(10.2) / np.float64(10.0))  # 保留同一收益率公式和精度。
    assert np.isnan(actual.y[1:4]).all()  # 其余候选按原规则使用 NaN 收益率。


@pytest.mark.parametrize(  # 比较预测后停牌与不相交边界的不同结果。
    ("halt_boundary", "entry_missing", "target_missing", "reason"),  # 参数给出停牌边界、两端价格状态和预期原因。
    (  # 定义独立的单条标签案例。
        ("inside", True, True, "halt_before_target"),  # 预测后停牌优先于买入和退出异常。
        ("starts_at_target", False, True, "missing_target"),  # 恰好在退出分钟收盘时开始的停牌不重合。
        ("ends_at_prediction", True, True, "missing_entry"),  # 两端都缺价时记录买入端原因。
        ("ends_at_prediction", False, False, "accepted"),  # 恰好在预测分钟末结束的停牌不重合。
    ),  # 完成停牌边界与双端异常的组合。
)  # 分别验收原单条标签接口。
def test_scalar_target_label_preserves_halt_precedence(halt_boundary, entry_missing, target_missing, reason):  # 验证单条调用的优先级与收益率。
    """输入停牌边界、两端是否缺价和预期原因，核对单条标签结果。

    单条调用在边界、缺失判断或有效收益率上发生变化时，抛出断言异常。
    """
    minutes = samples.SESSIONS["morning"]  # 使用同一上午连续竞价时段。
    t = samples.WINDOW - 1  # 选取第一个可预测时点。
    target = t + samples.HORIZON  # 根据固定标签间隔找到目标分钟前的闭合时点。
    bars = np.tile([10.0, 10.0, 10.0, 10.0, 100.0, 1000.0, 1.0], (len(minutes), 1))  # 构造正值完整的分钟行情。
    bars[t, samples.CLOSE] = 10.05  # 预测时点的收盘价与买入开盘价不同。
    bars[t + samples.ENTRY_OFFSET, samples.OPEN] = np.nan if entry_missing else 10.1  # 设置下一分钟的开盘价。
    bars[target, samples.CLOSE] = np.nan if target_missing else 10.2  # 设置退出分钟的收盘价。
    prediction_seconds = minutes[t] * 60  # 将预测分钟的结束时间转为秒。
    target_seconds = minutes[target] * 60  # 将目标分钟的结束时间转为秒。
    intervals = {  # 设置跨预测到目标区间的三种停牌关系。
        "inside": (prediction_seconds + 60, target_seconds - 60),  # 区间落在预测之后且目标之前。
        "starts_at_target": (target_seconds, target_seconds + 60),  # 区间从目标结束时才开始。
        "ends_at_prediction": (prediction_seconds - 60, prediction_seconds),  # 区间在预测结束时结束。
    }  # 已确定可直接传入原接口的停牌区间。
    actual_reason, label = samples.label_for_target(bars, minutes, t, target, (intervals[halt_boundary],))  # 使用原单条标签方法。
    assert actual_reason == reason  # 核对停牌边界决定的第一标签状态。
    if reason == "accepted":  # 只有输入有效且未来停牌不重合才计算收益。
        assert label == np.log(np.float64(10.2) / np.float64(10.1))  # 用退出收盘价和买入开盘价核对对数收益。
    else:  # 缺失和停牌场景均无法观察实际收益。
        assert np.isnan(label)  # 原接口仍以 NaN 返回无效标签。
