"""检查批量归一化与原逐窗口算法的数值及样本状态一致。

输入包括不同精度、正成交奇偶数、零成交及拒绝条件重合的合成行情。
测试逐位比较有效特征，并检查候选的第一剔除原因和事后标签。
"""

from decimal import Decimal  # 用十进制数构造精确的前收盘价。

import numpy as np  # 构造合成窗口并逐位比较归一化结果。
import pytest  # 为每种输入精度和剔除顺序运行独立的测试。

from src.features import samples  # 访问批量归一化入口、行情字段和正式时段样本接口。


def _scalar_reference(window):  # 用逐窗口公式生成独立的归一化基准。
    """输入一个形状为（30，7）的有效行情窗口。

    返回形状为（30，6）的 float32 归一化特征，缺少正股数或正成交额时返回 None。
    输入的形状或类型不支持数组运算时，由 NumPy 抛出异常。
    """
    shares = window[:, samples.VOLUME] * window[:, samples.FACTOR]  # 以逐元素乘法将成交量换为原始股数。
    amount = window[:, samples.AMOUNT]  # 读取以元计的成交额。
    positive_shares = shares[shares > 0]  # 按原规则排除零股数，再求基准。
    positive_amount = amount[amount > 0]  # 单独筛选正成交额，股数列的正值位置不必相同。
    if not len(positive_shares) or not len(positive_amount):  # 识别任一特征缺少正值的窗口。
        return None  # 无法按原规则完成归一化。
    prices = np.log(window[:, :4] / np.median(window[:, samples.CLOSE]))  # 以标量收盘价中位数归一化四个价格字段。
    scaled_shares = np.log1p(shares / np.median(positive_shares))  # 分别计算每分钟的成交股数特征。
    scaled_amount = np.log1p(amount / np.median(positive_amount))  # 分别计算每分钟的成交额特征。
    return np.column_stack((prices, scaled_shares, scaled_amount)).astype(np.float32)  # 按模型约定拼列并转为单精度。


@pytest.mark.parametrize("dtype", (np.float32, np.float64))  # 覆盖不同输入精度在求中位数和对数时的舍入。
def test_batch_normalization_matches_scalar_bit_for_bit(dtype):  # 验证批内各窗口之间没有共享统计量。
    """输入 pytest 提供的行情精度，逐位核对批量特征与原公式；无返回值。

    批量特征、有效性掩码、输入不变性或输出精度发生偏差时，抛出断言异常。
    """
    rng = np.random.default_rng(821)  # 固定随机种子，产生顺序打乱且不同尺度的正值。
    base = rng.lognormal(0, 2, size=(14, samples.WINDOW, 7)).astype(dtype)  # 生成不重复的正价格和成交字段。
    windows = base[::2]  # 模拟按步长切出的非连续窗口批。
    windows[1, :9, samples.VOLUME] = 0  # 在第二行留下 21 个正股数，覆盖奇数中位数。
    windows[1, 8:18, samples.AMOUNT] = 0  # 在第二行留下 20 个正金额，覆盖偶数中位数。
    windows[2, 5:15, samples.VOLUME] = 0  # 在第三行留下 20 个正股数，交换奇偶组合。
    windows[2, :9, samples.AMOUNT] = 0  # 在第三行留下 21 个正金额。
    windows[3, 1:, samples.VOLUME] = 0  # 在第四行保留一个股数基准。
    windows[3, :-1, samples.AMOUNT] = 0  # 在另一分钟保留唯一成交额基准。
    windows[4, :, samples.VOLUME] = 0  # 第五行没有正股数，应标记无有效归一化。
    windows[5, :, samples.AMOUNT] = 0  # 第六行没有正金额，应标记无有效归一化。
    windows[6, :, [samples.VOLUME, samples.AMOUNT]] = 0  # 第七行两个基准均缺失。
    original = windows.copy()  # 保存输入副本，用于检测批量计算是否改写行情。
    expected_rows = [_scalar_reference(window) for window in windows]  # 逐窗口独立计算基准。
    expected_mask = [row is not None for row in expected_rows]  # 标记原算法可处理的窗口。
    features, valid = samples._normalize_windows(windows)  # 调用生产入口批量计算特征。
    assert features.shape == (7, samples.WINDOW, 6) and features.dtype == np.float32  # 维持批量行对齐及模型输入精度。
    assert valid.shape == (7,) and valid.dtype == np.bool_  # 每条输入窗口应有一个布尔标记。
    np.testing.assert_array_equal(valid, expected_mask)  # 核对缺少正基准的三条窗口都无效。
    np.testing.assert_array_equal(windows, original)  # 确认排序或选位过程中未原地修改输入。
    expected = np.stack([row for row in expected_rows if row is not None])  # 按原顺序收集全部有效行。
    np.testing.assert_array_equal(features[valid].view(np.uint32), expected.view(np.uint32))  # 比较位模式而不放宽浮点误差。


@pytest.mark.parametrize(  # 将容易先后颠倒的拒绝条件组合为同一候选。
    ("conditions", "expected"),  # 本行参数分别给出冲突条件和正确的首个原因。
    (  # 列出需覆盖的具体冲突。
        (("excluded", "input_halt", "missing"), "market_trading_anomaly"),  # 已排除的市场时段优先。
        (("input_halt", "missing", "invalid"), "input_overlaps_halt"),  # 停牌相交优先于行情错误。
        (("missing", "invalid", "limit"), "missing_input"),  # 输入缺失优先于输入无效与限价。
        (("invalid", "limit", "prediction_zero"), "invalid_input"),  # 输入无效优先于限价与预测端零成交。
        (("limit", "prediction_zero", "no_median"), "close_at_price_limit"),  # 限价优先于零成交和缺少中位数。
        (("prediction_zero", "no_median", "future_missing"), "zero_volume_at_prediction"),  # 预测端零成交优先于归一化与目标缺失。
        (("no_median", "future_missing", "future_halt"), "missing_positive_median"),  # 归一化失败应当先于所有事后标签原因。
        (("future_missing", "future_halt"), "halt_before_target"),  # 合格候选在未来停牌与目标缺失重合时保留输入。
    ),  # 完成全部组合定义。
)  # 让 pytest 对各组合运行一次验收。
def test_session_keeps_first_applicable_reason(conditions, expected):  # 确认批量预计算没有改变时段内原因顺序。
    """输入冲突条件和预期首个原因，检查指定时点的样本状态；无返回值。

    优先级、可保留窗口或事后标签与原规则不一致时，抛出断言异常。
    """
    minutes = samples.SESSIONS["morning"]  # 选用上午连续竞价的时钟序列。
    bars = np.tile([10.0, 10.0, 10.0, 10.0, 100.0, 1000.0, 1.0], (len(minutes), 1))  # 构造全部合法且可归一化的行情。
    t = samples.WINDOW - 1  # 选取第一个完整输入窗口的结束位置（10:00）。
    target = t + samples.HORIZON  # 定位 5 分钟后的标签时点。
    halts = []  # 收集本组条件需要的已知停复牌区间。
    if "input_halt" in conditions:  # 根据测试条件在历史输入中插入停牌。
        halts.append(((minutes[t] - 10) * 60, (minutes[t] - 9) * 60))  # 将区间完全放在预测之前。
    if "missing" in conditions:  # 根据测试条件制造一个输入缺失字段。
        bars[2, samples.AMOUNT] = np.nan  # 选用窗口早期的成交额，避免改变预测端字段。
    if "invalid" in conditions:  # 根据测试条件制造另一条无效输入。
        bars[3, samples.VOLUME] = -1  # 用负成交量触发基础有效性检查。
    if "limit" in conditions:  # 根据测试条件令预测时点触及涨停。
        bars[t, :4] = 11  # 前收 10 元，涨停为 11 元。
    if "prediction_zero" in conditions:  # 根据测试条件令预测分钟没有成交。
        bars[t, samples.VOLUME] = 0  # 将预测端成交量设为零。
    if "no_median" in conditions:  # 根据测试条件去除整个输入窗口的正金额。
        bars[:t + 1, samples.AMOUNT] = 0  # 保留成交量字段，单独制造金额中位数缺失。
    if "future_missing" in conditions:  # 根据测试条件令目标端无法获得收盘价。
        bars[target, samples.CLOSE] = np.nan  # 只改未来标签相关字段。
    if "future_halt" in conditions:  # 根据测试条件在预测后、目标前插入停牌。
        halts.append(((minutes[t] + 1) * 60, (minutes[t] + 2) * 60))  # 该区间不会与历史窗口重合。
    bounds = samples.limit_cents(Decimal("10.00"))  # 获取研究口径的预测端限价。
    session = samples.sample_session(  # 生成连续竞价时段的候选数组和逐分钟原因。
        bars, minutes, bounds, halts,  # 传入全部合成行情和同一时段的业务条件。
        excluded="excluded" in conditions,  # 由组合决定是否排除整个时段。
    )  # 完成该时段的样本构造。
    assert len(session.reasons) == len(minutes) and session.reasons[0] == "insufficient_history"  # 开头仍逐行记录历史不足。
    assert session.reasons[-1] == "target_outside_session"  # 结尾仍先记录目标跨出时段。
    assert session.reasons[t] == expected  # 核对指定分钟的首个剔除原因或事后标签状态。
    if expected == "halt_before_target":  # 预测输入有效而目标标签受事后停牌影响。
        assert session.positions[0] == t  # 首个时间合格的预测位置应仍出现在候选数组中。
        expected_features = _scalar_reference(bars[:t + 1])  # 重新用独立公式计算截至预测时点的特征。
        np.testing.assert_array_equal(session.x[0], expected_features)  # 未来异常不能改变预测输入。
        assert not session.label_valid[0] and np.isnan(session.y[0])  # 将对应收益率保留为缺失标签。
    else:  # 其余原因都在预测时点排除候选。
        assert t not in session.positions  # 输入不合格的分钟不应出现在候选数组中。
