"""核对预测输入的批量限价检查及筛选顺序。

用合成行情覆盖交易时段、停牌和质量边界，并记录限价计算收到的数组形状。
各检查均验证原始分钟顺序和预测端审计原因。
"""

import numpy as np  # 构造合成行情，观测数组限价运算。
import pytest  # 独立执行各组筛选边界，并临时记录限价运算。

from src.features import samples  # 复用正式筛选入口和行情列位置。


def _healthy_bars(count):  # 生成常价且持续有成交的合成分钟行情。
    """输入所需分钟数 count，返回形状为（分钟数，7）的浮点行情。

    各分钟的价格均为 10 元，且汇总成交量、成交额和缩放因子均为正。
    """
    return np.tile([10.0, 10.0, 10.0, 10.0, 100.0, 1000.0, 1.0], (count, 1))  # 各分钟使用相同的合格原始值。


def test_input_limit_checks_all_eligible_predictions_in_one_array(monkeypatch):  # 确认限价取整按时段批量执行。
    """输入 pytest 的替换工具 monkeypatch，观测限价取整接收的行情数组。

    如果基础无效的预测参与除法，或合格位置没有一次批量检查，则测试失败。
    """
    minutes = samples.SESSIONS["morning"]  # 选择完整的上午连续竞价时段。
    bars = _healthy_bars(len(minutes))  # 初始化全部分钟的合成行情。
    first = samples.WINDOW - 1  # 定位第一个时间合格的预测分钟。
    bars[first, samples.FACTOR] = 0  # 该无效因子应跳过限价除法，并使后续含本行的窗口无效。
    floor = np.floor  # 保存真实取整函数，供观测时继续计算。
    checked_shapes = []  # 记录每次限价取整接收的数据形状。

    def observe_floor(values):  # 捕获限价取整的输入范围。
        """输入 values 为取整前的分值，记录其形状并返回真实取整结果。"""
        checked_shapes.append(np.shape(values))  # 核对限价取整调用是否覆盖一组预测分钟。
        return floor(values)  # 使用实际取整函数继续筛选。

    monkeypatch.setattr(samples.np, "floor", observe_floor)  # 让正式筛选入口经过可观测的限价取整。
    with np.errstate(divide="raise", invalid="raise"):  # 若错误地对无效因子除法，立即令测试失败。
        reasons = list(samples._input_reasons(bars, minutes, (900, 1100), (), False))  # 运行当前时段的输入筛选。
    expected_count = len(minutes) - samples.HORIZON - first - samples.WINDOW  # 扣除时间不合格与含无效因子的窗口。
    assert checked_shapes == [(expected_count,)]  # 剩余预测端应在一个一维数组中一起取整。
    assert reasons[first:first + samples.WINDOW] == ["invalid_input"] * samples.WINDOW  # 确认所有包含无效行的窗口均拒绝。


def test_input_window_quality_precedes_limit_and_zero_volume():  # 检查重叠输入窗口中各原因的优先级与恢复位置。
    """使用合成行情检查缺失、无效、限价和预测端零成交；无返回值。

    筛选原因、窗口恢复位置或输入原数据发生变化时，触发断言异常。
    """
    minutes = samples.SESSIONS["morning"]  # 采用上午交易时段的分钟末时刻。
    bars = _healthy_bars(len(minutes))  # 构造干净的分钟行情作为对照。
    first = samples.WINDOW - 1  # 获取首个能够形成完整输入的预测行号。
    bars[0, samples.FACTOR] = 0  # 首行无效，和下一个缺失行同时进入首个窗口。
    bars[1, samples.AMOUNT] = np.nan  # 第二行缺失，最后进入预测行号为 first + 1 的窗口。
    bars[2, samples.VOLUME] = -1  # 第三行无效，最后进入预测行号为 first + 2 的窗口。
    bars[first + 3, :4] = 11.01  # 下一时点收盘价超出涨停价。
    bars[first + 3:first + 5, samples.VOLUME] = 0  # 前者应优先记录限价，后者则记录预测端零成交。
    original = bars.copy()  # 保存原始值，用于确认筛选保持行情不变。
    reasons = list(samples._input_reasons(bars, minutes, (900, 1100), (), False))  # 一次取得结果用于检查时间和优先级。
    assert reasons[first:first + 6] == [  # 按时序检查异常离开窗口后各预测时点的首因。
        "missing_input",  # 缺失优先于窗口内同行外的无效值。
        "missing_input",  # 缺失行仍位于下一个窗口。
        "invalid_input",  # 缺失行离开后继续检查仍在窗口内的无效值。
        "close_at_price_limit",  # 超限与零成交同时发生时优先报告超限。
        "zero_volume_at_prediction",  # 成交量为零的当前预测时点单独拒绝。
        None,  # 已过零成交的预测端，历史中的零成交分钟仍可留在输入窗口。
    ]  # 完成相邻预测窗口的逐行核对。
    np.testing.assert_array_equal(bars, original)  # 批量筛选不应修改原始行情。


def test_input_price_limit_compares_rounded_ticks():  # 核对上下限边缘需要先按报价精度取整。
    """使用略低于和略高于半分的合成收盘价核对两侧限价；无返回值。

    限价比较忽略还原因子或使用未按分取整的价格时，触发断言异常。
    """
    minutes = samples.SESSIONS["morning"]  # 采用上午时段，保证四个测试分钟的窗口均完整。
    bars = _healthy_bars(len(minutes))  # 初始化有成交的有效行情。
    first = samples.WINDOW - 1  # 从首个满足时间条件的预测分钟开始插入价格。
    raw_prices = np.array([9.0049, 9.0051, 10.9949, 10.9951])  # 给出两侧限价边缘的未复权价格。
    factors = np.array([0.25, 0.5, 2.0, 4.0])  # 各行使用不同缩放系数。
    bars[first:first + 4, samples.FACTOR] = factors  # 保存还原价格所需的分钟级缩放因子。
    bars[first:first + 4, :4] = (raw_prices * factors)[:, None]  # 用 Qlib 口径存储开高低收各列。
    reasons = list(samples._input_reasons(bars, minutes, (900, 1100), (), False))  # 按分比较两侧限价。
    assert reasons[first:first + 4] == ["close_at_price_limit", None, None, "close_at_price_limit"]  # 核对取整后严格在区间内的两个位置。


@pytest.mark.parametrize(  # 分别核对停牌区间恰好相切和亚秒级重叠。
    ("edge", "overlap", "expected"),  # 指定窗口的哪一侧、相交秒数及首个拒绝原因。
    (  # 将停牌时刻移到输入窗口边界两侧。
        ("start", 0, "missing_input"),  # 停牌恰在首根分钟开始时结束，两者不重叠。
        ("start", 0.125, "input_overlaps_halt"),  # 停牌晚于首根分钟开始 0.125 秒，优先报告停牌。
        ("end", 0, "missing_input"),  # 停牌恰在预测分钟结束时开始，两者不重叠。
        ("end", 0.125, "input_overlaps_halt"),  # 停牌早于预测分钟结束 0.125 秒，优先报告停牌。
    ),  # 完成两个时间边界的案例设定。
)  # 分别运行每种时间关系的生产筛选。
def test_input_halts_preserve_subminute_boundaries(edge, overlap, expected):  # 核对停牌重叠时的秒数精度和判断优先级。
    """输入窗口侧 edge、重叠秒数 overlap 和期望的首因 expected；无返回值。

    停牌边界或与缺失、市场异常的优先级有误时，触发断言异常。
    """
    minutes = samples.SESSIONS["morning"]  # 使用正常上午时段。
    bars = _healthy_bars(len(minutes))  # 创建没有成交和价格异常的行情。
    bars[0, samples.AMOUNT] = np.nan  # 所测试窗口同时缺少一个字段，用于判断原因优先级。
    first = samples.WINDOW - 1  # 选取首个输入历史完整的预测时点。
    start_seconds = (minutes[0] - 1) * 60  # 获取输入窗口首根分钟的开始秒数。
    end_seconds = minutes[first] * 60  # 获取窗口末根分钟的结束秒数。
    intervals = {  # 明确构造相切或轻微重叠的停牌区间。
        "start": (start_seconds - 60, start_seconds + overlap),  # 停牌从输入开始前延续至窗口开端。
        "end": (end_seconds - overlap, end_seconds + 60),  # 停牌在预测时点附近开始并延续至未来。
    }  # 获取可传入筛选入口的真实单位区间。
    halts = (intervals[edge],)  # 仅传入当前案例对应的停牌区间。
    bounds = (900, 1100)  # 指定以分为单位的跌停和涨停价。
    reasons = list(samples._input_reasons(bars, minutes, bounds, halts, False))  # 检查输入窗口与停牌的相交关系。
    assert reasons[first] == expected  # 仅实际重叠的停牌优先于行情缺失。
    excluded = list(samples._input_reasons(bars, minutes, bounds, halts, True))  # 同一时段另有市场异常时检查最高优先级。
    assert excluded[first - 1:first + 1] == ["insufficient_history", "market_trading_anomaly"]  # 时间边界先于市场异常，市场异常先于停牌。
    assert excluded[-samples.HORIZON:] == ["target_outside_session"] * samples.HORIZON  # 末尾时间边界同样先于市场异常。


@pytest.mark.parametrize("count", (0, samples.WINDOW - 1, samples.WINDOW, samples.WINDOW + samples.HORIZON - 1))  # 各种无法产生候选的时段长度。
def test_input_handles_sessions_without_eligible_windows(count):  # 无合格窗口时仍逐行给出时间边界原因。
    """输入合成时段的分钟数 count，验证没有可采样窗口时的原因；无返回值。

    产生意外候选、遗漏分钟或颠倒时间原因优先级时，触发断言异常。
    """
    minutes = samples.SESSIONS["morning"][:count]  # 从正式上午时段取需要的分钟数。
    bars = _healthy_bars(count)  # 令行情质量合格，只让时间条件起作用。
    history_count = min(count, samples.WINDOW - 1)  # 输入历史不足的行应优先得到对应原因。
    outside_count = count - history_count  # 其余行只有可能因目标超出时段被拒绝。
    reasons = list(samples._input_reasons(bars, minutes, (900, 1100), (), True))  # 故意同时排除市场异常，检查时间边界仍然优先。
    assert reasons == ["insufficient_history"] * history_count + ["target_outside_session"] * outside_count  # 核对全部分钟的首因。
