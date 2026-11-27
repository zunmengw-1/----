"""用少量手算行情核验窗口和 5 分钟标签。

输入是连续竞价时段的固定价格和成交量，先生成有效标签，再改变未来行情或插入未来停牌。
测试检查标签的数值和是否可用，同时确认所有预测输入保持一致。
"""

from decimal import Decimal  # 导入精确表示以元计价格的十进制数类型。
from math import log  # 导入自然对数，用于写出可手算的特征和标签示例。

import numpy as np  # 导入数组运算与数组比对工具。
import pytest  # 导入测试框架，用于分开运行正常和异常目标时点的测试。

from src.features.samples import (  # 使用与正式样本相同的列位置、时段和生成函数。
    CLOSE,  # 获取收盘价的列索引，用于制造目标端价格缺失。
    ENTRY_OFFSET,  # 定位预测时点之后的买入分钟。
    HORIZON,  # 获取预测时点到退出的分钟距离。
    OPEN,  # 获取开盘价列索引，用于制造买入端价格异常。
    SESSIONS,  # 获取上午连续竞价每根分钟线的结束时刻。
    VOLUME,  # 获取分钟成交量的列索引，用于制造买入端零成交。
    WINDOW,  # 获取输入窗口包含的分钟线数量。
    limit_cents,  # 从股票的前收盘价计算整数分表示的上下限。
    sample_session,  # 导入正式的样本生成的时段入口。
)  # 完成样本规则接口的导入。


@pytest.mark.parametrize(  # 对比买入端、退出端和停牌异常对标签的影响。
    "target_state",  # 指定预测时点之后的行情状态。
    ("available", "missing_entry", "invalid_entry", "zero_volume_at_entry", "missing_target", "halt_before_target"),  # 覆盖买入端三类异常及其他原因。
)  # 对每种未来状态生成独立的样本。
def test_future_state_changes_label_without_changing_input(target_state):  # 通过价格已知的合成行情核验特征、标签和归属边界。
    """输入 pytest 提供的目标状态，检查手算收益率与事后标签筛选；无返回值。

    出现错误收益、未来信息影响输入或未按状态标记标签时抛出断言异常。
    """
    minutes = SESSIONS["morning"]  # 使用上午时段，避免午休对时间对齐造成影响。
    start_index = WINDOW - 1  # 选择上午第一条拥有完整历史输入的预测时点。
    entry_index = start_index + ENTRY_OFFSET  # 定位预测后的首根分钟线。
    target_index = start_index + HORIZON  # 定位预测之后第五根分钟线。
    bars = np.tile([10.0, 10.0, 10.0, 10.0, 100.0, 1000.0, 1.0], (len(minutes), 1))  # 创建价格稳定且有成交的分钟行情。
    bars[start_index, :4] = 10.2  # 设定预测端收盘价，窗口归一化的收盘价中位数仍为 10 元。
    bars[entry_index, :4] = [10.1, 10.15, 10.08, 10.12]  # 买入开盘价不同于预测端收盘价及买入分钟的收盘价。
    bars[target_index, :4] = 10.3  # 设定 10:05 的目标价格，与起始价格不同。
    bounds = limit_cents(Decimal("10.00"))  # 设置 ±10% 的涨跌幅限制，两个价格均未触及。
    halts = []  # 默认整个时段都没有已知停牌。
    if target_state == "missing_entry":  # 未来买入开盘价缺失时只标记标签无效。
        bars[entry_index, OPEN] = np.nan  # 修改未来字段，不改变截至预测分钟的已知量价。
    if target_state == "invalid_entry":  # 未来买入开盘价无效时仍保留合格输入。
        bars[entry_index, OPEN] = 0  # 用非正价格触发买入价无效。
    if target_state == "zero_volume_at_entry":  # 确认开盘价存在也不能代替成交量检查。
        bars[entry_index, VOLUME] = 0  # 令未来买入时段没有成交。
    if target_state == "missing_target":  # 退出数据缺失时仍须保留预测候选。
        bars[target_index, CLOSE] = np.nan  # 仅删除标签依赖的未来收盘价。
    if target_state == "halt_before_target":  # 预测后发生停牌时，候选保留而标签不可用。
        halts.append((10 * 3600 + 60, 10 * 3600 + 120))  # 插入 10:01 至 10:02 的盘中停牌区间。
    session = sample_session(bars, minutes, bounds, halts)  # 生成整段的候选数组及审计状态。
    assert session.positions[0] == start_index  # 所有未来状态下都必须保留原始的首个预测时点。
    window = session.x[0]  # 根据首个候选的位置取出其输入窗口。
    label = session.y[0]  # 取得首个窗口对应的买入至退出对数收益率。
    reason = session.reasons[start_index]  # 用原时段的分钟行索引取得标签状态。
    assert window.shape == (30, 6)  # 核对保留的输入窗口包含三十分钟和六个特征。
    np.testing.assert_allclose(window[0], [0.0] * 4 + [log(2.0)] * 2, atol=1e-7)  # 核对窗口第一根分钟线的中位数归一化。
    np.testing.assert_allclose(window[-1], [log(1.02)] * 4 + [log(2.0)] * 2, atol=1e-7)  # 核对窗口在预测时点结束且不受目标状态影响。
    if target_state == "available":  # 当目标有成交且没有停牌时，应给出连续数值标签。
        assert session.label_valid[0] and reason == "accepted"  # 标签可用标记与逐分钟原因一致。
        assert label == pytest.approx(log(10.3 / 10.1))  # 分子为退出收盘价，分母为买入开盘价。
    else:  # 买入或退出端异常及预测后停牌都会使标签不可用。
        assert not session.label_valid[0] and np.isnan(label)  # 不可用的事后标签标为 NaN。
        assert reason == target_state  # 核对每种未来异常分别保留对应原因。
