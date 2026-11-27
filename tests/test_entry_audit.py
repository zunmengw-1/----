"""检验下一分钟开盘入场的样本口径和独立零成交统计。"""

import csv  # 按列读取逐分钟决策明细。
import json  # 读取样本口径和审计汇总。
from decimal import Decimal  # 构造精确到分的前收参考价。
from types import SimpleNamespace  # 构造小型行情及元数据接口。

import numpy as np  # 创建合成分钟行情并读取样本。

from src.data.minutes import (  # 定位合成行情里需要调整的字段。
    AMOUNT,  # 成交额列。
    CLOSE,  # 收盘价列。
    OPEN,  # 开盘价列。
    VOLUME,  # 成交量列。
)  # 完成行情字段导入。
from src.features import sample_audit  # 调用正式审计构造及统计接口。


def test_zero_volume_keeps_prediction_target_counts_separate_from_entry():  # 对照各端点独立口径。
    """传入两个含缺失和零值的时段，核对独立入场统计以及原两端的统计。"""
    volume = np.ones((2, 36))  # 每个时段可在第 29 和 30 行（从零起）预测。
    volume[0, [29, 30, 34]] = 0  # 第一段预测端两次零成交，目标端一次，入场端一次。
    volume[0, [31, 35]] = np.nan  # 第二个候选入场与目标端成交量都缺失。
    volume[1, 29] = -1  # 第二段首个候选的预测端成交量无效。
    volume[1, [31, 34, 35]] = 0  # 第二段入场端一次、目标端两次零成交。

    counts = sample_audit.zero_volume_counts(volume)  # 汇总时间合格的四个候选。
    summary = sample_audit.summarize_zero_volume(counts)  # 将各计数转成比例。
    assert summary["candidates_checked"] == 4  # 保留基于窗口和目标时间边界的分母。
    assert summary["candidates_with_unavailable_volume"] == 2  # 旧字段只涉及预测端和目标端。
    assert [summary[field]["count"] for field in ("prediction", "target", "both", "either")] == [2, 3, 1, 4]  # 检查原两端计数。
    assert summary["entry"] == {"count": 2, "share": 0.5}  # 入场端单独统计。
    assert summary["candidates_with_unavailable_entry_volume"] == 1  # 入场端缺失不改变原字段。


def test_build_samples_records_entry_and_return_definition(tmp_path, monkeypatch):  # 核对验收产物。
    """构造一个有两条候选的时段，验证 CSV、NPZ 和 JSON 一致记录入场信息。"""
    day = "2024-06-03"  # 采用一条合成训练交易日。
    code = "SH600001"  # 给样本定位一个普通主板股票代码。
    library = SimpleNamespace(dates={day})  # 提供样本入口需要的交易日期。
    metadata = SimpleNamespace(historical_code=lambda stock, _: stock)  # 让历史代码与请求代码相同。
    daily = SimpleNamespace(preclose=Decimal("10.00"), intraday_halts=())  # 提供前收参考和空停复牌列表。
    minutes = tuple(range(571, 607))  # 使用 09:31 至 10:06，时间合格预测时点为 10:00 和 10:01。
    bars = np.tile([10.0, 10.3, 9.8, 10.0, 2.0, 20.0, 1.0], (len(minutes), 1))  # 构造基础有效的分钟记录。
    bars[30, OPEN] = 10.1  # 首个预测的入场开盘价与预测收盘价不同。
    bars[34, CLOSE] = 10.2  # 首个预测的目标收盘价为 10.2。
    bars[31, [VOLUME, AMOUNT]] = 0  # 第二个预测的入场分钟零成交。
    monkeypatch.setattr(sample_audit, "stock_day_reason", lambda *_: ("eligible", daily))  # 提供合格股票日。
    monkeypatch.setattr(sample_audit, "stock_day_sessions", lambda *_: [("morning", minutes, bars, False)])  # 提供待审计的实际时段。

    assert sample_audit.build_samples(library, metadata, [(code, day)], tmp_path) == 2  # 标签不可用的候选也写入。
    with (tmp_path / "decisions.csv").open(encoding="utf-8", newline="") as file:  # 逐列读取决策记录。
        decisions = {row["t"]: row for row in csv.DictReader(file)}  # 按预测分钟快速定位候选。
    first, second = decisions["10:00"], decisions["10:01"]  # 取出两条实际写入的候选记录。
    assert (first["entry"], first["target"], first["label_valid"]) == ("10:01", "10:05", "True")  # 检查两端时刻。
    assert (second["entry"], second["reason"], second["label_valid"]) == ("10:02", "zero_volume_at_entry", "False")  # 记入入场原因。
    with np.load(tmp_path / "samples.npz") as archive:  # 核验模型读取的样本数组。
        np.testing.assert_array_equal(archive["label_valid"], [True, False])  # 保留第一条有效标签并掩蔽第二条。
        np.testing.assert_allclose(archive["y"], [np.log(10.2 / 10.1), np.nan])  # 确认收益以入场开盘为分母。
        label = archive["return_label"].item()  # 读取无需反序列化对象的口径标识。
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))  # 读取此次建样口径。
    assert label == audit["return_label"] == "next_open_to_t_plus_5_close"  # 输出使用一致的标签定义。
    assert audit["entry_offset_minutes"] == 1 and audit["horizon_minutes"] == 5  # 两端时点都相对预测分钟定义。
    assert audit["zero_volume_occurrences"]["entry"] == {"count": 1, "share": 0.5}  # 独立统计入场端。
    assert audit["splits"]["train"]["zero_volume_at_entry_share_of_written_candidates"] == 0.5  # 分开统计标签状态。
