"""把少量指定股票日的候选样本写入文件，并审计筛选结果。

本模块从 Qlib 读取行情和辅助元数据，调用共享采样规则生成候选。
输出股票日及逐分钟决策、价量越界、零成交汇总和样本数组。
"""

import csv  # 写出逐条审计明细。
import json  # 写出按数据集汇总的审计结果。
from collections import Counter  # 累加各种筛选原因和质量检查结果。
from contextlib import ExitStack  # 一起管理多张审计表的文件句柄。

import numpy as np  # 构建样本数组并统计分钟状态。

from src.data.minutes import (  # 获取价量检查涉及的字段和时间格式。
    AMOUNT,  # 成交额列的索引。
    FACTOR,  # Qlib 缩放因子列的索引。
    HIGH,  # 最高价列的索引。
    LOW,  # 最低价列的索引。
    VOLUME,  # 成交量列的索引。
    clock_text,  # 将分钟位置格式化为 HH:MM。
)  # 完成原始行情符号的导入。
from src.features.samples import (  # 复用训练读取使用的采样规则。
    ENTRY_OFFSET,  # 入场分钟线距预测时点的分钟数。
    FEATURES,  # 特征名和排列顺序。
    HORIZON,  # 目标分钟线距预测时点的分钟数。
    RETURN_LABEL,  # 标识下一分钟开盘至第五分钟收盘的收益率口径。
    TICK,  # 最小报价单位。
    WINDOW,  # 输入窗口的分钟数。
    input_quality,  # 各分钟行情的基础有效性检查。
    sample_session,  # 连续竞价时段的候选及标签生成。
    sampling_conditions,  # 同一股票日的限价和停牌条件。
    split_for,  # 按交易日分配数据集。
    stock_day_reason,  # 股票日筛选和日级元数据读取。
    stock_day_sessions,  # 读取股票日行情并按时段切分。
)  # 完成共享采样接口的导入。


def audit_price_volume(bars, valid, minutes, key, writer, stats):  # 记录分钟均价越界情况。
    """检查有成交的有效分钟线，更新计数并把越界记录写入 CSV。

    bars 是形状为（分钟数，7）的 Qlib 行情，valid 是同长度的基础有效性掩码。
    minutes 是同一时段各行的分钟末时刻，单位为从北京时间午夜起算的分钟。
    key 指定股票代码、交易日、数据集和时段，writer 是已打开的 CSV 写入器。
    stats 是当前数据集的质量计数器。本函数更新传入对象，无返回值。
    """
    for index in np.flatnonzero(valid & (bars[:, VOLUME] > 0)):  # 只检查有效且有成交的分钟。
        row = bars[index]  # 取出当前需要检查的分钟线。
        average = row[AMOUNT] / (row[VOLUME] * row[FACTOR])  # 用成交额和股数求实际成交均价。
        low, high = row[[LOW, HIGH]] / row[FACTOR]  # 将最高价和最低价还原成元/股。
        gap = max(low - average, average - high)  # 正差值表示均价落在价格区间外。
        stats["price_volume_checked_minutes"] += 1  # 每根受检分钟都进入分母。
        if gap > 0:  # 越界只记录，不改变候选状态。
            stats["price_volume_outside_range"] += 1  # 累计出现越界的分钟数。
            # 另计越界幅度达到一个最小报价单位的分钟。
            stats["price_volume_outside_by_at_least_one_tick"] += int(gap >= float(TICK))
            # 保留股票日和分钟末时刻，以便定位原始行情。
            writer.writerow((*key, clock_text(minutes[index]), average, low, high, gap))


def zero_volume_counts(volume):  # 分别统计预测、入场和目标分钟的零成交。
    """接收 Qlib 成交量数组 volume，最后一维为单一连续竞价时段的分钟。

    返回时间合格候选数和各时点的零成交计数。两端同时或任一端零成交、
    原有成交量不可用统计均只涉及预测和目标分钟。入场成交量不可用另计。
    """
    first = WINDOW - 1  # 第一个具备完整输入窗口的行位置。
    prediction = volume[..., first:-HORIZON]  # 截取能在当前时段内找到目标的预测端。
    target = volume[..., first + HORIZON:]  # 对齐各预测端之后第 5 分钟的目标端。
    entry_start = first + ENTRY_OFFSET  # 定位时间合格候选的首个入场分钟。
    entry = volume[..., entry_start:entry_start + prediction.shape[-1]]  # 对齐同一批候选的入场成交量。
    prediction_zero = prediction == 0  # 标记预测端为零的候选。
    target_zero = target == 0  # 标记目标端为零的候选。
    prediction_count = int(np.count_nonzero(prediction_zero))  # 统计预测端零成交次数。
    target_count = int(np.count_nonzero(target_zero))  # 统计目标端零成交次数。
    both_count = int(np.count_nonzero(prediction_zero & target_zero))  # 两端同时为零只计一次。
    # 预测与目标两端都须有有限非负的成交量，才能完整判断原有两端状态。
    observed = np.isfinite(prediction) & (prediction >= 0) & np.isfinite(target) & (target >= 0)
    entry_observed = np.isfinite(entry) & (entry >= 0)  # 独立检查入场成交量是否可用。
    return Counter(  # 返回可按时段和数据集累加的计数。
        candidates_checked=prediction.size,  # 所有符合窗口和目标时间条件的候选。
        # 预测和目标两端至少有一端的成交量缺失或无效。
        candidates_with_unavailable_volume=prediction.size - int(np.count_nonzero(observed)),
        candidates_with_unavailable_entry_volume=entry.size - int(np.count_nonzero(entry_observed)),  # 入场成交量不可用。
        prediction=prediction_count,  # 预测端零成交。
        target=target_count,  # 目标端零成交。
        both=both_count,  # 两端同时零成交。
        either=prediction_count + target_count - both_count,  # 预测与目标至少一端零成交的去重数量。
        entry=int(np.count_nonzero(entry == 0)),  # 单独统计入场分钟零成交。
    )  # 完成各项零成交计数。


def summarize_zero_volume(counts):  # 将累计计数转换为可写入 JSON 的汇总。
    """接收 zero_volume_counts 的累计计数 counts，返回各端点的数量和占比。

    占比使用时间合格候选总数作为分母，分母为零时保留 None。
    """
    checked = counts["candidates_checked"]  # 记录计算占比使用的统一分母。
    return {  # 组织零成交审计的固定字段。
        "candidates_checked": checked,  # 时间合格候选总数。
        # 分开报告预测或目标成交量不可用，以及入场成交量不可用。
        "candidates_with_unavailable_volume": counts["candidates_with_unavailable_volume"],  # 预测或目标端。
        "candidates_with_unavailable_entry_volume": counts["candidates_with_unavailable_entry_volume"],  # 入场端。
        **{  # 展开预测与目标两端的原有结果，再单列入场。
            endpoint: {  # 为当前端点口径记录次数和占比。
                "count": counts[endpoint],  # 零成交次数。
                "share": counts[endpoint] / checked if checked else None,  # 候选中的发生比例。
            }  # 完成单个端点的汇总。
            for endpoint in ("prediction", "target", "both", "either", "entry")  # 入场独立于原四项。
        },  # 完成全部端点汇总。
    }  # 返回最终审计字典。


def build_samples(library, metadata, requests, output):  # 建样并写出小批量验收产物。
    """为选定股票日生成候选样本并写出审计结果，返回写入的候选数。

    library 是 Qlib 分钟行情读取器，metadata 是关联日线及交易事件的元数据读取器。
    requests 是规范化后的（Qlib 股票代码，YYYY-MM-DD 交易日）序列。
    output 是已经存在的输出目录，类型为 pathlib.Path。
    写出 x 为（样本数，30，6）的候选窗口，以及收益率、标签有效性、代码和时间。
    日期不在 Qlib 交易日历、元数据无法解析或文件读写失败时，传递对应错误。
    """
    # 按相同次序累计窗口、标签和每条样本的定位字段。
    windows, labels, codes, timestamps, splits = [], [], [], [], []
    per_split = {  # 为训练、验证和测试分别维护计数。
        name: {  # 当前数据集的股票日、分钟、质量和标签计数。
            "stock_days": Counter(),  # 股票日筛选原因的计数。
            "minutes": Counter(),  # 各预测分钟筛选或标签状态的计数。
            "quality": Counter(),  # 价量交叉核对的计数。
            "zero_volume_occurrences": Counter(),  # 符合时间边界的窗口端点零成交。
            "candidates_written": 0,  # 通过预测时点筛选的候选数量。
            "labels_available": 0,  # 保存的候选中标签有效的数量。
        }  # 完成一个数据集的统计初始化。
        for name in ("train", "validation", "test")  # 按固定顺序初始化三个数据集。
    }  # 完成分组审计计数的初始化。
    with ExitStack() as stack:  # 处理完成或发生异常时自动关闭所有明细表。
        tables = {}  # 按表名保存对应的 CSV 写入器。
        for name, columns in {  # 按固定顺序声明表名和各表的 CSV 列。
            "stock_days": (  # 每个股票日一条筛选结果。
                "code",  # Qlib 使用的证券代码。
                "historical_code",  # 交易所当日实际使用的证券代码。
                "date",  # 北京时间交易日。
                "split",  # 交易日所属数据集。
                "reason",  # 股票日的保留或排除原因。
            ),  # 完成股票日筛选表头。
            "decisions": (  # 预测时点核验结果与标签状态。
                "code",  # Qlib 使用的证券代码。
                "date",  # 北京时间交易日。
                "split",  # 交易日所属数据集。
                "session",  # 预测分钟所属连续竞价时段。
                "t",  # 预测分钟的结束时刻。
                "input_start",  # 输入窗口第一根分钟线的结束时刻。
                "entry",  # 提供入场开盘价的分钟线时间戳。
                "target",  # 提供退出收盘价的分钟线时间戳。
                "reason",  # 输入排除原因或通过筛选后的标签状态。
                "candidate_kept",  # 预测时点是否通过筛选。
                "label_valid",  # 入样候选的目标标签是否可用。
            ),  # 完成逐分钟决策表头。
            "price_volume_anomalies": (  # 记录均价超出最高价或最低价的分钟。
                "code",  # Qlib 使用的证券代码。
                "date",  # 北京时间交易日。
                "split",  # 交易日所属数据集。
                "session",  # 被核对分钟所属的连续竞价时段。
                "minute_end",  # 被核对分钟的结束时刻。
                "average_price",  # 按成交额和实际成交股数计算的均价。
                "low",  # 还原成元/股的最低价。
                "high",  # 还原成元/股的最高价。
                "outside_by",  # 均价到被越过的价格边界的差值。
            ),  # 完成价量越界表头。
        }.items():  # 依次创建并初始化各张审计表。
            stream = stack.enter_context(  # 将当前表格的句柄加入自动关闭列表。
                (output / f"{name}.csv").open("w", newline="", encoding="utf-8")  # 写出规范 CSV。
            )  # 获得当前表格的可写句柄。
            tables[name] = csv.writer(stream)  # 按表名保留写入器。
            tables[name].writerow(columns)  # 先写表头，便于核对各列。

        for code, day in requests:  # 独立处理每个指定的股票日。
            split = split_for(day)  # 按交易日归入相应的数据集。
            if day not in library.dates:  # 检查请求日期是否有 Qlib 交易日历。
                raise ValueError(f"{day} is absent from the Qlib trading calendar")  # 报出无效日期。
            historical_code = metadata.historical_code(code, day)  # 获取该日实际使用的股票代码。
            key = (code, day, split)  # 保存写明细时共用的证券、日期和数据集。
            stats = per_split[split]  # 取得该股票日所属数据集的计数对象。
            reason, daily = stock_day_reason(library, metadata, code, day)  # 先筛选股票日。
            stats["stock_days"][reason] += 1  # 统计通过或排除的具体原因。
            tables["stock_days"].writerow((code, historical_code, day, split, reason))  # 记录该日结论。
            if reason != "eligible":  # 不满足股票日条件时不读分钟行情。
                continue  # 处理下一个股票日。

            bounds, halts = sampling_conditions(daily)  # 将前收和停复牌时间转成筛选条件。
            # 读取股票日行情并按时段切分，避免窗口跨午休。
            for session, minutes, bars, excluded in stock_day_sessions(library, code, day, historical_code):
                session_key = (*key, session)  # 明细行同时记录连续竞价时段。
                if not excluded:  # 市场异常时段不进入零成交和价量越界审计。
                    stats["zero_volume_occurrences"].update(zero_volume_counts(bars[:, VOLUME]))  # 端点计数。
                    _, valid = input_quality(bars)  # 价量越界只检查通过基础有效性检查的分钟。
                    audit_price_volume(  # 写入越界行情及分组计数。
                        bars,  # 当前时段的分钟行情。
                        valid,  # 当前时段各分钟的有效性标记。
                        minutes,  # 用来定位行情的分钟结束时刻。
                        session_key,  # 股票日与时段的组合键。
                        tables["price_volume_anomalies"],  # 越界记录写入的 CSV 表。
                        stats["quality"],  # 当前数据集的质量计数器。
                    )  # 完成当前时段的价量核验。
                samples = sample_session(bars, minutes, bounds, halts, excluded)  # 从同一次采样取得候选数组和逐分钟原因。
                kept = np.zeros(len(minutes), dtype=bool)  # 为时段每根分钟线记录是否保留。
                kept[samples.positions] = True  # 通过输入筛选的预测时点记为保留。
                valid_by_minute = np.full(len(minutes), "", dtype=object)  # 排除的分钟在标签列留空。
                valid_by_minute[samples.positions] = samples.label_valid  # 将候选的标签可用性写到对应分钟。
                stats["minutes"].update(samples.reasons)  # 累计时段每个分钟的最终分类。
                for t, reason in enumerate(samples.reasons):  # 写出逐分钟结果，包括市场异常原因。
                    minute = minutes[t]  # t 是行索引，minute 是时钟分钟位置。
                    # 窗口不足 30 分钟时不填写输入起点。
                    input_start = clock_text(minute - WINDOW + 1) if t >= WINDOW - 1 else ""
                    entry = clock_text(minute + ENTRY_OFFSET)  # 根据预测分钟计算入场开盘价所属分钟线。
                    target = clock_text(minute + HORIZON)  # 记录退出收盘价所属分钟线。
                    tables["decisions"].writerow(  # 落盘逐分钟筛选或标签判定。
                        (  # 组装逐分钟决策表的一行。
                            *session_key,  # 展开证券代码、日期、数据集和时段。
                            clock_text(minute),  # 写出预测时点。
                            input_start,  # 写出输入窗口起点或留空。
                            entry,  # 写出入场分钟线时间戳。
                            target,  # 写出退出分钟线时间戳。
                            reason,  # 写出本分钟的筛选或标签判定原因。
                            kept[t],  # 记录输入是否保留。
                            valid_by_minute[t],  # 记录入样后的目标标签可用性。
                        ),  # 按表头顺序提交各字段。
                    )  # 结束当前行的写入。
                count = len(samples.y)  # 得到当前时段通过预测时点筛选的候选数。
                stats["candidates_written"] += count  # 汇总入样候选数量。
                stats["labels_available"] += int(np.count_nonzero(samples.label_valid))  # 汇总可用标签数量。
                windows.append(samples.x)  # 保存时段的归一化窗口数组。
                labels.append(samples.y)  # 保存收益率数组和其中的缺失标记。
                codes.extend([code] * count)  # 为所有候选记录对应的 Qlib 代码。
                timestamps.extend(f"{day}T{clock_text(minutes[t])}" for t in samples.positions)  # 记录候选的北京时间。
                splits.extend([split] * count)  # 为所有候选记录所属数据集。

    labels = np.concatenate(labels) if labels else np.empty(0, dtype=np.float64)  # 合并各时段的收益率数组。
    label_valid = np.isfinite(labels)  # 单独保存标签有效掩码。
    np.savez_compressed(  # 写入本次小批量验收的压缩样本库。
        output / "samples.npz",  # 指定小批量验收的样本文件路径。
        # 空结果保留正常的三维输入形状。
        x=np.concatenate(windows) if windows else np.empty((0, WINDOW, len(FEATURES)), np.float32),
        y=labels,  # 以原精度保存标签或 NaN。
        label_valid=label_valid,  # 拟合和评估使用的标签掩码。
        code=np.array(codes, dtype="U8"),  # 保存股票代码。
        t=np.array(timestamps, dtype="datetime64[m]"),  # 保存精确到分钟的预测时点。
        split=np.array(splits, dtype="U10"),  # 保存训练、验证或测试的标识。
        features=np.array(FEATURES),  # 保存输入各列的名称。
        return_label=np.array(RETURN_LABEL),  # 保存无须反序列化对象的标签口径标识。
    )  # 完成样本文件写入。
    all_zero_volume = Counter()  # 汇总所有数据集的端点计数。
    for counts in per_split.values():  # 依次生成每个数据集的审计统计。
        zero_volume = counts["zero_volume_occurrences"]  # 取出尚未转成比例的原始计数。
        all_zero_volume.update(zero_volume)  # 累加全量的端点事件数。
        counts["zero_volume_occurrences"] = summarize_zero_volume(zero_volume)  # 计算端点比例。
        minutes = counts["minutes"]  # 读取已确定原因的分钟计数。
        # 时间合格候选只扣除历史不足和目标越过时段的分钟。
        time_eligible = sum(minutes.values()) - minutes["insufficient_history"] - minutes["target_outside_session"]
        written = counts["candidates_written"]  # 读取通过预测时点筛选的候选数量。
        available = counts["labels_available"]  # 读取入样后标签可得的数量。
        counts["time_eligible_candidates"] = time_eligible  # 保存时间合格候选数。
        counts["labels_unavailable"] = written - available  # 计算已保留输入但标签不可得的数量。
        counts["label_coverage"] = available / written if written else None  # 按入样候选计算覆盖率。
        counts["zero_volume_at_prediction_share_of_time_eligible"] = (  # 按首个剔除原因统计预测端。
            minutes["zero_volume_at_prediction"] / time_eligible if time_eligible else None  # 时间候选为分母。
        )  # 完成预测时点剔除比例计算。
        counts["zero_volume_at_target_share_of_written_candidates"] = (  # 按标签状态统计目标端。
            minutes["zero_volume_at_target"] / written if written else None  # 写入的候选为分母。
        )  # 完成目标端标签不可用比例计算。
        counts["zero_volume_at_entry_share_of_written_candidates"] = (  # 记录首个标签状态为入场零成交的比例。
            minutes["zero_volume_at_entry"] / written if written else None  # 使用全部写入候选作为分母。
        )  # 完成入场端标签不可用比例计算。
    available = int(label_valid.sum())  # 统计跨数据集的有效标签总数。
    report = {  # 以固定字段保存运行口径和审计结果。
        "timezone": "Asia/Shanghai",  # 输出分钟按北京时间解释。
        "window_minutes": WINDOW,  # 保存输入窗口的分钟数。
        "return_label": RETURN_LABEL,  # 标识使用入场开盘和目标收盘计算对数收益。
        "entry_offset_minutes": ENTRY_OFFSET,  # 保存入场分钟线相对预测时点的偏移。
        "horizon_minutes": HORIZON,  # 保存目标分钟线相对预测时点的偏移。
        "features": FEATURES,  # 明确输入列的顺序。
        "stock_days_requested": len(requests),  # 记录请求处理的股票日总数。
        "samples_written": len(labels),  # 记录保留候选总数。
        "labels_available": available,  # 记录有效标签总数。
        "labels_unavailable": len(labels) - available,  # 记录目标无法观测的候选数。
        "splits": per_split,  # 附上各数据集的详细计数和比例。
        "price_volume_audit": "record_only",  # 表示价量越界只记审计。
        "zero_volume_occurrences": summarize_zero_volume(all_zero_volume),  # 全体端点零成交。
        # 明确端点独立统计的候选范围和例外。
        "zero_volume_scope": "time-eligible windows on eligible stock-days outside excluded market sessions",
        # 明确逐分钟决策保留的是首个淘汰原因或预测通过后的标签状态。
        "decision_counting": "first candidate rejection or resulting label status per prediction minute",
    }  # 完成总审计内容。
    (output / "audit.json").write_text(  # 将结果写入验收目录。
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",  # 按原有缩进和换行生成 JSON。
        encoding="utf-8",  # 保留 UTF-8 编码。
    )  # 完成审计文件写入。
    return len(labels)  # 返回候选样本总数供入口报告。
