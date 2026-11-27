"""构建训练股票日的连续竞价行情缓存。

从策略目录运行以下小批试跑命令。输入是本机验收的元数据清单和 13 个股票日。
    trial_dir=$(mktemp -d /private/tmp/tcn-stock-day-cache.XXXXXX)
    python scripts/build_stock_day_cache.py --metadata /private/tmp/qlib-sample-acceptance/metadata.json --stock-days /private/tmp/qlib-sample-acceptance/required_cases.csv --output "$trial_dir/cache"
终端报告股票日筛选计数，并在成功时显示缓存股票日数量和目录。省略股票日名单时，
从 Qlib 覆盖范围和正式训练日期中选择全部合格股票日。
"""

import argparse  # 解析缓存输入与输出参数。
import json  # 读取元数据清单，输出筛选计数。
import sys  # 支持从策略目录直接运行脚本。
from bisect import bisect_left, bisect_right  # 在交易日列表中定位证券覆盖范围。
from collections import Counter  # 记录保留及排除股票日的各类原因。
from pathlib import Path  # 表示行情目录、名单和缓存目录。


def _training_calendar(library):  # 取得 Qlib 日历里的正式训练交易日。
    """library 为已加载的 Qlib 读取器，返回按时间排序的训练日字符串列表。"""
    from src.features.samples import split_for  # 复用正式的训练、验证和测试日期切分。

    dates = []  # 保存属于训练期的 Qlib 日期。
    for day in sorted(library.dates):  # 逐日检查 Qlib 日历。
        try:  # 共享行情库允许有项目研究范围之外的日期。
            split = split_for(day)  # 以正式切分判断该日是否属于训练期。
        except ValueError:  # 研究范围之外的日历日期无需写入训练缓存。
            continue  # 跳过该日并检查后续日期。
        if split == "train":  # 只为正式训练日期建缓存。
            dates.append(day)  # 保留升序日期供二分查找覆盖范围。
    return dates  # 供显式名单和全目录选择共用。


def _catalog_requests(library, train_dates):  # 生成 Qlib 覆盖范围内的训练股票日。
    """传入 Qlib 读取器和有序训练日列表，逐项返回证券代码与交易日。"""
    for code, (first, last) in library.spans.items():  # 遍历 Qlib 的证券覆盖清单。
        start = bisect_left(train_dates, first[:10])  # 保留覆盖起始交易日。
        stop = bisect_right(train_dates, last[:10])  # 将覆盖末日也纳入范围。
        for day in train_dates[start:stop]:  # 只在有效覆盖区间内枚举实际训练交易日。
            yield code, day  # 交给正式股票日规则判断当天是否合格。


def _select_requests(library, metadata, stock_days):  # 通过现有股票日规则筛选建缓存范围。
    """传入行情读取器、元数据和可选股票日 CSV，返回合格清单和各原因计数。

    显式清单将非训练期或非交易日归为排除项。两个请求映射到同一规范股票日时，
    显式清单报错，Qlib 全目录枚举则只保留一个。行情与元数据错误交由调用方处理。
    """
    from src.features.samples import (  # 复用名单校验与股票日筛选规则。
        read_requests,  # 读取并校验显式股票日名单。
        stock_day_reason,  # 获取正式的股票日筛选原因。
    )  # 完成样本规则导入。

    train_dates = _training_calendar(library)  # 取得允许建缓存的真实训练交易日。
    train_date_set = set(train_dates)  # 以常数时间判断输入日期是否属于训练期。
    candidates = _catalog_requests(library, train_dates) if stock_days is None else read_requests(stock_days)  # 确定请求范围。
    requests = []  # 存放合格股票日的规范键。
    counts = Counter()  # 统计候选和各排除原因。
    seen = set()  # 记录规范化后的键，避免别名导致重复写入。
    for code, day in candidates:  # 逐项检查指定或枚举所得的股票日。
        if day not in train_date_set:  # 只保留 Qlib 日历内的真实训练日。
            reason = "outside_training_split" if day in library.dates else "outside_qlib_calendar"  # 区分排除起因。
            counts[reason] += 1  # 报告非训练日期或非交易日。
            continue  # 跳过该请求。
        canonical = metadata.canonical_code(code, day)  # 将旧证券代码统一到 Qlib 口径。
        key = canonical, day  # 用规范代码和日期共同定位一个行情块。
        if key in seen:  # 处理规范化后重复出现的股票日。
            if stock_days is not None:  # 显式请求的重复项与正式训练读取器一样报错。
                raise ValueError(f"Multiple requested codes refer to the same stock-day: {canonical} {day}")  # 提示需要修正的键。
            counts["duplicate_canonical_stock_day"] += 1  # 记录 Qlib 全量枚举中重合的代码。
            continue  # 已处理的规范股票日无需重复检查和写入。
        seen.add(key)  # 将该股票日加入已经处理的范围。
        reason, _ = stock_day_reason(library, metadata, canonical, day)  # 使用正式规则筛选训练股票日。
        counts[reason] += 1  # 汇总保留项和各类排除原因。
        if reason == "eligible":  # 仅将正式合格的股票日写入缓存。
            requests.append(key)  # 记录用于读取七个行情字段的规范键。
    return requests, counts  # 将合格清单和完整筛选计数交给命令行入口。


def main():  # 构建并发布选定训练股票日的行情缓存。
    """读取命令行参数并报告输出数量。输入或读写错误以状态码 1 退出。"""
    project_root = Path(__file__).resolve().parents[1]  # 定位含 src 包的策略目录。
    if str(project_root) not in sys.path:  # 检查直接运行脚本时能否导入策略源码。
        sys.path.insert(0, str(project_root))  # 支持从策略目录直接运行。

    from src.data.metadata import Metadata  # 加载日线与停牌等辅助数据。
    from src.data.minutes import (  # 使用正式行情接口。
        FIELDS,  # 取正式 Qlib 字段顺序。
        MinuteLibrary,  # 初始化 Qlib 行情读取器。
    )  # 完成分钟行情接口导入。
    from src.data.stock_day_cache import (  # 导入缓存写入入口。
        build_stock_day_cache,  # 委托数据层构建行情缓存。
    )  # 完成缓存构建接口导入。
    from src.features.samples import TRADING_MINUTES  # 使用现有连续竞价网格。

    parser = argparse.ArgumentParser(description=__doc__)  # 将数据来源和输出位置交由调用方指定。
    parser.add_argument("--qlib", type=Path, default=project_root.parent / "data/qlib_cn_1min", help="Qlib minute library directory")  # 默认共享分钟行情库。
    parser.add_argument("--metadata", type=Path, required=True, help="JSON manifest of audited daily and event sources")  # 必须提供核验元数据清单。
    parser.add_argument("--stock-days", type=Path, help="Optional CSV with Qlib code and date columns")  # 传入小批名单时限制缓存范围。
    parser.add_argument("--output", type=Path, required=True, help="New directory for the stock-day cache")  # 指定缓存输出路径。
    args = parser.parse_args()  # 检查命令行参数。

    try:  # 将输入或数据写入错误转换为可读的命令行提示。
        if args.output.exists() or args.output.is_symlink():  # 在扫描全目录之前检查输出路径。
            raise ValueError(f"Output directory already exists: {args.output}")  # 阻止覆盖已有缓存。
        print("Selecting training stock-days", flush=True)  # 在全量目录扫描开始时提示所处阶段。
        library = MinuteLibrary(args.qlib)  # 加载交易日历和证券覆盖范围。
        spec = json.loads(args.metadata.read_text(encoding="utf-8"))  # 读取元数据来源定义。
        with Metadata(spec, base_dir=args.metadata.resolve().parent) as metadata:  # 选择范围期间保持元数据连接打开。
            requests, counts = _select_requests(library, metadata, args.stock_days)  # 汇总合格训练股票日。
        print(json.dumps({"requested": counts.total(), "eligible": len(requests), "excluded": counts.total() - len(requests), "reasons": counts}, sort_keys=True), flush=True)  # 在读行情之前报告筛选数量。
        count = build_stock_day_cache(library, requests, TRADING_MINUTES, FIELDS, args.output)  # 发布完整的股票日缓存。
    except (OSError, ValueError) as error:  # 处理无法读取行情、元数据或输出的情况。
        parser.exit(1, f"{error}\n")  # 报告具体原因并返回失败状态。
    print(f"Wrote {count:,} stock-days to {args.output}", flush=True)  # 输出缓存目录与包含的股票日数。


if __name__ == "__main__":  # 作为脚本执行时进入命令行流程。
    main()  # 执行一次股票日缓存构建。
