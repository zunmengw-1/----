"""读取指定股票日并生成样本与审计文件。

从策略目录运行以下小批试跑命令。输入是本机验收清单和 13 个股票日。
    trial_dir=$(mktemp -d /private/tmp/tcn-sample-trial.XXXXXX)
    python scripts/build_samples.py --metadata /private/tmp/qlib-sample-acceptance/metadata.json --stock-days /private/tmp/qlib-sample-acceptance/required_cases.csv --output "$trial_dir/results"
成功时在结果目录写入 795 条候选及审计文件，终端显示写入条数与输出路径。
"""

import argparse  # 引入命令行参数解析器。
import json  # 引入 JSON 解析工具，用于读取元数据清单。
import sys  # 引入模块查找路径，支持从策略目录直接运行。
from pathlib import Path  # 引入路径类型，用于定位行情库与结果目录。
from tempfile import TemporaryDirectory  # 引入临时目录，用于完整写入结果后再交付。


def main():  # 执行一次指定股票日的样本生成任务。
    """读取命令行参数，构造样本并发布结果目录。

    路径输入来自命令行。默认 Qlib 路径相对于策略目录，为 ../data/qlib_cn_1min。
    该函数通过终端报告结果。输入无效、重复或写盘失败时以状态码 1 退出。
    """
    project_root = Path(__file__).resolve().parents[1]  # 定位包含 src 源码包的策略目录。
    if str(project_root) not in sys.path:  # 检查直接运行脚本时能否找到策略目录。
        sys.path.insert(0, str(project_root))  # 让命令行入口能通过 src 导入项目源码。

    from src.data.metadata import Metadata  # 加载交易日线与停牌等辅助数据。
    from src.data.minutes import MinuteLibrary  # 加载 Qlib 分钟行情读取器。
    from src.features.sample_audit import (  # 导入样本和审计的输出接口。
        build_samples,  # 调用包内的样本审计流程。
    )  # 完成审计接口的导入。
    from src.features.samples import (  # 导入股票日名单的读取接口。
        read_requests,  # 调用包内的股票日名单验证逻辑。
    )  # 完成名单读取接口的导入。

    parser = argparse.ArgumentParser(description=__doc__)  # 创建命令行解析器，并显示试跑说明。
    parser.add_argument(  # 注册共享行情库路径参数。
        "--qlib",  # 指定 Qlib 行情库路径。
        type=Path,  # 将输入转为文件系统路径。
        default=Path(__file__).resolve().parents[2] / "data/qlib_cn_1min",  # 使用策略目录上级的共享行情库。
    )  # 完成行情库路径参数注册。
    parser.add_argument("--metadata", type=Path, required=True, help="JSON manifest of audited daily and event sources")  # 注册元数据清单。
    parser.add_argument("--stock-days", type=Path, required=True, help="CSV with Qlib code and date columns")  # 注册股票日 CSV。
    parser.add_argument("--output", type=Path, required=True, help="New directory for samples and audit files")  # 注册新结果目录。
    args = parser.parse_args()  # 读取并验证命令行参数。

    try:  # 将可预见的输入或文件错误转换为命令行提示。
        requests = read_requests(args.stock_days)  # 读取试跑要求处理的股票日。
        library = MinuteLibrary(args.qlib)  # 加载分钟交易日历及行情位置。
        spec = json.loads(args.metadata.read_text(encoding="utf-8"))  # 从清单取得数据来源配置。
        if args.output.exists() or args.output.is_symlink():  # 检查结果路径是否已被占用。
            raise ValueError(f"Output directory already exists: {args.output}")  # 拒绝覆盖之前的结果。
        args.output.parent.mkdir(parents=True, exist_ok=True)  # 准备结果目录所属的父目录。
        with (  # 在处理期间管理临时目录与元数据读取器。
            TemporaryDirectory(dir=args.output.parent) as staged,  # 在同一文件系统准备临时输出位置。
            Metadata(spec, base_dir=args.metadata.resolve().parent) as metadata,  # 以清单目录解释数据相对路径。
        ):  # 完成读取和输出资源的准备。
            requests = [(metadata.canonical_code(code, day), day) for code, day in requests]  # 统一 Qlib 证券代码。
            if len(set(requests)) != len(requests):  # 检查别名统一后是否出现重复股票日。
                raise ValueError("Multiple requested codes refer to the same stock-day")  # 提示调用方去除重复请求。
            pending = Path(staged) / "output"  # 设置本轮的临时结果目录。
            pending.mkdir()  # 创建临时结果目录。
            count = build_samples(library, metadata, requests, pending)  # 委托样本模块写入样本及审计。
            pending.rename(args.output)  # 全部文件成功写入后交付结果目录。
    except (OSError, ValueError) as error:  # 捕获数据验证与文件读写错误。
        parser.exit(1, f"{error}\n")  # 保留原命令的失败退出方式。
    print(f"Wrote {count:,} samples to {args.output}")  # 向调用方报告写入条数与位置。


if __name__ == "__main__":  # 直接执行此文件时运行命令行入口。
    main()  # 启动一次样本生成任务。
