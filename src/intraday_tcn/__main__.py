"""Command-line entry point for the intraday research pipeline."""

import argparse
import sys
from importlib import import_module

PIPELINE = {
    "train": ("training.train", "训练共享 TCN"),
    "score": ("evaluation.scores", "生成 30 分钟网格的预测及基线"),
    "adapt": ("training.adapt", "在训练期内拟合逐股预测头"),
    "evaluate": ("evaluation.stock_heads", "配对评估共享头和逐股预测头"),
    "calibrate-close": ("replay.signal_close", "校准信号分钟收盘入场"),
    "replay": ("replay.stock_heads", "逐分钟对照两种预测头的成本表现"),
}
DIAGNOSTICS = {
    "ic-distribution": ("evaluation.rank_ic", "分析 IC 分布及配对差异"),
    "groups": ("evaluation.groups", "按股票类型检查分位表现"),
    "calibrate-baselines": ("replay.calibration", "校准共享模型及简单基线"),
    "replay-baselines": ("replay.execution", "对照上分位信号与简单基线"),
    "replay-baselines-gated": ("replay.cost_gate", "按成本门槛回放 30 分钟网格"),
    "replay-baselines-minute": ("replay.minute", "按成本门槛逐分钟回放基线"),
}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    commands = PIPELINE | DIAGNOSTICS
    help_lines = ["研究流程", *(f"  {name:<26} {text}" for name, (_, text) in PIPELINE.items())]
    help_lines.extend(["", "补充分析", *(f"  {name:<26} {text}" for name, (_, text) in DIAGNOSTICS.items())])
    parser = argparse.ArgumentParser(
        prog="intraday-tcn",
        description="沪深主板分钟级量价因子研究。使用 intraday-tcn <command> --help 查看参数。",
        epilog="\n".join(help_lines),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=commands, metavar="<command>")
    if not args:
        parser.print_help()
        return
    command = parser.parse_args(args[:1]).command
    module_name, _ = commands[command]
    import_module(f"intraday_tcn.{module_name}").main(args[1:])


if __name__ == "__main__":
    main()
