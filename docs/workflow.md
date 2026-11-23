# 数据格式与运行流程

## 数据

准备 2025 年分钟行情和用于主板筛选及涨跌停估计的日行情。

| 文件 | 要求 |
|---|---|
| `calendars/1min.txt` | Qlib 交易日历。每行为北京时间的分钟线结束标签 |
| `features/<symbol>/<field>.1min.bin` | Qlib 小端 float32 二进制。首值为日历起始索引，目录名如 `sh600000`。字段为 `open`、`high`、`low`、`close`、`volume`、`amount`，成本回放另需 `factor` |
| 日行情 Parquet | 每股每交易日一行。字段为时间戳 `date`、形如 `600000.SH` 的 `code`、历史证券简称 `name`、未复权 `close`、`volume`、`amount` 和复权因子 `factor` |

分钟价和量额须同口径。回放以价格除以分钟因子、成交量乘以分钟因子还原未复权值。涨跌停参考价用前日收盘和相邻交易日的日因子近似计算。

## 训练与评估

先按[项目首页](../README.md#运行)安装项目。替换下列行情路径，选择一个尚未存在的输出目录。各命令的参数通过 `intraday-tcn <命令> --help` 查询。各入口也可用 `python -m intraday_tcn <命令>` 调用。

```sh
minute_data=/path/to/qlib_cn_1min
daily_data=/path/to/daily_summary.parquet
run_dir=output/tcn_mainboard_joint_01

intraday-tcn train --history-mainboard --shared-only --fast-cpu --device cpu \
  --threads 1 --epochs 2 --batch-size 512 --train-cap-per-stock 256 \
  --eval-sample-every 5 --provider "$minute_data" --daily-source "$daily_data" --out "$run_dir"

intraday-tcn score --run-dir "$run_dir" --provider "$minute_data"
intraday-tcn adapt --run-dir "$run_dir" --provider "$minute_data"
intraday-tcn evaluate --run-dir "$run_dir" --provider "$minute_data"
```

评估输出在 `$run_dir/stock_adaptation/`。`ic_comparison.json` 保存配对统计，`daily_ic.csv` 保存逐日结果。

## 成本回放

下一分钟开盘入场使用建模后的默认校准。信号分钟收盘入场先用相同训练期信号重新校准。

```sh
intraday-tcn replay --run-dir "$run_dir" --provider "$minute_data" \
  --daily-source "$daily_data" --entry next_open

intraday-tcn calibrate-close --run-dir "$run_dir" --provider "$minute_data"
intraday-tcn replay --run-dir "$run_dir" --provider "$minute_data" \
  --daily-source "$daily_data" --entry signal_close
```

回放输出分别在 `$run_dir/stock_adaptation/cost_gate_1m/` 和 `cost_gate_signal_close_1m/`，包含汇总、逐日结果和分钟预测示例。补充分析包括 IC 分布、分组检验、动量和反转基线，入口见 `intraday-tcn --help`。
