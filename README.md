# 沪深主板分钟级量价因子研究

用过去 30 根分钟 K 线预测未来 5 分钟的收盘到收盘对数收益。项目包含共享时序卷积网络（TCN）、逐股预测头、非重叠窗口的 Rank IC 评估，以及逐分钟成本回放。

## 方法

- 输入为每分钟 20 个通道，包含相对 OHLC 和成交量、日内时段、5/10/20 分钟收益、实现波动率与偏度、量价相关性及 VWAP 偏离等。窗口在同一交易时段内，模型使用出分时已结束的分钟线。
- 共享模型由两个因果卷积残差块组成，使用 WeightNorm 和 LeakyReLU。可学习的卷积将时间维从 30 压缩至 15、5、1，再用 16 维表示预测单一收益。冻结共享编码器后，使用 Huber 损失和指向共享头的 L2 正则化拟合每股的 17 参数线性头。
- 2025 年行情按交易日切分为 170 天训练、49 天开发测试和 24 天最终验证。单股预测头通常在前 140 个训练日拟合，后 30 个训练日用于收益分桶校准。IC 使用非重叠输入窗口，交易回放每分钟检查一次信号。
- 回放按校准预期收益与往返成本设定入场条件。在已结算底仓上模拟回转，计入佣金、印花税、过户费和不同档位的滑点。

## 开发期结果

共享训练覆盖 3,134 只股票。开发期在 3,054 只股票上配对评估。

| 49 个开发交易日 | 共享头 | 逐股微调头 |
|---|---:|---:|
| 收盘到收盘的日均单股日内 Rank IC | 0.1705 | 0.1609 |
| 下一分钟开盘入场口径的日均 Rank IC | 0.03995 | 0.03817 |
| 信号收盘价入场、零滑点的扣费净收益 | −4.58 bp | −3.07 bp |

净收益按累计模拟买入金额归一。用完整分钟线出分后以同一根收盘价成交属于回放假设。股票池依据历史简称和行情近似筛选。最后 24 个交易日留作最终验证，表中为开发期结果。

## 数据与环境

需要 Python 3.12 及自行准备的 2025 年行情。

- 分钟行情采用 Qlib 二进制布局。交易日历位于 `calendars/1min.txt`，时间为北京时间，按分钟线结束标签对齐。股票位于 `features/<symbol>`，例如 `features/sh600000`。每个字段文件为 `<field>.1min.bin`，以小端 float32 存储，首值为日历起始索引。字段为 `open`、`high`、`low`、`close`、`volume`、`amount`，成本回放另需 `factor`。价格与量额须同口径，回放以价格除以因子、成交量乘以因子还原未复权值。
- 日行情 Parquet 每股每交易日一行。`date` 为交易日期对应的时间戳，`code` 格式如 `600000.SH`，其余列为历史证券简称 `name`、未复权收盘价 `close`、`volume`、`amount` 及复权因子 `factor`。涨跌停参考价使用前日收盘价及相邻交易日的因子比例近似计算。

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py --help
```

## 运行流程

从仓库根目录执行，先替换下列行情路径。共享训练要求 `run_dir` 为新目录。

```sh
minute_data=/path/to/qlib_cn_1min
daily_data=/path/to/daily_summary.parquet
run_dir=output/tcn_mainboard_joint_01

python src/train.py --history-mainboard --shared-only --fast-cpu --device cpu \
  --threads 1 --epochs 2 --batch-size 512 --train-cap-per-stock 256 \
  --eval-sample-every 5 --provider "$minute_data" --daily-source "$daily_data" --out "$run_dir"

python src/score_economics.py --run-dir "$run_dir" --provider "$minute_data"
python src/adapt_stock_heads.py --run-dir "$run_dir" --provider "$minute_data"
python src/evaluate_stock_adaptation.py --run-dir "$run_dir" --provider "$minute_data"

python src/replay_stock_adaptation.py --run-dir "$run_dir" --provider "$minute_data" \
  --daily-source "$daily_data" --entry next_open

python src/calibrate_signal_close.py --run-dir "$run_dir" --provider "$minute_data"
python src/replay_stock_adaptation.py --run-dir "$run_dir" --provider "$minute_data" \
  --daily-source "$daily_data" --entry signal_close
```

`src/` 中另有简单动量、反转基线和 IC 分布分析入口。输出写到 `output/`，包括模型权重、训练与股票池审计、逐股 IC 及扣成本结果。
