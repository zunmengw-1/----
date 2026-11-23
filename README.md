# 沪深主板分钟级量价因子研究

用过去 30 根分钟 K 线和因果时序卷积网络（TCN）预测未来 5 分钟收益。项目比较共享模型与逐股微调，并检验预测信号能否覆盖交易成本。

## 开发期表现

49 个开发交易日、3,054 只股票上的配对结果如下。

| 指标 | 共享模型 | 逐股微调 |
|---|---:|---:|
| 收盘到收盘收益的日均单股日内 Rank IC | 0.1705 | 0.1609 |
| 信号收盘价入场、零滑点的扣费净收益 | −4.58 bp | −3.07 bp |

回放按累计模拟买入金额归一，假设出分后能以信号分钟的收盘价成交。完整口径和开盘入场结果见[研究方法与结果](docs/method.md)。

## 运行

需要 Python 3.12 和自行准备的 2025 年行情。

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
intraday-tcn --help
```

常用流程是 `train → score → adapt → evaluate → replay`。从[数据格式与运行流程](docs/workflow.md)复制完整命令。

## 代码结构

| 位置 | 主要内容 |
|---|---|
| [data.py](src/intraday_tcn/data.py) · [features.py](src/intraday_tcn/features.py) | 行情读取、股票池、特征与标签 |
| [training](src/intraday_tcn/training) | TCN 模型、联合训练、逐股微调 |
| [evaluation](src/intraday_tcn/evaluation) | 打分、Rank IC、分组分析 |
| [replay](src/intraday_tcn/replay) | 收益校准、成本门槛、逐分钟成交模拟 |
