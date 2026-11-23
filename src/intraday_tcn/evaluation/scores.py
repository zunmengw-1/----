"""Score the frozen joint TCN and calibrate training-only signal quintiles."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn.utils.parametrize import cached

from intraday_tcn.data import history_mainboard_eligibility, load_bars, read_dates
from intraday_tcn.features import FEATURE_NAMES, encode_windows, find_targets
from intraday_tcn.paths import DEFAULT_PROVIDER, DEFAULT_RUN
from intraday_tcn.training.fast_tcn import FastReturnTCN
from intraday_tcn.training.train import Scaler, encode

SIGNALS = ("tcn", "reversal", "momentum")
QUANTILES = (20, 40, 60, 80)
THRESHOLD_COLUMNS = tuple(f"{name}_q{q}" for name in SIGNALS for q in QUANTILES)
PROFILE_COLUMNS = (
    "symbol", "calibration_n", "train_median_amount", "train_volatility_bp", *THRESHOLD_COLUMNS,
)
SCORE_SCHEMA = pa.schema([
    ("symbol", pa.string()), ("date", pa.string()), ("slot", pa.int16()),
    ("tcn_bp", pa.float64()), ("reversal_bp", pa.float64()), ("momentum_bp", pa.float64()),
    ("label_bp", pa.float64()), ("execution_return_bp", pa.float64()),
])
WINDOW = 30
HORIZON = 5
SAMPLE_EVERY = 30


def _score_parameters(checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = FastReturnTCN()
    model.load_state_dict(state["model_state_dict"])
    scaler_state = state["scaler"]
    if tuple(scaler_state["feature_names"]) != FEATURE_NAMES:
        raise ValueError("冻结模型的特征列与当前编码器不相符")
    scaler = Scaler(
        np.asarray(scaler_state["center"], dtype=np.float32),
        np.asarray(scaler_state["scale"], dtype=np.float32),
        scaler_state["label_scale_bp"],
    )
    model.eval()
    weight = model.head.weight.detach().numpy().ravel().astype(np.float64)
    bias = model.head.bias.item()
    return model, scaler, weight, bias


def _date_range(calendar: list[date], split: dict) -> list[date]:
    start, end = date.fromisoformat(split["start"]), date.fromisoformat(split["end"])
    dates = [day for day in calendar if start <= day <= end]
    if len(dates) != split["days"]:
        raise ValueError(f"交易日日历和原模型区间不一致: {split}")
    return dates


def _development_days(path: Path, allowed_days: list[date]) -> dict[str, list[date]]:
    eligible_days = set(allowed_days)
    dates_per_symbol: dict[str, set[date]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            day = date.fromisoformat(row["date"])
            if day not in eligible_days:
                raise ValueError(f"开发评分文件包含区间外日期: {day}")
            symbol_days = dates_per_symbol.setdefault(row["symbol"], set())
            if day in symbol_days:
                raise ValueError(f"开发评分文件包含重复股票日: {row['symbol']} {day}")
            symbol_days.add(day)
    return {symbol: sorted(days) for symbol, days in sorted(dates_per_symbol.items())}


def _raw_symbol(symbol: str) -> str:
    return f"{symbol[2:]}.{symbol[:2]}"


def _training_amounts(
    source: Path, train_dates: list[date], universe: dict[str, np.ndarray], symbols: set[str],
) -> dict[str, float]:
    supported = {_raw_symbol(symbol): symbol for symbol in symbols if symbol in universe}
    panel = pq.read_table(
        source,
        columns=["date", "code", "amount"],
        filters=[
            ("date", ">=", datetime.combine(train_dates[0], datetime.min.time())),
            ("date", "<=", datetime.combine(train_dates[-1], datetime.min.time())),
            ("code", "in", list(supported)),
        ],
    ).to_pandas()
    calendar_index = np.asarray(train_dates, dtype="datetime64[us]")
    result = {}
    for code, group in panel.groupby("code", sort=False):
        symbol = supported[code]
        daily = group.set_index("date")["amount"].reindex(calendar_index).to_numpy(dtype=np.float64)
        daily = daily[universe[symbol] & np.isfinite(daily) & (daily >= 0)]
        if len(daily):
            result[symbol] = float(np.median(daily))
    return result


def _signals(bars: np.ndarray, coords: np.ndarray, parameters) -> tuple[np.ndarray, ...]:
    model, scaler, weight, bias = parameters
    x = encode_windows(bars, coords)
    encoded = encode(model, x, scaler, 512, torch.device("cpu"))
    tcn = (encoded @ weight + bias) * scaler.label_scale
    d, slot = coords.T
    last = np.log(bars[d, slot, 3].astype(np.float64))
    previous = np.log(bars[d, slot - 1, 3].astype(np.float64))
    first_open = np.log(bars[d, slot - WINDOW + 1, 0].astype(np.float64))
    return tcn, 10_000 * (previous - last), 10_000 * (last - first_open)


def _execution_returns(bars: np.ndarray, coords: np.ndarray) -> np.ndarray:
    d, slot = coords.T
    entry = bars[d, slot + 1, 0].astype(np.float64)
    exit_price = bars[d, slot + HORIZON, 3].astype(np.float64)
    entry_volume, exit_volume = bars[d, slot + 1, 4], bars[d, slot + HORIZON, 4]
    valid = (
        np.isfinite(entry) & (entry > 0) & np.isfinite(exit_price) & (exit_price > 0)
        & np.isfinite(entry_volume) & (entry_volume > 0)
        & np.isfinite(exit_volume) & (exit_volume > 0)
    )
    returns = np.full(len(coords), np.nan)
    returns[valid] = 10_000 * (np.log(exit_price[valid]) - np.log(entry[valid]))
    return returns


def _score_development(
    provider: Path, symbol: str, dates: list[date], parameters,
) -> pa.Table:
    bars = load_bars(provider, symbol, dates)
    y, coords = find_targets(bars, sample_every=SAMPLE_EVERY)
    scores = _signals(bars, coords, parameters)
    execution = _execution_returns(bars, coords)
    iso_days = np.asarray([day.isoformat() for day in dates])
    return pa.Table.from_arrays(
        [
            pa.array([symbol] * len(coords), type=pa.string()),
            pa.array(iso_days[coords[:, 0]], type=pa.string()),
            pa.array(coords[:, 1], type=pa.int16()),
            *(pa.array(score, type=pa.float64()) for score in scores),
            pa.array(y, type=pa.float64()),
            pa.array(execution, mask=~np.isfinite(execution), type=pa.float64()),
        ],
        schema=SCORE_SCHEMA,
    )


def _stock_profile(
    provider: Path, symbol: str, dates: list[date], median_amount: float | None, seed: int, parameters,
) -> dict:
    bars = load_bars(provider, symbol, dates)
    y, coords = find_targets(bars, sample_every=SAMPLE_EVERY)
    _, starts, counts = np.unique(coords[:, 0], return_index=True, return_counts=True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(symbol[2:]), 0 if symbol[:2] == "SH" else 1]))
    picked = starts + rng.integers(counts)
    result = dict.fromkeys(PROFILE_COLUMNS)
    result.update(symbol=symbol, calibration_n=len(picked), train_median_amount=median_amount)
    if len(picked):
        result["train_volatility_bp"] = float(np.std(y[picked], dtype=np.float64))
    if len(picked) >= 30:
        for name, score in zip(SIGNALS, _signals(bars, coords[picked], parameters), strict=True):
            values = np.quantile(score, np.asarray(QUANTILES) / 100)
            result.update({f"{name}_q{q}": float(value) for q, value in zip(QUANTILES, values, strict=True)})
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    run = json.loads((args.run_dir / "metrics.json").read_text())
    if run["universe_mode"] != "historical_mainboard_proxy":
        raise ValueError("经济分析依赖，且只复用，原联合模型的近似主板股票池")
    source = Path(run["universe_source"])
    calendar = read_dates(args.provider)
    train_dates = _date_range(calendar, run["split"]["train"])
    dev_dates = _date_range(calendar, run["split"]["development_test"])
    if train_dates[-1] >= dev_dates[0] or dev_dates[-1] >= date.fromisoformat(
        run["split"]["sealed_final_validation"]["start"]
    ):
        raise ValueError("原模型训练、开发和封存区间存在重合")
    development = _development_days(args.run_dir / "shared_daily.csv", dev_dates)
    if not development:
        raise ValueError("原模型缺少开发评分股票日")
    universe, _ = history_mainboard_eligibility(args.provider, source, train_dates)
    medians = _training_amounts(source, train_dates, universe, set(development))
    parameters = _score_parameters(args.run_dir / "model.pt")
    out = args.run_dir / "economics"
    out.mkdir(exist_ok=True)

    tables, profiles = [], []
    rows = execution_missing = 0
    with cached(), pq.ParquetWriter(out / "signal_scores.parquet", SCORE_SCHEMA, compression="zstd") as writer:
        for index, (symbol, dates) in enumerate(development.items(), 1):
            table = _score_development(args.provider, symbol, dates, parameters)
            tables.append(table)
            rows += len(table)
            execution_missing += table.column("execution_return_bp").null_count
            eligible = universe.get(symbol, np.zeros(len(train_dates), dtype=bool))
            calibration = [day for day, keep in zip(train_dates, eligible, strict=True) if keep]
            profiles.append(_stock_profile(
                args.provider, symbol, calibration, medians.get(symbol), run["seed"], parameters,
            ))
            if len(tables) >= 32:
                writer.write_table(pa.concat_tables(tables))
                tables.clear()
            if index == 1 or index % 250 == 0 or index == len(development):
                print(f"经济信号 {index}/{len(development)} 只，累计 {rows:,} 个开发窗口", flush=True)
        if tables:
            writer.write_table(pa.concat_tables(tables))
    with (out / "stock_profiles.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        writer.writerows(profiles)
    print(json.dumps({
        "development_windows": rows, "missing_execution_returns": execution_missing,
        "calibrated_stocks": sum(row["calibration_n"] >= 30 for row in profiles),
        "profile_stocks": len(profiles),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
