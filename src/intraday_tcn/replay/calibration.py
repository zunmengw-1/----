"""Export training-only executable returns and frozen scores for the cost gate."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn.utils.parametrize import cached

from intraday_tcn.data import history_mainboard_eligibility, load_bars, read_dates
from intraday_tcn.evaluation.scores import (
    SAMPLE_EVERY,
    SCORE_SCHEMA,
    _date_range,
    _execution_returns,
    _score_parameters,
    _signals,
)
from intraday_tcn.features import find_targets
from intraday_tcn.paths import DEFAULT_PROVIDER, DEFAULT_RUN

CALIBRATION_SCHEMA = pa.schema([field for field in SCORE_SCHEMA if field.name != "label_bp"])


def _one_window_per_day(coords: np.ndarray, symbol: str, seed: int) -> np.ndarray:
    _, starts, counts = np.unique(coords[:, 0], return_index=True, return_counts=True)
    exchange_flag = 0 if symbol[:2] == "SH" else 1
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(symbol[2:]), exchange_flag]))
    return coords[starts + rng.integers(counts)]


def _calibrate_stock(
    provider: Path, symbol: str, dates: list[date], seed: int, parameters,
) -> pa.Table:
    bars = load_bars(provider, symbol, dates)
    _, coords = find_targets(bars, sample_every=SAMPLE_EVERY)
    picked = _one_window_per_day(coords, symbol, seed)
    scores = _signals(bars, picked, parameters)
    execution = _execution_returns(bars, picked)
    iso_days = np.asarray([day.isoformat() for day in dates])
    return pa.Table.from_arrays(
        [
            pa.array([symbol] * len(picked), type=pa.string()),
            pa.array(iso_days[picked[:, 0]], type=pa.string()),
            pa.array(picked[:, 1], type=pa.int16()),
            *(pa.array(score, type=pa.float64()) for score in scores),
            pa.array(execution, mask=~np.isfinite(execution), type=pa.float64()),
        ],
        schema=CALIBRATION_SCHEMA,
    )


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
        raise ValueError("校准需要原联合模型的历史主板股票池")
    train_dates = _date_range(read_dates(args.provider), run["split"]["train"])
    if train_dates[-1] >= date.fromisoformat(run["split"]["development_test"]["start"]):
        raise ValueError("校准区间和开发区间存在重合")
    universe, _ = history_mainboard_eligibility(
        args.provider, Path(run["universe_source"]), train_dates,
    )
    parameters = _score_parameters(args.run_dir / "model.pt")
    out = args.run_dir / "economics" / "cost_gate"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "calibration_signals.parquet"

    tables = []
    rows = execution_missing = 0
    with cached(), pq.ParquetWriter(path, CALIBRATION_SCHEMA, compression="zstd") as writer:
        for index, (symbol, mask) in enumerate(sorted(universe.items()), 1):
            dates = [day for day, keep in zip(train_dates, mask, strict=True) if keep]
            table = _calibrate_stock(args.provider, symbol, dates, run["seed"], parameters)
            tables.append(table)
            rows += len(table)
            execution_missing += table.column("execution_return_bp").null_count
            if len(tables) >= 32:
                writer.write_table(pa.concat_tables(tables))
                tables.clear()
            if index == 1 or index % 500 == 0 or index == len(universe):
                print(f"成本校准 {index}/{len(universe)} 只，累计 {rows:,} 个训练窗口", flush=True)
        if tables:
            writer.write_table(pa.concat_tables(tables))
    print(json.dumps({
        "calibration_windows": rows, "missing_execution_returns": execution_missing,
        "eligible_symbols": len(universe), "first_date": train_dates[0].isoformat(),
        "last_date": train_dates[-1].isoformat(),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
