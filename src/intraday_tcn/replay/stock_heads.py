"""Replay shared and stock-specific forecasts using matched training-only cost gates."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from intraday_tcn.data import read_dates
from intraday_tcn.evaluation.scores import _score_parameters
from intraday_tcn.features import encode_windows
from intraday_tcn.paths import DEFAULT_DAILY_SOURCE, DEFAULT_PROVIDER, DEFAULT_RUN
from intraday_tcn.replay.cost_gate import QUANTILES, RETURN_COLUMN, fit_return_buckets
from intraday_tcn.replay.execution import END, SLOTS, START
from intraday_tcn.replay.minute import (
    collect_candidates,
    eligible_stock_days,
    save_results,
)
from intraday_tcn.training.train import encode

METHODS = ("shared", "adapted", "static")
SCORES = ["shared_bp", "adapted_bp"]


def _configure_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _collect_with_heads(provider, days, universe, heads, checkpoint, calibration, example_symbol):
    model, scaler, weight, bias = _score_parameters(checkpoint)
    device = torch.device("cpu")

    def score_forecasts(symbol, bars, coords):
        encoded = encode(model, encode_windows(bars, coords), scaler, 512, device)
        stock = heads[symbol]
        return scaler.label_scale * np.column_stack((encoded @ weight + bias, encoded @ stock[:-1] + stock[-1]))

    return collect_candidates(provider, days, universe, score_forecasts, calibration, example_symbol, METHODS)


def _combine_candidate_runs(runs):
    tables, daily, examples, audits = zip(*runs, strict=True)
    combined_tables = {}
    for slippage in tables[0]:
        nonempty = [table[slippage] for table in tables if not table[slippage].empty]
        combined_tables[slippage] = (
            pd.concat(nonempty, ignore_index=True).sort_values(["symbol", "day", "slot"])
            if nonempty else tables[0][slippage]
        )
    bound_key = "maximum_expected_margin_upper_bound_bp"
    audit = {
        key: sum(shard[key] for shard in audits)
        for key in ("scored_minute_windows", "skipped_without_order_possibility")
    }
    audit[bound_key] = {}
    for slippage in audits[0][bound_key]:
        audit[bound_key][slippage] = {
            method: max(
                (value for shard in audits if (value := shard[bound_key][slippage][method]) is not None),
                default=None,
            )
            for method in METHODS[:-1]
        }
    return combined_tables, np.sum(daily, axis=0), [row for shard in examples for row in shard], audit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--daily-source", type=Path, default=DEFAULT_DAILY_SOURCE)
    parser.add_argument("--example-symbol", default="SH600000")
    parser.add_argument("--entry", choices=("next_open", "signal_close"), default="next_open")
    parser.add_argument("--workers", type=int, default=1, help="Independent CPU processes across stocks")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    provider = args.provider.resolve()
    dates = read_dates(provider)
    days = [day for day in dates if START <= day <= END]
    training = [day for day in dates if date(2025, 1, 2) <= day < START]
    if len(days) != 49 or len(training) != 170 or training[-1] != date(2025, 9, 11):
        raise ValueError("交易日历与联合训练或开发期的区间不符")

    adapt_dir = args.run_dir / "stock_adaptation"
    signal_close = args.entry == "signal_close"
    output = adapt_dir / ("cost_gate_signal_close_1m" if signal_close else "cost_gate_1m")
    with np.load(adapt_dir / "heads.npz", allow_pickle=False) as saved:
        theta = saved["theta"]
        symbols = saved["symbols"]
    if (symbols.ndim != 1 or len(set(symbols)) != len(symbols)
            or theta.shape != (len(symbols), 17) or not np.isfinite(theta).all()):
        raise ValueError("股票预测头无效")
    heads = dict(zip(symbols, theta, strict=True))
    universe = eligible_stock_days(args.run_dir / "economics", days, heads)
    if not universe:
        raise ValueError("开发期缺少有微调头的股票")

    columns = ["symbol", "date", "slot", *SCORES, RETURN_COLUMN]
    calibration_path = output / "calibration.parquet" if signal_close else adapt_dir / "calibration.parquet"
    calibration = pd.read_parquet(calibration_path, columns=columns)
    allowed = [day.isoformat() for day in training[-30:]]
    if (set(calibration["date"]) != set(allowed)
            or not calibration["symbol"].isin(heads).all()
            or not calibration["slot"].isin(SLOTS).all()
            or calibration.duplicated(["symbol", "date", "slot"]).any()):
        raise ValueError("校准窗口与训练期末30日或既定采样不符")
    cutoffs, levels, bucket_audit = fit_return_buckets(calibration, SCORES)
    upper = np.asarray([np.nanmax(values) for values in levels])
    print(f"训练期校准 {len(calibration):,} 条，分桶最高预期收益 {upper.round(4).tolist()} bp", flush=True)
    checkpoint = args.run_dir / "model.pt"
    fitted = (cutoffs, levels, upper)
    example_symbol = args.example_symbol.upper()
    worker_count = min(args.workers, len(universe))
    if worker_count == 1:
        _configure_worker()
        result = _collect_with_heads(provider, days, universe, heads, checkpoint, fitted, example_symbol)
    else:
        stocks = list(universe.items())
        shards = [dict(stocks[i::worker_count]) for i in range(worker_count)]
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_configure_worker) as pool:
            tasks = [
                pool.submit(_collect_with_heads, provider, days, shard, heads, checkpoint, fitted, example_symbol)
                for shard in shards
            ]
            result = _combine_candidate_runs([task.result() for task in tasks])
    tables, valid_per_day, example, audit = result
    report = save_results(
        provider, days, args.daily_source, tables, valid_per_day, example, output, METHODS,
        entry_at_signal_close=signal_close,
    )
    audit.update({
        "development_dates": [days[0].isoformat(), days[-1].isoformat()], "development_days": len(days),
        "symbols": len(universe), "eligible_stock_days": sum(map(len, universe.values())),
        "eligible_minute_windows": int(valid_per_day.sum()), "sample_every_minutes": 1, "window_minutes": 30,
        "label_horizon_minutes": 5,
        "worker_count": worker_count,
        "calibration_dates": [allowed[0], allowed[-1]], "calibration_signals": len(calibration),
        "calibration_with_defined_executable_return": int(calibration.dropna(subset=[*SCORES, RETURN_COLUMN]).shape[0]),
        "training_percentile_edges": list(QUANTILES), "calibration_tables": bucket_audit,
        "max_calibrated_execution_return_bp": dict(zip(METHODS[:-1], upper.tolist(), strict=True)),
        "execution": "Same eligibility, modeled fills and costs for both heads. An overlay can enter only when the frozen calibrated net estimate is positive.",
        "entry": args.entry,
        "exit": "Close of the fifth bar after the signal or the first eligible later exit.",
        "entry_price_assumption": (
            "Assumes a fill at the observed signal bar close even though the complete bar is used for the forecast."
            if signal_close else "Uses the opening price of the first bar after the signal as a fill proxy."
        ),
        "calibration_prices": "signal close to t+5 close" if signal_close else "next open to t+5 close",
        "execution_rules": "../../economics/cost_assumptions.json",
        "bound_rule": "A minute is skipped in network inference only if no bucket can cover costs or the signal-time price is invalid. The example day is scored throughout.",
        "example_forecast_rows": len(example), "no_future_quote_filter": True, "final_24_days_read": False,
    })
    (output / "replay_assumptions.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(report[["strategy", "slippage_bp_per_side", "trigger_signals", "entries", "closed", "unclosed",
                  "positive_net_days", "net_bp_on_buy_notional", "net_pnl_yuan"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
