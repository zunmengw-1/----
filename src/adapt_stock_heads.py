"""Fit stock-specific linear heads on a frozen common minute encoder."""

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

from data import history_mainboard_eligibility, load_bars, read_dates
from features import encode_windows, find_signal_coords, find_targets
from score_economics import (
    SAMPLE_EVERY,
    _date_range,
    _execution_returns,
    _score_parameters,
)
from train import ROOT, encode, fit_stock_head, sample_training_rows

FIT_DAYS = 140
CALIBRATION_DAYS = 30
FIT_CAP = 512
PENALTY = 0.01
CPU = torch.device("cpu")
SCHEMA = pa.schema([
    ("symbol", pa.string()), ("date", pa.string()), ("slot", pa.int16()),
    ("shared_bp", pa.float64()), ("adapted_bp", pa.float64()), ("execution_return_bp", pa.float64()),
])


def _design(model, scaler, bars: np.ndarray, coords: np.ndarray) -> np.ndarray:
    x = encode_windows(bars, coords)
    features = encode(model, x, scaler, FIT_CAP, CPU)
    return np.column_stack((features, np.ones(len(coords))))


def _calibration_table(
    symbol: str, dates: list[date], bars: np.ndarray, model, scaler, shared: np.ndarray, theta: np.ndarray,
) -> pa.Table:
    coords = find_signal_coords(bars, sample_every=SAMPLE_EVERY)
    design = _design(model, scaler, bars, coords)
    execution = _execution_returns(bars, coords)
    iso_days = np.asarray([day.isoformat() for day in dates], dtype="U10")
    return pa.Table.from_arrays([
        pa.array([symbol] * len(coords), type=pa.string()),
        pa.array(iso_days[coords[:, 0]], type=pa.string()),
        pa.array(coords[:, 1], type=pa.int16()),
        pa.array((design @ shared) * scaler.label_scale, type=pa.float64()),
        pa.array((design @ theta) * scaler.label_scale, type=pa.float64()),
        pa.array(execution, mask=~np.isfinite(execution), type=pa.float64()),
    ], schema=SCHEMA)


def _adapt_stock(
    provider: Path, symbol: str, early_dates: list[date], tail_dates: list[date],
    model, scaler, shared: np.ndarray, seed: int,
) -> tuple[np.ndarray | None, pa.Table | None, dict]:
    dates = early_dates + tail_dates
    early_count = len(early_dates)
    bars = load_bars(provider, symbol, dates)
    fit_bars, fit_dates = bars[:early_count], early_dates
    y, coords = find_targets(fit_bars)
    fit_period = "early_140"
    if not len(y):
        fit_period = "fallback_full_170"
        fit_bars, fit_dates = bars, dates
        y, coords = find_targets(fit_bars)
    record = {
        "symbol": symbol,
        "eligible_early_stock_days": early_count,
        "eligible_tail_stock_days": len(tail_dates),
        "fit_period": fit_period if len(y) else None,
        "fit_candidate_windows": len(y),
        "fit_used_windows": 0,
        "fit_days_represented": 0,
        "calibration_windows": 0,
        "calibration_executable_windows": 0,
    }
    if not len(y):
        record["status"] = "no_labeled_training_windows"
        return None, None, record

    exchange_flag = 0 if symbol[:2] == "SH" else 1
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(symbol[2:]), exchange_flag]))
    picked = sample_training_rows(coords, FIT_CAP, True, rng)
    fit_day_indices = coords[picked, 0]
    record.update(
        fit_used_windows=len(picked),
        fit_days_represented=int(np.unique(fit_day_indices).size),
        first_fit_date=fit_dates[fit_day_indices.min()].isoformat(),
        last_fit_date=fit_dates[fit_day_indices.max()].isoformat(),
    )
    design = _design(model, scaler, fit_bars, coords[picked])
    labels = y[picked].astype(np.float64) / scaler.label_scale
    try:
        theta, converged = fit_stock_head(design, labels, shared, PENALTY)
        record["status"] = "fitted" if converged else "finite_head_optimizer_warning"
    except RuntimeError as exc:
        theta = shared.copy()
        record["status"] = "shared_head_optimizer_fallback"
        record["error"] = str(exc)
    record["parameter_delta_l2"] = float(np.linalg.norm(theta - shared))

    calibration = None
    if fit_period == "early_140":
        calibration = _calibration_table(symbol, tail_dates, bars[early_count:], model, scaler, shared, theta)
        record["calibration_windows"] = len(calibration)
        record["calibration_executable_windows"] = len(calibration) - calibration.column("execution_return_bp").null_count
    return theta, calibration, record


def _period(dates: list[date]) -> dict:
    return {"start": dates[0].isoformat(), "end": dates[-1].isoformat(), "days": len(dates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "output/tcn_mainboard_joint_01")
    parser.add_argument("--provider", type=Path, default=ROOT / "data/qlib_cn_1min")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    run = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    if run["universe_mode"] != "historical_mainboard_proxy":
        raise ValueError("单股适配需要原联合模型的历史主板股票池")
    train_dates = _date_range(read_dates(args.provider), run["split"]["train"])
    if len(train_dates) != FIT_DAYS + CALIBRATION_DAYS:
        raise ValueError("原联合训练期必须包含 140 个拟合日和 30 个校准日")
    if train_dates[-1] >= date.fromisoformat(run["split"]["development_test"]["start"]):
        raise ValueError("原联合训练期和开发期存在重叠")
    early, tail = train_dates[:FIT_DAYS], train_dates[FIT_DAYS:]
    source = Path(run["universe_source"])
    universe, _ = history_mainboard_eligibility(args.provider, source, train_dates)
    checkpoint = args.run_dir / "model.pt"
    model, scaler, weight, bias = _score_parameters(checkpoint)
    shared = np.r_[weight, bias]
    out = args.run_dir / "stock_adaptation"
    out.mkdir(exist_ok=True)
    tables, symbols, weights, records = [], [], [], []
    calibrated = 0
    with cached(), pq.ParquetWriter(out / "calibration.parquet", SCHEMA, compression="zstd") as writer:
        for index, (symbol, mask) in enumerate(sorted(universe.items()), 1):
            eligible_early = [day for day, valid in zip(early, mask[:FIT_DAYS], strict=True) if valid]
            eligible_tail = [day for day, valid in zip(tail, mask[FIT_DAYS:], strict=True) if valid]
            theta, calibration, record = _adapt_stock(
                args.provider, symbol, eligible_early, eligible_tail, model, scaler, shared, run["seed"],
            )
            records.append(record)
            if theta is not None:
                symbols.append(symbol)
                weights.append(theta)
            if calibration is not None and len(calibration):
                tables.append(calibration)
                calibrated += len(calibration)
            if len(tables) >= 32:
                writer.write_table(pa.concat_tables(tables))
                tables.clear()
            if index == 1 or index % 250 == 0 or index == len(universe):
                print(f"单股头适配 {index}/{len(universe)} 只，已产出 {len(symbols)} 个头，校准 {calibrated:,} 条", flush=True)
        if tables:
            writer.write_table(pa.concat_tables(tables))
    if not weights:
        raise ValueError("训练期没有可用于单股头拟合的窗口")
    np.savez_compressed(out / "heads.npz", symbols=np.asarray(symbols, dtype=np.str_), theta=np.vstack(weights))

    status_counts = {}
    for record in records:
        name = record["status"]
        status_counts[name] = status_counts.get(name, 0) + 1
    audit = {
        "shared_checkpoint": str(checkpoint),
        "historical_universe_source": str(source),
        "fit_period": _period(early),
        "fallback_fit_period": _period(train_dates),
        "calibration_period": _period(tail),
        "calibration_held_out_from_stock_head_fits": True,
        "shared_encoder_original_training_includes_calibration_dates": True,
        "eligible_symbols": len(universe),
        "heads_saved": len(symbols),
        "head_status_counts": status_counts,
        "window_cap_per_stock": FIT_CAP,
        "fit_sample_every_minutes": 1,
        "fit_sampling": "day-stratified, with at least one window from each valid day",
        "seed": run["seed"],
        "penalty_l2_to_shared_head": PENALTY,
        "huber_delta_scaled_labels": 1.0,
        "label_scale_bp": scaler.label_scale,
        "calibration_sample_every_minutes": SAMPLE_EVERY,
        "fallback_fit_symbols_excluded_from_calibration": [
            row["symbol"] for row in records if row["fit_period"] == "fallback_full_170"
        ],
        "fit_candidate_windows": sum(row["fit_candidate_windows"] for row in records),
        "fit_used_windows": sum(row["fit_used_windows"] for row in records),
        "fit_represented_stock_days": sum(row["fit_days_represented"] for row in records),
        "calibration_windows": sum(row["calibration_windows"] for row in records),
        "calibration_executable_windows": sum(row["calibration_executable_windows"] for row in records),
        "stocks": records,
    }
    (out / "head_fit_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in audit.items() if key != "stocks"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
