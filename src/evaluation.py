"""Evaluate minute-return predictions in basis points, by stock and date."""

import csv
from collections import defaultdict
from datetime import date
from math import fsum, isfinite, sqrt
from pathlib import Path

import numpy as np


DAILY_COLUMNS = (
    "symbol",
    "date",
    "n",
    "rank_ic",
    "mae_bp",
    "sq_error_bp2",
    "mean_prediction_bp",
    "mean_actual_bp",
)
MIN_RANK_IC_SAMPLES = 20
SIGNAL_SLOTS_BY_SESSION = (86, 83)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    last_ranks = np.cumsum(counts)
    return (last_ranks - (counts - 1) / 2)[inverse]


def _spearman(prediction: np.ndarray, actual: np.ndarray) -> float | None:
    if len(prediction) < MIN_RANK_IC_SAMPLES:
        return None
    prediction_ranks = _average_ranks(prediction)
    actual_ranks = _average_ranks(actual)
    prediction_ranks -= prediction_ranks.mean()
    actual_ranks -= actual_ranks.mean()
    denominator = np.linalg.norm(prediction_ranks) * np.linalg.norm(actual_ranks)
    if denominator == 0:
        return None
    return float(np.clip(np.dot(prediction_ranks, actual_ranks) / denominator, -1, 1))


def daily_metrics(
    symbol: str,
    dates: list[date],
    coords: np.ndarray,
    prediction: np.ndarray,
    actual: np.ndarray,
) -> list[dict]:
    """Compute metrics for each date with at least one finite prediction/label pair."""
    coords = np.asarray(coords)
    prediction = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if prediction.ndim != 1 or prediction.shape != actual.shape:
        raise ValueError("prediction and actual must be equally sized one-dimensional arrays")
    if coords.shape != (len(prediction), 2):
        raise ValueError("coords must have one (date index, minute index) pair per prediction")
    if len(coords) == 0:
        return []
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("coords must contain integers")

    day_indices = coords[:, 0]
    if np.any(day_indices < 0) or np.any(day_indices >= len(dates)):
        raise ValueError("coords contain a date index outside dates")

    valid = np.flatnonzero(np.isfinite(prediction) & np.isfinite(actual))
    if len(valid) == 0:
        return []
    order = valid[np.argsort(day_indices[valid], kind="stable")]
    group_boundaries = np.flatnonzero(np.diff(day_indices[order])) + 1
    rows = []
    for indices in np.split(order, group_boundaries):
        day_prediction = prediction[indices]
        day_actual = actual[indices]
        error = day_prediction - day_actual
        rows.append({
            "symbol": symbol,
            "date": dates[day_indices[indices[0]]].isoformat(),
            "n": len(indices),
            "rank_ic": _spearman(day_prediction, day_actual),
            "mae_bp": float(np.abs(error).mean()),
            "sq_error_bp2": float(np.square(error).mean()),
            "mean_prediction_bp": float(day_prediction.mean()),
            "mean_actual_bp": float(day_actual.mean()),
        })
    return rows


def summarize_daily(rows: list[dict], eligible_stock_days: int, sample_every: int = 1) -> dict:
    """Summarize IC, pooled errors, and coverage at the per-session sampling step."""
    eligible_stock_days = int(eligible_stock_days)
    if eligible_stock_days < 0:
        raise ValueError("eligible_stock_days must be nonnegative")
    if not isinstance(sample_every, (int, np.integer)) or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    sample_every = int(sample_every)

    ic_by_day = defaultdict(list)
    for row in rows:
        rank_ic = row["rank_ic"]
        if rank_ic is not None and isfinite(rank_ic):
            ic_by_day[row["date"]].append(rank_ic)

    daily_rank_ics = [fsum(values) / len(values) for values in ic_by_day.values()]
    n_ic_days = len(daily_rank_ics)
    n = sum(int(row["n"]) for row in rows)
    slots_per_day = sum(1 + (slots - 1) // sample_every for slots in SIGNAL_SLOTS_BY_SESSION)
    expected_n = slots_per_day * eligible_stock_days
    return {
        "n": n,
        "expected_n": expected_n,
        "sample_coverage": n / expected_n if expected_n else None,
        "eligible_stock_days": eligible_stock_days,
        "observed_stock_days": len(rows),
        "n_days": len({row["date"] for row in rows}),
        "n_ic_days": n_ic_days,
        "mean_rank_ic": fsum(daily_rank_ics) / n_ic_days if n_ic_days else None,
        "positive_rank_ic_day_fraction": (
            sum(ic > 0 for ic in daily_rank_ics) / n_ic_days if n_ic_days else None
        ),
        "mae_bp": fsum(row["n"] * row["mae_bp"] for row in rows) / n if n else None,
        "rmse_bp": (
            sqrt(fsum(row["n"] * row["sq_error_bp2"] for row in rows) / n) if n else None
        ),
    }


def write_daily(path: Path, rows: list[dict]) -> None:
    """Write the stock-day metrics as CSV, with missing metrics left blank."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAILY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
