"""Measure intraday rank IC, its cross-day stability, and comparable baselines."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

from intraday_tcn.paths import DEFAULT_RUN

DEVELOPMENT_START = "2025-09-12"
DEVELOPMENT_END = "2025-11-27"
SIGNAL_SLOTS = (29, 59, 89, 149, 179, 209)
STOCK_DAY_KEYS = ["symbol", "date"]
SIGNAL_KEYS = [*STOCK_DAY_KEYS, "slot"]
RESEARCH_METHODS = {
    "tcn_rank_ic": "tcn_bp",
    "reversal_rank_ic": "reversal_bp",
    "momentum_rank_ic": "momentum_bp",
}
EXECUTION_METRIC = "tcn_execution_rank_ic"
METRICS = (*RESEARCH_METHODS, EXECUTION_METRIC)
SCORE_COLUMNS = [*SIGNAL_KEYS, *RESEARCH_METHODS.values(), "label_bp", "execution_return_bp"]
BLOCK_LENGTH = 5
BOOTSTRAP_DRAWS = 10_000
DEFAULT_SCORES = DEFAULT_RUN / "economics/signal_scores.parquet"


def _rank_ic(scores: np.ndarray, target_ranks: np.ndarray) -> np.ndarray:
    score_ranks = rankdata(scores, axis=1, method="average", nan_policy="propagate")
    center = (len(SIGNAL_SLOTS) + 1) / 2
    score_ranks -= center
    centered_target = target_ranks - center
    denominator = np.sqrt((score_ranks**2).sum(axis=1) * (centered_target**2).sum(axis=1))
    return np.divide(
        (score_ranks * centered_target).sum(axis=1),
        denominator,
        out=np.full(len(scores), np.nan),
        where=denominator > 0,
    )


def stock_day_ic(scores: pd.DataFrame) -> pd.DataFrame:
    if not scores["slot"].isin(SIGNAL_SLOTS).all():
        raise ValueError("signal scores include a slot outside the six-point grid")
    if scores.duplicated(SIGNAL_KEYS).any():
        raise ValueError("duplicate symbol/date/slot in signal scores")

    scores = scores.sort_values(SIGNAL_KEYS)
    groups = scores.groupby(STOCK_DAY_KEYS, sort=False).size().rename("n_signals").reset_index()
    complete = groups["n_signals"] == len(SIGNAL_SLOTS)
    matrices = {}
    rows = scores.loc[np.repeat(complete.to_numpy(), groups["n_signals"].to_numpy())]
    for col in SCORE_COLUMNS[len(SIGNAL_KEYS):]:
        values = rows[col].to_numpy(dtype=np.float64, na_value=np.nan).reshape(-1, len(SIGNAL_SLOTS))
        matrices[col] = np.where(np.isfinite(values), values, np.nan)

    research = rankdata(matrices["label_bp"], axis=1, method="average", nan_policy="propagate")
    execution = rankdata(matrices["execution_return_bp"], axis=1, method="average", nan_policy="propagate")
    for metric, column in RESEARCH_METHODS.items():
        groups[metric] = np.nan
        groups.loc[complete, metric] = _rank_ic(matrices[column], research)
    groups[EXECUTION_METRIC] = np.nan
    groups.loc[complete, EXECUTION_METRIC] = _rank_ic(matrices["tcn_bp"], execution)
    return groups


def _distribution(values: np.ndarray) -> dict:
    probabilities = {
        "min": 0, "p05": .05, "p10": .1, "p25": .25, "median": .5,
        "p75": .75, "p90": .9, "p95": .95, "max": 1,
    }
    values = values[np.isfinite(values)]
    estimates = (
        np.quantile(values, list(probabilities.values())).tolist()
        if len(values) else [None] * len(probabilities)
    )
    quantiles = dict(zip(probabilities, estimates))
    n_positive = int((values > 0).sum())
    return {
        "n": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        **quantiles,
        "iqr": quantiles["p75"] - quantiles["p25"] if len(values) else None,
        "n_positive": n_positive,
        "positive_fraction": n_positive / len(values) if len(values) else None,
    }


def _moving_block_interval(daily: pd.Series, n_expected_dates: int) -> list[float] | None:
    values = daily.dropna().to_numpy(dtype=np.float64)
    if len(values) != n_expected_dates or len(values) < 2 * BLOCK_LENGTH:
        return None
    rng = np.random.default_rng(2025)
    n_blocks = (len(values) + BLOCK_LENGTH - 1) // BLOCK_LENGTH
    starts = rng.integers(0, len(values) - BLOCK_LENGTH + 1, size=(BOOTSTRAP_DRAWS, n_blocks))
    draws = values[(starts[..., None] + np.arange(BLOCK_LENGTH)).reshape(BOOTSTRAP_DRAWS, -1)]
    return np.quantile(draws[:, :len(values)].mean(axis=1), [.025, .975]).tolist()


def _summary(groups: pd.DataFrame, metric: str, n_dates: int) -> tuple[dict, pd.DataFrame]:
    date_groups = groups.groupby("date")[metric]
    daily = date_groups.agg(["mean", "count"])
    individual = groups.groupby("symbol")[metric].mean()
    return {
        "stock_day": _distribution(groups[metric].to_numpy()),
        "daily_equal_stock_weight": _distribution(daily["mean"].to_numpy()),
        "stock_mean_equal_available_day_weight": _distribution(individual.to_numpy()),
        "daily_mean_95pct_moving_block_interval": _moving_block_interval(daily["mean"], n_dates),
    }, daily


def _compare(groups: pd.DataFrame, metrics: tuple[str, ...], n_dates: int) -> tuple[dict, dict]:
    same_rows = groups.dropna(subset=list(metrics))
    summaries, series = {}, {}
    for metric in metrics:
        summaries[metric], series[metric] = _summary(same_rows, metric, n_dates)
    return {"valid_common_stock_days": len(same_rows), "metrics": summaries}, series


def _daily_difference(daily_a: pd.DataFrame, daily_b: pd.DataFrame, n_dates: int) -> dict:
    difference = daily_a["mean"] - daily_b["mean"]
    return {
        **_distribution(difference.to_numpy()),
        "mean_95pct_moving_block_interval": _moving_block_interval(difference, n_dates),
    }


def analyze(scores: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if scores.empty:
        raise ValueError("development signal table is empty")
    groups = stock_day_ic(scores)
    n_dates = scores["date"].nunique()
    independent, independent_daily = {}, {}
    for metric in METRICS:
        independent[metric], independent_daily[metric] = _summary(groups, metric, n_dates)

    common_research, research_daily = _compare(groups, tuple(RESEARCH_METHODS), n_dates)
    common_research["tcn_minus_reversal_daily"] = _daily_difference(
        research_daily["tcn_rank_ic"], research_daily["reversal_rank_ic"], n_dates
    )
    common_research["tcn_minus_momentum_daily"] = _daily_difference(
        research_daily["tcn_rank_ic"], research_daily["momentum_rank_ic"], n_dates
    )
    common_execution, execution_daily = _compare(groups, ("tcn_rank_ic", EXECUTION_METRIC), n_dates)
    common_execution["execution_minus_research_daily"] = _daily_difference(
        execution_daily[EXECUTION_METRIC], execution_daily["tcn_rank_ic"], n_dates
    )

    daily = pd.concat([
        frame.add_prefix(f"{sample}_{metric}_")
        for sample, metric_frames in (
            ("independent", independent_daily), ("common_research", research_daily),
            ("common_execution", execution_daily),
        )
        for metric, frame in metric_frames.items()
    ], axis=1).sort_index()
    result = {
        "period": {"start": scores["date"].min(), "end": scores["date"].max(), "days": n_dates},
        "signal_slots": list(SIGNAL_SLOTS),
        "n_signals": len(scores),
        "n_stock_days": len(groups),
        "n_complete_stock_days": int((groups["n_signals"] == len(SIGNAL_SLOTS)).sum()),
        "bootstrap": {"block_trading_days": BLOCK_LENGTH, "draws": BOOTSTRAP_DRAWS, "seed": 2025},
        "independent_available_stock_days": independent,
        "common_research_stock_days": common_research,
        "common_execution_stock_days": common_execution,
    }
    return result, groups, daily


def _check_reference(scores_path: Path, result: dict) -> None:
    path = scores_path.parent.parent / "ic_stability/input_disjoint_30m.json"
    if not path.is_file():
        return
    reference = json.loads(path.read_text(encoding="utf-8"))
    tcn = result["independent_available_stock_days"]["tcn_rank_ic"]
    observed = tcn["daily_equal_stock_weight"]["mean"]
    expected = reference["disjoint_input_mean_rank_ic"]
    n_stock_days = tcn["stock_day"]["n"]
    if n_stock_days != reference["valid_ic_stock_days"] or abs(observed - expected) > 5e-5:
        raise ValueError(
            f"reference IC {expected} with {reference['valid_ic_stock_days']} stock-days, "
            f"new IC {observed} with {n_stock_days} stock-days"
        )
    result["reference_check"] = {"mean_daily_rank_ic": expected, "difference": observed - expected}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    out = args.out or args.scores.parent

    scores = pq.read_table(
        args.scores, columns=SCORE_COLUMNS,
        filters=[("date", ">=", DEVELOPMENT_START), ("date", "<=", DEVELOPMENT_END)],
    ).to_pandas()
    result, stock_days, daily = analyze(scores)
    _check_reference(args.scores, result)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ic_distribution.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    stock_days.to_parquet(out / "stock_day_ic.parquet", index=False)
    daily.to_csv(out / "daily_ic.csv", index_label="date")


if __name__ == "__main__":
    main()
