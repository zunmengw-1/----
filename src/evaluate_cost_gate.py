"""Trigger an overlay only when calibrated executable returns exceed modeled costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import read_dates
from evaluate_costs import (
    COMMISSION,
    END,
    METHODS,
    MIN_COMMISSION,
    ORDER_SHARES,
    RESULTS,
    ROOT,
    SCORES,
    SLIPPAGE,
    SLOTS,
    STAMP_DUTY,
    START,
    THRESHOLDS,
    TRANSFER_FEE,
    account,
    load_price_references,
    raw_market,
    simulate,
    summary,
)

QUANTILES = (.20, .40, .60, .80, .90, .95, .975, .99, .995, .999)
RETURN_COLUMN = "execution_return_bp"


def fit_return_buckets(calibration: pd.DataFrame, score_names):
    # Costs are charged on notionals, so average arithmetic returns inside each bucket.
    usable = calibration.dropna(subset=[*score_names, RETURN_COLUMN])
    realized = 10_000 * np.expm1(usable[RETURN_COLUMN].to_numpy() / 10_000)
    if not len(usable) or not np.isfinite(realized).all():
        raise ValueError("缺少有效的训练期可执行收益")
    cutoffs_by_method, levels_by_method = [], []
    audit = {}
    for name in score_names:
        score = usable[name].to_numpy()
        if not np.isfinite(score).all():
            raise ValueError(f"{name} 校准分数包含非有限值")
        cutoffs = np.unique(np.quantile(score, QUANTILES))
        buckets = np.searchsorted(cutoffs, score, side="right")
        n_buckets = len(cutoffs) + 1
        counts = np.bincount(buckets, minlength=n_buckets)
        total = np.bincount(buckets, weights=realized, minlength=n_buckets)
        means = np.divide(total, counts, where=counts > 0, out=np.full(n_buckets, np.nan))
        cutoffs_by_method.append(cutoffs)
        levels_by_method.append(means)
        training_dates = usable.assign(bucket=buckets).groupby("bucket")["date"].nunique()
        audit[name] = {
            "cutoffs_bp": cutoffs.tolist(),
            "buckets": [
                {"train_rows": int(counts[i]), "train_days": int(training_dates.get(i, 0)),
                 "expected_arithmetic_return_bp": float(means[i]) if np.isfinite(means[i]) else None}
                for i in range(n_buckets)
            ],
        }
    return cutoffs_by_method, levels_by_method, audit


def estimate_returns(calibration: pd.DataFrame, scores: pd.DataFrame) -> tuple[np.ndarray, dict]:
    cutoffs, levels, audit = fit_return_buckets(calibration, SCORES)
    expected = np.empty((len(scores), len(SCORES)))
    for column, (name, edges, means) in enumerate(zip(SCORES, cutoffs, levels, strict=True)):
        dev_buckets = np.searchsorted(edges, scores[name].to_numpy(), side="right")
        expected[:, column] = means[dev_buckets]
        dev_counts = np.bincount(dev_buckets, minlength=len(means))
        for bucket, count in zip(audit[name]["buckets"], dev_counts, strict=True):
            bucket["development_rows"] = int(count)
    return expected, audit


def signal_close_prices(provider: Path, scores: pd.DataFrame, days: list) -> np.ndarray:
    prices = np.full(len(scores), np.nan)
    for symbol, group in scores.groupby("symbol", sort=False):
        day_indices, row = np.unique(group["day"].to_numpy(dtype=np.intp), return_inverse=True)
        _, close, _ = raw_market(provider, symbol, [days[index] for index in day_indices])
        prices[group.index.to_numpy()] = close[row, group["slot"].to_numpy(dtype=np.intp)]
    return prices


def projected_margin_bp(reference: np.ndarray, gross_bp: np.ndarray, slippage_bp: int) -> np.ndarray:
    """Compute the expected net round trip using only signal-time prices and forecasts."""
    slip = slippage_bp / 10_000
    buy = reference[:, None] * ORDER_SHARES * (1 + slip)
    sell = reference[:, None] * ORDER_SHARES * (1 + gross_bp / 10_000) * (1 - slip)
    commission = np.maximum(MIN_COMMISSION, buy * COMMISSION) + np.maximum(MIN_COMMISSION, sell * COMMISSION)
    fees = commission + (buy + sell) * TRANSFER_FEE + sell * STAMP_DUTY
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10_000 * (sell - buy - fees) / buy


def _finite_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.max()) if len(finite) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--economics", type=Path, default=RESULTS)
    parser.add_argument("--provider", type=Path, default=ROOT / "data/qlib_cn_1min")
    parser.add_argument("--daily-source", type=Path, default=ROOT / "市场微观结构研究/cache/daily_summary.parquet")
    args = parser.parse_args()
    data = args.economics
    output = data / "cost_gate"
    calibration = pd.read_parquet(output / "calibration_signals.parquet", columns=["date", *SCORES, RETURN_COLUMN])
    if calibration["date"].min() != "2025-01-02" or calibration["date"].max() != "2025-09-11":
        raise ValueError("校准文件必须限定在已约定的前 170 个交易日")
    scores = pd.read_parquet(data / "signal_scores.parquet", columns=["symbol", "date", "slot", *SCORES])
    profiles = pd.read_csv(data / "stock_profiles.csv", usecols=["symbol", *THRESHOLDS])
    # Use the same comparison universe as the original upper-quintile execution experiment.
    common = profiles.loc[np.isfinite(profiles[THRESHOLDS]).all(axis=1), "symbol"]
    scores = scores.loc[scores["symbol"].isin(common) & np.isfinite(scores[SCORES]).all(axis=1)].reset_index(drop=True)
    days = [day for day in read_dates(args.provider) if START <= day <= END]
    day_index = {day.isoformat(): i for i, day in enumerate(days)}
    scores["day"] = scores["date"].map(day_index)
    if (len(days) != 49 or scores["day"].isna().any() or not scores["slot"].isin(SLOTS).all()
            or scores.duplicated(["symbol", "day", "slot"]).any()):
        raise ValueError("开发信号与约定采样或日期不一致")
    per_day = np.bincount(scores["day"].to_numpy(dtype=np.intp), minlength=len(days))
    expected, calibration_stats = estimate_returns(calibration, scores)
    reference = signal_close_prices(args.provider.resolve(), scores, days)
    price_available = np.isfinite(reference) & (reference > 0)
    references = load_price_references(args.daily_source, days)
    measures, trigger_stats, unclosed = [], {}, []
    for slippage in SLIPPAGE:
        net_margin = projected_margin_bp(reference, expected, slippage)
        flags = price_available[:, None] & np.isfinite(net_margin) & (net_margin > 0)
        candidates = scores.loc[flags.any(axis=1), ["symbol", "day", "slot"]].copy()
        candidates[list(METHODS[:-1])] = flags[flags.any(axis=1)]
        candidates = candidates.sort_values(["symbol", "day", "slot"])
        counters, trades, open_positions = simulate(args.provider.resolve(), days, candidates, references)
        daily = account(days, per_day, counters, trades)
        frame = daily.loc[daily["slippage_bp_per_side"].eq(slippage)]
        measures.append(frame)
        unclosed.extend((slippage, *position) for position in open_positions)
        trigger_stats[str(slippage)] = {
            method: {
                "candidates_triggered": int(flags[:, i].sum()),
                "calibrated_signals": int(np.isfinite(expected[:, i]).sum()),
                "max_expected_gross_bp": _finite_max(expected[:, i]),
                "max_expected_net_bp": _finite_max(net_margin[:, i]),
            }
            for i, method in enumerate(METHODS[:-1])
        }
        print(f"每边滑点 {slippage} bp，触发数 {flags.sum(axis=0).tolist()}", flush=True)

    daily = pd.concat(measures, ignore_index=True)
    report = summary(daily)
    report.to_csv(output / "cost_gate_comparison.csv", index=False)
    daily.to_csv(output / "cost_gate_daily.csv", index=False)
    pd.DataFrame(unclosed, columns=["slippage_bp", "date", "strategy", "symbol", "signal_bar_end",
                                   "entry_bar_end", "planned_exit_bar_end", "buy_raw_yuan", "eod_mark_yuan",
                                   "mark_source"]).to_csv(output / "cost_gate_unclosed.csv", index=False)
    audit = {
        "calibration_dates": [calibration["date"].min(), calibration["date"].max()],
        "calibration_signals": len(calibration),
        "calibration_with_defined_executable_return": int(calibration[RETURN_COLUMN].notna().sum()),
        "development_dates": [START.isoformat(), END.isoformat()], "development_signals": len(scores),
        "development_symbols": int(scores["symbol"].nunique()), "price_available_signals": int(price_available.sum()),
        "gate": "Expected arithmetic return from next-minute open to t+5 close, converted to yuan using the signal close. Trade when the projected round-trip margin is positive.",
        "calibration": "Frozen model scores and training-period execution proxy only, pooled by score percentile, conditional on an available entry and exit proxy.",
        "training_percentile_edges": list(QUANTILES),
        "calibration_tables": calibration_stats, "trigger_audit_by_slippage": trigger_stats,
        "exit": "Same t+5 or earliest later modeled exit if blocked, using settled base holding",
        "execution_rules": "../cost_assumptions.json", "final_24_days_read": False,
    }
    (output / "cost_gate_assumptions.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(report[["strategy", "slippage_bp_per_side", "trigger_signals", "entries", "net_bp_on_buy_notional", "net_pnl_yuan"]].to_string(index=False))


if __name__ == "__main__":
    main()
