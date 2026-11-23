"""Check stock-type coverage and quintile ordering on frozen development scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path(__file__).resolve().parents[1] / "output/tcn_mainboard_joint_01/economics"
QUINTILES = range(1, 6)
CUTOFFS = ["tcn_q20", "tcn_q40", "tcn_q60", "tcn_q80"]


def profiles_by_type(profiles: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    profiles = profiles.copy()
    profiles["venue"] = profiles["symbol"].str[:2]
    groups = {}
    for column, dimension in (
        ("train_median_amount", "training_liquidity"),
        ("train_volatility_bp", "training_volatility"),
    ):
        valid = profiles[column].dropna()
        lower, upper = valid.quantile([1 / 3, 2 / 3]).tolist()
        if not np.isfinite([lower, upper]).all() or not lower < upper:
            raise ValueError(f"训练期 {column} 无法分成三组")
        profiles[dimension] = pd.cut(
            profiles[column], [-np.inf, lower, upper, np.inf], labels=["low", "middle", "high"]
        )
        groups[dimension] = {"source": column, "cutoffs": [lower, upper], "stock_count": len(valid)}
    return profiles, groups


def quintile_curve(rows: pd.DataFrame, outcome: str) -> tuple[dict, list[dict]]:
    valid = rows.loc[np.isfinite(rows[outcome])]
    by_stock_day = valid.groupby(["date", "symbol", "quintile"], sort=False)[outcome].mean()
    by_day = by_stock_day.groupby(level=["date", "quintile"]).mean().unstack().reindex(columns=QUINTILES)
    by_day = by_day.dropna()
    if by_day.empty:
        raise ValueError(f"{outcome} 没有覆盖五组的交易日")
    curve = by_day.mean(axis=0).to_numpy()
    spread = by_day[5] - by_day[1]
    index = np.arange(1, len(curve) + 1)
    fit = np.polyval(np.polyfit(index, curve, 1), index)
    variation = float(np.sum((curve - curve.mean()) ** 2))
    return {
        "windows": len(valid), "days_with_five_buckets": len(by_day),
        "quintile_means_bp": curve.tolist(), "q5_minus_q1_bp": float(spread.mean()),
        "positive_spread_day_fraction": float(np.mean(spread > 0)),
        "rising_adjacent_steps": int(np.count_nonzero(np.diff(curve) > 0)),
        "bucket_index_linear_r2": 1 - float(np.sum((curve - fit) ** 2)) / variation if variation else None,
    }, [
        {"date": day, "quintile": quintile, "mean_return_bp": float(row[quintile])}
        for day, row in by_day.iterrows() for quintile in QUINTILES
    ]


def rank_ic_values(ic: pd.DataFrame, column: str) -> dict:
    finite = ic.loc[ic[column].notna()]
    daily = finite.groupby("date")[column].mean()
    return {
        "stock_days": len(finite), "day_count": len(daily),
        "mean": float(daily.mean()) if len(daily) else None,
        "positive_days": int(daily.gt(0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=OUTPUT)
    directory = parser.parse_args().directory
    profiles, cutoffs = profiles_by_type(pd.read_csv(directory / "stock_profiles.csv"))
    scores = pd.read_parquet(directory / "signal_scores.parquet")
    ic = pd.read_parquet(directory / "stock_day_ic.parquet")
    assert scores["date"].min() == "2025-09-12" and scores["date"].max() == "2025-11-27"
    assigned = scores.merge(profiles, on="symbol", how="left", validate="many_to_one")
    rank_columns = ["symbol", "venue", "training_liquidity", "training_volatility"]
    ic = ic.merge(profiles[rank_columns], on="symbol", how="left", validate="many_to_one")
    thresholds = assigned[CUTOFFS].to_numpy()
    valid_thresholds = np.isfinite(thresholds).all(axis=1) & np.isfinite(assigned["tcn_bp"])
    usable = assigned.loc[valid_thresholds].copy()
    usable["quintile"] = 1 + (usable["tcn_bp"].to_numpy()[:, None] >= thresholds[valid_thresholds]).sum(axis=1)
    # paired curves use the same observations when comparing the two entry references
    paired = usable.loc[np.isfinite(usable["execution_return_bp"])]
    results, daily_rows = [], []
    for dimension, groups in (
        ("all", ["all"]), ("venue", ["SH", "SZ"]),
        ("training_liquidity", ["low", "middle", "high"]),
        ("training_volatility", ["low", "middle", "high"]),
    ):
        for group in groups:
            signals = paired if dimension == "all" else paired.loc[paired[dimension].eq(group)]
            stock_day = ic if dimension == "all" else ic.loc[ic[dimension].eq(group)]
            result = {
                "dimension": dimension, "group": group, "quintile_stock_count": signals["symbol"].nunique(),
                "label_rank_ic": rank_ic_values(stock_day, "tcn_rank_ic"),
                "execution_rank_ic": rank_ic_values(stock_day, "tcn_execution_rank_ic"),
            }
            for kind, outcome in (("label", "label_bp"), ("execution", "execution_return_bp")):
                result[kind], curve = quintile_curve(signals, outcome)
                daily_rows.extend({"dimension": dimension, "group": group, "return_kind": kind, **row} for row in curve)
            results.append(result)

    output = {
        "calibration": "fixed per-stock training quintile cutoffs; threshold ties go to the upper bucket",
        "averaging": "stock-day-bucket means, then equal stocks within date and equal dates across days with five buckets",
        "label_curve_used_executable_proxy_subset": True,
        "all_signal_count": len(scores), "signals_with_training_cutoffs": len(usable),
        "paired_gross_return_signals": len(paired), "stock_type_cutoffs": cutoffs, "groups": results,
    }
    (directory / "grouped_linearity.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(daily_rows).to_csv(directory / "group_quintile_daily.csv", index=False)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
