"""Compare shared and stock-specific heads on the same six daily development signals."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata
from torch.nn.utils.parametrize import cached

from data import load_bars, read_dates
from evaluate_ic_distribution import (
    BLOCK_LENGTH,
    BOOTSTRAP_DRAWS,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    SIGNAL_KEYS,
    SIGNAL_SLOTS,
    STOCK_DAY_KEYS,
    _compare,
    _daily_difference,
    _distribution,
    _rank_ic,
    _summary,
)
from features import encode_windows
from score_economics import _date_range, _score_parameters
from train import encode

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "output/tcn_mainboard_joint_01"
SCORE_COLUMNS = [*SIGNAL_KEYS, "tcn_bp", "label_bp", "execution_return_bp"]
TARGETS = {"research": "label_bp", "execution": "execution_return_bp"}
SCORES = {"shared": "tcn_bp", "adapted": "adapted_bp"}
DEVELOPMENT_DAYS = 49


def load_heads(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        symbols = archive["symbols"]
        theta = archive["theta"]
    if (symbols.ndim != 1 or symbols.dtype.kind not in "US"
            or theta.shape != (len(symbols), 17) or not len(symbols)):
        raise ValueError("heads.npz must hold a string symbol vector and a nonempty [N, 17] theta matrix")
    symbols = symbols.astype(str)
    if len(np.unique(symbols)) != len(symbols) or not np.isfinite(theta).all():
        raise ValueError("stock heads have duplicate symbols or nonfinite parameters")
    return dict(zip(symbols.tolist(), theta.astype(np.float64, copy=False), strict=True))


def _score_adapted(scores: pd.DataFrame, heads: dict, provider: Path, parameters) -> tuple[np.ndarray, dict]:
    model, scaler, shared_weight, shared_bias = parameters
    adapted = np.full(len(scores), np.nan)
    compared_signals = 0
    max_error = 0.0
    device = torch.device("cpu")
    groups = scores.groupby("symbol", sort=False).indices
    with cached():
        for i, (symbol, rows) in enumerate(groups.items(), start=1):
            if symbol in heads:
                block = scores.iloc[rows]
                dates, day_indices = np.unique(block["date"].to_numpy(), return_inverse=True)
                bars = load_bars(provider, symbol, [date.fromisoformat(day) for day in dates])
                coords = np.column_stack((day_indices, block["slot"].to_numpy(dtype=np.intp)))
                z = encode(model, encode_windows(bars, coords), scaler, 512, device)
                reference = block["tcn_bp"].to_numpy(dtype=np.float64)
                shared_repeat = (z @ shared_weight + shared_bias) * scaler.label_scale
                np.testing.assert_allclose(shared_repeat, reference, rtol=1e-5, atol=5e-5, err_msg=symbol)
                max_error = max(max_error, float(np.max(np.abs(shared_repeat - reference))))
                theta = heads[symbol]
                adapted[rows] = (z @ theta[:16] + theta[16]) * scaler.label_scale
                compared_signals += len(rows)
            if i == 1 or i % 500 == 0 or i == len(groups):
                print(f"逐股预测 {i}/{len(groups)}，重算共享以核对 {compared_signals:,} 个信号", flush=True)
    if not compared_signals or not np.isfinite(adapted[scores["symbol"].isin(heads)]).all():
        raise ValueError("no finite stock-specific scores on their development signals")
    n_stocks_with_head = int(scores.loc[np.isfinite(adapted), "symbol"].nunique())
    return adapted, {
        "n_stock_specific_heads": len(heads),
        "n_development_stocks_with_head": n_stocks_with_head,
        "n_development_stocks_without_head": int(scores["symbol"].nunique()) - n_stocks_with_head,
        "n_development_signals_with_head": compared_signals,
        "max_abs_recomputed_shared_forecast_diff_bp": max_error,
    }


def paired_stock_day_ic(scores: pd.DataFrame) -> pd.DataFrame:
    if not scores["slot"].isin(SIGNAL_SLOTS).all() or scores.duplicated(SIGNAL_KEYS).any():
        raise ValueError("development scores do not match the unique six-point grid")
    scores = scores.sort_values(SIGNAL_KEYS)
    groups = scores.groupby(STOCK_DAY_KEYS, sort=False).size().rename("n_signals").reset_index()
    complete = groups["n_signals"] == len(SIGNAL_SLOTS)
    rows = scores.loc[np.repeat(complete.to_numpy(), groups["n_signals"].to_numpy())]

    matrices = {}
    for column in (*TARGETS.values(), *SCORES.values()):
        values = rows[column].to_numpy(dtype=np.float64, na_value=np.nan).reshape(-1, len(SIGNAL_SLOTS))
        matrices[column] = np.where(np.isfinite(values), values, np.nan)
    for target, target_column in TARGETS.items():
        target_ranks = rankdata(matrices[target_column], axis=1, method="average", nan_policy="propagate")
        for model, score_column in SCORES.items():
            metric = f"{model}_{target}_rank_ic"
            groups[metric] = np.nan
            groups.loc[complete, metric] = _rank_ic(matrices[score_column], target_ranks)
    return groups


def analyze(scores: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    groups = paired_stock_day_ic(scores)
    n_dates = scores["date"].nunique()
    paired, independent, daily_frames = {}, {}, {}
    for target in TARGETS:
        shared, adapted = (f"{model}_{target}_rank_ic" for model in SCORES)
        paired[target], daily = _compare(groups, (shared, adapted), n_dates)
        paired[target]["adapted_minus_shared_daily"] = _daily_difference(daily[adapted], daily[shared], n_dates)
        paired[target]["adapted_minus_shared_stock_day"] = _distribution((groups[adapted] - groups[shared]).to_numpy())
        independent[target] = {
            model: _summary(groups, f"{model}_{target}_rank_ic", n_dates)[0] for model in SCORES
        }
        daily_frames[target] = pd.DataFrame({
            f"{target}_paired_stock_days": daily[shared]["count"],
            f"{target}_shared_rank_ic": daily[shared]["mean"],
            f"{target}_adapted_rank_ic": daily[adapted]["mean"],
            f"{target}_difference": daily[adapted]["mean"] - daily[shared]["mean"],
        })
    result = {
        "period": {"start": scores["date"].min(), "end": scores["date"].max(), "days": n_dates},
        "sampling_minutes": 30,
        "minimum_points_per_stock_day": len(SIGNAL_SLOTS),
        "n_stocks": scores["symbol"].nunique(),
        "n_input_signals": len(scores),
        "n_input_stock_days": len(groups),
        "n_complete_stock_days": int((groups["n_signals"] == len(SIGNAL_SLOTS)).sum()),
        "bootstrap": {"block_trading_days": BLOCK_LENGTH, "draws": BOOTSTRAP_DRAWS, "seed": 2025},
        "paired": paired,
        "independent": independent,
    }
    return result, groups, pd.concat(daily_frames.values(), axis=1).sort_index()


def _load_development(run_dir: Path, provider: Path) -> tuple[pd.DataFrame, dict | None]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    period = metrics["split"]["development_test"]
    if (period["start"], period["end"], period["days"]) != (
        DEVELOPMENT_START, DEVELOPMENT_END, DEVELOPMENT_DAYS,
    ):
        raise ValueError("metrics.json must describe the original 49-day development period")
    days = _date_range(read_dates(provider), period)
    scores = pq.read_table(
        run_dir / "economics/signal_scores.parquet", columns=SCORE_COLUMNS,
        filters=[("date", ">=", period["start"]), ("date", "<=", period["end"])],
    ).to_pandas()
    if set(scores["date"]) != {day.isoformat() for day in days}:
        raise ValueError("development scores do not cover the run's expected trading dates")

    reference_path = run_dir / "ic_stability/input_disjoint_30m.json"
    reference = None
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference["period"] != period or len(scores) != reference["scoreable_windows"]:
            raise ValueError("development signal table differs from the original 30-minute replay")
    return scores, reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--provider", type=Path, default=ROOT / "data/qlib_cn_1min")
    parser.add_argument("--heads", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(args.threads)
    out = args.run_dir / "stock_adaptation"
    heads_path = args.heads or out / "heads.npz"
    heads = load_heads(heads_path)
    scores, development = _load_development(args.run_dir, args.provider)
    parameters = _score_parameters(args.run_dir / "model.pt")
    scores["adapted_bp"], score_audit = _score_adapted(scores, heads, args.provider.resolve(), parameters)
    result, groups, daily = analyze(scores)
    shared = result["independent"]["research"]["shared"]
    reference = None if development is None else development["disjoint_input_mean_rank_ic"]
    shared_ic = shared["daily_equal_stock_weight"]["mean"]
    if development is not None and (
        shared["stock_day"]["n"] != development["valid_ic_stock_days"]
        or shared_ic is None or abs(shared_ic - reference) > 5e-5
    ):
        raise ValueError("shared-head IC differs from the existing 30-minute baseline")
    result.update(score_audit)
    result["reference_shared_research_rank_ic"] = reference
    result["last_24_market_days_read"] = False
    out.mkdir(parents=True, exist_ok=True)
    (out / "ic_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    groups.to_parquet(out / "stock_day_ic.parquet", index=False)
    daily.to_csv(out / "daily_ic.csv", index_label="date")


if __name__ == "__main__":
    main()
