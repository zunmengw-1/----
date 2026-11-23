"""Replay frozen cost-gated signals at every available continuous-auction minute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.parametrize import cached

from intraday_tcn.data import load_bars, read_dates
from intraday_tcn.evaluation.scores import (
    _development_days,
    _score_parameters,
    _signals,
)
from intraday_tcn.features import find_signal_coords
from intraday_tcn.paths import DEFAULT_DAILY_SOURCE, DEFAULT_PROVIDER
from intraday_tcn.replay.cost_gate import projected_margin_bp
from intraday_tcn.replay.execution import (
    END,
    METHODS,
    RESULTS,
    SCORES,
    SLIPPAGE,
    START,
    THRESHOLDS,
    account,
    load_price_references,
    raw_market,
    simulate,
    summary,
    time_label,
)


def load_calibration(path):
    frozen = json.loads(path.read_text())
    if frozen["calibration_dates"][1] >= START.isoformat():
        raise ValueError("成本校准数据须早于开发测试期")
    table = frozen["calibration_tables"]
    cutoffs = [np.asarray(table[name]["cutoffs_bp"]) for name in SCORES]
    levels = [np.asarray([x["expected_arithmetic_return_bp"] for x in table[name]["buckets"]], dtype=float)
              for name in SCORES]
    for edges, values in zip(cutoffs, levels, strict=True):
        if len(edges) + 1 != len(values) or not np.isfinite(values).any() or not np.all(np.diff(edges) > 0):
            raise ValueError("冻结校准表的分桶无效")
    upper = np.asarray([np.nanmax(values) for values in levels])
    return cutoffs, levels, upper


def mapped_returns(scores, cutoffs, levels):
    expected = np.full(scores.shape, np.nan)
    for column, (edges, values) in enumerate(zip(cutoffs, levels, strict=True)):
        mask = np.isfinite(scores[:, column])
        expected[mask, column] = values[np.searchsorted(edges, scores[mask, column], side="right")]
    return expected


def eligible_stock_days(economics, days, symbols=None):
    if symbols is None:
        profiles = pd.read_csv(economics / "stock_profiles.csv", usecols=["symbol", *THRESHOLDS])
        symbols = set(profiles.loc[np.isfinite(profiles[THRESHOLDS]).all(axis=1), "symbol"])
    dates = {day: i for i, day in enumerate(days)}
    by_stock = _development_days(economics.parent / "shared_daily.csv", days)
    return {symbol: np.asarray([dates[day] for day in stock_days], dtype=np.intp)
            for symbol, stock_days in by_stock.items() if symbol in symbols}


def _scoring_upper_bounds(prices, upper):
    expected_max = np.broadcast_to(upper, (len(prices), len(upper)))
    upper_margins = {s: projected_margin_bp(prices, expected_max, s) for s in SLIPPAGE}
    # Close to the cutoff, score conservatively rather than pruning on roundoff.
    maybe_positive = np.logical_or.reduce([(margin >= -1e-8).any(axis=1) for margin in upper_margins.values()])
    return upper_margins, maybe_positive


def collect_candidates(provider, days, universe, score_forecasts, calibration, example_symbol, methods=METHODS):
    cutoffs, levels, upper = calibration
    candidates = {s: [] for s in SLIPPAGE}
    max_margin_upper = {s: np.full(len(upper), -np.inf) for s in SLIPPAGE}
    valid_per_day = np.zeros(len(days), dtype=np.int64)
    total_signals, score_count, cutoff_pruned = 0, 0, 0
    example_rows = []
    example_methods = ("tcn",) if methods == METHODS else methods[:-1]

    with cached():
        for index, (symbol, active) in enumerate(universe.items(), 1):
            dates = [days[i] for i in active]
            bars = load_bars(provider, symbol, dates)
            coords = find_signal_coords(bars)
            valid_per_day[active] += np.bincount(coords[:, 0], minlength=len(dates))
            total_signals += len(coords)
            if not len(coords):
                continue

            _, close, _ = raw_market(provider, symbol, dates)
            reference = close[coords[:, 0], coords[:, 1]]
            price_ok = np.isfinite(reference) & (reference > 0)
            upper_margins, possible = _scoring_upper_bounds(reference, upper)
            for slip, margin in upper_margins.items():
                maxima = np.where(np.isfinite(margin), margin, -np.inf).max(axis=0)
                max_margin_upper[slip] = np.maximum(max_margin_upper[slip], maxima)
            force_example = (coords[:, 0] == 0) if symbol == example_symbol else np.zeros(len(coords), dtype=bool)
            score_mask = (price_ok & possible) | force_example
            cutoff_pruned += int((~score_mask).sum())
            if score_mask.any():
                selected = coords[score_mask]
                raw_scores = score_forecasts(symbol, bars, selected)
                expected = mapped_returns(raw_scores, cutoffs, levels)
                prices = reference[score_mask]
                valid_price = price_ok[score_mask]
                margin_by_slippage = {}
                for slippage in SLIPPAGE:
                    margin = projected_margin_bp(prices, expected, slippage)
                    margin_by_slippage[slippage] = margin
                    flags = valid_price[:, None] & np.isfinite(margin) & (margin > 0)
                    assert not flags[~possible[score_mask]].any()
                    keep = flags.any(axis=1)
                    if keep.any():
                        records = pd.DataFrame({
                            "symbol": symbol, "day": active[selected[keep, 0]], "slot": selected[keep, 1],
                        })
                        records[list(methods[:-1])] = flags[keep]
                        candidates[slippage].append(records)
                score_count += len(selected)

                if force_example.any():
                    for i in np.flatnonzero(force_example[score_mask]):
                        d, slot = selected[i]
                        row = {"symbol": symbol, "date": dates[d].isoformat(), "signal_bar_end": time_label(slot)}
                        for column, method in enumerate(example_methods):
                            row[f"raw_{method}_forecast_bp"] = raw_scores[i, column]
                            prefix = "" if methods == METHODS else f"{method}_"
                            row[f"{prefix}expected_execution_return_bp"] = expected[i, column]
                            row.update({f"{prefix}expected_net_bp_slip_{s}": margin_by_slippage[s][i, column] for s in SLIPPAGE})
                        example_rows.append(row)
            if index == 1 or index % 250 == 0 or index == len(universe):
                print(f"分钟回放准备 {index}/{len(universe)} 股，合法时点 {total_signals:,}，已编码 {score_count:,}", flush=True)

    assert cutoff_pruned + score_count == total_signals
    tables = {s: (pd.concat(candidates[s], ignore_index=True).sort_values(["symbol", "day", "slot"])
                  if candidates[s] else pd.DataFrame(columns=["symbol", "day", "slot", *methods[:-1]]))
              for s in SLIPPAGE}
    audit = {
        "scored_minute_windows": score_count, "skipped_without_order_possibility": cutoff_pruned,
        "maximum_expected_margin_upper_bound_bp": {
            str(s): {method: float(value) if np.isfinite(value) else None
                     for method, value in zip(methods[:-1], margins, strict=True)}
            for s, margins in max_margin_upper.items()
        },
    }
    return tables, valid_per_day, example_rows, audit


def save_results(
    provider, days, daily_source, tables, valid_per_day, example, output, methods=METHODS, *,
    entry_at_signal_close=False,
):
    references = load_price_references(daily_source, days)
    measures, open_rows = [], []
    for slippage, candidates in tables.items():
        counters, trades, open_positions = simulate(
            provider, days, candidates, references, methods=methods, entry_at_signal_close=entry_at_signal_close,
        )
        daily = account(days, valid_per_day, counters, trades, methods=methods)
        measures.append(daily.loc[daily["slippage_bp_per_side"].eq(slippage)])
        open_rows.extend((slippage, *position) for position in open_positions)
    daily = pd.concat(measures, ignore_index=True)
    report = summary(daily)
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / "comparison.csv", index=False)
    daily.to_csv(output / "daily.csv", index=False)
    pd.DataFrame(open_rows, columns=["slippage_bp", "date", "strategy", "symbol", "signal_bar_end",
                                     "entry_bar_end", "planned_exit_bar_end", "buy_raw_yuan", "eod_mark_yuan",
                                     "mark_source"]).to_csv(output / "unclosed.csv", index=False)
    pd.DataFrame(example).to_csv(output / "minute_forecasts_example.csv", index=False)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--economics", type=Path, default=RESULTS)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--daily-source", type=Path, default=DEFAULT_DAILY_SOURCE)
    parser.add_argument("--example-symbol", default="SH600000")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    provider = args.provider.resolve()
    economics = args.economics.resolve()
    days = [day for day in read_dates(provider) if START <= day <= END]
    if len(days) != 49:
        raise ValueError("开发期应为原实验的 49 个交易日")
    calibration = load_calibration(economics / "cost_gate/cost_gate_assumptions.json")
    parameters = _score_parameters(economics.parent / "model.pt")
    universe = eligible_stock_days(economics, days)

    def score_forecasts(_symbol, bars, coords):
        return np.column_stack(_signals(bars, coords, parameters))

    tables, valid_per_day, example, audit = collect_candidates(
        provider, days, universe, score_forecasts, calibration, args.example_symbol.upper(),
    )

    output = economics / "cost_gate_1m"
    report = save_results(provider, days, args.daily_source, tables, valid_per_day, example, output)
    audit.update({
        "development_dates": [START.isoformat(), END.isoformat()], "development_days": len(days),
        "symbols": len(universe), "eligible_stock_days": sum(map(len, universe.values())),
        "eligible_minute_windows": int(valid_per_day.sum()), "sample_every_minutes": 1, "window_minutes": 30,
        "label_horizon_minutes": 5,
        "calibration": "Frozen 30-minute-grid training-only calibration. This isolates the change in signal cadence.",
        "no_future_quote_filter": True, "initial_budget_and_fee_assumptions": "../cost_assumptions.json",
        "bound_rule": "A forecast is skipped in replay only if no maximum calibrated return can cover the modeled round-trip cost, or the signal-time price is invalid. The example day is always fully scored.",
        "example_forecast_rows": len(example), "final_24_days_read": False,
    })
    (output / "replay_assumptions.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print(report[["strategy", "slippage_bp_per_side", "trigger_signals", "entries", "closed", "unclosed", "skipped_overlap",
                  "net_bp_on_buy_notional", "net_pnl_yuan"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
