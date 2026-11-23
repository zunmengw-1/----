"""Compare fixed-share, buy-first overlays on a settled A-share base holding."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from intraday_tcn.data import (
    BAR_FIELDS,
    MINUTES,
    SESSION_LENGTHS,
    _calendar,
    load_bars,
    read_dates,
)
from intraday_tcn.paths import DEFAULT_DAILY_SOURCE, DEFAULT_PROVIDER, DEFAULT_RUN

RESULTS = DEFAULT_RUN / "economics"
START = date(2025, 9, 12)
END = date(2025, 11, 27)
METHODS = ("tcn", "reversal", "momentum", "static")
SCORES = [f"{method}_bp" for method in METHODS[:-1]]
THRESHOLDS = [f"{method}_q80" for method in METHODS[:-1]]
LOOKBACK, HORIZON, STRIDE = 30, 5, 30
MORNING, AFTERNOON = SESSION_LENGTHS
SLOTS = (*range(LOOKBACK - 1, MORNING - HORIZON, STRIDE),
         *range(MORNING + LOOKBACK - 1, MORNING + AFTERNOON - HORIZON, STRIDE))
DAY_END = len(MINUTES)
BASE_SHARES, ORDER_SHARES = 10_000, 1_000
COMMISSION, MIN_COMMISSION, STAMP_DUTY, TRANSFER_FEE = 0.00025, 5.0, 0.0005, 0.00001
SLIPPAGE = (0, 2, 5)
BUY_FAILURES = ("missing_entry_price", "no_entry_volume", "insufficient_entry_volume",
                "unknown_price_limit", "suspected_limit_up")
COUNTS = ("trigger_signals", "entries", "closed", "delayed_exits", "unclosed",
          "skipped_overlap", "skipped_t1", *BUY_FAILURES)
COUNT_INDEX = {name: i for i, name in enumerate(COUNTS)}


def load_signals(scores_path: Path, profile_path: Path, days: list[date]):
    scores = pq.read_table(scores_path, columns=["symbol", "date", "slot", *SCORES]).to_pandas()
    profiles = pd.read_csv(profile_path, usecols=["symbol", "calibration_n", *THRESHOLDS])
    dates = {day.isoformat(): index for index, day in enumerate(days)}
    scores["day"] = scores["date"].astype(str).map(dates)
    if scores["day"].isna().any():
        raise ValueError("分数包含开发测试期以外的日期")
    if not scores["slot"].isin(SLOTS).all():
        raise ValueError(f"分数包含约定的 30 分钟网格以外的时点: {SLOTS}")
    if scores.duplicated(["symbol", "day", "slot"]).any():
        raise ValueError("分数包含重复股票、日期或分钟")
    if profiles["symbol"].duplicated().any():
        raise ValueError("校准分位数表包含重复股票")

    joined = scores.merge(profiles, on="symbol", how="left", validate="many_to_one")
    sufficient = joined["calibration_n"].gt(0).to_numpy() & np.isfinite(joined[THRESHOLDS]).all(axis=1).to_numpy()
    complete = np.isfinite(joined[SCORES]).all(axis=1).to_numpy()
    common = sufficient & complete
    scores_by_day = np.bincount(joined.loc[common, "day"].to_numpy(dtype=np.intp), minlength=len(days))
    table = joined.loc[common, ["symbol", "day", "slot"]].copy()
    triggers = joined.loc[common, SCORES].to_numpy() >= joined.loc[common, THRESHOLDS].to_numpy()
    table[list(METHODS[:-1])] = triggers
    table = table.loc[triggers.any(axis=1)].sort_values(["symbol", "day", "slot"])
    if not common.any():
        raise ValueError("数据没有三种分数和校准门槛都齐备的候选分钟")
    symbol_list = joined.loc[common, "symbol"].unique()
    audit = {
        "input_rows": len(scores),
        "common_signal_rows": int(common.sum()),
        "common_signal_symbols": len(symbol_list),
        "rows_without_calibrated_q80_for_all_methods": int((~sufficient).sum()),
        "rows_with_calibration_but_missing_score": int((sufficient & ~complete).sum()),
    }
    return table, scores_by_day, audit


def round_cents(prices: np.ndarray) -> np.ndarray:
    return np.floor(prices * 100.0 + 0.5 + 1e-9) / 100.0


def load_price_references(source: Path, days: list[date]) -> dict[str, np.ndarray]:
    end_exclusive = datetime.combine(days[-1] + timedelta(days=1), datetime.min.time())
    daily = pq.read_table(source, columns=["date", "code", "close", "factor"],
                          filters=[("date", "<", end_exclusive)]).to_pandas()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    mainboard_hint = daily["code"].str.endswith((".SH", ".SZ"), na=False)
    daily = daily.loc[mainboard_hint].sort_values(["code", "date"])
    if daily.duplicated(["code", "date"]).any():
        raise ValueError("日行情表包含重复股票日")
    has_reference = np.isfinite(daily["close"]) & daily["close"].gt(0)
    has_reference &= np.isfinite(daily["factor"]) & daily["factor"].gt(0)
    previous = daily[["close", "factor"]].where(has_reference, np.nan)
    previous = previous.groupby(daily["code"], sort=False).ffill()
    previous = previous.groupby(daily["code"], sort=False).shift()
    with np.errstate(divide="ignore", invalid="ignore"):
        reference = round_cents((previous["close"] * previous["factor"] / daily["factor"]).to_numpy())
    reference[~np.isfinite(reference) | (reference <= 0)] = np.nan
    daily["limit_up"] = round_cents(reference * 1.10)
    daily["limit_down"] = round_cents(reference * 0.90)
    stamps = pd.DatetimeIndex(days)
    daily = daily.loc[daily["date"].isin(stamps)]
    result = {}
    for code, stock in daily.groupby("code", sort=False):
        symbol = code[-2:] + code[:6]
        result[symbol] = stock.set_index("date").reindex(stamps)[["limit_up", "limit_down", "close"]].to_numpy()
    return result


def raw_market(provider: Path, symbol: str, days: list[date]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bars = load_bars(provider, symbol, days)
    path = provider / "features" / symbol.lower() / "factor.1min.bin"
    factors = np.memmap(path, dtype="<f4", mode="r")
    offset = float(factors[0])
    if not offset.is_integer() or offset < 0:
        raise ValueError(f"Qlib 日历偏移无效: {path}")
    original = np.stack([_calendar(provider)[day] for day in days]) - int(offset) + 1
    present = (original >= 1) & (original < len(factors))
    factor = np.full(present.shape, np.nan, dtype=np.float64)
    factor[present] = factors[original[present]]
    factor[~np.isfinite(factor) | (factor <= 0)] = np.nan
    entry = round_cents(bars[..., BAR_FIELDS.index("open")] / factor)
    close = round_cents(bars[..., BAR_FIELDS.index("close")] / factor)
    volume = np.rint(bars[..., BAR_FIELDS.index("volume")] * factor)
    return entry, close, volume


def execution_grid(entry: np.ndarray, close: np.ndarray, volume: np.ndarray, references: np.ndarray):
    upper, lower, _ = references.T
    enter_failure = np.zeros(entry.shape, dtype=np.int8)
    failures = (
        ~np.isfinite(entry) | (entry <= 0),
        ~np.isfinite(volume) | (volume <= 0),
        volume < ORDER_SHARES,
        ~np.isfinite(upper[:, None]),
        entry >= upper[:, None] - 0.005,
    )
    for reason, mask in enumerate(failures, start=1):
        enter_failure[(enter_failure == 0) & mask] = reason
    can_sell = (np.isfinite(close) & (close > 0) & (volume >= ORDER_SHARES)
                & np.isfinite(lower[:, None]) & (close > lower[:, None] + 0.005))
    latest = np.where(can_sell, np.arange(DAY_END), DAY_END)
    next_sale = np.minimum.accumulate(latest[:, ::-1], axis=1)[:, ::-1]
    return enter_failure, next_sale


def end_mark(close: np.ndarray, entry_price: float, entry_slot: int, summary_close: float):
    if np.isfinite(summary_close) and summary_close > 0:
        return round(summary_close, 2), "daily_close"
    later = close[entry_slot:]
    traded_marks = later[np.isfinite(later) & (later > 0)]
    if len(traded_marks):
        return traded_marks[-1], "last_minute_close"
    return entry_price, "entry_price_fallback"


def time_label(slot: int) -> str:
    hour, minute = divmod(MINUTES[slot], 60)
    return f"{hour:02d}:{minute:02d}"


def simulate(
    provider: Path,
    days: list[date],
    table: pd.DataFrame,
    price_references: dict[str, np.ndarray],
    methods: tuple[str, ...] = METHODS,
    *,
    entry_at_signal_close: bool = False,
):
    """Replay signals at the next open or, optionally, the signal close."""
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("策略列表必须非空，且名称不能重复")
    active_methods = [(index, name) for index, name in enumerate(methods) if name != "static"]
    signal_columns = [name for _, name in active_methods]
    entry_lag = 0 if entry_at_signal_close else 1
    counters = np.zeros((len(days), len(methods), len(COUNTS)), dtype=np.int64)
    trades, open_positions = [], []
    missing_ref = np.full((len(days), 3), np.nan)
    for symbol, rows in table.groupby("symbol", sort=False):
        values = rows[["day", "slot", *signal_columns]].to_numpy(dtype=np.int16)
        day_values = values[:, 0]
        stock_days, starts = np.unique(day_values, return_index=True)
        stops = np.r_[starts[1:], len(values)]
        open_price, close, volume = raw_market(provider, symbol, [days[index] for index in stock_days])
        entry = close if entry_at_signal_close else open_price
        refs = price_references.get(symbol, missing_ref)[stock_days]
        failed_entry, next_sale = execution_grid(entry, close, volume, refs)
        for day_row, (day, start, stop) in enumerate(zip(stock_days, starts, stops, strict=True)):
            day_rows = values[start:stop]
            for column, (method, method_name) in enumerate(active_methods, start=2):
                counts = counters[day, method]
                triggered = day_rows[day_rows[:, column] != 0, 1]
                counts[COUNT_INDEX["trigger_signals"]] += len(triggered)
                sellable = BASE_SHARES
                open_until = -1
                for slot in triggered:
                    if slot < open_until:
                        counts[COUNT_INDEX["skipped_overlap"]] += 1
                        continue
                    if sellable < ORDER_SHARES:
                        counts[COUNT_INDEX["skipped_t1"]] += 1
                        continue
                    buy_slot = slot + entry_lag
                    failed = failed_entry[day_row, buy_slot]
                    if failed:
                        counts[COUNT_INDEX[BUY_FAILURES[failed - 1]]] += 1
                        continue
                    counts[COUNT_INDEX["entries"]] += 1
                    buy_price = entry[day_row, buy_slot]
                    due = slot + HORIZON
                    exit_slot = next_sale[day_row, due]
                    open_until = exit_slot
                    closed = exit_slot < DAY_END
                    if closed:
                        end_price = close[day_row, exit_slot]
                        sellable -= ORDER_SHARES
                        counts[COUNT_INDEX["closed"]] += 1
                        counts[COUNT_INDEX["delayed_exits"]] += int(exit_slot > due)
                    else:
                        counts[COUNT_INDEX["unclosed"]] += 1
                        end_price, source = end_mark(close[day_row], buy_price, buy_slot, refs[day_row, 2])
                        open_positions.append((days[day].isoformat(), method_name, symbol, time_label(slot),
                                               time_label(buy_slot), time_label(due), buy_price, end_price, source))
                    trades.append((day, method, buy_price, end_price, closed))
    return counters, trades, open_positions


def account(days, scores_by_day, counters, trades, methods: tuple[str, ...] = METHODS):
    """Apply transaction costs using the same method order as the replay."""
    if counters.shape != (len(days), len(methods), len(COUNTS)):
        raise ValueError("成本统计的策略维度和回放计数不一致")
    fills = np.array(trades, dtype=np.float64).reshape(-1, 5)
    group = (fills[:, 0] * len(methods) + fills[:, 1]).astype(np.intp)
    raw_buy = fills[:, 2] * ORDER_SHARES
    raw_value = fills[:, 3] * ORDER_SHARES
    closed = fills[:, 4].astype(bool)
    size = len(days) * len(methods)
    frame = pd.DataFrame(counters.reshape(size, -1), columns=COUNTS)
    frame["date"] = np.repeat([day.isoformat() for day in days], len(methods))
    frame["strategy"] = np.tile(methods, len(days))
    frame["common_signal_rows"] = np.repeat(scores_by_day, len(methods))
    frames = []
    for scenario in SLIPPAGE:
        rate = scenario / 10_000
        buy_notional = raw_buy * (1 + rate)
        sale_notional = np.where(closed, raw_value * (1 - rate), 0.0)
        commission = np.maximum(MIN_COMMISSION, buy_notional * COMMISSION)
        commission += np.where(closed, np.maximum(MIN_COMMISSION, sale_notional * COMMISSION), 0.0)
        transfer = (buy_notional + sale_notional) * TRANSFER_FEE
        tax = sale_notional * STAMP_DUTY
        slip = (buy_notional - raw_buy) + np.where(closed, raw_value - sale_notional, 0.0)
        fees = commission + transfer + tax
        gross = raw_value - raw_buy
        net = gross - fees - slip
        result = frame.copy()
        result["slippage_bp_per_side"] = scenario
        amounts = {
            "buy_notional_yuan": buy_notional,
            "gross_before_costs_yuan": gross,
            "commission_yuan": commission,
            "transfer_fee_yuan": transfer,
            "stamp_duty_yuan": tax,
            "slippage_yuan": slip,
            "total_cost_yuan": fees + slip,
            "net_pnl_yuan": net,
            "closed_net_pnl_yuan": np.where(closed, net, 0.0),
            "unclosed_marked_pnl_yuan": np.where(closed, 0.0, net),
        }
        for name, amount in amounts.items():
            result[name] = np.bincount(group, weights=amount, minlength=size)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def summary(daily: pd.DataFrame) -> pd.DataFrame:
    measures = [*COUNTS, "common_signal_rows", *[column for column in daily if column.endswith("_yuan")]]
    grouped = daily.groupby(["strategy", "slippage_bp_per_side"], sort=False)
    report = grouped[measures].sum()
    report["trading_days"] = grouped["date"].size()
    report["positive_net_days"] = grouped["net_pnl_yuan"].agg(lambda pnl: int(pnl.gt(0).sum()))
    report["active_days"] = grouped["entries"].agg(lambda entries: int(entries.gt(0).sum()))
    report["entry_fill_rate"] = report["entries"] / report["trigger_signals"].replace(0, np.nan)
    report["exit_fill_rate"] = report["closed"] / report["entries"].replace(0, np.nan)
    report["net_bp_on_buy_notional"] = 10_000 * report["net_pnl_yuan"] / report["buy_notional_yuan"].replace(0, np.nan)
    report.loc[report.index.get_level_values("strategy") == "static", "net_bp_on_buy_notional"] = 0.0
    return report.reset_index().sort_values(["slippage_bp_per_side", "strategy"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=RESULTS / "signal_scores.parquet")
    parser.add_argument("--profiles", type=Path, default=RESULTS / "stock_profiles.csv")
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--daily-source", type=Path, default=DEFAULT_DAILY_SOURCE)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    provider = args.provider.resolve()
    days = [day for day in read_dates(provider) if START <= day <= END]
    if len(days) != 49:
        raise ValueError("开发测试期分钟日历必须包含 49 个交易日")
    signals, scores_by_day, audit = load_signals(args.scores, args.profiles, days)
    references = load_price_references(args.daily_source, days)
    counts, trades, open_positions = simulate(provider, days, signals, references)
    daily = account(days, scores_by_day, counts, trades)
    results = summary(daily)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "cost_comparison.csv", index=False)
    daily.to_csv(args.out / "cost_daily.csv", index=False)
    pd.DataFrame(open_positions, columns=["date", "strategy", "symbol", "signal_bar_end", "entry_bar_end",
                                          "planned_exit_bar_end", "buy_raw_yuan", "eod_mark_yuan", "mark_source"]).to_csv(
        args.out / "cost_unclosed.csv", index=False,
    )
    audit.update({
        "scores_path": str(args.scores.resolve()),
        "profiles_path": str(args.profiles.resolve()),
        "start": START.isoformat(),
        "end": END.isoformat(),
        "entry": "t+1 raw open, price and positive full-order bar volume required",
        "exit": "t+5 raw close, retry every subsequent continuous auction bar through 14:57 if blocked",
        "limit_proxy": "10% of prior raw daily close adjusted by previous/current day factor; round to cents, block at suspected limit",
        "eligible_entry_without_known_limit": False,
        "execution_proxy": "Minute OHLC and total minute volume; depth, priority, queue and open-time volume unknown",
        "capital": "Independent stock-days with 10000 initially settled shares and sufficient cash; equal 1000-share orders, no shared capital constraint",
        "base_shares": BASE_SHARES,
        "order_shares": ORDER_SHARES,
        "commission_rate_per_side": COMMISSION,
        "commission_minimum_yuan_per_side": MIN_COMMISSION,
        "stamp_duty_sell_rate": STAMP_DUTY,
        "transfer_fee_rate_per_side": TRANSFER_FEE,
        "slippage_bp_per_side": SLIPPAGE,
        "unclosed": "Value at same-day daily close, then last valid close at or after entry, then raw entry price; no hypothetical sale or sale-side costs and no carry in the independently initialized next day's comparison",
        "return_denominator": "Total executed buy notional including slippage, including unclosed entries; static overlay P&L and comparison basis points are zero by definition",
        "entry_fill_rate_denominator": "Threshold-crossing signals including those ignored due to open position or T+1 capacity",
        "exit_fill_rate_denominator": "Executed buys",
        "final_24_days_read": False,
    })
    (args.out / "cost_assumptions.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    display = ["strategy", "slippage_bp_per_side", "entries", "closed", "unclosed", "net_bp_on_buy_notional", "net_pnl_yuan"]
    print(results[display].to_string(index=False))


if __name__ == "__main__":
    main()
