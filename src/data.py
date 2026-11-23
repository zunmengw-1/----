"""Minute-bar inputs and point-in-time stock-day eligibility for the CNN."""

from __future__ import annotations

import csv
import re
import unicodedata
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

BAR_FIELDS = ("open", "high", "low", "close", "volume", "amount")
SESSION_LENGTHS = (120, 117)
MINUTES = (*range(9 * 60 + 31, 11 * 60 + 31), *range(13 * 60 + 1, 14 * 60 + 58))
MINUTE_POSITIONS = {minute: i for i, minute in enumerate(MINUTES)}

# Pre-2025 listing evidence for this small, fixed pilot universe:
# SH: https://www.sse.com.cn/market/sseindex/diclosure/c/c_20150911_3984896.shtml
# SZ: https://docs.static.szse.cn/www/market/periodical/year/W020230915385920755705.pdf
PILOT_SYMBOLS = (
    "SH600000", "SZ000001", "SH600519", "SZ000333", "SH600036", "SZ000858",
    "SH600900", "SZ002415", "SH600030", "SZ002594", "SH600031", "SZ002714",
    "SH600028", "SZ000651", "SH600660", "SZ002475", "SH600887", "SZ002352",
    "SH600050", "SZ002142", "SH600009", "SZ000625", "SH600196", "SZ000063",
)


@lru_cache(maxsize=4)
def _calendar(provider: Path) -> dict[date, np.ndarray]:
    path = provider / "calendars" / "1min.txt"
    day_indices: dict[date, np.ndarray] = {}
    previous = None
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            stamp = datetime.fromisoformat(line.strip())
            if stamp.second or stamp.microsecond or stamp.tzinfo is not None:
                raise ValueError(f"日历时间须为本地整分钟: {stamp}")
            if previous is not None and stamp <= previous:
                raise ValueError(f"分钟日历必须严格递增: {stamp}")
            previous = stamp
            day = stamp.date()
            if day not in day_indices:
                day_indices[day] = np.full(len(MINUTES), -1, dtype=np.intp)
            position = MINUTE_POSITIONS.get(60 * stamp.hour + stamp.minute)
            if position is not None:
                day_indices[day][position] = index
    if not day_indices:
        raise ValueError(f"分钟日历为空: {path}")
    return day_indices


def read_dates(provider: Path) -> list[date]:
    """Read all date keys from the provider's local minute calendar."""
    return list(_calendar(Path(provider).resolve()))


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", symbol):
        number, exchange = symbol.split(".")
        return exchange + number
    if re.fullmatch(r"(?:SH|SZ|BJ)\d{6}", symbol):
        return symbol
    raise ValueError(f"证券代码无效: {symbol!r}")


def load_bars(provider: Path, symbol: str, dates: list[date]) -> np.ndarray:
    """Return Qlib OHLC, volume and amount as [day, minute, field] float32."""
    provider = Path(provider).resolve()
    symbol = normalize_symbol(symbol)
    directory = provider / "features" / symbol.lower()
    if not directory.is_dir():
        raise ValueError(f"分钟库缺少证券 {symbol}")
    bars = np.full((len(dates), len(MINUTES), len(BAR_FIELDS)), np.nan, dtype=np.float32)
    if not dates:
        return bars
    calendar = _calendar(provider)
    try:
        minute_indices = np.stack([calendar[day] for day in dates])
    except KeyError as exc:
        raise ValueError(f"日期不在分钟库日历中: {exc.args[0]}") from exc
    for column, field in enumerate(BAR_FIELDS):
        path = directory / f"{field}.1min.bin"
        if not path.is_file():
            continue
        # Qlib stores the starting calendar index in the first little-endian float.
        values = np.memmap(path, dtype="<f4", mode="r")
        offset = float(values[0])
        if not offset.is_integer() or offset < 0:
            raise ValueError(f"Qlib 日历偏移无效: {path}")
        positions = minute_indices - int(offset) + 1
        present = (positions >= 1) & (positions < len(values))
        bars[:, :, column][present] = values[positions[present]]
    return bars


def _mainboard(symbol: str) -> bool:
    return symbol.startswith(("SH600", "SH601", "SH603", "SH605", "SZ000", "SZ001", "SZ002", "SZ003"))


def read_eligibility(path: Path, provider: Path, dates: list[date]) -> dict[str, np.ndarray]:
    """Read externally verified eligible pairs, one `date,symbol` pair per CSV row."""
    provider = Path(provider).resolve()
    calendar = _calendar(provider)
    date_positions = {day: i for i, day in enumerate(dates)}
    masks: dict[str, np.ndarray] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"date", "symbol"}.issubset(rows.fieldnames or ()):
            raise ValueError("股票池 CSV 缺少 date 或 symbol 列")
        for row in rows:
            try:
                day = date.fromisoformat((row["date"] or "").strip())
                symbol = normalize_symbol(row["symbol"] or "")
            except ValueError as exc:
                raise ValueError(f"股票池 CSV 第 {rows.line_num} 行无效: {exc}") from exc
            if day not in calendar:
                raise ValueError(f"股票池 CSV 第 {rows.line_num} 行的日期不在分钟库日历中: {day}")
            if not _mainboard(symbol):
                raise ValueError(f"股票池 CSV 含非沪深主板 A 股代码: {symbol}")
            position = date_positions.get(day)
            if position is None:
                continue
            if symbol not in masks:
                if not (provider / "features" / symbol.lower()).is_dir():
                    raise ValueError(f"股票池证券不在分钟库中: {symbol}")
                masks[symbol] = np.zeros(len(dates), dtype=bool)
            if masks[symbol][position]:
                raise ValueError(f"股票池 CSV 的股票交易日重复: {symbol} {day}")
            masks[symbol][position] = True
    if not masks:
        raise ValueError("股票池 CSV 在所选日期没有可用证券")
    return masks


def _normalized_name(name: str | None) -> str:
    return "".join(unicodedata.normalize("NFKC", name or "").upper().split())


def _normal_name(name: str | None) -> bool:
    text = _normalized_name(name)
    return bool(text) and "退" not in text and not text.startswith(("*ST", "ST", "S*ST", "SST"))


def _pilot_names(source: Path, codes: list[str], days: list[date]) -> dict[tuple[date, str], set[str | None]]:
    import pyarrow.parquet as pq

    names: dict[tuple[date, str], set[str | None]] = {}
    if source.is_file():
        panel = pq.read_table(source, columns=["date", "code", "name"], filters=[("code", "in", codes)])
        for row in panel.to_pylist():
            day = row["date"]
            if isinstance(day, datetime):
                day = day.date()
            names.setdefault((day, row["code"]), set()).add(row["name"])
        return names

    for day in days:
        files = list(source.glob(f"{day}*/data.parquet"))
        if len(files) != 1:
            raise ValueError(f"{source} 中 {day} 的原始分钟分区数不是 1")
        panel = pq.read_table(files[0], columns=["code", "name"], filters=[("code", "in", codes)])
        for row in panel.group_by(["code", "name"]).aggregate([]).to_pylist():
            names.setdefault((day, row["code"]), set()).add(row["name"])
    return names


def pilot_eligibility(provider: Path, raw_root: Path, dates: list[date], count: int) -> dict[str, np.ndarray]:
    """Check 2025 pilot symbols against same-day names and observed session trading."""
    if not dates or any(day.year != 2025 for day in dates):
        raise ValueError("内置试跑股票池仅覆盖 2025 年")
    if not 1 <= count <= len(PILOT_SYMBOLS):
        raise ValueError(f"试跑股票数须为 1 至 {len(PILOT_SYMBOLS)}")
    chosen = PILOT_SYMBOLS[:count]
    codes = [f"{symbol[2:]}.{symbol[:2]}" for symbol in chosen]
    names = _pilot_names(Path(raw_root), codes, dates)
    masks: dict[str, np.ndarray] = {}
    for symbol, code in zip(chosen, codes, strict=True):
        mask = np.zeros(len(dates), dtype=bool)
        for i, day in enumerate(dates):
            observed = names.get((day, code), set())
            mask[i] = len(observed) == 1 and _normal_name(next(iter(observed)))
        volume = load_bars(provider, symbol, dates)[:, :, BAR_FIELDS.index("volume")]
        traded = np.any(np.isfinite(volume) & (volume > 0), axis=1)
        masks[symbol] = mask & traded
        if not masks[symbol].any():
            raise ValueError(f"试跑证券 {symbol} 在所选日期没有满足条件的股票日")
    return masks


def history_mainboard_eligibility(
    provider: Path,
    daily_source: Path,
    dates: list[date],
) -> tuple[dict[str, np.ndarray], dict]:
    """Build a broad main-board proxy from historical names and first observed quotes."""
    import pandas as pd
    import pyarrow.parquet as pq

    provider = Path(provider).resolve()
    source = Path(daily_source).resolve()
    calendar = {pd.Timestamp(day): i for i, day in enumerate(read_dates(provider))}
    selected_dates = {pd.Timestamp(day): i for i, day in enumerate(dates)}
    if not dates or dates != sorted(set(dates)) or not selected_dates.keys() <= calendar.keys():
        raise ValueError("股票池交易日须在分钟库日历内严格递增")

    # The history before the selected dates is useful, but subsequent data is sealed.
    stop = datetime.combine(dates[-1] + timedelta(days=1), datetime.min.time())
    panel = pq.read_table(
        source,
        columns=["date", "code", "name", "close", "volume"],
        filters=[("date", "<", stop)],
    ).to_pandas()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["calendar_index"] = panel["date"].map(calendar)
    panel = panel.loc[panel["calendar_index"].notna()].copy()
    if panel.empty:
        raise ValueError(f"日汇总与所选分钟库不存在重合日期: {source}")
    coverage_start = panel["calendar_index"].min()

    code_symbols = {}
    for code in panel["code"].dropna().unique():
        try:
            symbol = normalize_symbol(code)
        except ValueError:
            continue
        if _mainboard(symbol):
            code_symbols[code] = symbol
    panel["symbol"] = panel["code"].map(code_symbols)
    panel["selected_index"] = panel["date"].map(selected_dates)
    in_scope = panel["selected_index"].notna()
    candidate = panel["symbol"].notna()
    available = {directory.name.upper() for directory in (provider / "features").iterdir() if directory.is_dir()}
    in_provider = panel["symbol"].isin(available)
    eligible_panel = panel.loc[in_provider]
    if eligible_panel.duplicated(["symbol", "date"]).any():
        raise ValueError(f"主板日汇总包含重复股票日: {source}")

    has_quote = np.isfinite(eligible_panel["close"]) & eligible_panel["close"].gt(0)
    first_quote = eligible_panel.loc[has_quote].groupby("symbol")["calendar_index"].min()
    observed = eligible_panel.loc[eligible_panel["selected_index"].notna()].copy()
    first_seen = observed["symbol"].map(first_quote)
    established = first_seen.eq(coverage_start)
    beyond_buffer = observed["calendar_index"].ge(first_seen + 5)

    name_values = observed["name"].fillna("")
    names = name_values.map({name: _normalized_name(name) for name in name_values.unique()})
    warning_names = names.str.startswith(("*ST", "ST", "S*ST", "SST")) | names.str.contains("退", regex=False)
    # N/C followed by a CJK character (possibly separated by whitespace in the source).
    # Longer alphabetic names such as TCL科技 aren't initial-listing markers.
    new_names = names.str.match("^[NC][\u3400-\u9fff]")
    traded = np.isfinite(observed["volume"]) & observed["volume"].gt(0)
    quoted = np.isfinite(observed["close"]) & observed["close"].gt(0)

    # The first three dates of this panel are listing days 3–5 for 国货航.
    # Its normalized local name lacks the C marker.
    early_ipo = observed["symbol"].eq("SZ001391") & observed["date"].between(
        pd.Timestamp("2024-12-30"), pd.Timestamp("2025-01-06"),
    )
    checks = (
        ("missing_name", names.ne("")),
        ("risk_or_delisting_name", ~warning_names),
        ("initial_listing_name", ~new_names),
        ("nonpositive_or_missing_close", quoted),
        ("nonpositive_or_missing_volume", traded),
        ("known_year_boundary_ipo", ~early_ipo),
        ("first_observed_five_trading_days", established | beyond_buffer),
    )
    keep = np.ones(len(observed), dtype=bool)
    excluded_stock_days = {
        "non_mainboard_or_invalid_code": int((in_scope & ~candidate).sum()),
        "missing_qlib_symbol": int((in_scope & candidate & ~in_provider).sum()),
    }
    excluded_symbols = {
        "non_mainboard_or_invalid_code": int(panel.loc[in_scope & ~candidate, "code"].nunique()),
        "missing_qlib_symbol": int(panel.loc[in_scope & candidate & ~in_provider, "symbol"].nunique()),
    }
    for reason, condition in checks:
        passed = condition.to_numpy(dtype=bool)
        removed = keep & ~passed
        excluded_stock_days[reason] = int(removed.sum())
        excluded_symbols[reason] = int(observed.loc[removed, "symbol"].nunique())
        keep &= passed

    masks: dict[str, np.ndarray] = {}
    for symbol, rows in observed.loc[keep].groupby("symbol"):
        mask = np.zeros(len(dates), dtype=bool)
        mask[rows["selected_index"].to_numpy(dtype=np.intp)] = True
        masks[symbol] = mask
    if not masks:
        raise ValueError(f"日汇总在所选日期没有近似符合条件的沪深主板股票: {source}")

    active_symbols = set(observed["symbol"])
    audit = {
        "mode": "mainboard_history_proxy",
        "daily_source": str(source),
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "trading_days": len(dates),
        "source_stock_days": int(in_scope.sum()),
        "mainboard_symbols_in_qlib": len(active_symbols),
        "mainboard_stock_days_in_qlib": len(observed),
        "missing_summary_stock_days_for_candidate_symbols": len(active_symbols) * len(dates) - len(observed),
        "already_observed_at_history_start_symbols": int(first_quote.loc[first_quote.index.isin(active_symbols)].eq(coverage_start).sum()),
        "first_observed_after_history_start_symbols": int(first_quote.loc[first_quote.index.isin(active_symbols)].gt(coverage_start).sum()),
        "eligible_symbols": len(masks),
        "eligible_stock_days": int(keep.sum()),
        "eligible_symbols_by_exchange": {exchange: sum(symbol.startswith(exchange) for symbol in masks) for exchange in ("SH", "SZ")},
        "excluded_stock_days": excluded_stock_days,
        "symbols_affected_by_exclusion": excluded_symbols,
        "actual_listing_date_available": False,
        "daily_limit_ratio_available": False,
        "proxy_rules": {
            "present_at_history_start": "assume already listed where the daily name and trading checks pass",
            "first_seen_later": "exclude the first five positions on the trading calendar from the first observed valid quote",
            "security_names": "historical ST/delisting names and N/C flags where available; the source may omit initial-listing flags",
            "known_year_boundary_ipo": "SZ001391 through 2025-01-06, https://www.szse.cn/disclosure/notice/company/t20241227_611339.html",
        },
    }
    return masks, audit
