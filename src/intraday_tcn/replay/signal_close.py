"""Replace the entry return in the frozen stock-head calibration with signal-close returns."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from intraday_tcn.data import BAR_FIELDS, load_bars, read_dates
from intraday_tcn.paths import DEFAULT_PROVIDER, DEFAULT_RUN
from intraday_tcn.replay.cost_gate import RETURN_COLUMN, fit_return_buckets
from intraday_tcn.replay.execution import HORIZON, SLOTS

CALIBRATION_START = date(2025, 8, 1)
CALIBRATION_END = date(2025, 9, 11)
CALIBRATION_DAYS = 30
KEY_COLUMNS = ["symbol", "date", "slot"]
SCORES = ["shared_bp", "adapted_bp"]
COLUMNS = [*KEY_COLUMNS, *SCORES, RETURN_COLUMN]
CLOSE_COLUMN = BAR_FIELDS.index("close")
VOLUME_COLUMN = BAR_FIELDS.index("volume")


def _positive(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0)


def calibrate_labels(frame: pd.DataFrame, provider: Path) -> tuple[np.ndarray, dict]:
    old_defined = np.isfinite(frame[RETURN_COLUMN].to_numpy(dtype=np.float64))
    updated = np.full(len(frame), np.nan, dtype=np.float64)
    no_positive_signal_volume = 0
    failed_price_or_exit = 0
    for symbol, rows in frame.groupby("symbol", sort=False).indices.items():
        rows = rows[old_defined[rows]]
        if not len(rows):
            continue
        block = frame.iloc[rows]
        dates, day = np.unique(block["date"].to_numpy(), return_inverse=True)
        slot = block["slot"].to_numpy(dtype=np.intp)
        exit_slot = slot + HORIZON
        bars = load_bars(provider, symbol, [date.fromisoformat(value) for value in dates])
        entry_price = bars[day, slot, CLOSE_COLUMN].astype(np.float64)
        exit_price = bars[day, exit_slot, CLOSE_COLUMN].astype(np.float64)
        signal_volume_ok = _positive(bars[day, slot, VOLUME_COLUMN])
        other_ok = (
            _positive(entry_price) & _positive(exit_price) & _positive(bars[day, exit_slot, VOLUME_COLUMN])
        )
        valid = signal_volume_ok & other_ok
        updated[rows[valid]] = 10_000 * (np.log(exit_price[valid]) - np.log(entry_price[valid]))
        no_positive_signal_volume += int((~signal_volume_ok).sum())
        failed_price_or_exit += int((signal_volume_ok & ~other_ok).sum())

    kept = int(np.isfinite(updated).sum())
    assert int(old_defined.sum()) == kept + no_positive_signal_volume + failed_price_or_exit
    return updated, {
        "calibration_rows": len(frame),
        "previous_valid_labels": int(old_defined.sum()),
        "signal_close_valid_labels": kept,
        "lost_to_nonpositive_or_missing_signal_volume": no_positive_signal_volume,
        "lost_to_entry_price_or_exit_conditions_after_positive_signal_volume": failed_price_or_exit,
    }


def _bucket_summary(frame: pd.DataFrame) -> dict:
    _, _, audit = fit_return_buckets(frame, SCORES)
    return {
        name: {
            "highest_score_bucket": stats["buckets"][-1],
            "highest_score_bucket_cutoff_bp": stats["cutoffs_bp"][-1] if stats["cutoffs_bp"] else None,
            "highest_expected_return_bucket": max(
                (bucket for bucket in stats["buckets"] if bucket["expected_arithmetic_return_bp"] is not None),
                key=lambda bucket: bucket["expected_arithmetic_return_bp"],
            ),
        }
        for name, stats in audit.items()
    }


def _validate_calibration(frame: pd.DataFrame, provider: Path) -> None:
    calendar = [day for day in read_dates(provider) if CALIBRATION_START <= day <= CALIBRATION_END]
    if (len(calendar) != CALIBRATION_DAYS or calendar[0] != CALIBRATION_START or calendar[-1] != CALIBRATION_END
            or set(frame["date"]) != {day.isoformat() for day in calendar}
            or frame["symbol"].isna().any() or not frame["slot"].isin(SLOTS).all()
            or frame.duplicated(KEY_COLUMNS).any()):
        raise ValueError("校准表必须匹配 2025-08-01 至 2025-09-11 的30个交易日及原信号时点")


def calibrate_signal_close(calibration: pd.DataFrame, provider: Path) -> pd.DataFrame:
    """Preserve score rows and replace the label with the same-close return when defined."""
    _validate_calibration(calibration, provider)
    values, audit = calibrate_labels(calibration, provider)
    updated = calibration.assign(**{RETURN_COLUMN: values})
    updated.attrs["label_audit"] = audit
    return updated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    args = parser.parse_args(argv)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    provider = args.provider.resolve()
    adaptation = args.run_dir / "stock_adaptation"
    source = pq.read_table(adaptation / "calibration.parquet", columns=COLUMNS, use_threads=False)
    frame = source.to_pandas()
    old_buckets = _bucket_summary(frame)
    updated = calibrate_signal_close(frame, provider)
    new_buckets = _bucket_summary(updated)
    values = updated[RETURN_COLUMN].to_numpy()
    returns = pa.array(values, mask=~np.isfinite(values), type=pa.float64())
    result = source.set_column(source.schema.get_field_index(RETURN_COLUMN), RETURN_COLUMN, returns)
    output = adaptation / "cost_gate_signal_close_1m" / "calibration.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(result, temporary, compression="zstd")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        "calibration_start": CALIBRATION_START.isoformat(),
        "calibration_end": CALIBRATION_END.isoformat(),
        "calibration_days": CALIBRATION_DAYS,
        **updated.attrs["label_audit"],
        "next_open_label": old_buckets,
        "signal_close_label": new_buckets,
    }, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
