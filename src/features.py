"""Build fixed-window minute samples from one stock's aligned OHLCVA."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES: tuple[str, ...] = (
    "relative_open",
    "relative_high",
    "relative_low",
    "relative_close",
    "relative_volume",
    "is_afternoon",
    "session_progress",
    "realized_volatility",
    "realized_skewness",
    "return_5m",
    "return_10m",
    "return_20m",
    "high_low_range",
    "wick_asymmetry",
    "price_volume_corr",
    "vwap_deviation",
    "volume_slope",
    "return_5m_valid",
    "return_10m_valid",
    "return_20m_valid",
)

MORNING_BARS = 120
AFTERNOON_BARS = 117
DAY_BARS = MORNING_BARS + AFTERNOON_BARS
FEATURE_BATCH = 2_048
RETURN_HORIZONS = (5, 10, 20)


def _divide(numerator: np.ndarray, denominator: np.ndarray, valid=True) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=(denominator > 0) & valid,
    )


def _fill_features(out: np.ndarray, windows: np.ndarray, slots: np.ndarray) -> None:
    width = windows.shape[1]
    positions = np.arange(width)
    opens, highs, lows, closes, volume, amount = windows.transpose(2, 0, 1)

    log_prices = np.log(windows[..., :4])
    out[..., :4] = log_prices - log_prices[:, :1, :1]
    log_closes = log_prices[..., 3]
    positive_volume = np.where(volume > 0, volume, np.nan)
    volume_median = np.nanmedian(positive_volume, axis=1, keepdims=True)
    out[..., 4] = np.log1p(volume / volume_median)

    afternoon = slots >= MORNING_BARS
    out[..., 5] = afternoon
    out[..., 6] = np.where(
        afternoon, (slots - MORNING_BARS + 1) / AFTERNOON_BARS, (slots + 1) / MORNING_BARS
    )

    returns = np.diff(log_closes, axis=1)
    return_count = positions[1:]
    squared_returns = np.cumsum(returns**2, axis=1)
    out[:, 1:, 7] = np.sqrt(squared_returns)
    out[:, 1:, 8] = _divide(
        np.sqrt(return_count) * np.cumsum(returns**3, axis=1),
        squared_returns**1.5,
        return_count >= 3,
    )

    for index, lag in enumerate(RETURN_HORIZONS):
        if lag < width:
            out[:, lag:, 9 + index] = log_closes[:, lag:] - log_closes[:, :-lag]
        out[..., 17 + index] = positions >= lag

    out[..., 12] = (highs - lows) / closes
    upper_wick = highs - np.maximum(opens, closes)
    lower_wick = np.minimum(opens, closes) - lows
    out[..., 13] = (upper_wick - lower_wick) / closes

    # Referencing the first pair keeps the prefix moments stable on near-flat data.
    centered_returns = returns - returns[:, :1]
    centered_volume = volume[:, 1:] - volume[:, 1:2]
    sum_returns = np.cumsum(centered_returns, axis=1)
    sum_volume = np.cumsum(centered_volume, axis=1)
    variance_returns = return_count * np.cumsum(centered_returns**2, axis=1) - sum_returns**2
    variance_volume = return_count * np.cumsum(centered_volume**2, axis=1) - sum_volume**2
    covariance = return_count * np.cumsum(centered_returns * centered_volume, axis=1) - sum_returns * sum_volume
    spread = np.sqrt(np.maximum(variance_returns, 0)) * np.sqrt(np.maximum(variance_volume, 0))
    out[:, 1:, 14] = np.clip(_divide(covariance, spread, return_count >= 3), -1, 1)

    prefix_volume = np.cumsum(volume, axis=1)
    prefix_amount = np.cumsum(amount, axis=1)
    out[..., 15] = _divide(closes * prefix_volume, prefix_amount, prefix_volume > 0) - 1

    count = positions.astype(np.float64) + 1
    sum_squared_time = count * (count**2 - 1) / 12
    centered_volume = volume - volume[:, :1]
    sum_time_volume = (
        np.cumsum(positions * centered_volume, axis=1)
        - positions * np.cumsum(centered_volume, axis=1) / 2
    )
    out[..., 16] = _divide(count * sum_time_volume, sum_squared_time * prefix_volume)


def _as_bars(bars: np.ndarray) -> np.ndarray:
    bars = np.asarray(bars)
    if bars.ndim != 3 or bars.shape[1:] != (DAY_BARS, 6):
        raise ValueError(f"bars must have shape (days, {DAY_BARS}, 6)")
    if bars.dtype.kind not in "fiu" or bars.dtype.itemsize > 8:
        bars = bars.astype(np.float64)
    return bars


def _positive_integer(name: str, value: int) -> None:
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _window_counts(events: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    counts = np.zeros((len(events), DAY_BARS + 1), dtype=np.int16)
    np.cumsum(events, axis=1, dtype=counts.dtype, out=counts[:, 1:])
    return counts[:, starts + width] - counts[:, starts]


def find_signal_coords(
    bars: np.ndarray, width: int = 30, horizon: int = 5, sample_every: int = 1
) -> np.ndarray:
    """Find (day index, signal slot) coordinates using each input window alone.

    Input columns are open, high, low, close, volume, and amount. Each day
    contains 09:31–11:30 followed by 13:01–14:57. Unavailable bars must be
    represented by missing values. Sampling is anchored to the first possible
    signal in each session, even when a stock's first window is incomplete.
    The horizon reserves clock slots in the session; later quote availability
    does not determine signal eligibility.
    """
    bars = _as_bars(bars)
    _positive_integer("width", width)
    _positive_integer("horizon", horizon)
    _positive_integer("sample_every", sample_every)

    if len(bars) == 0 or width + horizon > MORNING_BARS:
        return np.empty((0, 2), dtype=np.intp)

    opens, highs, lows, closes, volume, amount = bars.transpose(2, 0, 1)
    good_bar = (
        np.isfinite(bars).all(axis=2)
        & (bars[..., :4] > 0).all(axis=2)
        & (highs >= np.maximum(opens, closes))
        & (lows <= np.minimum(opens, closes))
        & (volume >= 0)
        & (amount >= 0)
    )

    starts = np.concatenate((
        np.arange(0, MORNING_BARS - width - horizon + 1, sample_every, dtype=np.intp),
        MORNING_BARS + np.arange(0, AFTERNOON_BARS - width - horizon + 1, sample_every, dtype=np.intp),
    ))
    endpoints = starts + width - 1
    complete = _window_counts(good_bar, starts, width) == width
    traded = _window_counts(volume > 0, starts, width) > 0
    day_index, column = np.nonzero(complete & traded)
    return np.column_stack((day_index, endpoints[column]))


def find_targets(
    bars: np.ndarray, width: int = 30, horizon: int = 5, sample_every: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Find valid realized labels in basis points and their signal coordinates."""
    bars = _as_bars(bars)
    coords = find_signal_coords(bars, width, horizon, sample_every)
    if not len(coords):
        return np.empty(0, dtype=np.float32), coords
    day_index, signal_slot = coords.T
    closes = bars[..., 3]
    next_close = closes[day_index, signal_slot + horizon]
    valid = np.isfinite(next_close) & (next_close > 0)
    labels = 10_000 * (
        np.log(next_close[valid].astype(np.float64, copy=False))
        - np.log(closes[day_index[valid], signal_slot[valid]].astype(np.float64, copy=False))
    )
    return labels.astype(np.float32), coords[valid]


def encode_windows(bars: np.ndarray, coords: np.ndarray, width: int = 30) -> np.ndarray:
    """Encode eligible signal coordinates, preserving their order.

    Amount divided by volume must use the same price basis as OHLC. Summary
    features use the window prefix at their row, apart from volume's fixed
    per-window scale. Undefined statistics remain NaN.
    """
    bars = _as_bars(bars)
    _positive_integer("width", width)
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.dtype.kind not in "iu":
        raise ValueError("coords must have integer shape (samples, 2)")
    if not len(coords):
        return np.empty((0, width, len(FEATURE_NAMES)), dtype=np.float32)

    day_index, signal_slot = coords.astype(np.intp, copy=False).T
    start_index = signal_slot - width + 1
    session_start = np.where(signal_slot < MORNING_BARS, 0, MORNING_BARS)
    if (np.any((day_index < 0) | (day_index >= len(bars)))
            or np.any((start_index < session_start) | (signal_slot >= DAY_BARS))):
        raise ValueError("coords must refer to an in-session window on an available day")

    features = np.full((len(coords), width, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    positions = np.arange(width)
    for first in range(0, len(coords), FEATURE_BATCH):
        batch = slice(first, first + FEATURE_BATCH)
        slots = start_index[batch, None] + positions
        windows = bars[day_index[batch, None], slots].astype(np.float64, copy=False)
        _fill_features(features[batch], windows, slots)
    return features


def build_samples(
    bars: np.ndarray, width: int = 30, horizon: int = 5, sample_every: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return float32 features and labels alongside integer sample coordinates."""
    labels, coords = find_targets(bars, width, horizon, sample_every)
    return encode_windows(bars, coords, width), labels, coords
