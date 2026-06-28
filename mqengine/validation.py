from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schemas import ValidationIssue, ValidationReport


def _issue(errors: list[ValidationIssue], code: str, message: str, count: int = 1) -> None:
    errors.append(ValidationIssue(code=code, message=message, severity="error", count=int(count)))


def _warning(warnings: list[ValidationIssue], code: str, message: str, count: int = 1) -> None:
    warnings.append(ValidationIssue(code=code, message=message, severity="warning", count=int(count)))


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def validate_ohlc(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    expected_freq: str | pd.Timedelta | None = None,
    allow_empty: bool = False,
) -> ValidationReport:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if df.empty and not allow_empty:
        _issue(errors, "empty_ohlc", "OHLC dataframe is empty.")
        return ValidationReport(errors=errors, warnings=warnings)

    required = [ts_col, open_col, high_col, low_col, close_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        _issue(errors, "missing_ohlc_columns", f"Missing OHLC columns: {missing}.", len(missing))
        return ValidationReport(errors=errors, warnings=warnings)

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.isna().any():
        _issue(errors, "invalid_timestamp", "OHLC timestamp column contains invalid values.", int(ts.isna().sum()))
    if not ts.is_monotonic_increasing:
        _issue(errors, "timestamp_not_monotonic", "OHLC timestamps must be monotonic increasing.")
    dup_count = int(ts.duplicated().sum())
    if dup_count:
        _issue(errors, "duplicate_timestamps", "OHLC timestamps contain duplicates.", dup_count)

    open_s = _numeric_series(df, open_col)
    high_s = _numeric_series(df, high_col)
    low_s = _numeric_series(df, low_col)
    close_s = _numeric_series(df, close_col)

    for col, series in [(open_col, open_s), (high_col, high_s), (low_col, low_s), (close_col, close_s)]:
        bad_count = int(series.isna().sum())
        if bad_count:
            _issue(errors, "non_numeric_ohlc", f"{col} contains non-numeric values.", bad_count)

    close_bad = int((close_s <= 0.0).sum())
    if close_bad:
        _issue(errors, "non_positive_close", "close must be greater than zero.", close_bad)

    high_bad = int((high_s < pd.concat([open_s, close_s], axis=1).max(axis=1)).sum())
    if high_bad:
        _issue(errors, "high_below_open_or_close", "high must be greater than or equal to max(open, close).", high_bad)

    low_bad = int((low_s > pd.concat([open_s, close_s], axis=1).min(axis=1)).sum())
    if low_bad:
        _issue(errors, "low_above_open_or_close", "low must be less than or equal to min(open, close).", low_bad)

    if volume_col in df.columns:
        volume_s = _numeric_series(df, volume_col)
        bad_volume = int((volume_s < 0.0).sum() + volume_s.isna().sum())
        if bad_volume:
            _issue(errors, "invalid_volume", "volume must be numeric and non-negative when present.", bad_volume)

    if expected_freq is not None and len(ts.dropna()) > 1:
        sorted_ts = ts.dropna().sort_values()
        expected = pd.date_range(sorted_ts.iloc[0], sorted_ts.iloc[-1], freq=expected_freq)
        missing_count = int(len(expected.difference(pd.DatetimeIndex(sorted_ts))))
        if missing_count:
            _warning(warnings, "missing_bars", "Expected frequency implies missing OHLC bars.", missing_count)

    return ValidationReport(errors=errors, warnings=warnings)


def validate_signal(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    signal_col: str = "signal",
    event_ts_col: str | None = None,
    available_ts_col: str | None = None,
    duplicate_policy: str = "error",
    nan_policy: str = "warn",
    inf_policy: str = "error",
    stale_after: str | pd.Timedelta | None = None,
) -> ValidationReport:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if df.empty:
        _issue(errors, "empty_signal", "Signal dataframe is empty.")
        return ValidationReport(errors=errors, warnings=warnings)

    missing = [col for col in [ts_col, signal_col] if col not in df.columns]
    if missing:
        _issue(errors, "missing_signal_columns", f"Missing signal columns: {missing}.", len(missing))
        return ValidationReport(errors=errors, warnings=warnings)

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.isna().any():
        _issue(errors, "invalid_signal_timestamp", "Signal timestamp column contains invalid values.", int(ts.isna().sum()))

    dup_count = int(ts.duplicated().sum())
    if dup_count:
        if duplicate_policy == "error":
            _issue(errors, "duplicate_signal_timestamps", "Signal timestamps contain duplicates.", dup_count)
        elif duplicate_policy == "warn":
            _warning(warnings, "duplicate_signal_timestamps", "Signal timestamps contain duplicates.", dup_count)

    sig = pd.to_numeric(df[signal_col], errors="coerce")
    nan_count = int(sig.isna().sum())
    if nan_count:
        if nan_policy == "error":
            _issue(errors, "nan_signal", "Signal contains NaN or non-numeric values.", nan_count)
        elif nan_policy == "warn":
            _warning(warnings, "nan_signal", "Signal contains NaN or non-numeric values.", nan_count)

    inf_count = int(np.isinf(sig.fillna(0.0)).sum())
    if inf_count:
        if inf_policy == "error":
            _issue(errors, "inf_signal", "Signal contains infinite values.", inf_count)
        elif inf_policy == "warn":
            _warning(warnings, "inf_signal", "Signal contains infinite values.", inf_count)

    if event_ts_col and event_ts_col in df.columns:
        event_ts = pd.to_datetime(df[event_ts_col], errors="coerce")
        bad_event = int(event_ts.isna().sum())
        if bad_event:
            _issue(errors, "invalid_event_ts", "event_ts contains invalid values.", bad_event)
    else:
        event_ts = None

    if available_ts_col and available_ts_col in df.columns:
        available_ts = pd.to_datetime(df[available_ts_col], errors="coerce")
        bad_available = int(available_ts.isna().sum())
        if bad_available:
            _issue(errors, "invalid_available_ts", "available_ts contains invalid values.", bad_available)
    else:
        available_ts = None

    if event_ts is not None and available_ts is not None:
        before_event = int((available_ts < event_ts).sum())
        if before_event:
            _warning(warnings, "available_before_event_ts", "available_ts is earlier than event_ts for some signals.", before_event)

    if stale_after is not None and available_ts is not None:
        stale_delta = pd.Timedelta(stale_after)
        stale_count = int(((ts - available_ts) > stale_delta).sum())
        if stale_count:
            _warning(warnings, "stale_signal", "Signal is older than the configured stale_after threshold.", stale_count)

    return ValidationReport(errors=errors, warnings=warnings)
