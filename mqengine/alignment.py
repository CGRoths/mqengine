from __future__ import annotations

import warnings

import pandas as pd


def _resolve_signal_time_col(signal_df: pd.DataFrame, preferred: str, fallback: str = "ts") -> str:
    if preferred in signal_df.columns:
        return preferred
    if fallback in signal_df.columns:
        return fallback
    raise ValueError(f"Signal dataframe must include {preferred!r} or fallback {fallback!r}.")


def align_signal_to_bars(
    execution_bars: pd.DataFrame,
    signal_df: pd.DataFrame,
    *,
    execution_ts_col: str = "ts",
    signal_event_ts_col: str = "event_ts",
    signal_available_ts_col: str = "available_ts",
    signal_value_col: str = "signal_value",
    output_col: str = "signal",
    direction: str = "backward",
    tolerance: str | pd.Timedelta | None = "24h",
    min_lag: str | pd.Timedelta = "0s",
    closed_bar_only: bool = True,
    stale_policy: str = "nan",
) -> pd.DataFrame:
    """
    Align signals to executable bars using availability time, not event time.

    A signal is usable only when available_ts <= execution_ts - min_lag.
    If available_ts is missing, event_ts is used as a fallback and a warning is
    emitted because this is only safe when event timestamps are true availability
    timestamps.
    """
    if direction not in {"backward", "forward", "nearest"}:
        raise ValueError("direction must be one of: backward, forward, nearest.")
    if stale_policy not in {"nan", "drop", "keep"}:
        raise ValueError("stale_policy must be one of: nan, drop, keep.")
    if execution_ts_col not in execution_bars.columns:
        raise ValueError(f"execution_bars is missing {execution_ts_col!r}.")
    if signal_value_col not in signal_df.columns:
        raise ValueError(f"signal_df is missing {signal_value_col!r}.")

    left = execution_bars.copy()
    left["_mq_order"] = range(len(left))
    left["_mq_execution_ts"] = pd.to_datetime(left[execution_ts_col], errors="coerce")
    min_lag_delta = pd.Timedelta(min_lag)
    left["_mq_alignment_key"] = left["_mq_execution_ts"] - min_lag_delta

    audit_cols = {
        output_col: pd.NA,
        "signal_event_ts": pd.NaT,
        "signal_available_ts": pd.NaT,
        "signal_age_seconds": pd.NA,
        "is_stale_signal": False,
        "alignment_method": f"merge_asof_{direction}_available_ts",
        "lookahead_safe": False,
    }

    if left.empty:
        for col, value in audit_cols.items():
            left[col] = value
        return left.drop(columns=[c for c in left.columns if c.startswith("_mq_")])

    if signal_df.empty:
        for col, value in audit_cols.items():
            left[col] = value
        return left.drop(columns=[c for c in left.columns if c.startswith("_mq_")])

    event_col = _resolve_signal_time_col(signal_df, signal_event_ts_col)
    if signal_available_ts_col in signal_df.columns:
        available_col = signal_available_ts_col
        availability_fallback = False
    else:
        available_col = event_col
        availability_fallback = True
        warnings.warn(
            "signal_available_ts_col is missing; falling back to event_ts. "
            "This is only lookahead-safe if event_ts is the true availability time.",
            RuntimeWarning,
            stacklevel=2,
        )

    right = signal_df.copy()
    right["_mq_event_ts"] = pd.to_datetime(right[event_col], errors="coerce")
    right["_mq_available_ts"] = pd.to_datetime(right[available_col], errors="coerce")
    right["_mq_signal_value"] = pd.to_numeric(right[signal_value_col], errors="coerce")
    right = right.dropna(subset=["_mq_available_ts", "_mq_signal_value"]).sort_values("_mq_available_ts")

    if right.empty:
        for col, value in audit_cols.items():
            left[col] = value
        out = left.sort_values("_mq_order").drop(columns=[c for c in left.columns if c.startswith("_mq_")])
        out.attrs["alignment_warnings"] = ["no_valid_signal_rows"]
        return out

    merged = pd.merge_asof(
        left.sort_values("_mq_alignment_key"),
        right[["_mq_available_ts", "_mq_event_ts", "_mq_signal_value"]].sort_values("_mq_available_ts"),
        left_on="_mq_alignment_key",
        right_on="_mq_available_ts",
        direction=direction,
    )

    merged[output_col] = merged["_mq_signal_value"]
    merged["signal_event_ts"] = merged["_mq_event_ts"]
    merged["signal_available_ts"] = merged["_mq_available_ts"]

    age = (merged["_mq_execution_ts"] - merged["_mq_available_ts"]).dt.total_seconds()
    merged["signal_age_seconds"] = age
    safe = merged["_mq_available_ts"].notna() & (merged["_mq_available_ts"] <= merged["_mq_alignment_key"])
    if closed_bar_only:
        safe = safe & merged["_mq_execution_ts"].notna()
    merged["lookahead_safe"] = safe.fillna(False).astype(bool)

    if tolerance is None:
        stale = pd.Series(False, index=merged.index)
    else:
        tolerance_seconds = pd.Timedelta(tolerance).total_seconds()
        stale = age > tolerance_seconds
        stale = stale.fillna(False)
    merged["is_stale_signal"] = stale.astype(bool)
    merged["alignment_method"] = f"merge_asof_{direction}_available_ts"

    merged.loc[~merged["lookahead_safe"], output_col] = pd.NA
    if stale_policy == "nan":
        merged.loc[merged["is_stale_signal"], output_col] = pd.NA
    elif stale_policy == "drop":
        merged = merged.loc[~merged["is_stale_signal"]].copy()

    out = merged.sort_values("_mq_order").drop(columns=[c for c in merged.columns if c.startswith("_mq_")])
    if availability_fallback:
        out.attrs["alignment_warnings"] = ["available_ts_missing_fallback_to_event_ts"]
    return out
