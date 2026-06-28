from __future__ import annotations

from typing import Optional
import pandas as pd

from .alignment import align_signal_to_bars

try:
    from sqlalchemy import text
except ImportError:  # pragma: no cover - exercised only when optional SQLAlchemy is absent.
    def text(sql: str) -> str:
        return sql


def fetch_sql_signal(conn_or_engine, *, table_name: str, ts_col: str, value_col: str, start_ts, end_ts, where_sql: str = "") -> pd.DataFrame:
    sql = text(f"""
        SELECT
            {ts_col} AS signal_ts,
            {value_col} AS signal_value
        FROM {table_name}
        WHERE {ts_col} >= :start_ts
          AND {ts_col} <= :end_ts
          AND {value_col} IS NOT NULL
          {where_sql}
        ORDER BY {ts_col} ASC
    """)
    df = pd.read_sql(sql, conn_or_engine, params={"start_ts": pd.Timestamp(start_ts).to_pydatetime(), "end_ts": pd.Timestamp(end_ts).to_pydatetime()})
    if df.empty:
        return pd.DataFrame(columns=["ts", "signal_value"])
    df["ts"] = pd.to_datetime(df["signal_ts"], utc=True, errors="coerce").dt.tz_convert(None)
    df["signal_value"] = pd.to_numeric(df["signal_value"], errors="coerce")
    return df.dropna(subset=["ts", "signal_value"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)[["ts", "signal_value"]]


def fetch_sql_ohlc(conn_or_engine, *, table_name: str, open_time_col: str, close_time_col: str, open_col: str = "open", high_col: str = "high", low_col: str = "low", close_col: str = "close", volume_col: Optional[str] = None, start_ms: Optional[int] = None, end_ms: Optional[int] = None, extra_where_sql: str = "", extra_params: Optional[dict] = None) -> pd.DataFrame:
    volume_select = f", {volume_col} AS volume" if volume_col else ""
    sql = text(f"""
        SELECT
            {open_time_col} AS open_time,
            {close_time_col} AS close_time,
            {open_col} AS open,
            {high_col} AS high,
            {low_col} AS low,
            {close_col} AS close
            {volume_select}
        FROM {table_name}
        WHERE (:start_ms IS NULL OR {open_time_col} >= :start_ms)
          AND (:end_ms IS NULL OR {open_time_col} <= :end_ms)
          {extra_where_sql}
        ORDER BY {open_time_col} ASC
    """)
    params = {"start_ms": start_ms, "end_ms": end_ms}
    if extra_params:
        params.update(extra_params)
    df = pd.read_sql(sql, conn_or_engine, params=params)
    if df.empty:
        return pd.DataFrame(columns=["ts", "close_ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce").dt.tz_convert(None)
    df["close_ts"] = pd.to_datetime(df["close_time"], unit="ms", utc=True, errors="coerce").dt.tz_convert(None)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def merge_signal_backward(execution_bars: pd.DataFrame, signal_df: pd.DataFrame, *, execution_ts_col: str = "ts", signal_ts_col: str = "ts", signal_value_col: str = "signal_value", output_col: str = "signal") -> pd.DataFrame:
    left = execution_bars.copy().sort_values(execution_ts_col).reset_index(drop=True)
    right = signal_df.copy().sort_values(signal_ts_col).reset_index(drop=True)
    if left.empty or right.empty:
        out = left.copy()
        out[output_col] = pd.NA
        return out
    merged = pd.merge_asof(left, right[[signal_ts_col, signal_value_col]], left_on=execution_ts_col, right_on=signal_ts_col, direction="backward")
    merged = merged.rename(columns={signal_value_col: output_col, signal_ts_col: f"{output_col}_ts"})
    return merged
