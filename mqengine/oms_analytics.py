from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from .risk import compute_period_metrics


def _to_df(records: Any) -> pd.DataFrame:
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    rows = []
    for item in records:
        if is_dataclass(item):
            rows.append(asdict(item))
        elif isinstance(item, dict):
            rows.append(dict(item))
        else:
            rows.append(vars(item))
    return pd.DataFrame(rows)


def _ratio(num: float, den: float) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


def compute_oms_metrics(
    *,
    order_events=None,
    fill_events=None,
    position_snapshots=None,
    equity_snapshots=None,
    target_position_snapshots=None,
    calendar: str = "crypto_365",
) -> dict[str, Any]:
    orders = _to_df(order_events)
    fills = _to_df(fill_events)
    positions = _to_df(position_snapshots)
    equity = _to_df(equity_snapshots)
    targets = _to_df(target_position_snapshots)

    metrics: dict[str, Any] = {
        "fill_ratio": 0.0,
        "cancel_ratio": 0.0,
        "reject_ratio": 0.0,
        "average_slippage_bps": 0.0,
        "median_slippage_bps": 0.0,
        "implementation_shortfall": 0.0,
        "target_vs_actual_position_drift": 0.0,
        "latency_ms": 0.0,
        "fee_drag": 0.0,
        "funding_drag": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "OMS_period_sharpe": 0.0,
        "live_rolling_drawdown": 0.0,
        "exposure_by_symbol": {},
        "execution_quality_by_venue": {},
    }

    if not orders.empty and "status" in orders.columns:
        status = orders["status"].astype(str).str.lower()
        total = len(status)
        metrics["fill_ratio"] = _ratio(status.isin(["filled", "partially_filled", "partial"]).sum(), total)
        metrics["cancel_ratio"] = _ratio(status.isin(["cancelled", "canceled"]).sum(), total)
        metrics["reject_ratio"] = _ratio(status.isin(["rejected", "reject"]).sum(), total)

    if not fills.empty:
        if "fee" in fills.columns:
            metrics["fee_drag"] = round(float(pd.to_numeric(fills["fee"], errors="coerce").fillna(0.0).sum()), 6)
        if "latency_ms" in fills.columns:
            metrics["latency_ms"] = round(float(pd.to_numeric(fills["latency_ms"], errors="coerce").dropna().mean()), 6)
        if {"fill_price", "arrival_price"}.issubset(fills.columns):
            fill_price = pd.to_numeric(fills["fill_price"], errors="coerce")
            arrival = pd.to_numeric(fills["arrival_price"], errors="coerce")
            side = fills.get("side", pd.Series("buy", index=fills.index)).astype(str).str.lower()
            signed = np.where(side.isin(["sell", "short"]), arrival - fill_price, fill_price - arrival)
            slippage_bps = np.where(arrival.abs() > 1e-12, signed / arrival * 10000.0, np.nan)
            slippage = pd.Series(slippage_bps).replace([np.inf, -np.inf], np.nan).dropna()
            if not slippage.empty:
                metrics["average_slippage_bps"] = round(float(slippage.mean()), 6)
                metrics["median_slippage_bps"] = round(float(slippage.median()), 6)
                metrics["implementation_shortfall"] = round(float(slippage.mean()), 6)
        if "venue" in fills.columns and {"fill_price", "arrival_price"}.issubset(fills.columns):
            quality = {}
            for venue, group in fills.groupby("venue"):
                fp = pd.to_numeric(group["fill_price"], errors="coerce")
                ap = pd.to_numeric(group["arrival_price"], errors="coerce")
                slip = ((fp - ap) / ap * 10000.0).replace([np.inf, -np.inf], np.nan).dropna()
                quality[str(venue)] = {"avg_slippage_bps": round(float(slip.mean()), 6) if not slip.empty else 0.0, "fills": int(len(group))}
            metrics["execution_quality_by_venue"] = quality

    if not positions.empty:
        for col in ["realized_pnl", "unrealized_pnl"]:
            if col in positions.columns:
                metrics[col] = round(float(pd.to_numeric(positions[col], errors="coerce").fillna(0.0).sum()), 6)
        if {"symbol", "quantity", "mark_price"}.issubset(positions.columns):
            latest_ts = pd.to_datetime(positions["ts"], errors="coerce").max() if "ts" in positions.columns else None
            latest = positions[pd.to_datetime(positions["ts"], errors="coerce") == latest_ts] if latest_ts is not None else positions
            exposure = latest.assign(
                exposure=pd.to_numeric(latest["quantity"], errors="coerce").fillna(0.0)
                * pd.to_numeric(latest["mark_price"], errors="coerce").fillna(0.0)
            ).groupby("symbol")["exposure"].sum()
            metrics["exposure_by_symbol"] = {str(k): round(float(v), 6) for k, v in exposure.items()}

    if not positions.empty and not targets.empty and {"ts", "symbol", "quantity"}.issubset(positions.columns) and {"ts", "symbol", "target_quantity"}.issubset(targets.columns):
        pos = positions.copy()
        tgt = targets.copy()
        pos["ts"] = pd.to_datetime(pos["ts"], errors="coerce")
        tgt["ts"] = pd.to_datetime(tgt["ts"], errors="coerce")
        merged = pd.merge_asof(
            pos.sort_values("ts"),
            tgt.sort_values("ts"),
            on="ts",
            by="symbol",
            direction="backward",
        )
        drift = (pd.to_numeric(merged["quantity"], errors="coerce") - pd.to_numeric(merged["target_quantity"], errors="coerce")).abs()
        metrics["target_vs_actual_position_drift"] = round(float(drift.dropna().mean()), 6) if not drift.dropna().empty else 0.0

    if not equity.empty and {"ts", "equity"}.issubset(equity.columns):
        eq = equity.copy()
        eq["ts"] = pd.to_datetime(eq["ts"], errors="coerce")
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        eq = eq.dropna(subset=["ts", "equity"]).sort_values("ts")
        period = compute_period_metrics(eq["equity"], eq["ts"], calendar=calendar)
        metrics["OMS_period_sharpe"] = period["period_sharpe"]
        metrics["live_rolling_drawdown"] = period["max_drawdown_pct"]

    return metrics
