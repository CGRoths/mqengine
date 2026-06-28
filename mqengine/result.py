from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
import json
import numpy as np
import pandas as pd


def to_iso_strings(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, pd.Series):
        seq = values.tolist()
    elif isinstance(values, np.ndarray):
        seq = values.tolist()
    else:
        seq = list(values)
    out = []
    for v in seq:
        ts = pd.Timestamp(v)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        out.append(ts.strftime("%Y-%m-%dT%H:%M:%S"))
    return out


def downsample_xy(x_vals: list[str], y_vals: list[float], max_points: int):
    n = len(y_vals)
    if n <= max_points:
        return x_vals, y_vals
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    idx = np.unique(idx)
    return [x_vals[i] for i in idx], [y_vals[i] for i in idx]


def downsample_multiple(x_vals: list[str], series_map: dict[str, list[float]], max_points: int):
    lengths = [len(v) for v in series_map.values()] if series_map else [0]
    n = min([len(x_vals)] + lengths)
    if n <= max_points:
        return x_vals, series_map
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    idx = np.unique(idx)
    x_out = [x_vals[i] for i in idx]
    out = {key: [vals[i] for i in idx] for key, vals in series_map.items()}
    return x_out, out


def build_debug_summary_from_trades(trades: list[dict]) -> dict:
    if not trades:
        return {
            "num_trades": 0,
            "avg_bars_held": 0.0,
            "max_win_pct": 0.0,
            "max_loss_pct": 0.0,
            "avg_allocated_capital": 0.0,
            "avg_units": 0.0,
            "avg_account_return_pct": 0.0,
        }
    pnl_list = [float(t.get("pnl_pct", 0.0)) for t in trades]
    bars_held_list = [float(t.get("bars_held", 0.0)) for t in trades]
    alloc_list = [float(t.get("allocated_capital", 0.0)) for t in trades]
    units_list = [float(t.get("units", 0.0)) for t in trades]
    acct_list = [float(t.get("account_return_pct", 0.0)) for t in trades]
    return {
        "num_trades": len(trades),
        "avg_bars_held": round(float(np.mean(bars_held_list)), 3) if bars_held_list else 0.0,
        "max_win_pct": round(float(np.max(pnl_list)), 4) if pnl_list else 0.0,
        "max_loss_pct": round(float(np.min(pnl_list)), 4) if pnl_list else 0.0,
        "avg_allocated_capital": round(float(np.mean(alloc_list)), 4) if alloc_list else 0.0,
        "avg_units": round(float(np.mean(units_list)), 8) if units_list else 0.0,
        "avg_account_return_pct": round(float(np.mean(acct_list)), 4) if acct_list else 0.0,
    }


@dataclass
class BacktestResult:
    name: str
    strategy_id: str
    params: dict[str, Any]
    data: pd.DataFrame
    trades: list[dict]
    metrics: dict
    price_col: str
    signal_col: str
    benchmark_col: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    research: dict[str, Any] = field(default_factory=dict)

    def debug_summary(self) -> dict:
        return build_debug_summary_from_trades(self.trades)

    def to_payload(self) -> dict[str, Any]:
        payload_df = self.data.copy()
        payload_df["ts"] = pd.to_datetime(payload_df["ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "name": self.name,
            "strategy_id": self.strategy_id,
            "params": self.params,
            "metrics": self.metrics,
            "rows": payload_df.to_dict(orient="records"),
            "trades": self.trades,
            "price_col": self.price_col,
            "signal_col": self.signal_col,
            "benchmark_col": self.benchmark_col,
            "meta": self.meta,
            "validation": self.validation,
            "research": self.research,
        }

    def build_chart_payload(self, max_points: int = 1800) -> dict[str, Any]:
        dates = to_iso_strings(self.data["ts"])
        chart_dates, chart_series = downsample_multiple(
            dates,
            {
                "close": self.data[self.price_col].astype(float).tolist(),
                "signal": self.data[self.signal_col].astype(float).tolist(),
            },
            max_points=max_points,
        )

        long_entry_dates, long_entry_prices = [], []
        short_entry_dates, short_entry_prices = [], []
        exit_dates, exit_prices = [], []
        for trade in self.trades:
            entry_date = trade.get("entry_date")
            exit_date = trade.get("exit_date")
            entry_price = trade.get("entry_price")
            exit_price = trade.get("exit_price")
            side = trade.get("side")
            if entry_date is not None and entry_price is not None:
                if side == "LONG":
                    long_entry_dates.append(str(entry_date).replace(" ", "T"))
                    long_entry_prices.append(float(entry_price))
                elif side == "SHORT":
                    short_entry_dates.append(str(entry_date).replace(" ", "T"))
                    short_entry_prices.append(float(entry_price))
            if exit_date is not None and exit_price is not None:
                exit_dates.append(str(exit_date).replace(" ", "T"))
                exit_prices.append(float(exit_price))

        strategy_equity_dates, strategy_equity = downsample_xy(
            dates,
            self.data["equity"].astype(float).tolist(),
            max_points=max_points,
        )

        benchmark_dates = []
        benchmark_equity = []
        if self.benchmark_col and self.benchmark_col in self.data.columns:
            benchmark_dates, benchmark_equity = downsample_xy(
                dates,
                self.data[self.benchmark_col].astype(float).tolist(),
                max_points=max_points,
            )

        threshold_dates = []
        long_thr_line = []
        short_thr_line = []
        exit_thr_line = []
        if chart_dates:
            threshold_dates = [chart_dates[0], chart_dates[-1]]
            if "long_threshold" in self.params:
                long_thr_line = [self.params["long_threshold"], self.params["long_threshold"]]
            if "short_threshold" in self.params:
                short_thr_line = [self.params["short_threshold"], self.params["short_threshold"]]
            if self.params.get("exit_signal") is not None:
                exit_thr_line = [self.params["exit_signal"], self.params["exit_signal"]]

        return {
            "equity_chart": {
                "strategy_dates": strategy_equity_dates,
                "strategy_equity": strategy_equity,
                "benchmark_dates": benchmark_dates,
                "benchmark_equity": benchmark_equity,
            },
            "price_signal_chart": {
                "dates": chart_dates,
                "close": chart_series["close"],
                "signal": chart_series["signal"],
                "signal_name": self.signal_col,
                "long_threshold_dates": threshold_dates,
                "long_threshold": long_thr_line,
                "short_threshold_dates": threshold_dates,
                "short_threshold": short_thr_line,
                "exit_threshold_dates": threshold_dates,
                "exit_threshold": exit_thr_line,
                "entry_long_dates": long_entry_dates,
                "entry_long_prices": long_entry_prices,
                "entry_short_dates": short_entry_dates,
                "entry_short_prices": short_entry_prices,
                "exit_dates": exit_dates,
                "exit_prices": exit_prices,
            },
        }

    def to_detail_payload(self, max_points: int = 1800, benchmark_name: str = "Benchmark") -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy_id": self.strategy_id,
            "params": self.params,
            "metrics": self.metrics,
            "validation": self.validation,
            "research": self.research,
            "usable_start": pd.Timestamp(self.data["ts"].iloc[0]).strftime("%Y-%m-%d %H:%M:%S"),
            "usable_end": pd.Timestamp(self.data["ts"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S"),
            "usable_rows": int(len(self.data)),
            "trades": self.trades[-500:],
            "debug_summary": self.debug_summary(),
            "chart_data": self.build_chart_payload(max_points=max_points),
            "benchmark": {
                "name": benchmark_name,
                "metrics": {},
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload())

    def to_research_report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy_id": self.strategy_id,
            "params": self.params,
            "metrics": self.metrics,
            "validation": self.validation,
            "research": self.research,
            "trades": self.trades,
            "debug_summary": self.debug_summary(),
        }

    def to_flask_app(self):
        from .dashboard import build_single_dashboard_app
        return build_single_dashboard_app(self)


@dataclass
class SweepResult:
    name: str
    strategy_results: list[BacktestResult]
    results_df: pd.DataFrame
    param_columns: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def filter_options(self) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = {}
        for col in self.param_columns:
            if col in self.results_df.columns:
                vals = self.results_df[col].drop_duplicates().tolist()
                try:
                    vals = sorted(vals, key=lambda x: (str(type(x)), x))
                except Exception:
                    vals = sorted(vals, key=lambda x: str(x))
                out[col] = vals
        return out

    def to_flask_app(self):
        from .dashboard import build_sweep_dashboard_app
        return build_sweep_dashboard_app(self)
