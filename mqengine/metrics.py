from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import compute_period_metrics, compute_trade_metrics, compute_trade_sharpe


def compute_equity_drawdown_stats(equity: np.ndarray) -> tuple[float, float]:
    if len(equity) == 0:
        return 0.0, 0.0
    peaks = np.maximum.accumulate(equity)
    safe_peaks = np.where(peaks <= 0, np.nan, peaks)
    drawdowns = (safe_peaks - equity) / safe_peaks
    drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)
    max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0
    if equity[0] > 0 and equity[-1] > 0:
        total_return = float(equity[-1] / equity[0] - 1.0)
    else:
        total_return = 0.0
    return total_return, max_dd


def compute_metrics(
    equity: np.ndarray,
    trades: list[dict],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    timestamps=None,
    periods_per_year: float | None = None,
    calendar: str = "crypto_365",
) -> dict:
    total_return, max_dd = compute_equity_drawdown_stats(equity)
    net_profit = float(equity[-1] - equity[0]) if len(equity) else 0.0
    total_days = max((end_ts - start_ts).total_seconds() / 86400.0, 1.0)

    if timestamps is None:
        if len(equity) > 1:
            timestamps = pd.date_range(pd.Timestamp(start_ts), pd.Timestamp(end_ts), periods=len(equity))
        elif len(equity) == 1:
            timestamps = [pd.Timestamp(start_ts)]
        else:
            timestamps = []

    period_metrics = compute_period_metrics(
        equity,
        timestamps,
        periods_per_year=periods_per_year,
        calendar=calendar,
    )
    trade_metrics = compute_trade_metrics(
        trades,
        total_days=total_days,
        equity_points=len(equity),
    )

    trade_sharpe = float(trade_metrics["trade_sharpe"])
    cagr_pct = float(period_metrics["cagr_pct"])
    calmar = float(period_metrics["calmar"])
    profit_factor = float(trade_metrics["profit_factor"])
    win_rate = float(trade_metrics["win_rate"])
    avg_trade_pct = float(trade_metrics["avg_trade_return"])
    num_trades = int(trade_metrics["num_trades"])
    trades_per_year = float(trade_metrics["trades_per_year"])

    out = {
        "return_pct": round(total_return * 100.0, 2),
        "sharpe": round(trade_sharpe, 3),
        "trade_sharpe": round(trade_sharpe, 3),
        "max_drawdown": round(max_dd * 100.0, 2),
        "net_profit": round(net_profit, 2),
        "cagr_pct": round(cagr_pct, 2),
        "calmar": round(calmar, 3),
        "num_trades": num_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(float(profit_factor), 3),
        "avg_trade_pct": round(avg_trade_pct, 2),
        "trades_per_year": round(float(trades_per_year), 3),
    }

    out.update({
        "period_sharpe": round(float(period_metrics["period_sharpe"]), 3),
        "sortino": round(float(period_metrics["sortino"]), 3),
        "rolling_sharpe": round(float(period_metrics["rolling_sharpe"]), 3),
        "rolling_sortino": round(float(period_metrics["rolling_sortino"]), 3),
        "volatility_annualized": round(float(period_metrics["volatility_annualized"]), 3),
        "downside_volatility": round(float(period_metrics["downside_volatility"]), 3),
        "cagr": round(float(period_metrics["cagr"]), 6),
        "max_drawdown_pct": round(float(period_metrics["max_drawdown_pct"]), 3),
        "max_drawdown_start": period_metrics["max_drawdown_start"],
        "max_drawdown_end": period_metrics["max_drawdown_end"],
        "max_drawdown_recovery": period_metrics["max_drawdown_recovery"],
        "max_drawdown_duration_days": round(float(period_metrics["max_drawdown_duration_days"]), 3),
        "median_trade_return": round(float(trade_metrics["median_trade_return"]), 3),
        "best_trade": round(float(trade_metrics["best_trade"]), 3),
        "worst_trade": round(float(trade_metrics["worst_trade"]), 3),
        "avg_trade_return": round(float(trade_metrics["avg_trade_return"]), 3),
        "avg_holding_bars": round(float(trade_metrics["avg_holding_bars"]), 3),
        "avg_holding_time": round(float(trade_metrics["avg_holding_time"]), 3),
        "time_in_market_pct": round(float(trade_metrics["time_in_market_pct"]), 3),
        "turnover": round(float(trade_metrics["turnover"]), 3),
        "fee_drag": round(float(trade_metrics["fee_drag"]), 6),
        "slippage_drag": round(float(trade_metrics["slippage_drag"]), 6),
        "gross_pnl": round(float(trade_metrics["gross_pnl"]), 6),
        "net_pnl": round(float(trade_metrics["net_pnl"]), 6),
        "VaR_95": round(float(period_metrics["VaR_95"]), 3),
        "CVaR_95": round(float(period_metrics["CVaR_95"]), 3),
        "skew": round(float(period_metrics["skew"]), 3),
        "kurtosis": round(float(period_metrics["kurtosis"]), 3),
    })

    return out
