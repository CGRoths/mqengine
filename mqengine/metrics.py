from __future__ import annotations

import math
from typing import Iterable
import numpy as np
import pandas as pd


def compute_trade_sharpe(trade_account_returns: Iterable[float], trades_per_year: float) -> float:
    arr = np.asarray(list(trade_account_returns), dtype=float)
    if arr.size < 2:
        return 0.0
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr, ddof=0))
    if std_r <= 1e-12:
        return 0.0
    return (mean_r / std_r) * math.sqrt(max(trades_per_year, 1e-12))


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


def compute_metrics(equity: np.ndarray, trades: list[dict], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict:
    total_return, max_dd = compute_equity_drawdown_stats(equity)
    net_profit = float(equity[-1] - equity[0]) if len(equity) else 0.0
    total_days = max((end_ts - start_ts).total_seconds() / 86400.0, 1.0)

    if trades:
        trade_account_returns = [float(t["account_return_pct"]) / 100.0 for t in trades]
        trades_per_year = (len(trades) / total_days) * 365.0
        sharpe = compute_trade_sharpe(trade_account_returns, trades_per_year)
    else:
        trade_account_returns = []
        trades_per_year = 0.0
        sharpe = 0.0

    if len(equity) >= 2 and equity[0] > 0 and equity[-1] > 0:
        years = total_days / 365.0
        log_ratio = math.log(equity[-1] / equity[0])
        annual_log_return = log_ratio / max(years, 1e-12)
        annual_log_return = max(min(annual_log_return, 700), -700)
        cagr = math.exp(annual_log_return) - 1.0
    else:
        cagr = 0.0

    calmar = (cagr / max_dd) if max_dd > 1e-12 else 0.0

    if trade_account_returns:
        arr = np.asarray(trade_account_returns, dtype=float)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        gross_win = float(wins.sum()) if len(wins) else 0.0
        gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 1e-12 else (gross_win if gross_win > 0 else 0.0)
        win_rate = float((arr > 0).mean() * 100.0)
        avg_trade_pct = float(arr.mean() * 100.0)
        num_trades = int(len(arr))
    else:
        profit_factor = 0.0
        win_rate = 0.0
        avg_trade_pct = 0.0
        num_trades = 0

    return {
        "return_pct": round(total_return * 100.0, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd * 100.0, 2),
        "net_profit": round(net_profit, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "calmar": round(calmar, 3),
        "num_trades": num_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(float(profit_factor), 3),
        "avg_trade_pct": round(avg_trade_pct, 2),
        "trades_per_year": round(float(trades_per_year), 3),
    }
