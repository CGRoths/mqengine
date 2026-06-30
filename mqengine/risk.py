from __future__ import annotations

import math
from typing import Iterable, Sequence, Any

import numpy as np
import pandas as pd


EPS = 1e-12


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _round(value: Any, digits: int = 6) -> float:
    return round(_finite_float(value), digits)


def _safe_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def infer_periods_per_year(timestamps: Sequence[Any] | pd.Series | pd.Index | None, calendar: str = "crypto_365") -> float:
    """
    Infer annualization from timestamp spacing.

    Crypto defaults to a continuous 365-day year, so daily bars annualize by 365,
    4h bars by 6 * 365, 1h bars by 24 * 365, and so on.
    """
    if timestamps is None:
        return 365.0 if calendar == "crypto_365" else 252.0

    values = list(timestamps)
    if not values:
        return 365.0 if calendar == "crypto_365" else 252.0
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
        return 365.0 if calendar == "crypto_365" else 252.0

    ts = pd.Series(pd.to_datetime(values, errors="coerce")).dropna().sort_values()
    if len(ts) < 2:
        return 365.0 if calendar == "crypto_365" else 252.0

    diffs = ts.diff().dt.total_seconds().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 365.0 if calendar == "crypto_365" else 252.0

    median_seconds = float(diffs.median())
    if calendar == "trading_252":
        seconds_per_year = 252.0 * 24.0 * 60.0 * 60.0
    else:
        seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
    return max(seconds_per_year / max(median_seconds, EPS), 1.0)


def _timedelta_seconds(value: str | pd.Timedelta) -> float:
    try:
        delta = value if isinstance(value, pd.Timedelta) else pd.Timedelta(value)
    except Exception:
        delta = pd.Timedelta("1D")
    seconds = float(delta.total_seconds())
    return seconds if seconds > EPS else 86400.0


def _infer_adrs_interval_seconds(
    timestamps: Sequence[Any] | pd.Series | pd.Index,
    *,
    base_period: str | pd.Timedelta,
) -> float:
    fallback = _timedelta_seconds(base_period)
    values = list(timestamps)
    if len(values) < 2:
        return fallback

    ts = pd.Series(pd.to_datetime(values, errors="coerce")).dropna()
    if len(ts) < 2:
        return fallback

    diffs = ts.diff().dt.total_seconds().dropna()
    positive_diffs = diffs[diffs > 0.0]
    if positive_diffs.empty:
        return fallback

    median_seconds = float(positive_diffs.median())
    if math.isfinite(median_seconds) and median_seconds > EPS:
        return median_seconds

    last_seconds = float(positive_diffs.iloc[-1])
    if math.isfinite(last_seconds) and last_seconds > EPS:
        return last_seconds
    return fallback


def compute_adrs_compatible_metrics(
    pnl: Sequence[float] | pd.Series | np.ndarray,
    timestamps: Sequence[Any] | pd.Series | pd.Index,
    equity: Sequence[float] | pd.Series | np.ndarray | None = None,
    *,
    num_periods: int = 365,
    base_period: str | pd.Timedelta = "1D",
) -> dict[str, Any]:
    """
    Compute ADRS-compatible return metrics for MQEngine results.

    ``pnl`` is interpreted as a period return series. Timestamp spacing uses the
    median positive observed diff in series order, which is more robust to one
    missed bar than ADRS' last-diff rule while preserving the same annualization
    model. If no valid diff exists, ``base_period`` is used as the interval.
    When ``equity`` is omitted, total return and CAGR are compounded from
    ``pnl`` as a return series.
    """
    pnl_arr = np.asarray(list(pnl), dtype=float)
    pnl_arr = pnl_arr[np.isfinite(pnl_arr)]

    interval_seconds = _infer_adrs_interval_seconds(timestamps, base_period=base_period)
    base_seconds = _timedelta_seconds(base_period)
    period_multiplier = base_seconds / max(interval_seconds, EPS)
    annualization = max(float(num_periods) * period_multiplier, EPS)

    if pnl_arr.size == 0:
        return {
            "adrs_sharpe": 0.0,
            "adrs_sortino": 0.0,
            "adrs_annualized_return": 0.0,
            "adrs_total_return": 0.0,
            "adrs_cagr": 0.0,
            "adrs_interval_seconds": _round(interval_seconds, 6),
            "adrs_period_multiplier": _round(period_multiplier, 8),
            "adrs_num_periods": int(num_periods),
        }

    mean_r = float(np.mean(pnl_arr))
    std_r = float(np.std(pnl_arr, ddof=1)) if pnl_arr.size > 1 else 0.0
    downside = np.where(pnl_arr < 0.0, pnl_arr, 0.0)
    downside_dev = math.sqrt(float(np.mean(downside**2))) if downside.size else 0.0

    adrs_sharpe = (mean_r / std_r) * math.sqrt(annualization) if std_r > EPS else 0.0
    adrs_sortino = (mean_r / downside_dev) * math.sqrt(annualization) if downside_dev > EPS else 0.0
    annualized_return = mean_r * annualization

    if equity is not None:
        equity_arr = np.asarray(list(equity), dtype=float)
        equity_arr = equity_arr[np.isfinite(equity_arr)]
        if equity_arr.size >= 2 and abs(float(equity_arr[0])) > EPS:
            total_return = float(equity_arr[-1] / equity_arr[0] - 1.0)
        else:
            total_return = 0.0
    else:
        gross = float(np.prod(1.0 + pnl_arr))
        total_return = gross - 1.0 if math.isfinite(gross) else float(np.sum(pnl_arr))

    years = max(float(pnl_arr.size) / annualization, 1.0 / max(float(num_periods), 1.0))
    if total_return <= -1.0:
        cagr = -1.0
    else:
        cagr = math.exp(max(min(math.log1p(total_return) / max(years, EPS), 700.0), -700.0)) - 1.0

    return {
        "adrs_sharpe": _round(adrs_sharpe, 6),
        "adrs_sortino": _round(adrs_sortino, 6),
        "adrs_annualized_return": _round(annualized_return, 8),
        "adrs_total_return": _round(total_return, 8),
        "adrs_cagr": _round(cagr, 8),
        "adrs_interval_seconds": _round(interval_seconds, 6),
        "adrs_period_multiplier": _round(period_multiplier, 8),
        "adrs_num_periods": int(num_periods),
    }


def compute_trade_sharpe(trade_account_returns: Iterable[float], trades_per_year: float) -> float:
    arr = np.asarray(list(trade_account_returns), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    std_r = float(np.std(arr, ddof=0))
    if std_r <= EPS:
        return 0.0
    return float(np.mean(arr) / std_r) * math.sqrt(max(float(trades_per_year), EPS))


def drawdown_details(equity: Sequence[float], timestamps: Sequence[Any] | None = None) -> dict[str, Any]:
    equity_arr = np.asarray(list(equity), dtype=float)
    if equity_arr.size == 0:
        return {
            "max_drawdown_pct": 0.0,
            "max_drawdown_start": None,
            "max_drawdown_end": None,
            "max_drawdown_recovery": None,
            "max_drawdown_duration_days": 0.0,
        }

    if timestamps is None:
        ts = pd.Series(pd.RangeIndex(len(equity_arr)))
    else:
        ts = pd.Series(list(timestamps)).reset_index(drop=True)
        if len(ts) != len(equity_arr):
            ts = pd.Series(pd.RangeIndex(len(equity_arr)))

    peaks = np.maximum.accumulate(equity_arr)
    safe_peaks = np.where(peaks <= 0, np.nan, peaks)
    drawdowns = (equity_arr - safe_peaks) / safe_peaks
    drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)

    trough_idx = int(np.argmin(drawdowns))
    peak_value = peaks[trough_idx]
    peak_candidates = np.where(equity_arr[: trough_idx + 1] >= peak_value - EPS)[0]
    start_idx = int(peak_candidates[0]) if len(peak_candidates) else trough_idx

    recovery_idx = None
    if trough_idx < len(equity_arr) - 1:
        recovered = np.where(equity_arr[trough_idx + 1 :] >= peak_value - EPS)[0]
        if len(recovered):
            recovery_idx = int(trough_idx + 1 + recovered[0])

    duration_end_idx = recovery_idx if recovery_idx is not None else len(equity_arr) - 1

    def _duration_days(a: Any, b: Any) -> float:
        ta = _safe_timestamp(a)
        tb = _safe_timestamp(b)
        if ta is None or tb is None:
            return float(max(duration_end_idx - start_idx, 0))
        return max(float((tb - ta).total_seconds() / 86400.0), 0.0)

    def _ts_out(value: Any) -> Any:
        t = _safe_timestamp(value)
        if t is None:
            try:
                return int(value)
            except Exception:
                return value
        return t.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "max_drawdown_pct": _round(abs(float(drawdowns[trough_idx])) * 100.0, 6),
        "max_drawdown_start": _ts_out(ts.iloc[start_idx]),
        "max_drawdown_end": _ts_out(ts.iloc[trough_idx]),
        "max_drawdown_recovery": _ts_out(ts.iloc[recovery_idx]) if recovery_idx is not None else None,
        "max_drawdown_duration_days": _round(_duration_days(ts.iloc[start_idx], ts.iloc[duration_end_idx]), 6),
    }


def compute_period_metrics(
    equity: Sequence[float] | pd.Series | np.ndarray,
    timestamps: Sequence[Any] | pd.Series | pd.Index | None = None,
    *,
    periods_per_year: float | None = None,
    calendar: str = "crypto_365",
    rolling_window: int = 30,
) -> dict[str, Any]:
    equity_arr = np.asarray(list(equity), dtype=float)
    equity_arr = equity_arr[np.isfinite(equity_arr)]
    if equity_arr.size == 0:
        return {
            "period_sharpe": 0.0,
            "sortino": 0.0,
            "rolling_sharpe": 0.0,
            "rolling_sortino": 0.0,
            "volatility_annualized": 0.0,
            "downside_volatility": 0.0,
            "cagr": 0.0,
            "cagr_pct": 0.0,
            "calmar": 0.0,
            "return_pct": 0.0,
            "VaR_95": 0.0,
            "CVaR_95": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            **drawdown_details([]),
        }

    if timestamps is None:
        ts_values = list(range(len(equity_arr)))
    else:
        ts_values = list(timestamps)
        if len(ts_values) != len(equity_arr):
            ts_values = list(range(len(equity_arr)))

    ppy = float(periods_per_year) if periods_per_year is not None else infer_periods_per_year(ts_values, calendar=calendar)
    equity_s = pd.Series(equity_arr)
    returns = equity_s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    returns = returns[np.isfinite(returns)]

    total_return = float(equity_arr[-1] / equity_arr[0] - 1.0) if equity_arr[0] > EPS else 0.0

    start_ts = _safe_timestamp(ts_values[0]) if ts_values else None
    end_ts = _safe_timestamp(ts_values[-1]) if ts_values else None
    if start_ts is not None and end_ts is not None:
        years = max(float((end_ts - start_ts).total_seconds() / (365.0 * 86400.0)), 1.0 / 365.0)
    else:
        years = max(float(len(equity_arr) - 1) / max(ppy, 1.0), 1.0 / 365.0)

    if equity_arr[0] > EPS and equity_arr[-1] > EPS:
        cagr = math.exp(max(min(math.log(equity_arr[-1] / equity_arr[0]) / max(years, EPS), 700), -700)) - 1.0
    else:
        cagr = 0.0

    if returns.empty:
        period_sharpe = 0.0
        sortino = 0.0
        volatility = 0.0
        downside_vol = 0.0
        rolling_sharpe = 0.0
        rolling_sortino = 0.0
        var_95 = 0.0
        cvar_95 = 0.0
        skew = 0.0
        kurtosis = 0.0
    else:
        mean_r = float(returns.mean())
        std_r = float(returns.std(ddof=0))
        downside = returns[returns < 0.0]
        downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
        period_sharpe = (mean_r / std_r) * math.sqrt(ppy) if std_r > EPS else 0.0
        sortino = (mean_r / downside_std) * math.sqrt(ppy) if downside_std > EPS else 0.0
        volatility = std_r * math.sqrt(ppy)
        downside_vol = downside_std * math.sqrt(ppy)

        window = max(int(rolling_window), 2)
        roll_mean = returns.rolling(window).mean()
        roll_std = returns.rolling(window).std(ddof=0)
        roll_downside_std = returns.where(returns < 0.0, 0.0).rolling(window).std(ddof=0)
        rolling_sharpe_s = (roll_mean / roll_std.replace(0.0, np.nan)) * math.sqrt(ppy)
        rolling_sortino_s = (roll_mean / roll_downside_std.replace(0.0, np.nan)) * math.sqrt(ppy)
        rolling_sharpe = _finite_float(rolling_sharpe_s.dropna().iloc[-1] if not rolling_sharpe_s.dropna().empty else 0.0)
        rolling_sortino = _finite_float(rolling_sortino_s.dropna().iloc[-1] if not rolling_sortino_s.dropna().empty else 0.0)

        var_95 = float(returns.quantile(0.05))
        tail = returns[returns <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) else var_95
        skew = _finite_float(returns.skew())
        kurtosis = _finite_float(returns.kurtosis())

    dd = drawdown_details(equity_arr, ts_values)
    max_dd = float(dd["max_drawdown_pct"]) / 100.0
    calmar = (cagr / max_dd) if max_dd > EPS else 0.0

    return {
        "period_sharpe": _round(period_sharpe, 6),
        "sortino": _round(sortino, 6),
        "rolling_sharpe": _round(rolling_sharpe, 6),
        "rolling_sortino": _round(rolling_sortino, 6),
        "volatility_annualized": _round(volatility * 100.0, 6),
        "downside_volatility": _round(downside_vol * 100.0, 6),
        "cagr": _round(cagr, 8),
        "cagr_pct": _round(cagr * 100.0, 6),
        "calmar": _round(calmar, 6),
        "return_pct": _round(total_return * 100.0, 6),
        "VaR_95": _round(var_95 * 100.0, 6),
        "CVaR_95": _round(cvar_95 * 100.0, 6),
        "skew": _round(skew, 6),
        "kurtosis": _round(kurtosis, 6),
        **dd,
    }


def compute_trade_metrics(
    trades: list[dict],
    *,
    total_days: float,
    equity_points: int | None = None,
) -> dict[str, Any]:
    if not trades:
        return {
            "trade_sharpe": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "median_trade_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_holding_bars": 0.0,
            "avg_holding_time": 0.0,
            "time_in_market_pct": 0.0,
            "turnover": 0.0,
            "fee_drag": 0.0,
            "slippage_drag": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "num_trades": 0,
            "trades_per_year": 0.0,
        }

    account_returns = np.asarray([_finite_float(t.get("account_return_pct")) / 100.0 for t in trades], dtype=float)
    trade_returns_pct = np.asarray([_finite_float(t.get("pnl_pct")) for t in trades], dtype=float)
    pnl_cash = np.asarray([_finite_float(t.get("pnl_cash")) for t in trades], dtype=float)
    wins = pnl_cash[pnl_cash > 0.0]
    losses = pnl_cash[pnl_cash < 0.0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    trades_per_year = (len(trades) / max(float(total_days), 1.0)) * 365.0

    bars = np.asarray([_finite_float(t.get("bars_held")) for t in trades], dtype=float)
    if equity_points is not None and equity_points > 0:
        time_in_market = min(float(bars.sum()) / max(float(equity_points), 1.0), 1.0) * 100.0
    else:
        time_in_market = 0.0

    holding_days: list[float] = []
    for trade in trades:
        entry_ts = _safe_timestamp(trade.get("entry_date"))
        exit_ts = _safe_timestamp(trade.get("exit_date"))
        if entry_ts is not None and exit_ts is not None:
            holding_days.append(max(float((exit_ts - entry_ts).total_seconds() / 86400.0), 0.0))

    fee_drag = float(sum(_finite_float(t.get("fee_cash")) for t in trades))
    slippage_drag = float(sum(_finite_float(t.get("slippage_cash")) for t in trades))
    turnover = float(sum(abs(_finite_float(t.get("allocated_capital"))) for t in trades))

    return {
        "trade_sharpe": _round(compute_trade_sharpe(account_returns, trades_per_year), 6),
        "profit_factor": _round((gross_win / gross_loss) if gross_loss > EPS else (gross_win if gross_win > 0.0 else 0.0), 6),
        "win_rate": _round(float((account_returns > 0.0).mean() * 100.0), 6),
        "avg_trade_return": _round(float(np.mean(trade_returns_pct)), 6),
        "median_trade_return": _round(float(np.median(trade_returns_pct)), 6),
        "best_trade": _round(float(np.max(trade_returns_pct)), 6),
        "worst_trade": _round(float(np.min(trade_returns_pct)), 6),
        "avg_holding_bars": _round(float(np.mean(bars)), 6),
        "avg_holding_time": _round(float(np.mean(holding_days)), 6) if holding_days else 0.0,
        "time_in_market_pct": _round(time_in_market, 6),
        "turnover": _round(turnover, 6),
        "fee_drag": _round(fee_drag, 6),
        "slippage_drag": _round(slippage_drag, 6),
        "gross_pnl": _round(gross_win, 6),
        "net_pnl": _round(float(pnl_cash.sum()), 6),
        "num_trades": int(len(trades)),
        "trades_per_year": _round(trades_per_year, 6),
    }
