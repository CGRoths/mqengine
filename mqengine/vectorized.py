from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .result import BacktestResult


def _required_frame(df: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    out = df[columns].copy()
    out[columns[0]] = pd.to_datetime(out[columns[0]], errors="coerce")
    out = out.dropna(subset=[columns[0]]).sort_values(columns[0]).reset_index(drop=True)
    return out


def run_vectorized_signal_backtest(
    price_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    price_col: str = "close",
    signal_col: str = "signal",
    fee_pct: float = 0.0,
    initial_capital: float = 100.0,
    signal_lag: int = 1,
    calendar: str = "crypto_365",
) -> BacktestResult:
    """
    Fast signal-return evaluator for alpha research.

    This path uses lagged target signals and close-to-close returns. It does not
    replace MQEngine's OHLC event execution model.
    """
    lag = int(signal_lag)
    if lag < 0:
        raise ValueError("signal_lag must be non-negative")

    price = _required_frame(price_df, [ts_col, price_col], "price_df")
    price[price_col] = pd.to_numeric(price[price_col], errors="coerce")
    price = price.dropna(subset=[price_col]).reset_index(drop=True)

    if price.empty:
        now = pd.Timestamp.utcnow()
        metrics = compute_metrics(np.asarray([], dtype=float), [], now, now, timestamps=[], calendar=calendar)
        return BacktestResult(
            name="vectorized_signal_return",
            strategy_id="vectorized_signal_return",
            params={
                "fee_pct": float(fee_pct),
                "initial_capital": float(initial_capital),
                "signal_lag": lag,
                "calendar": calendar,
            },
            data=pd.DataFrame(columns=[ts_col, price_col, signal_col, "prev_signal", "returns", "trade_delta", "pnl_return", "equity"]),
            trades=[],
            metrics=metrics,
            price_col=price_col,
            signal_col=signal_col,
            meta={"execution_mode": "vectorized_signal_return", "calendar": calendar},
        )

    signal = _required_frame(signal_df, [ts_col, signal_col], "signal_df")
    signal[signal_col] = pd.to_numeric(signal[signal_col], errors="coerce")
    signal = signal.dropna(subset=[signal_col]).reset_index(drop=True)

    if signal.empty:
        aligned = price.copy()
        aligned[signal_col] = 0.0
    else:
        aligned = pd.merge_asof(price, signal, on=ts_col, direction="backward")
        aligned[signal_col] = pd.to_numeric(aligned[signal_col], errors="coerce").ffill().fillna(0.0)

    aligned["prev_signal"] = aligned[signal_col].shift(lag).fillna(0.0)
    aligned["returns"] = aligned[price_col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    aligned["trade_delta"] = aligned[signal_col] - aligned["prev_signal"]
    aligned["pnl_return"] = aligned["prev_signal"] * aligned["returns"] - aligned["trade_delta"].abs() * float(fee_pct)
    aligned["equity"] = float(initial_capital) * (1.0 + aligned["pnl_return"]).cumprod()

    metric_equity = np.concatenate([[float(initial_capital)], aligned["equity"].to_numpy(dtype=float)])
    first_ts = pd.Timestamp(aligned[ts_col].iloc[0])
    metric_timestamps: list[Any] = [first_ts] + pd.to_datetime(aligned[ts_col]).tolist()
    pnl_returns = aligned["pnl_return"].to_numpy(dtype=float)
    metrics = compute_metrics(
        metric_equity,
        trades=[],
        start_ts=first_ts,
        end_ts=pd.Timestamp(aligned[ts_col].iloc[-1]),
        timestamps=metric_timestamps,
        pnl=pnl_returns,
        calendar=calendar,
    )

    turnover = float(aligned["trade_delta"].abs().sum())
    fee_drag_return = float(aligned["trade_delta"].abs().mul(float(fee_pct)).sum())
    metrics.update({
        "vectorized_turnover": round(turnover, 6),
        "vectorized_fee_drag_return": round(fee_drag_return, 8),
    })

    return BacktestResult(
        name="vectorized_signal_return",
        strategy_id="vectorized_signal_return",
        params={
            "fee_pct": float(fee_pct),
            "initial_capital": float(initial_capital),
            "signal_lag": lag,
            "calendar": calendar,
        },
        data=aligned,
        trades=[],
        metrics=metrics,
        price_col=price_col,
        signal_col=signal_col,
        meta={
            "execution_mode": "vectorized_signal_return",
            "calendar": calendar,
            "turnover": round(turnover, 6),
            "fee_drag_return": round(fee_drag_return, 8),
        },
    )
