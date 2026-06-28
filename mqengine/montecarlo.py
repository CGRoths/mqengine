from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .risk import compute_period_metrics


def _returns_from_trades(trades_or_returns: list[dict] | Iterable[float]) -> np.ndarray:
    values = list(trades_or_returns)
    if not values:
        return np.asarray([], dtype=float)
    first = values[0]
    if isinstance(first, dict):
        return np.asarray([float(t.get("account_return_pct", t.get("pnl_pct", 0.0))) / 100.0 for t in values], dtype=float)
    return np.asarray(values, dtype=float)


def simulate_trade_paths(
    trades_or_returns: list[dict] | Iterable[float],
    *,
    n_sims: int = 1000,
    seed: int | None = None,
    method: str = "bootstrap",
    initial_equity: float = 1.0,
) -> np.ndarray:
    returns = _returns_from_trades(trades_or_returns)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return np.empty((0, 0), dtype=float)
    rng = np.random.default_rng(seed)
    n = len(returns)
    sims = np.empty((int(n_sims), n), dtype=float)
    for i in range(int(n_sims)):
        if method == "shuffle":
            sampled = rng.permutation(returns)
        elif method == "bootstrap":
            sampled = rng.choice(returns, size=n, replace=True)
        else:
            raise ValueError("method must be 'bootstrap' or 'shuffle'.")
        sims[i, :] = float(initial_equity) * np.cumprod(1.0 + sampled)
    return sims


def _max_drawdown(path: np.ndarray) -> float:
    peaks = np.maximum.accumulate(path)
    drawdowns = np.where(peaks > 0.0, (peaks - path) / peaks, 0.0)
    return float(np.max(drawdowns)) if len(drawdowns) else 0.0


def monte_carlo_trade_robustness(
    trades_or_returns: list[dict] | Iterable[float],
    *,
    n_sims: int = 1000,
    seed: int | None = None,
    method: str = "bootstrap",
    initial_equity: float = 1.0,
) -> dict[str, Any]:
    paths = simulate_trade_paths(
        trades_or_returns,
        n_sims=n_sims,
        seed=seed,
        method=method,
        initial_equity=initial_equity,
    )
    if paths.size == 0:
        return {
            "n_sims": int(n_sims),
            "method": method,
            "probability_of_loss": 0.0,
            "probability_sharpe_below_zero": 0.0,
            "expected_worst_drawdown": 0.0,
            "terminal_equity_p05": float(initial_equity),
            "terminal_equity_p50": float(initial_equity),
            "terminal_equity_p95": float(initial_equity),
            "drawdown_p95": 0.0,
        }

    terminal = paths[:, -1]
    max_drawdowns = np.asarray([_max_drawdown(path) for path in paths], dtype=float)
    sharpes = []
    for path in paths:
        metrics = compute_period_metrics(path, range(len(path)))
        sharpes.append(float(metrics["period_sharpe"]))
    sharpes_arr = np.asarray(sharpes, dtype=float)

    return {
        "n_sims": int(n_sims),
        "method": method,
        "probability_of_loss": round(float((terminal < initial_equity).mean()), 6),
        "probability_sharpe_below_zero": round(float((sharpes_arr < 0.0).mean()), 6),
        "expected_worst_drawdown": round(float(max_drawdowns.mean() * 100.0), 6),
        "terminal_equity_p05": round(float(np.quantile(terminal, 0.05)), 6),
        "terminal_equity_p50": round(float(np.quantile(terminal, 0.50)), 6),
        "terminal_equity_p95": round(float(np.quantile(terminal, 0.95)), 6),
        "drawdown_p95": round(float(np.quantile(max_drawdowns, 0.95) * 100.0), 6),
    }
