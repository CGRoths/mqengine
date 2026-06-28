from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .risk import compute_period_metrics
from .validation import validate_ohlc


@dataclass
class PortfolioResult:
    name: str
    equity: pd.DataFrame
    positions: pd.DataFrame
    metrics: dict[str, Any]
    attribution: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        equity_df = self.equity.copy()
        if "ts" in equity_df.columns:
            equity_df["ts"] = pd.to_datetime(equity_df["ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        positions_df = self.positions.copy()
        if "ts" in positions_df.columns:
            positions_df["ts"] = pd.to_datetime(positions_df["ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "name": self.name,
            "metrics": self.metrics,
            "equity": equity_df.to_dict(orient="records"),
            "positions": positions_df.to_dict(orient="records"),
            "attribution": self.attribution,
            "meta": self.meta,
        }


class PortfolioBacktester:
    def __init__(
        self,
        *,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        weights: pd.DataFrame | None = None,
        initial_capital: float = 10000.0,
        fee_pct: float = 0.0,
        calendar: str = "crypto_365",
        name: str = "portfolio",
    ):
        self.prices = prices.copy()
        self.signals = signals.copy()
        self.weights = weights.copy() if weights is not None else pd.DataFrame(columns=["strategy_id", "weight"])
        self.initial_capital = float(initial_capital)
        self.fee_pct = float(fee_pct)
        self.calendar = calendar
        self.name = name

    def _prepare(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        required_price_cols = {"ts", "symbol", "open", "high", "low", "close"}
        missing_price = required_price_cols.difference(self.prices.columns)
        if missing_price:
            raise ValueError(f"prices is missing required columns: {sorted(missing_price)}")
        required_signal_cols = {"ts", "strategy_id", "symbol"}
        missing_signal = required_signal_cols.difference(self.signals.columns)
        if missing_signal:
            raise ValueError(f"signals is missing required columns: {sorted(missing_signal)}")

        for _, symbol_df in self.prices.groupby("symbol"):
            validate_ohlc(symbol_df.sort_values("ts"), allow_empty=False).raise_if_errors()

        prices = self.prices.copy()
        prices["ts"] = pd.to_datetime(prices["ts"])
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
        prices = prices.dropna(subset=["ts", "symbol", "close"]).sort_values(["ts", "symbol"])
        close = prices.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last").sort_index().ffill()

        signals = self.signals.copy()
        signals["ts"] = pd.to_datetime(signals["ts"])
        if "target_weight" not in signals.columns:
            if "signal_value" not in signals.columns:
                raise ValueError("signals must include target_weight or signal_value.")
            signals["target_weight"] = pd.to_numeric(signals["signal_value"], errors="coerce")
        signals["target_weight"] = pd.to_numeric(signals["target_weight"], errors="coerce").fillna(0.0)

        if self.weights.empty:
            strategy_weights = pd.Series(1.0, index=signals["strategy_id"].drop_duplicates())
        else:
            strategy_weights = self.weights.set_index("strategy_id")["weight"].astype(float)
        signals["strategy_weight"] = signals["strategy_id"].map(strategy_weights).fillna(0.0)
        signals["weighted_target"] = signals["target_weight"] * signals["strategy_weight"]
        targets = (
            signals.groupby(["ts", "symbol"], as_index=False)["weighted_target"].sum()
            .pivot_table(index="ts", columns="symbol", values="weighted_target", aggfunc="last")
            .sort_index()
        )
        targets = targets.reindex(close.index).ffill().fillna(0.0)
        targets = targets.reindex(columns=close.columns).fillna(0.0)
        return close, targets

    def run(self) -> PortfolioResult:
        close, targets = self._prepare()
        if close.empty:
            empty = pd.DataFrame(columns=["ts", "equity", "cash", "gross_exposure", "net_exposure", "leverage", "turnover"])
            return PortfolioResult(self.name, empty, pd.DataFrame(), {}, {}, {"calendar": self.calendar})

        symbols = list(close.columns)
        cash = float(self.initial_capital)
        positions = pd.Series(0.0, index=symbols)
        last_equity = float(self.initial_capital)
        rows: list[dict[str, Any]] = []
        pos_rows: list[dict[str, Any]] = []
        fees_by_symbol = {symbol: 0.0 for symbol in symbols}
        pnl_by_symbol = {symbol: 0.0 for symbol in symbols}
        prev_prices: pd.Series | None = None
        rebalance_count = 0

        for ts, prices in close.iterrows():
            prices = prices.astype(float)
            if prev_prices is not None:
                pnl = positions * (prices - prev_prices)
                for symbol in symbols:
                    pnl_by_symbol[symbol] += float(pnl[symbol])

            pre_trade_equity = cash + float((positions * prices).sum())
            target_weights = targets.loc[ts].astype(float).reindex(symbols).fillna(0.0)
            desired_value = target_weights * pre_trade_equity
            desired_positions = desired_value / prices.replace(0.0, np.nan)
            desired_positions = desired_positions.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            delta = desired_positions - positions
            trade_value = (delta.abs() * prices).fillna(0.0)
            fee = trade_value * self.fee_pct

            if float(trade_value.sum()) > 1e-12:
                rebalance_count += 1

            cash -= float((delta * prices).sum())
            cash -= float(fee.sum())
            for symbol in symbols:
                fees_by_symbol[symbol] += float(fee[symbol])
            positions = desired_positions
            equity = cash + float((positions * prices).sum())
            exposure_values = positions * prices
            gross_exposure = float(exposure_values.abs().sum())
            net_exposure = float(exposure_values.sum())
            leverage = gross_exposure / equity if equity > 0.0 else 0.0
            turnover = float(trade_value.sum()) / max(last_equity, 1e-12)
            drift = (exposure_values / max(equity, 1e-12) - target_weights).abs()

            rows.append({
                "ts": pd.Timestamp(ts),
                "equity": float(equity),
                "cash": float(cash),
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "leverage": float(leverage),
                "turnover": turnover,
                "rebalance_count": rebalance_count,
                "drift_from_target": float(drift.mean()),
            })
            for symbol in symbols:
                actual_weight = float(exposure_values[symbol] / equity) if equity > 0.0 else 0.0
                pos_rows.append({
                    "ts": pd.Timestamp(ts),
                    "symbol": symbol,
                    "position": float(positions[symbol]),
                    "price": float(prices[symbol]),
                    "target_weight": float(target_weights[symbol]),
                    "actual_weight": actual_weight,
                    "pnl_by_symbol": float(pnl_by_symbol[symbol]),
                    "fee_by_symbol": float(fees_by_symbol[symbol]),
                })

            last_equity = float(equity)
            prev_prices = prices

        equity_df = pd.DataFrame(rows)
        positions_df = pd.DataFrame(pos_rows)
        metrics = compute_period_metrics(equity_df["equity"], equity_df["ts"], calendar=self.calendar)
        metrics.update({
            "gross_exposure": round(float(equity_df["gross_exposure"].iloc[-1]), 6),
            "net_exposure": round(float(equity_df["net_exposure"].iloc[-1]), 6),
            "leverage": round(float(equity_df["leverage"].iloc[-1]), 6),
            "turnover": round(float(equity_df["turnover"].sum()), 6),
            "rebalance_count": int(rebalance_count),
            "drift_from_target": round(float(equity_df["drift_from_target"].mean()), 6),
        })
        attribution = {
            "pnl_by_symbol": {k: round(float(v), 6) for k, v in pnl_by_symbol.items()},
            "fee_by_symbol": {k: round(float(v), 6) for k, v in fees_by_symbol.items()},
            "pnl_by_strategy": {},
            "per_strategy_attribution": {},
            "per_symbol_attribution": {k: {"pnl": round(float(pnl_by_symbol[k]), 6), "fee": round(float(fees_by_symbol[k]), 6)} for k in symbols},
        }
        return PortfolioResult(
            name=self.name,
            equity=equity_df,
            positions=positions_df,
            metrics=metrics,
            attribution=attribution,
            meta={"calendar": self.calendar, "initial_capital": self.initial_capital, "fee_pct": self.fee_pct},
        )
