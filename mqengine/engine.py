from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Optional, Any

import numpy as np
import pandas as pd

from .conditions import BaseCondition, Above, Below, CrossAbove, CrossBelow
from .metrics import compute_metrics
from .result import BacktestResult
from .transforms import transform_registry
from .types import PreparedData, StrategyConfig


def _strategy_id_from_name_and_params(name: str, params: dict[str, Any]) -> str:
    if not params:
        return name.replace(" ", "_")
    parts = [name.replace(" ", "_")]
    for key, value in params.items():
        sval = "NONE" if value is None else str(value)
        sval = sval.replace(" ", "_")
        parts.append(f"{key}-{sval}")
    return "__".join(parts)


def _condition_mask(cond: BaseCondition | None, signal_arr: np.ndarray, signal_series: pd.Series | None = None) -> np.ndarray:
    """
    Fast vectorized condition evaluation.

    The old engine called cond.triggered(series, i) inside the bar loop.
    That repeatedly hit pandas Series.iloc, which is extremely slow for large sweeps.
    This function precomputes boolean masks once per strategy.
    """
    n = len(signal_arr)
    if cond is None:
        return np.zeros(n, dtype=bool)

    if isinstance(cond, Above):
        if cond.inclusive:
            return signal_arr >= cond.level
        return signal_arr > cond.level

    if isinstance(cond, Below):
        if cond.inclusive:
            return signal_arr <= cond.level
        return signal_arr < cond.level

    if isinstance(cond, CrossAbove):
        out = np.zeros(n, dtype=bool)
        if n == 0:
            return out

        if cond.inclusive:
            cur_hit = signal_arr >= cond.level
        else:
            cur_hit = signal_arr > cond.level

        if cond.trigger_on_first_bar:
            out[0] = bool(cur_hit[0])

        if n > 1:
            prev = signal_arr[:-1]
            cur = signal_arr[1:]
            if cond.inclusive:
                out[1:] = (prev < cond.level) & (cur >= cond.level)
            else:
                out[1:] = (prev < cond.level) & (cur > cond.level)

        return out

    if isinstance(cond, CrossBelow):
        out = np.zeros(n, dtype=bool)
        if n == 0:
            return out

        if cond.inclusive:
            cur_hit = signal_arr <= cond.level
        else:
            cur_hit = signal_arr < cond.level

        if cond.trigger_on_first_bar:
            out[0] = bool(cur_hit[0])

        if n > 1:
            prev = signal_arr[:-1]
            cur = signal_arr[1:]
            if cond.inclusive:
                out[1:] = (prev > cond.level) & (cur <= cond.level)
            else:
                out[1:] = (prev > cond.level) & (cur < cond.level)

        return out

    # Fallback for future custom plugin conditions.
    # Slower, but keeps extensibility.
    if signal_series is None:
        signal_series = pd.Series(signal_arr)

    return np.asarray([bool(cond.triggered(signal_series, i)) for i in range(n)], dtype=bool)


class BacktestEngine:
    def __init__(self, initial_capital: float = 100.0):
        self.initial_capital = float(initial_capital)

    def run(self, prepared: PreparedData, config: StrategyConfig) -> BacktestResult:
        """
        Optimized array-based engine.

        Major speed fixes:
        - no df.iloc inside the bar loop
        - no pandas Series.iloc condition calls inside the bar loop
        - entry/exit conditions are precomputed as boolean NumPy masks
        - equity_curve is preallocated as a NumPy array
        """
        df = prepared.df.copy().reset_index(drop=True)

        n = len(df)
        if n == 0:
            empty_metrics = compute_metrics(
                equity=np.asarray([], dtype=float),
                trades=[],
                start_ts=pd.Timestamp.utcnow(),
                end_ts=pd.Timestamp.utcnow(),
            )
            strategy_id = _strategy_id_from_name_and_params(config.name, config.params)
            return BacktestResult(
                name=config.name,
                strategy_id=strategy_id,
                params=deepcopy(config.params),
                data=df,
                trades=[],
                metrics=empty_metrics,
                price_col=prepared.price_col,
                signal_col=prepared.signal_col,
                benchmark_col=None,
                meta={"benchmark_name": "BUY_AND_HOLD"},
            )

        signal_series = df[prepared.signal_col].astype(float)
        signal_arr = signal_series.to_numpy(dtype=float, copy=False)
        close_arr = df[prepared.price_col].astype(float).to_numpy(dtype=float, copy=False)

        if "high" in df.columns:
            high_arr = df["high"].astype(float).to_numpy(dtype=float, copy=False)
        else:
            high_arr = close_arr

        if "low" in df.columns:
            low_arr = df["low"].astype(float).to_numpy(dtype=float, copy=False)
        else:
            low_arr = close_arr

        ts_arr = pd.to_datetime(df["ts"]).to_numpy()

        long_entry_mask = _condition_mask(config.long_entry, signal_arr, signal_series)
        short_entry_mask = _condition_mask(config.short_entry, signal_arr, signal_series)
        long_exit_mask = _condition_mask(config.long_exit, signal_arr, signal_series)
        short_exit_mask = _condition_mask(config.short_exit, signal_arr, signal_series)

        equity_curve = np.empty(n, dtype=float)
        trades: list[dict] = []

        equity = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_idx = 0
        entry_time = None
        entry_signal = None
        allocated_capital = 0.0
        units = 0.0

        fee_pct = float(config.risk.fee_pct)
        slippage_pct = float(config.risk.slippage_pct)
        position_size_pct = float(config.risk.position_size_pct)
        take_profit_pct = config.risk.take_profit_pct
        stop_loss_pct = config.risk.stop_loss_pct
        allow_same_bar_reentry = bool(config.risk.allow_same_bar_reentry)

        for i in range(n):
            ts = pd.Timestamp(ts_arr[i])
            high = float(high_arr[i])
            low = float(low_arr[i])
            close = float(close_arr[i])
            sig = float(signal_arr[i])

            exited_this_bar = False
            exit_reason = None
            exit_price = None

            if position != 0:
                if position == 1:
                    if stop_loss_pct is not None:
                        sl_price = entry_price * (1.0 + float(stop_loss_pct) / 100.0)
                        if low <= sl_price:
                            exit_reason = "stop_loss"
                            exit_price = sl_price

                    if exit_reason is None and take_profit_pct is not None:
                        tp_price = entry_price * (1.0 + float(take_profit_pct) / 100.0)
                        if high >= tp_price:
                            exit_reason = "take_profit"
                            exit_price = tp_price

                    if exit_reason is None and long_exit_mask[i]:
                        exit_reason = "signal_exit"
                        exit_price = close

                elif position == -1:
                    if stop_loss_pct is not None:
                        sl_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
                        if high >= sl_price:
                            exit_reason = "stop_loss"
                            exit_price = sl_price

                    if exit_reason is None and take_profit_pct is not None:
                        tp_price = entry_price * (1.0 - float(take_profit_pct) / 100.0)
                        if low <= tp_price:
                            exit_reason = "take_profit"
                            exit_price = tp_price

                    if exit_reason is None and short_exit_mask[i]:
                        exit_reason = "signal_exit"
                        exit_price = close

                if exit_reason is not None:
                    fee_cash = allocated_capital * fee_pct
                    slip_cash = allocated_capital * slippage_pct
                    pnl_cash = units * (float(exit_price) - entry_price) * position - fee_cash - slip_cash

                    equity_before = equity
                    account_return_pct = ((equity_before + pnl_cash) / equity_before - 1.0) * 100.0 if equity_before > 0 else 0.0
                    pnl_pct_trade = (pnl_cash / allocated_capital) * 100.0 if allocated_capital > 0 else 0.0

                    equity = equity + pnl_cash

                    trades.append({
                        "entry_date": pd.Timestamp(entry_time).strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_date": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "side": "LONG" if position == 1 else "SHORT",
                        "entry_price": round(float(entry_price), 6),
                        "exit_price": round(float(exit_price), 6),
                        "bars_held": int(i - entry_idx),
                        "position_size_pct": round(float(position_size_pct), 6),
                        "allocated_capital": round(float(allocated_capital), 6),
                        "units": round(float(units), 10),
                        "pnl_pct": round(float(pnl_pct_trade), 6),
                        "pnl_cash": round(float(pnl_cash), 6),
                        "account_return_pct": round(float(account_return_pct), 6),
                        "entry_signal": round(float(entry_signal), 6) if entry_signal is not None else None,
                        "exit_signal": round(float(sig), 6),
                        "exit_reason": exit_reason,
                    })

                    position = 0
                    entry_price = 0.0
                    entry_idx = 0
                    entry_time = None
                    entry_signal = None
                    allocated_capital = 0.0
                    units = 0.0
                    exited_this_bar = True

            can_reenter = (not exited_this_bar) or allow_same_bar_reentry

            if position == 0 and can_reenter:
                if long_entry_mask[i]:
                    position = 1
                elif short_entry_mask[i]:
                    position = -1

                if position != 0:
                    entry_price = close
                    entry_idx = i
                    entry_time = ts
                    entry_signal = sig
                    allocated_capital = equity * position_size_pct
                    units = allocated_capital / entry_price if entry_price > 0 else 0.0

                    fee_cash = allocated_capital * fee_pct
                    slip_cash = allocated_capital * slippage_pct
                    equity = equity - fee_cash - slip_cash

            mtm_equity = equity
            if position != 0 and entry_price > 0:
                floating_pnl = units * (close - entry_price) * position
                mtm_equity = equity + floating_pnl

            equity_curve[i] = float(mtm_equity)

        if position != 0:
            final_ts = pd.Timestamp(ts_arr[-1])
            final_price = float(close_arr[-1])
            final_signal = float(signal_arr[-1])

            fee_cash = allocated_capital * fee_pct
            slip_cash = allocated_capital * slippage_pct
            pnl_cash = units * (final_price - entry_price) * position - fee_cash - slip_cash

            equity_before = equity
            account_return_pct = ((equity_before + pnl_cash) / equity_before - 1.0) * 100.0 if equity_before > 0 else 0.0
            pnl_pct_trade = (pnl_cash / allocated_capital) * 100.0 if allocated_capital > 0 else 0.0

            equity = equity + pnl_cash
            equity_curve[-1] = float(equity)

            trades.append({
                "entry_date": pd.Timestamp(entry_time).strftime("%Y-%m-%d %H:%M:%S"),
                "exit_date": final_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "side": "LONG" if position == 1 else "SHORT",
                "entry_price": round(float(entry_price), 6),
                "exit_price": round(float(final_price), 6),
                "bars_held": int(n - 1 - entry_idx),
                "position_size_pct": round(float(position_size_pct), 6),
                "allocated_capital": round(float(allocated_capital), 6),
                "units": round(float(units), 10),
                "pnl_pct": round(float(pnl_pct_trade), 6),
                "pnl_cash": round(float(pnl_cash), 6),
                "account_return_pct": round(float(account_return_pct), 6),
                "entry_signal": round(float(entry_signal), 6) if entry_signal is not None else None,
                "exit_signal": round(float(final_signal), 6),
                "exit_reason": "forced_end",
            })

        df["equity"] = equity_curve.tolist()

        benchmark_col = prepared.benchmark_col
        benchmark_equity_col = None
        if benchmark_col:
            px0 = float(df[benchmark_col].iloc[0])
            benchmark_equity_col = "benchmark_equity"
            df[benchmark_equity_col] = (df[benchmark_col].astype(float) / px0) * self.initial_capital

        metrics = compute_metrics(
            equity=np.asarray(equity_curve, dtype=float),
            trades=trades,
            start_ts=pd.Timestamp(ts_arr[0]),
            end_ts=pd.Timestamp(ts_arr[-1]),
        )

        strategy_id = _strategy_id_from_name_and_params(config.name, config.params)

        return BacktestResult(
            name=config.name,
            strategy_id=strategy_id,
            params=deepcopy(config.params),
            data=df,
            trades=trades,
            metrics=metrics,
            price_col=prepared.price_col,
            signal_col=prepared.signal_col,
            benchmark_col=benchmark_equity_col,
            meta={"benchmark_name": "BUY_AND_HOLD"},
        )


class StrategyRunner:
    def __init__(
        self,
        price: pd.Series,
        signal: pd.Series,
        benchmark: Optional[pd.Series] = None,
        *,
        open_: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        name: str = "strategy",
        initial_capital: float = 100.0,
    ):
        self._raw_price = price.astype(float).copy()
        self._raw_signal = signal.astype(float).copy()
        self._raw_benchmark = benchmark.astype(float).copy() if benchmark is not None else price.astype(float).copy()
        self._raw_open = open_.astype(float).copy() if open_ is not None else price.astype(float).copy()
        self._raw_high = high.astype(float).copy() if high is not None else price.astype(float).copy()
        self._raw_low = low.astype(float).copy() if low is not None else price.astype(float).copy()
        self._engine = BacktestEngine(initial_capital=initial_capital)
        self._signal_pipeline: list[tuple[str, dict]] = []
        self._config = StrategyConfig(name=name)

    def clone(self) -> "StrategyRunner":
        new = StrategyRunner(
            price=self._raw_price.copy(),
            signal=self._raw_signal.copy(),
            benchmark=self._raw_benchmark.copy() if self._raw_benchmark is not None else None,
            open_=self._raw_open.copy(),
            high=self._raw_high.copy(),
            low=self._raw_low.copy(),
            name=self._config.name,
            initial_capital=self._engine.initial_capital,
        )
        new._signal_pipeline = deepcopy(self._signal_pipeline)
        new._config = deepcopy(self._config)
        return new

    def named(self, name: str):
        self._config.name = name
        return self

    def params(self, **kwargs):
        self._config.params.update(kwargs)
        return self

    def with_ohlc(self, *, open_: Optional[pd.Series] = None, high: Optional[pd.Series] = None, low: Optional[pd.Series] = None, close: Optional[pd.Series] = None):
        if close is not None:
            self._raw_price = close.astype(float).copy()
        if open_ is not None:
            self._raw_open = open_.astype(float).copy()
        if high is not None:
            self._raw_high = high.astype(float).copy()
        if low is not None:
            self._raw_low = low.astype(float).copy()
        return self

    def clear_transforms(self):
        self._signal_pipeline = []
        return self

    def norm(self, name: str, **kwargs):
        if name not in transform_registry:
            raise KeyError(f"Unknown transform: {name}")
        self._signal_pipeline.append((name, kwargs))
        return self

    def stand(self, name: str, **kwargs):
        return self.norm(name, **kwargs)

    def entry(self, *, long: BaseCondition | None = None, short: BaseCondition | None = None):
        self._config.long_entry = long
        self._config.short_entry = short
        return self

    def exit(self, *, long: BaseCondition | None = None, short: BaseCondition | None = None):
        self._config.long_exit = long
        self._config.short_exit = short
        return self

    def risk(self, **kwargs):
        self._config.risk = replace(self._config.risk, **kwargs)
        return self

    def prepare(self) -> PreparedData:
        price = self._raw_price.sort_index()
        signal = self._raw_signal.sort_index()
        benchmark = self._raw_benchmark.sort_index() if self._raw_benchmark is not None else None
        open_ = self._raw_open.sort_index()
        high = self._raw_high.sort_index()
        low = self._raw_low.sort_index()

        df = pd.concat([
            price.rename("close"),
            signal.rename("signal_raw"),
            open_.rename("open"),
            high.rename("high"),
            low.rename("low"),
        ], axis=1)

        if benchmark is not None:
            df = pd.concat([df, benchmark.rename("benchmark_price")], axis=1)

        sig = df["signal_raw"].copy()
        last_name = "signal_raw"

        for name, kwargs in self._signal_pipeline:
            sig = transform_registry[name](sig, **kwargs)
            last_name = f"signal_{name}"

        df[last_name] = sig
        df["ts"] = pd.to_datetime(df.index)
        df = df.dropna(subset=[last_name, "close", "open", "high", "low"]).copy()

        return PreparedData(
            df=df.reset_index(drop=True),
            signal_col=last_name,
            price_col="close",
            benchmark_col="benchmark_price" if benchmark is not None else None,
        )

    def run(self) -> BacktestResult:
        prepared = self.prepare()
        return self._engine.run(prepared, self._config)
