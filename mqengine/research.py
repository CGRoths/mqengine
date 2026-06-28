from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict, replace
from typing import Any

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .risk import compute_period_metrics


@dataclass(slots=True)
class ResearchProtocol:
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    in_sample: tuple[str | pd.Timestamp, str | pd.Timestamp] | None = None
    out_sample: tuple[str | pd.Timestamp, str | pd.Timestamp] | None = None
    calendar: str = "crypto_365"
    periods_per_year: float | None = None
    min_oos_trades: int = 5
    sharpe_decay_warn: float = 0.50
    return_decay_warn: float = 0.50
    drawdown_expansion_warn: float = 1.50

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ["start", "end"]:
            if out[key] is not None:
                out[key] = pd.Timestamp(out[key]).strftime("%Y-%m-%d %H:%M:%S")
        for key in ["in_sample", "out_sample"]:
            if out[key] is not None:
                out[key] = tuple(pd.Timestamp(v).strftime("%Y-%m-%d %H:%M:%S") for v in out[key])
        return out


@dataclass(slots=True)
class WalkForwardResult:
    folds: list[dict[str, Any]]
    combined_oos_metrics: dict[str, Any]
    best_params_by_fold: list[dict[str, Any]]
    stability: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": self.folds,
            "combined_oos_metrics": self.combined_oos_metrics,
            "best_params_by_fold": self.best_params_by_fold,
            "stability": self.stability,
            "meta": self.meta,
        }


def research_protocol(**kwargs: Any) -> ResearchProtocol:
    return ResearchProtocol(**kwargs)


def _range_tuple(range_value: tuple[Any, Any] | None, fallback_start: Any, fallback_end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    if range_value is None:
        return pd.Timestamp(fallback_start), pd.Timestamp(fallback_end)
    return pd.Timestamp(range_value[0]), pd.Timestamp(range_value[1])


def _filter_trades_by_exit(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    out = []
    for trade in trades:
        exit_ts = pd.to_datetime(trade.get("exit_date"), errors="coerce")
        if pd.notna(exit_ts) and start <= pd.Timestamp(exit_ts) <= end:
            out.append(deepcopy(trade))
    return out


def metrics_for_result_range(result, start: Any, end: Any, *, calendar: str = "crypto_365", periods_per_year: float | None = None) -> dict[str, Any]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    data = result.data.copy()
    data["ts"] = pd.to_datetime(data["ts"])
    subset = data[(data["ts"] >= start_ts) & (data["ts"] <= end_ts)].copy()
    trades = _filter_trades_by_exit(result.trades, start_ts, end_ts)

    if subset.empty:
        return compute_metrics(
            equity=np.asarray([], dtype=float),
            trades=[],
            start_ts=start_ts,
            end_ts=end_ts,
            timestamps=[],
            periods_per_year=periods_per_year,
            calendar=calendar,
        )

    return compute_metrics(
        equity=subset["equity"].to_numpy(dtype=float),
        trades=trades,
        start_ts=pd.Timestamp(subset["ts"].iloc[0]),
        end_ts=pd.Timestamp(subset["ts"].iloc[-1]),
        timestamps=subset["ts"],
        periods_per_year=periods_per_year,
        calendar=calendar,
    )


def _decay(is_value: float, oos_value: float) -> float:
    if abs(is_value) <= 1e-12:
        return 0.0 if abs(oos_value) <= 1e-12 else -1.0
    return float((is_value - oos_value) / abs(is_value))


def evaluate_is_oos_validation(
    in_sample_metrics: dict[str, Any],
    out_sample_metrics: dict[str, Any],
    protocol: ResearchProtocol,
) -> dict[str, Any]:
    is_sharpe = float(in_sample_metrics.get("period_sharpe", in_sample_metrics.get("sharpe", 0.0)) or 0.0)
    oos_sharpe = float(out_sample_metrics.get("period_sharpe", out_sample_metrics.get("sharpe", 0.0)) or 0.0)
    is_return = float(in_sample_metrics.get("return_pct", 0.0) or 0.0)
    oos_return = float(out_sample_metrics.get("return_pct", 0.0) or 0.0)
    is_mdd = float(in_sample_metrics.get("max_drawdown_pct", in_sample_metrics.get("max_drawdown", 0.0)) or 0.0)
    oos_mdd = float(out_sample_metrics.get("max_drawdown_pct", out_sample_metrics.get("max_drawdown", 0.0)) or 0.0)

    sharpe_decay = _decay(is_sharpe, oos_sharpe)
    return_decay = _decay(is_return, oos_return)
    mdd_expansion = (oos_mdd / max(is_mdd, 1e-12)) - 1.0 if oos_mdd > 0.0 else 0.0

    warnings: list[str] = []
    if int(out_sample_metrics.get("num_trades", 0) or 0) < protocol.min_oos_trades:
        warnings.append("oos_trades_too_few")
    if is_sharpe > 0.0 and sharpe_decay > protocol.sharpe_decay_warn:
        warnings.append("oos_sharpe_drops_too_much")
    if is_return > 0.0 and return_decay > protocol.return_decay_warn:
        warnings.append("oos_return_collapses")
    if is_mdd > 0.0 and mdd_expansion > protocol.drawdown_expansion_warn:
        warnings.append("oos_drawdown_much_worse_than_is")

    score = 100.0
    score -= max(sharpe_decay, 0.0) * 35.0
    score -= max(return_decay, 0.0) * 30.0
    score -= max(mdd_expansion, 0.0) * 15.0
    score -= max(protocol.min_oos_trades - int(out_sample_metrics.get("num_trades", 0) or 0), 0) * 3.0
    score = max(min(score, 100.0), 0.0)

    return {
        "oos_sharpe_decay": round(float(sharpe_decay), 6),
        "oos_return_decay": round(float(return_decay), 6),
        "oos_mdd_expansion": round(float(mdd_expansion), 6),
        "is_oos_consistency_score": round(float(score), 3),
        "parameter_stability_score": None,
        "oos_pass": len(warnings) == 0,
        "warnings": warnings,
    }


def apply_research_protocol(result, protocol: ResearchProtocol):
    data = result.data.copy()
    data["ts"] = pd.to_datetime(data["ts"])
    if data.empty:
        start_ts = pd.Timestamp(protocol.start or pd.Timestamp.utcnow())
        end_ts = pd.Timestamp(protocol.end or start_ts)
    else:
        start_ts = pd.Timestamp(protocol.start) if protocol.start is not None else pd.Timestamp(data["ts"].iloc[0])
        end_ts = pd.Timestamp(protocol.end) if protocol.end is not None else pd.Timestamp(data["ts"].iloc[-1])

    full_metrics = metrics_for_result_range(
        result,
        start_ts,
        end_ts,
        calendar=protocol.calendar,
        periods_per_year=protocol.periods_per_year,
    )
    is_start, is_end = _range_tuple(protocol.in_sample, start_ts, end_ts)
    oos_start, oos_end = _range_tuple(protocol.out_sample, is_end, end_ts)
    is_metrics = metrics_for_result_range(
        result,
        is_start,
        is_end,
        calendar=protocol.calendar,
        periods_per_year=protocol.periods_per_year,
    )
    oos_metrics = metrics_for_result_range(
        result,
        oos_start,
        oos_end,
        calendar=protocol.calendar,
        periods_per_year=protocol.periods_per_year,
    )
    validation = evaluate_is_oos_validation(is_metrics, oos_metrics, protocol)
    metrics = {
        "full": full_metrics,
        "in_sample": is_metrics,
        "out_sample": oos_metrics,
        "validation": validation,
    }
    research = {
        "protocol": protocol.to_dict(),
        "in_sample_range": (is_start.strftime("%Y-%m-%d %H:%M:%S"), is_end.strftime("%Y-%m-%d %H:%M:%S")),
        "out_sample_range": (oos_start.strftime("%Y-%m-%d %H:%M:%S"), oos_end.strftime("%Y-%m-%d %H:%M:%S")),
    }
    return replace(result, metrics=metrics, validation=validation, research=research)


def flatten_research_metrics(result) -> dict[str, Any]:
    if not isinstance(result.metrics, dict) or "full" not in result.metrics:
        return dict(result.metrics)

    row = dict(result.metrics["full"])
    for prefix in ["in_sample", "out_sample"]:
        for key, value in result.metrics.get(prefix, {}).items():
            row[f"{prefix}_{key}"] = value
    validation = result.metrics.get("validation", {})
    for key, value in validation.items():
        if key != "warnings":
            row[f"validation_{key}"] = value
    row["validation_warnings"] = ",".join(validation.get("warnings", []))
    return row


def _slice_series(series: pd.Series | None, start: pd.Timestamp, end: pd.Timestamp, *, inclusive_end: bool) -> pd.Series | None:
    if series is None:
        return None
    idx = pd.to_datetime(series.index)
    if inclusive_end:
        mask = (idx >= start) & (idx <= end)
    else:
        mask = (idx >= start) & (idx < end)
    return series.loc[mask].copy()


def slice_runner(runner, start: Any, end: Any, *, inclusive_end: bool = True):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    new = runner.clone()
    new._raw_price = _slice_series(new._raw_price, start_ts, end_ts, inclusive_end=inclusive_end)
    new._raw_signal = _slice_series(new._raw_signal, start_ts, end_ts, inclusive_end=inclusive_end)
    new._raw_benchmark = _slice_series(new._raw_benchmark, start_ts, end_ts, inclusive_end=inclusive_end)
    new._raw_open = _slice_series(new._raw_open, start_ts, end_ts, inclusive_end=inclusive_end)
    new._raw_high = _slice_series(new._raw_high, start_ts, end_ts, inclusive_end=inclusive_end)
    new._raw_low = _slice_series(new._raw_low, start_ts, end_ts, inclusive_end=inclusive_end)
    return new


def _objective_from_row(row: pd.Series, objective: str) -> float:
    if objective in row:
        return float(row[objective])
    if f"full_{objective}" in row:
        return float(row[f"full_{objective}"])
    return float(row.get("sharpe", 0.0))


def _runner_time_bounds(runner) -> tuple[pd.Timestamp, pd.Timestamp]:
    idx = pd.to_datetime(runner._raw_price.index)
    return pd.Timestamp(idx.min()), pd.Timestamp(idx.max())


def run_walkforward(
    runner_or_sweep,
    *,
    train_window: str | pd.Timedelta = "730D",
    test_window: str | pd.Timedelta = "180D",
    step: str | pd.Timedelta = "180D",
    objective: str = "period_sharpe",
    calendar: str = "crypto_365",
) -> WalkForwardResult:
    from .sweep import SweepRunner

    train_delta = pd.Timedelta(train_window)
    test_delta = pd.Timedelta(test_window)
    step_delta = pd.Timedelta(step)
    base_runner = runner_or_sweep._base_runner if isinstance(runner_or_sweep, SweepRunner) else runner_or_sweep
    data_start, data_end = _runner_time_bounds(base_runner)

    folds: list[dict[str, Any]] = []
    best_params_by_fold: list[dict[str, Any]] = []
    oos_equity_paths: list[np.ndarray] = []
    oos_timestamps: list[pd.Timestamp] = []
    fold_start = data_start
    fold_index = 0

    while True:
        train_start = fold_start
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta
        if test_start > data_end:
            break
        if train_end <= train_start or test_end <= test_start:
            break

        if isinstance(runner_or_sweep, SweepRunner):
            train_base = slice_runner(runner_or_sweep._base_runner, train_start, train_end, inclusive_end=False)
            train_sweep = SweepRunner(train_base, name=runner_or_sweep._name)
            train_sweep._grid = deepcopy(runner_or_sweep._grid)
            train_sweep._builder = runner_or_sweep._builder
            train_result = train_sweep.run()
            if train_result.results_df.empty:
                fold_start = fold_start + step_delta
                continue
            scored = train_result.results_df.copy()
            scored["_objective"] = scored.apply(lambda row: _objective_from_row(row, objective), axis=1)
            best_row = scored.sort_values(["_objective", "return_pct"], ascending=[False, False]).iloc[0]
            best_params = {col: best_row[col] for col in runner_or_sweep.param_columns if col in best_row}
            best_strategy_id = str(best_row["strategy_id"])

            test_runner = slice_runner(runner_or_sweep._base_runner, test_start, min(test_end, data_end + pd.Timedelta("1ns")), inclusive_end=False)
            test_runner.params(**best_params)
            if runner_or_sweep._builder is not None:
                runner_or_sweep._builder(test_runner, **best_params)
            test_result = test_runner.run()
            train_match = next((r for r in train_result.strategy_results if r.strategy_id == best_strategy_id), None)
            train_metrics = train_match.metrics if train_match is not None else {}
        else:
            best_params = deepcopy(base_runner._config.params)
            train_result = slice_runner(base_runner, train_start, train_end, inclusive_end=False).run()
            test_result = slice_runner(base_runner, test_start, min(test_end, data_end + pd.Timedelta("1ns")), inclusive_end=False).run()
            train_metrics = train_result.metrics

        test_metrics = test_result.metrics
        fold = {
            "fold": fold_index,
            "train_start": train_start.strftime("%Y-%m-%d %H:%M:%S"),
            "train_end": train_end.strftime("%Y-%m-%d %H:%M:%S"),
            "test_start": test_start.strftime("%Y-%m-%d %H:%M:%S"),
            "test_end": min(test_end, data_end).strftime("%Y-%m-%d %H:%M:%S"),
            "best_params": best_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        folds.append(fold)
        best_params_by_fold.append(best_params)

        if not test_result.data.empty and "equity" in test_result.data.columns:
            eq = test_result.data["equity"].to_numpy(dtype=float)
            if len(eq) > 1 and eq[0] > 0.0:
                returns = pd.Series(eq).pct_change().fillna(0.0).to_numpy(dtype=float)
                oos_equity_paths.append(returns)
                oos_timestamps.extend(pd.to_datetime(test_result.data["ts"]).tolist())

        fold_index += 1
        fold_start = fold_start + step_delta

    if oos_equity_paths:
        capital = float(base_runner._engine.initial_capital)
        combined: list[float] = []
        for returns in oos_equity_paths:
            path = capital * np.cumprod(1.0 + returns)
            combined.extend(path.tolist())
            capital = float(path[-1])
        combined_oos_metrics = compute_period_metrics(combined, oos_timestamps[: len(combined)], calendar=calendar)
    else:
        combined_oos_metrics = compute_period_metrics([], [], calendar=calendar)

    stability = {
        "fold_count": len(folds),
        "unique_best_param_sets": len({tuple(sorted(params.items())) for params in best_params_by_fold}),
    }
    if best_params_by_fold:
        stability["param_reuse_ratio"] = round(1.0 - (stability["unique_best_param_sets"] - 1) / max(len(best_params_by_fold), 1), 6)
    else:
        stability["param_reuse_ratio"] = 0.0

    return WalkForwardResult(
        folds=folds,
        combined_oos_metrics=combined_oos_metrics,
        best_params_by_fold=best_params_by_fold,
        stability=stability,
        meta={
            "train_window": str(train_delta),
            "test_window": str(test_delta),
            "step": str(step_delta),
            "objective": objective,
        },
    )
