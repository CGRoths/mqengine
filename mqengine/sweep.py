from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Optional
import pandas as pd

from .engine import StrategyRunner
from .research import ResearchProtocol, flatten_research_metrics
from .result import SweepResult
from .stability import compute_parameter_stability

BuilderFn = Callable[..., None]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return float(value)
    return value


class SweepRunner:
    def __init__(self, base_runner: StrategyRunner, name: str = "sweep"):
        self._base_runner = base_runner
        self._name = name
        self._grid: dict[str, list[Any]] = {}
        self._builder: Optional[Callable[..., None]] = None
        self._meta: dict[str, Any] = {
            "page_title": f"MQENGINE · {name}",
            "dataset_name": name,
            "notes": [
                "Parameter sweep dashboard built from MQENGINE SweepRunner.",
                "Metrics are computed trade-by-trade using account-level trade returns.",
                "Detail page shows equity vs benchmark plus price/signal/entry/exit overlays.",
            ],
        }

    def metadata(self, **kwargs):
        self._meta.update(kwargs)
        return self

    def grid(self, **kwargs):
        self._grid = {k: list(v) for k, v in kwargs.items()}
        return self

    def builder(self, fn: Callable[..., None]):
        self._builder = fn
        return fn

    @property
    def param_columns(self) -> list[str]:
        return list(self._grid.keys())

    def run(self) -> SweepResult:
        if self._builder is None:
            raise ValueError("SweepRunner requires a strategy builder. Use @sweep.builder")

        param_names = list(self._grid.keys())
        param_values = [self._grid[k] for k in param_names]
        strategy_results = []
        rows = []

        for combo in product(*param_values):
            params = {k: _normalize_value(v) for k, v in zip(param_names, combo)}
            runner = self._base_runner.clone()
            runner.params(**params)
            self._builder(runner, **params)
            if runner._config.name == self._base_runner._config.name:
                runner.named(self._base_runner._config.name)
            result = runner.run()
            strategy_results.append(result)
            rows.append({
                "strategy_id": result.strategy_id,
                "name": result.name,
                **result.params,
                "usable_start": pd.Timestamp(result.data["ts"].iloc[0]).strftime("%Y-%m-%d %H:%M:%S"),
                "usable_end": pd.Timestamp(result.data["ts"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S"),
                "usable_rows": int(len(result.data)),
                **result.metrics,
            })

        results_df = pd.DataFrame(rows)
        if not results_df.empty:
            results_df = results_df.sort_values(["sharpe", "return_pct"], ascending=[False, False]).reset_index(drop=True)
            best = results_df.iloc[0].to_dict()
            self._meta.update({
                "total_strategies": int(len(results_df)),
                "best_strategy": {
                    "strategy_id": best["strategy_id"],
                    "sharpe": best["sharpe"],
                    "return_pct": best["return_pct"],
                    "max_drawdown": best["max_drawdown"],
                    "num_trades": best["num_trades"],
                },
            })
        return SweepResult(
            name=self._name,
            strategy_results=strategy_results,
            results_df=results_df,
            param_columns=param_names,
            meta=self._meta,
        )

    def run_research(self, protocol: ResearchProtocol | dict, objective: str = "period_sharpe") -> SweepResult:
        if self._builder is None:
            raise ValueError("SweepRunner requires a strategy builder. Use @sweep.builder")
        if isinstance(protocol, dict):
            protocol = ResearchProtocol(**protocol)

        param_names = list(self._grid.keys())
        param_values = [self._grid[k] for k in param_names]
        strategy_results = []
        rows = []

        for combo in product(*param_values):
            params = {k: _normalize_value(v) for k, v in zip(param_names, combo)}
            runner = self._base_runner.clone()
            runner.params(**params)
            self._builder(runner, **params)
            if runner._config.name == self._base_runner._config.name:
                runner.named(self._base_runner._config.name)
            result = runner.run_research(protocol)
            strategy_results.append(result)
            flat_metrics = flatten_research_metrics(result)
            rows.append({
                "strategy_id": result.strategy_id,
                "name": result.name,
                **result.params,
                "usable_start": pd.Timestamp(result.data["ts"].iloc[0]).strftime("%Y-%m-%d %H:%M:%S") if not result.data.empty else None,
                "usable_end": pd.Timestamp(result.data["ts"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S") if not result.data.empty else None,
                "usable_rows": int(len(result.data)),
                **flat_metrics,
            })

        results_df = pd.DataFrame(rows)
        stability = compute_parameter_stability(results_df, param_names, objective=objective)
        if not results_df.empty:
            sort_col = objective if objective in results_df.columns else "sharpe"
            results_df = results_df.sort_values([sort_col, "return_pct"], ascending=[False, False]).reset_index(drop=True)
            best = results_df.iloc[0].to_dict()
            self._meta.update({
                "total_strategies": int(len(results_df)),
                "research_protocol": protocol.to_dict(),
                "objective": objective,
                "stability": stability,
                "best_strategy": {
                    "strategy_id": best["strategy_id"],
                    "objective": best.get(sort_col),
                    "sharpe": best.get("sharpe"),
                    "period_sharpe": best.get("period_sharpe"),
                    "return_pct": best.get("return_pct"),
                    "max_drawdown": best.get("max_drawdown"),
                    "num_trades": best.get("num_trades"),
                },
            })
        return SweepResult(
            name=self._name,
            strategy_results=strategy_results,
            results_df=results_df,
            param_columns=param_names,
            meta=self._meta,
        )
