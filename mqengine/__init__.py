from .facade import btdash
from .engine import BacktestEngine, StrategyRunner
from .conditions import cond, BaseCondition, register_condition, condition_registry
from .transforms import norm, stand, register_transform, transform_registry
from .result import BacktestResult, SweepResult
from .research import ResearchProtocol, WalkForwardResult, research_protocol, run_walkforward
from .portfolio import PortfolioBacktester, PortfolioResult
from .risk import compute_period_metrics
from .sweep import SweepRunner

__all__ = [
    "btdash",
    "BacktestEngine",
    "StrategyRunner",
    "SweepRunner",
    "BacktestResult",
    "SweepResult",
    "ResearchProtocol",
    "WalkForwardResult",
    "research_protocol",
    "run_walkforward",
    "PortfolioBacktester",
    "PortfolioResult",
    "compute_period_metrics",
    "cond",
    "BaseCondition",
    "register_condition",
    "condition_registry",
    "norm",
    "stand",
    "register_transform",
    "transform_registry",
]

from .alignment import align_signal_to_bars
from .io_sql import fetch_sql_signal, fetch_sql_ohlc, merge_signal_backward
from .montecarlo import monte_carlo_trade_robustness, simulate_trade_paths
from .oms_analytics import compute_oms_metrics
from .stability import compute_parameter_stability
from .validation import validate_ohlc, validate_signal

__all__ += [
    "align_signal_to_bars",
    "fetch_sql_signal",
    "fetch_sql_ohlc",
    "merge_signal_backward",
    "monte_carlo_trade_robustness",
    "simulate_trade_paths",
    "compute_oms_metrics",
    "compute_parameter_stability",
    "validate_ohlc",
    "validate_signal",
]
