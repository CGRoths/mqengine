from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
import pandas as pd

@dataclass(slots=True)
class RiskConfig:
    position_size_pct: float = 0.10
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    fee_pct: float = 0.0
    slippage_pct: float = 0.0
    allow_same_bar_reentry: bool = False
    allow_flip: bool = False

@dataclass(slots=True)
class StrategyConfig:
    long_entry: Any = None
    short_entry: Any = None
    long_exit: Any = None
    short_exit: Any = None
    risk: RiskConfig = field(default_factory=RiskConfig)
    name: str = "strategy"
    params: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class PreparedData:
    df: pd.DataFrame
    signal_col: str
    price_col: str
    benchmark_col: Optional[str] = None
