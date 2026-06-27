from __future__ import annotations

from .conditions import cond
from .engine import StrategyRunner
from .sweep import SweepRunner
from .transforms import norm, stand

class BTDashFacade:
    cond = cond
    norm = norm
    stand = stand

    def new(self, *, price, signal, benchmark=None, open_=None, high=None, low=None, name: str = "strategy", initial_capital: float = 100.0) -> StrategyRunner:
        return StrategyRunner(
            price=price,
            signal=signal,
            benchmark=benchmark,
            open_=open_,
            high=high,
            low=low,
            name=name,
            initial_capital=initial_capital,
        )

    def sweep(self, *, price, signal, benchmark=None, open_=None, high=None, low=None, name: str = "sweep", initial_capital: float = 100.0) -> SweepRunner:
        runner = self.new(
            price=price,
            signal=signal,
            benchmark=benchmark,
            open_=open_,
            high=high,
            low=low,
            name=name,
            initial_capital=initial_capital,
        )
        return SweepRunner(runner, name=name)

btdash = BTDashFacade()
