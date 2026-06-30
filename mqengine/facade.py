from __future__ import annotations

from .conditions import cond
from .engine import StrategyRunner
from .portfolio import PortfolioBacktester
from .research import ResearchProtocol, apply_research_protocol, research_protocol, run_walkforward
from .sweep import SweepRunner
from .transforms import norm, stand
from .vectorized import run_vectorized_signal_backtest

class BTDashFacade:
    cond = cond
    norm = norm
    stand = stand
    ResearchProtocol = ResearchProtocol

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

    def research_protocol(self, **kwargs) -> ResearchProtocol:
        return research_protocol(**kwargs)

    def research(self, **kwargs) -> ResearchProtocol:
        return research_protocol(**kwargs)

    def split(self, result, protocol: ResearchProtocol | dict):
        if isinstance(protocol, dict):
            protocol = ResearchProtocol(**protocol)
        return apply_research_protocol(result, protocol)

    def walkforward(self, runner_or_sweep, *, train_window="730D", test_window="180D", step="180D", objective="period_sharpe"):
        return run_walkforward(
            runner_or_sweep,
            train_window=train_window,
            test_window=test_window,
            step=step,
            objective=objective,
        )

    def vectorized_signal_backtest(self, price_df, signal_df, **kwargs):
        return run_vectorized_signal_backtest(price_df, signal_df, **kwargs)

    def portfolio(self, *, prices, signals, weights=None, initial_capital: float = 10000.0, fee_pct: float = 0.0, calendar: str = "crypto_365", name: str = "portfolio") -> PortfolioBacktester:
        return PortfolioBacktester(
            prices=prices,
            signals=signals,
            weights=weights,
            initial_capital=initial_capital,
            fee_pct=fee_pct,
            calendar=calendar,
            name=name,
        )

btdash = BTDashFacade()
