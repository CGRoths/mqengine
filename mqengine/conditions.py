from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict
import pandas as pd

class BaseCondition:
    def triggered(self, series: pd.Series, i: int) -> bool:
        raise NotImplementedError

@dataclass(slots=True)
class Above(BaseCondition):
    level: float
    inclusive: bool = True
    def triggered(self, series: pd.Series, i: int) -> bool:
        val = float(series.iloc[i])
        return val >= self.level if self.inclusive else val > self.level

@dataclass(slots=True)
class Below(BaseCondition):
    level: float
    inclusive: bool = True
    def triggered(self, series: pd.Series, i: int) -> bool:
        val = float(series.iloc[i])
        return val <= self.level if self.inclusive else val < self.level

@dataclass(slots=True)
class CrossAbove(BaseCondition):
    level: float
    inclusive: bool = True
    trigger_on_first_bar: bool = True
    def triggered(self, series: pd.Series, i: int) -> bool:
        cur = float(series.iloc[i])
        if i == 0:
            return self.trigger_on_first_bar and (cur >= self.level if self.inclusive else cur > self.level)
        prev = float(series.iloc[i - 1])
        return (prev < self.level) and (cur >= self.level if self.inclusive else cur > self.level)

@dataclass(slots=True)
class CrossBelow(BaseCondition):
    level: float
    inclusive: bool = True
    trigger_on_first_bar: bool = True
    def triggered(self, series: pd.Series, i: int) -> bool:
        cur = float(series.iloc[i])
        if i == 0:
            return self.trigger_on_first_bar and (cur <= self.level if self.inclusive else cur < self.level)
        prev = float(series.iloc[i - 1])
        return (prev > self.level) and (cur <= self.level if self.inclusive else cur < self.level)

ConditionFactory = Callable[..., BaseCondition]
condition_registry: Dict[str, ConditionFactory] = {}

def register_condition(name: str, factory: ConditionFactory):
    condition_registry[name] = factory
    return factory

register_condition("above", Above)
register_condition("below", Below)
register_condition("cross_above", CrossAbove)
register_condition("cross_below", CrossBelow)

class ConditionNamespace:
    def above(self, level: float, inclusive: bool = True) -> BaseCondition:
        return Above(level=level, inclusive=inclusive)
    def below(self, level: float, inclusive: bool = True) -> BaseCondition:
        return Below(level=level, inclusive=inclusive)
    def cross_above(self, level: float, inclusive: bool = True, trigger_on_first_bar: bool = True) -> BaseCondition:
        return CrossAbove(level=level, inclusive=inclusive, trigger_on_first_bar=trigger_on_first_bar)
    def cross_below(self, level: float, inclusive: bool = True, trigger_on_first_bar: bool = True) -> BaseCondition:
        return CrossBelow(level=level, inclusive=inclusive, trigger_on_first_bar=trigger_on_first_bar)

cond = ConditionNamespace()
