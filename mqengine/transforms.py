from __future__ import annotations

from typing import Callable, Dict
import numpy as np
import pandas as pd

TransformFunc = Callable[..., pd.Series]
transform_registry: Dict[str, TransformFunc] = {}

def register_transform(name: str):
    def decorator(func: TransformFunc):
        transform_registry[name] = func
        return func
    return decorator

@register_transform("identity")
def identity(series: pd.Series, **kwargs) -> pd.Series:
    return series.astype(float)

@register_transform("zscore")
def zscore(series: pd.Series, window: int = 20, min_periods: int | None = None, ddof: int = 0, **kwargs) -> pd.Series:
    s = series.astype(float)
    min_periods = window if min_periods is None else min_periods
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std(ddof=ddof)
    return ((s - mean) / std.replace(0, np.nan)).astype(float)

@register_transform("minmax")
def minmax(series: pd.Series, window: int | None = None, min_periods: int | None = None, **kwargs) -> pd.Series:
    s = series.astype(float)
    if window is None:
        lo = s.min()
        hi = s.max()
        denom = hi - lo
        if denom == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return ((s - lo) / denom).astype(float)
    min_periods = window if min_periods is None else min_periods
    lo = s.rolling(window, min_periods=min_periods).min()
    hi = s.rolling(window, min_periods=min_periods).max()
    return ((s - lo) / (hi - lo).replace(0, np.nan)).astype(float)

@register_transform("robust_zscore")
def robust_zscore(series: pd.Series, window: int = 20, min_periods: int | None = None, **kwargs) -> pd.Series:
    s = series.astype(float)
    min_periods = window if min_periods is None else min_periods
    med = s.rolling(window, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window, min_periods=min_periods).median()
    return (0.6745 * (s - med) / mad.replace(0, np.nan)).astype(float)

@register_transform("pct_change")
def pct_change(series: pd.Series, periods: int = 1, **kwargs) -> pd.Series:
    return series.astype(float).pct_change(periods=periods)

@register_transform("sma")
def sma(series: pd.Series, window: int = 20, min_periods: int | None = None, **kwargs) -> pd.Series:
    s = series.astype(float)
    min_periods = window if min_periods is None else min_periods
    return s.rolling(window, min_periods=min_periods).mean()

@register_transform("ema")
def ema(series: pd.Series, span: int = 20, adjust: bool = False, **kwargs) -> pd.Series:
    return series.astype(float).ewm(span=span, adjust=adjust).mean()

class TransformNamespace:
    def __getattr__(self, name: str):
        if name not in transform_registry:
            raise AttributeError(f"Unknown transform: {name}")
        return transform_registry[name]

norm = TransformNamespace()
stand = TransformNamespace()
