import numpy as np
import pandas as pd
from mqengine import btdash

np.random.seed(7)
n = 2500
idx = pd.date_range("2023-01-01", periods=n, freq="h")
close = pd.Series(20000 + np.cumsum(np.random.randn(n) * 15), index=idx, name="close")
open_ = close.shift(1).fillna(close.iloc[0])
high = pd.concat([close, open_], axis=1).max(axis=1) + np.abs(np.random.randn(n) * 5)
low = pd.concat([close, open_], axis=1).min(axis=1) - np.abs(np.random.randn(n) * 5)
signal_raw = pd.Series(np.cumsum(np.random.randn(n) * 0.2), index=idx, name="nvt_raw")

sweep = btdash.sweep(
    price=close,
    signal=signal_raw,
    benchmark=close,
    open_=open_,
    high=high,
    low=low,
    name="NVT Sweep Demo",
    initial_capital=100.0,
)

@sweep.builder
def build(ctx, rolling_window, long_threshold, short_threshold, exit_signal, take_profit_pct, stop_loss_pct):
    ctx.clear_transforms()
    ctx.norm("zscore", window=rolling_window)
    ctx.entry(
        long=btdash.cond.cross_above(long_threshold),
        short=btdash.cond.cross_below(short_threshold),
    )
    if exit_signal is None:
        ctx.exit(long=None, short=None)
    else:
        ctx.exit(
            long=btdash.cond.cross_below(exit_signal),
            short=btdash.cond.cross_above(exit_signal),
        )
    ctx.risk(
        position_size_pct=0.10,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        fee_pct=0.00055,
        allow_same_bar_reentry=False,
    )

sweep.grid(
    rolling_window=[10, 20, 30],
    long_threshold=[0.5, 1.0],
    short_threshold=[-0.5, -1.0],
    exit_signal=[0.0, None],
    take_profit_pct=[5.0, 10.0],
    stop_loss_pct=[-5.0, -10.0],
)

sweep.metadata(
    page_title="MQENGINE · NVT Sweep Demo",
    dataset_name="Synthetic NVT zscore + 1H OHLC execution",
)

result = sweep.run()
app = result.to_flask_app()
app.run(port=5000, debug=True)
