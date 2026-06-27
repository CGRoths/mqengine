from sqlalchemy import create_engine
import pandas as pd
from mqengine import btdash, fetch_sql_signal, fetch_sql_ohlc, merge_signal_backward

# Example only; adapt table names and DATABASE_URL for your environment.
DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5432/dbname"
engine = create_engine(DATABASE_URL)

start = pd.Timestamp("2023-01-01")
end = pd.Timestamp("2025-11-01")

signal_df = fetch_sql_signal(
    engine,
    table_name="nvt_4h",
    ts_col="bucket_start_utc",
    value_col="nvt_raw",
    start_ts=start - pd.Timedelta(days=20),
    end_ts=end,
)

price_df = fetch_sql_ohlc(
    engine,
    table_name="btcusd_1h",
    open_time_col="open_time",
    close_time_col="close_time",
    start_ms=int(start.timestamp() * 1000),
    end_ms=int((end + pd.Timedelta(days=1)).timestamp() * 1000),
    extra_where_sql="AND symbol = :symbol AND interval = :interval_value",
    extra_params={"symbol": "BTCUSD", "interval_value": "1h"},
)

merged = merge_signal_backward(price_df, signal_df, output_col="signal_raw")
merged = merged.dropna(subset=["signal_raw"]).set_index("ts")

runner = btdash.new(
    price=merged["close"],
    signal=merged["signal_raw"],
    benchmark=merged["close"],
    open_=merged["open"],
    high=merged["high"],
    low=merged["low"],
    name="NVT 1H + 4H NVT",
    initial_capital=100.0,
)

runner.norm("zscore", window=30)
runner.entry(
    long=btdash.cond.cross_above(1.0),
    short=btdash.cond.cross_below(-1.0),
)
runner.exit(
    long=btdash.cond.cross_below(0.0),
    short=btdash.cond.cross_above(0.0),
)
runner.risk(position_size_pct=0.10, take_profit_pct=10, stop_loss_pct=-10, fee_pct=0.00055)

result = runner.run()
app = result.to_flask_app()
app.run(port=5000, debug=True)
