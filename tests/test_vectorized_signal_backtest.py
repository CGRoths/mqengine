import unittest

import pandas as pd

from mqengine.vectorized import run_vectorized_signal_backtest


class VectorizedSignalBacktestTest(unittest.TestCase):
    def test_uses_previous_signal_to_avoid_same_bar_lookahead(self):
        price_df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=3, freq="D"),
            "close": [100.0, 200.0, 300.0],
        })
        signal_df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=3, freq="D"),
            "signal": [0.0, 1.0, 1.0],
        })

        result = run_vectorized_signal_backtest(price_df, signal_df, fee_pct=0.0, initial_capital=100.0)

        self.assertEqual(float(result.data.loc[1, "prev_signal"]), 0.0)
        self.assertEqual(float(result.data.loc[1, "pnl_return"]), 0.0)
        self.assertEqual(float(result.data.loc[2, "prev_signal"]), 1.0)
        self.assertAlmostEqual(float(result.data.loc[2, "pnl_return"]), 0.5)
        self.assertAlmostEqual(float(result.data.loc[2, "equity"]), 150.0)
        self.assertEqual(result.meta["execution_mode"], "vectorized_signal_return")
        self.assertIn("adrs_sharpe", result.metrics)


if __name__ == "__main__":
    unittest.main()
