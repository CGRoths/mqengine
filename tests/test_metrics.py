import unittest

import numpy as np
import pandas as pd

from mqengine.metrics import compute_metrics


class MetricsTest(unittest.TestCase):
    def test_trade_and_period_sharpe_are_distinct(self):
        timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
        equity = np.asarray([100.0, 101.0, 102.0, 101.0, 103.0])
        trades = [
            {"account_return_pct": 1.0, "pnl_pct": 10.0, "pnl_cash": 1.0, "bars_held": 1, "entry_date": timestamps[0], "exit_date": timestamps[1]},
            {"account_return_pct": -0.5, "pnl_pct": -5.0, "pnl_cash": -0.5, "bars_held": 1, "entry_date": timestamps[2], "exit_date": timestamps[3]},
            {"account_return_pct": 1.2, "pnl_pct": 12.0, "pnl_cash": 1.2, "bars_held": 1, "entry_date": timestamps[3], "exit_date": timestamps[4]},
        ]
        metrics = compute_metrics(equity, trades, timestamps[0], timestamps[-1], timestamps=timestamps)
        self.assertIn("trade_sharpe", metrics)
        self.assertIn("period_sharpe", metrics)
        self.assertEqual(metrics["sharpe"], metrics["trade_sharpe"])
        self.assertNotEqual(metrics["trade_sharpe"], metrics["period_sharpe"])

    def test_max_drawdown_duration_and_recovery(self):
        timestamps = pd.date_range("2024-01-01", periods=4, freq="D")
        metrics = compute_metrics(np.asarray([100.0, 90.0, 95.0, 110.0]), [], timestamps[0], timestamps[-1], timestamps=timestamps)
        self.assertEqual(metrics["max_drawdown_pct"], 10.0)
        self.assertEqual(metrics["max_drawdown_start"], "2024-01-01 00:00:00")
        self.assertEqual(metrics["max_drawdown_end"], "2024-01-02 00:00:00")
        self.assertEqual(metrics["max_drawdown_recovery"], "2024-01-04 00:00:00")
        self.assertEqual(metrics["max_drawdown_duration_days"], 3.0)


if __name__ == "__main__":
    unittest.main()
