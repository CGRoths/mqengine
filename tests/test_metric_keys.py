import unittest

import numpy as np
import pandas as pd

from mqengine.metrics import compute_metrics


class MetricKeysTest(unittest.TestCase):
    def test_compute_metrics_includes_dashboard_and_adrs_keys(self):
        timestamps = pd.date_range("2024-01-01", periods=6, freq="D")
        equity = np.asarray([100.0, 101.0, 103.0, 102.0, 104.0, 106.0])
        trades = [
            {"account_return_pct": 1.0, "pnl_pct": 10.0, "pnl_cash": 1.0, "bars_held": 1, "entry_date": timestamps[0], "exit_date": timestamps[1]},
            {"account_return_pct": -0.75, "pnl_pct": -7.5, "pnl_cash": -0.75, "bars_held": 1, "entry_date": timestamps[2], "exit_date": timestamps[3]},
            {"account_return_pct": 1.4, "pnl_pct": 14.0, "pnl_cash": 1.4, "bars_held": 1, "entry_date": timestamps[4], "exit_date": timestamps[5]},
        ]

        metrics = compute_metrics(equity, trades, timestamps[0], timestamps[-1], timestamps=timestamps)
        expected_keys = [
            "sharpe",
            "trade_sharpe",
            "period_sharpe",
            "adrs_sharpe",
            "sortino",
            "adrs_sortino",
            "rolling_sharpe",
            "return_pct",
            "max_drawdown",
            "max_drawdown_pct",
            "cagr_pct",
            "calmar",
            "profit_factor",
            "win_rate",
            "num_trades",
            "trades_per_year",
            "VaR_95",
            "CVaR_95",
            "skew",
            "kurtosis",
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)
        self.assertEqual(metrics["sharpe"], metrics["trade_sharpe"])


if __name__ == "__main__":
    unittest.main()
