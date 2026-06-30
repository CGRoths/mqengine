import math
import unittest

import numpy as np
import pandas as pd

from mqengine.risk import compute_adrs_compatible_metrics


class ADRSCompatMetricsTest(unittest.TestCase):
    def test_adrs_sharpe_uses_sample_std_and_annualization(self):
        pnl = np.asarray([0.01, -0.005, 0.02, -0.01], dtype=float)
        timestamps = pd.date_range("2024-01-01", periods=len(pnl), freq="D")

        metrics = compute_adrs_compatible_metrics(pnl, timestamps)
        expected = float(pnl.mean() / pnl.std(ddof=1) * math.sqrt(365.0))

        self.assertAlmostEqual(metrics["adrs_sharpe"], round(expected, 6), places=6)
        self.assertEqual(metrics["adrs_interval_seconds"], 86400.0)
        self.assertEqual(metrics["adrs_period_multiplier"], 1.0)
        self.assertEqual(metrics["adrs_num_periods"], 365)

    def test_adrs_total_return_compounds_return_series(self):
        pnl = np.asarray([0.10, -0.10, 0.05], dtype=float)
        timestamps = pd.date_range("2024-01-01", periods=len(pnl), freq="D")

        metrics = compute_adrs_compatible_metrics(pnl, timestamps)

        self.assertAlmostEqual(metrics["adrs_total_return"], (1.10 * 0.90 * 1.05) - 1.0, places=6)
        self.assertNotEqual(metrics["adrs_cagr"], 0.0)


if __name__ == "__main__":
    unittest.main()
