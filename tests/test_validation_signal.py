import unittest

import pandas as pd

from mqengine.validation import validate_strategy_signal


class StrategySignalValidationTest(unittest.TestCase):
    def test_default_bounds_are_configurable_for_zscore_signals(self):
        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=3, freq="D"),
            "signal": [-2.0, 0.0, 2.0],
        })

        bounded = validate_strategy_signal(df)
        unbounded = validate_strategy_signal(df, min_value=None, max_value=None)

        self.assertFalse(bounded["ok"])
        self.assertIn("signal_below_min:1", bounded["errors"])
        self.assertIn("signal_above_max:1", bounded["errors"])
        self.assertTrue(unbounded["ok"])
        self.assertEqual(unbounded["min_signal"], -2.0)
        self.assertEqual(unbounded["max_signal"], 2.0)

    def test_reports_null_and_duplicate_counts(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "signal": [0.5, None, 0.2],
        })

        report = validate_strategy_signal(df)

        self.assertFalse(report["ok"])
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["null_signal_count"], 1)
        self.assertEqual(report["duplicate_ts_count"], 1)
        self.assertIn("duplicate_timestamps:1", report["warnings"])


if __name__ == "__main__":
    unittest.main()
