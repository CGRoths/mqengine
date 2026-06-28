import unittest

import pandas as pd

from mqengine.alignment import align_signal_to_bars


class AlignmentTest(unittest.TestCase):
    def test_available_time_prevents_lookahead(self):
        bars = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [100.0, 101.0],
        })
        signals = pd.DataFrame({
            "event_ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "available_ts": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "signal_value": [1.0, 99.0],
        })

        out = align_signal_to_bars(bars, signals, tolerance="7D")

        self.assertEqual(float(out.loc[0, "signal"]), 1.0)
        self.assertNotEqual(out.loc[0, "signal"], 99.0)
        self.assertTrue(bool(out.loc[0, "lookahead_safe"]))
        self.assertLessEqual(out.loc[0, "signal_available_ts"], out.loc[0, "ts"])

    def test_min_lag_blocks_same_timestamp_signal(self):
        bars = pd.DataFrame({"ts": pd.to_datetime(["2024-01-02 00:00:00"])})
        signals = pd.DataFrame({
            "event_ts": pd.to_datetime(["2024-01-02 00:00:00"]),
            "available_ts": pd.to_datetime(["2024-01-02 00:00:00"]),
            "signal_value": [5.0],
        })

        out = align_signal_to_bars(bars, signals, min_lag="1s", tolerance="7D")

        self.assertTrue(pd.isna(out.loc[0, "signal"]))
        self.assertFalse(bool(out.loc[0, "lookahead_safe"]))


if __name__ == "__main__":
    unittest.main()
