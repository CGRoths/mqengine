import unittest

import numpy as np
import pandas as pd

from mqengine import btdash


def _runner():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    signal = pd.Series([-1.0] * 5 + [1.0] * 15 + [-1.0] * 10, index=idx)
    runner = btdash.new(price=close, signal=signal, benchmark=close, name="split")
    runner.entry(long=btdash.cond.cross_above(0.0))
    runner.exit(long=btdash.cond.cross_below(0.0))
    return runner


class ResearchSplitTest(unittest.TestCase):
    def test_research_protocol_returns_full_is_oos_metrics(self):
        protocol = btdash.research_protocol(
            start="2020-01-01",
            end="2020-01-30",
            in_sample=("2020-01-01", "2020-01-15"),
            out_sample=("2020-01-16", "2020-01-30"),
        )
        result = _runner().run_research(protocol)

        self.assertIn("full", result.metrics)
        self.assertIn("in_sample", result.metrics)
        self.assertIn("out_sample", result.metrics)
        self.assertIn("oos_pass", result.validation)
        self.assertEqual(result.research["in_sample_range"][0], "2020-01-01 00:00:00")
        self.assertEqual(result.research["out_sample_range"][0], "2020-01-16 00:00:00")


if __name__ == "__main__":
    unittest.main()
