import json
import unittest

import numpy as np
import pandas as pd

from mqengine import btdash


class EngineBackwardCompatTest(unittest.TestCase):
    def test_legacy_runner_flow_and_payload(self):
        idx = pd.date_range("2023-01-01", periods=25, freq="D")
        close = pd.Series(100.0 + np.cumsum(np.ones(len(idx))), index=idx)
        signal = pd.Series([-1.0] * 5 + [1.0] * 10 + [-1.0] * 10, index=idx)
        runner = btdash.new(price=close, signal=signal, benchmark=close, name="legacy", initial_capital=100.0)
        runner.norm("zscore", window=3)
        runner.entry(long=btdash.cond.cross_above(0.0))
        runner.exit(long=btdash.cond.cross_below(0.0))
        runner.risk(position_size_pct=0.10, fee_pct=0.00055)

        result = runner.run()
        payload = result.to_payload()
        json.dumps(payload)

        self.assertIn("sharpe", result.metrics)
        self.assertIn("trade_sharpe", result.metrics)
        self.assertEqual(result.metrics["sharpe"], result.metrics["trade_sharpe"])
        self.assertIn("rows", payload)
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("Flask is not installed in this test runtime.")
        self.assertTrue(hasattr(result.to_flask_app(), "test_client"))


if __name__ == "__main__":
    unittest.main()
