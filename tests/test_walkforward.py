import unittest

import numpy as np
import pandas as pd

from mqengine import btdash


class WalkForwardTest(unittest.TestCase):
    def test_walkforward_folds_do_not_overlap(self):
        idx = pd.date_range("2021-01-01", periods=40, freq="D")
        close = pd.Series(100.0 + np.arange(len(idx)), index=idx)
        signal = pd.Series(np.sin(np.arange(len(idx)) / 3.0), index=idx)
        runner = btdash.new(price=close, signal=signal, benchmark=close, name="wf")
        runner.entry(long=btdash.cond.cross_above(0.0))
        runner.exit(long=btdash.cond.cross_below(0.0))

        wf = btdash.walkforward(runner, train_window="10D", test_window="5D", step="5D")

        self.assertGreater(len(wf.folds), 0)
        for fold in wf.folds:
            train_end = pd.Timestamp(fold["train_end"])
            test_start = pd.Timestamp(fold["test_start"])
            self.assertLessEqual(train_end, test_start)
        self.assertIn("combined_oos_metrics", wf.to_dict())


if __name__ == "__main__":
    unittest.main()
