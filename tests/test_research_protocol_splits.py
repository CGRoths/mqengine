import unittest

import numpy as np
import pandas as pd

from mqengine import btdash
from mqengine.research import flatten_research_metrics


def _runner():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    signal = pd.Series([-1.0] * 5 + [1.0] * 10 + [-1.0] * 10 + [1.0] * 10 + [-1.0] * 5, index=idx)
    runner = btdash.new(price=close, signal=signal, benchmark=close, name="research-split")
    runner.entry(long=btdash.cond.cross_above(0.0))
    runner.exit(long=btdash.cond.cross_below(0.0))
    return runner


class ResearchProtocolSplitsTest(unittest.TestCase):
    def test_research_protocol_splits_include_required_metrics_and_validation(self):
        protocol = btdash.research_protocol(
            start="2020-01-01",
            end="2020-02-09",
            in_sample=("2020-01-01", "2020-01-20"),
            out_sample=("2020-01-21", "2020-02-09"),
            min_oos_trades=5,
        )

        result = _runner().run_research(protocol)

        for section in ["full", "in_sample", "out_sample"]:
            for key in [
                "trade_sharpe",
                "period_sharpe",
                "adrs_sharpe",
                "return_pct",
                "max_drawdown",
                "max_drawdown_pct",
                "cagr_pct",
                "calmar",
                "num_trades",
                "win_rate",
                "profit_factor",
            ]:
                self.assertIn(key, result.metrics[section])

        for key in [
            "oos_pass",
            "oos_sharpe_decay",
            "oos_period_sharpe_decay",
            "oos_trade_sharpe_decay",
            "oos_adrs_sharpe_decay",
            "oos_return_decay",
            "oos_mdd_expansion",
            "is_oos_consistency_score",
            "warnings",
        ]:
            self.assertIn(key, result.metrics["validation"])

        self.assertIn("oos_trades_too_few", result.metrics["validation"]["warnings"])
        flat = flatten_research_metrics(result)
        self.assertIn("out_sample_adrs_sharpe", flat)
        self.assertIn("validation_warnings", flat)


if __name__ == "__main__":
    unittest.main()
