import unittest

import pandas as pd

from mqengine import btdash


class PortfolioTest(unittest.TestCase):
    def test_single_symbol_portfolio_pnl(self):
        prices = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["BTC", "BTC"],
            "open": [100.0, 110.0],
            "high": [100.0, 110.0],
            "low": [100.0, 110.0],
            "close": [100.0, 110.0],
            "volume": [1.0, 1.0],
        })
        signals = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-01"]),
            "strategy_id": ["s1"],
            "symbol": ["BTC"],
            "target_weight": [1.0],
            "signal_value": [1.0],
        })
        weights = pd.DataFrame({"strategy_id": ["s1"], "weight": [1.0]})

        result = btdash.portfolio(prices=prices, signals=signals, weights=weights, initial_capital=1000.0).run()

        self.assertAlmostEqual(float(result.equity["equity"].iloc[-1]), 1100.0)
        self.assertAlmostEqual(result.metrics["return_pct"], 10.0)
        self.assertAlmostEqual(result.attribution["pnl_by_symbol"]["BTC"], 100.0)


if __name__ == "__main__":
    unittest.main()
