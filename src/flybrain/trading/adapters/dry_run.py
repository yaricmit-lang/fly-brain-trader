"""
Dry run: simulated like the paper account, but every trade is printed
loudly and explicitly marked as not sent to any exchange.

This is the default mode. The first live launch is always done through it.
"""

from __future__ import annotations

import numpy as np

from ...config import Config
from .paper import PaperAdapter


class DryRunAdapter(PaperAdapter):
    name = "dry_run"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.coins = [s.split("/")[0] for s in cfg.market.symbols]

    def sync(self, target: np.ndarray, prices: np.ndarray, ts=None) -> int:
        before = self.weights(prices)
        n = super().sync(target, prices, ts=ts)
        if n:
            moves = ", ".join(
                f"{c} {b * 100:.0f}%→{a * 100:.0f}%"
                for c, b, a in zip(self.coins, before, self.weights(prices))
                if abs(a - b) > 1e-4
            )
            print(f"  [dry run] {moves}   NOT sent to the exchange")
        return n
