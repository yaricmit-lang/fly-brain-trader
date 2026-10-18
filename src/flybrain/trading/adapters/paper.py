"""
Local paper account.

The source of truth for metrics: deterministic, independent of testnet
quirks, and reproducible from a seed.
"""

from __future__ import annotations

import numpy as np

from ...config import Config
from ..portfolio import Portfolio
from .base import Adapter


class PaperAdapter(Adapter):
    name = "paper"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.pf = Portfolio(
            coins=[s.split("/")[0] for s in cfg.market.symbols],
            initial_capital=cfg.trading.initial_capital,
            fee_rate=cfg.trading.fee_rate,
            slippage=cfg.trading.slippage,
        )

    def sync(self, target: np.ndarray, prices: np.ndarray, ts=None) -> int:
        return self.pf.rebalance(target, prices, ts=ts, min_change=0.0)

    def equity(self, prices: np.ndarray) -> float:
        return self.pf.equity(prices)

    def weights(self, prices: np.ndarray) -> np.ndarray:
        return self.pf.weights(prices)
