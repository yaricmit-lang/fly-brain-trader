"""Common interface for trade execution adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Adapter(ABC):
    """Where a decision goes: to a log, a simulator or an exchange."""

    name = "base"

    @abstractmethod
    def sync(self, target: np.ndarray, prices: np.ndarray, ts=None) -> int:
        """Bring positions to the target weights. Returns the number of trades."""

    @abstractmethod
    def equity(self, prices: np.ndarray) -> float:
        """Current account equity."""

    @abstractmethod
    def weights(self, prices: np.ndarray) -> np.ndarray:
        """Current capital weights per coin."""
