"""
Portfolio simulator.

This is the source of truth for all metrics. An exchange testnet may reset
the balance or serve different prices, whereas this is deterministic and
reproducible.

Execution rule: the decision is made at the close of candle t and the trade
fills at the price of candle t+1. Filling at the close of the same candle
the decision was made on is a classic look-ahead leak that makes a backtest
look better than it is.

Methodology v2 allows shorts. A coin's weight lies in [-1, 1] and negative
units mean a short position. There is no leverage: the sum of |weights| is
at most 1, so with 10,000 USDT the total position value is capped at
10,000, in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def clip_target(target: np.ndarray, allow_short: bool) -> np.ndarray:
    """
    Clip target weights to the feasible set: [0, 1] without shorts,
    [-1, 1] with shorts; the sum of |weights| is at most 1 (no leverage).

    A standalone function (not only a Portfolio method) because the same
    arithmetic is needed outside the portfolio: counterfactual positions of
    ensemble members in the agent reward (agent.py), without creating a
    throwaway Portfolio.
    """
    lo = -1.0 if allow_short else 0.0
    t = np.clip(np.asarray(target, dtype=np.float64), lo, 1.0)
    gross = np.abs(t).sum()
    if gross > 1.0:
        t = t / gross
    return t


@dataclass
class Portfolio:
    """Wallet: cash plus per-coin positions (long or short)."""

    coins: list[str]
    initial_capital: float = 10_000.0
    fee_rate: float = 0.0005
    slippage: float = 0.0002
    allow_short: bool = False

    cash: float = field(init=False)
    units: np.ndarray = field(init=False)          # coins held; < 0 = short
    equity_curve: list[float] = field(default_factory=list, init=False)
    trades: list[dict] = field(default_factory=list, init=False)
    fees_paid: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.cash = self.initial_capital
        self.units = np.zeros(len(self.coins), dtype=np.float64)

    # ------------------------------------------------------------------
    def equity(self, prices: np.ndarray) -> float:
        """
        Total value: cash plus positions at current prices.
        For a short, units < 0, so a price rise reduces equity, as it should.
        """
        return float(self.cash + (self.units * prices).sum())

    def weights(self, prices: np.ndarray) -> np.ndarray:
        """Current signed capital weights per coin."""
        eq = self.equity(prices)
        return (self.units * prices) / eq if eq > 0 else np.zeros(len(self.coins))

    def clip_target(self, target: np.ndarray) -> np.ndarray:
        """Clip target weights to the feasible set; see module-level clip_target()."""
        return clip_target(target, self.allow_short)

    # ------------------------------------------------------------------
    def rebalance(
        self,
        target: np.ndarray,
        prices: np.ndarray,
        ts=None,
        min_change: float = 0.0,
    ) -> int:
        """
        Rebalance the portfolio to the target weights.

        target      - desired capital weights per coin
        min_change  - do not trade while the deviation is below this
                      threshold. Without it the network would jitter the
                      position and feed the exchange fees.

        Returns the number of trades executed.
        """
        eq = self.equity(prices)
        if eq <= 0:
            return 0

        target = self.clip_target(target)
        current = self.weights(prices)
        diff = target - current
        # The diff must be clearly nonzero regardless of min_change. The
        # executor calls this with min_change=0.0 because the deadband is
        # already applied in guards.py (for coins the fly chose not to touch,
        # guards sets target=current, so diff is exactly 0). With only
        # `abs(diff) >= min_change`, abs(0) >= 0.0 is true and the portfolio
        # would record a phantom trade (zero cash, zero units, but a real
        # history entry such as "sold 0.000000 ZEC").
        act = (np.abs(diff) >= min_change) & (np.abs(diff) > 1e-9)
        if not act.any():
            return 0

        n_trades = 0
        for i in np.flatnonzero(act):
            delta_value = diff[i] * eq
            price = prices[i]
            if price <= 0:
                continue

            # slippage always works against us: buy higher, sell lower
            exec_price = price * (1 + self.slippage if delta_value > 0 else 1 - self.slippage)
            delta_units = delta_value / exec_price
            cost = delta_units * exec_price        # < 0 on sell/short: cash comes in
            fee = abs(cost) * self.fee_rate

            if delta_value > 0 and cost + fee > self.cash:      # insufficient cash
                affordable = max(self.cash, 0.0) / (1 + self.fee_rate)
                delta_units = affordable / exec_price
                cost = delta_units * exec_price
                fee = abs(cost) * self.fee_rate

            self.units[i] += delta_units
            self.cash -= cost + fee
            self.fees_paid += fee
            n_trades += 1
            self.trades.append({
                "ts": ts,
                "coin": self.coins[i],
                "side": "buy" if delta_units > 0 else "sell",
                "units": float(delta_units),
                "price": float(exec_price),
                "fee": float(fee),
                "target_weight": float(target[i]),
            })

        if not self.allow_short:
            self.units = np.maximum(self.units, 0.0)
            self.cash = max(self.cash, 0.0)
        return n_trades

    def mark(self, prices: np.ndarray) -> float:
        """Append current equity to the equity curve."""
        eq = max(self.equity(prices), 0.0)
        self.equity_curve.append(eq)
        return eq

    def liquidate(self, prices: np.ndarray, ts=None) -> None:
        """Move everything to cash. Used on stop-loss and at the end of a run."""
        self.rebalance(np.zeros(len(self.coins)), prices, ts=ts, min_change=0.0)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        eq = np.array(self.equity_curve, dtype=np.float64)
        return {
            "final_equity": float(eq[-1]) if len(eq) else self.initial_capital,
            "total_return": float(eq[-1] / self.initial_capital - 1) if len(eq) else 0.0,
            "n_trades": len(self.trades),
            "fees_paid": self.fees_paid,
        }
