"""
Guards (circuit breakers) between the network's decision and the exchange.

At the start of training the network outputs noise. Even on a demo account
this must be constrained, otherwise the logs turn into a mess and fees eat
the test deposit within a day.

Each guard either adjusts the decision or blocks it entirely, and always
states why. The reason goes to the log and the dashboard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Config


@dataclass
class GuardResult:
    """What remains of the decision after all checks."""
    target: np.ndarray                  # final fractions per coin
    blocked: bool = False               # trading fully blocked
    needs_approval: bool = False        # waiting for the dashboard button
    reasons: list[str] = field(default_factory=list)

    def explain(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "no remarks"


class Guards:
    """Set of constraints. State persists between calls (counters, stops)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.g = cfg.trading.guards
        self.coins = [s.split("/")[0] for s in cfg.market.symbols]

        self.trade_times: list[float] = []
        self.approved_count = 0
        self.killed = False
        self.kill_reason = ""
        self.day_start_equity: float | None = None
        self.day_key: str = ""

    # ------------------------------------------------------------------
    def check(
        self,
        raw_target: np.ndarray,
        current_weights: np.ndarray,
        equity: float,
        now: float | None = None,
        day: str = "",
    ) -> GuardResult:
        """Run the decision through all checks in order."""
        now = now or time.time()
        reasons: list[str] = []
        target = np.clip(np.asarray(raw_target, dtype=np.float64), 0.0, 1.0)

        # --- 1. daily drawdown stop ---
        if day and day != self.day_key:
            self.day_key, self.day_start_equity = day, equity
        if self.day_start_equity and self.day_start_equity > 0:
            dd = equity / self.day_start_equity - 1
            if dd <= -self.g.daily_drawdown_stop:
                self.killed = True
                self.kill_reason = (
                    f"daily drawdown {dd * 100:.1f}% exceeds limit of "
                    f"{-self.g.daily_drawdown_stop * 100:.0f}%"
                )

        if self.killed:
            return GuardResult(
                target=np.zeros_like(target),
                blocked=True,
                reasons=[f"STOP: {self.kill_reason}. Manual reset required."],
            )

        # --- 2. per-coin cap ---
        over = target > self.g.max_fraction_per_coin
        if over.any():
            names = ", ".join(np.array(self.coins)[over])
            reasons.append(
                f"capped fraction at {self.g.max_fraction_per_coin * 100:.0f}% for {names}"
            )
            target = np.minimum(target, self.g.max_fraction_per_coin)

        # --- 3. no leverage: sum of fractions <= 1 ---
        total = target.sum()
        if total > 1.0:
            target = target / total
            reasons.append(f"sum of fractions was {total:.2f}, scaled down to 1.00")

        # --- 4. skip tiny adjustments (deadband) ---
        delta = np.abs(target - current_weights)
        moving = delta >= self.g.min_position_change
        if not moving.any():
            return GuardResult(
                target=current_weights.copy(),
                reasons=reasons + [
                    f"change below {self.g.min_position_change * 100:.0f}%, "
                    f"no trades needed"
                ],
            )
        target = np.where(moving, target, current_weights)

        # --- 5. trade rate limit ---
        self.trade_times = [t for t in self.trade_times if now - t < 3600]
        if len(self.trade_times) >= self.g.max_trades_per_hour:
            return GuardResult(
                target=current_weights.copy(),
                blocked=True,
                reasons=reasons + [
                    f"limit of {self.g.max_trades_per_hour} trades per hour reached"
                ],
            )

        # --- 6. manual approval of the first trades ---
        needs = self.approved_count < self.g.manual_approval_first_n
        if needs:
            reasons.append(
                f"trade {self.approved_count + 1} of "
                f"{self.g.manual_approval_first_n}: approval required"
            )

        return GuardResult(target=target, needs_approval=needs, reasons=reasons)

    # ------------------------------------------------------------------
    def register_trade(self, n: int = 1, now: float | None = None) -> None:
        """Record executed trades. Called by the executor after execution."""
        now = now or time.time()
        self.trade_times.extend([now] * n)
        self.approved_count += n

    def reset_kill_switch(self) -> None:
        """Manually reset the kill switch. Only on explicit operator command."""
        self.killed = False
        self.kill_reason = ""
        self.day_start_equity = None
