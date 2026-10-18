"""
Bridge between the brain and the money.

The graph never talks to an exchange directly. It emits capital fractions;
the decision then passes through the guards, is logged, and only then
reaches the adapter. Every decision is therefore visible before it is
executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from .adapters.base import Adapter
from .guards import Guards, GuardResult


def make_adapter(cfg: Config) -> Adapter:
    """Pick the execution adapter from the config."""
    kind = cfg.trading.executor.adapter
    if kind == "dry_run":
        from .adapters.dry_run import DryRunAdapter  # noqa: PLC0415
        return DryRunAdapter(cfg)
    if kind == "paper":
        from .adapters.paper import PaperAdapter  # noqa: PLC0415
        return PaperAdapter(cfg)
    if kind == "binance_futures_testnet":
        from .adapters.binance_futures_testnet import (  # noqa: PLC0415
            BinanceFuturesTestnetAdapter,
        )
        return BinanceFuturesTestnetAdapter(cfg)
    if kind == "bybit_demo":
        from .adapters.bybit_demo import BybitDemoAdapter  # noqa: PLC0415
        return BybitDemoAdapter(cfg)
    raise ValueError(f"unknown adapter: {kind}")


@dataclass
class ExecutionRecord:
    """One decision record: what was requested, what was allowed, what was executed."""
    ts: object
    raw_target: list[float]
    final_target: list[float]
    executed: int
    equity: float
    blocked: bool
    needs_approval: bool
    reasons: list[str] = field(default_factory=list)


class Executor:
    """Decision -> guards -> log -> adapter."""

    def __init__(self, cfg: Config, adapter: Adapter | None = None) -> None:
        self.cfg = cfg
        self.coins = [s.split("/")[0] for s in cfg.market.symbols]
        self.guards = Guards(cfg)
        self.adapter = adapter or make_adapter(cfg)
        self.log: list[ExecutionRecord] = []
        self.pending: GuardResult | None = None

    # ------------------------------------------------------------------
    def submit(
        self,
        fractions: dict[str, float],
        prices: np.ndarray,
        ts=None,
        auto_approve: bool = False,
    ) -> ExecutionRecord:
        """
        Accept a decision from the network.

        auto_approve bypasses manual approval. Use it only in backtests,
        where no human is present.
        """
        raw = np.array([fractions.get(c, 0.0) for c in self.coins], dtype=np.float64)
        eq = self.adapter.equity(prices)
        cur = self.adapter.weights(prices)
        day = str(ts)[:10] if ts is not None else ""

        gr = self.guards.check(raw, cur, eq, day=day)

        executed = 0
        if gr.needs_approval and not auto_approve:
            self.pending = gr
        elif not gr.blocked:
            executed = self.adapter.sync(gr.target, prices, ts=ts)
            if executed:
                self.guards.register_trade(executed)

        rec = ExecutionRecord(
            ts=ts,
            raw_target=raw.tolist(),
            final_target=gr.target.tolist(),
            executed=executed,
            equity=eq,
            blocked=gr.blocked,
            needs_approval=gr.needs_approval and not auto_approve,
            reasons=gr.reasons,
        )
        self.log.append(rec)
        return rec

    def approve_pending(self, prices: np.ndarray, ts=None) -> int:
        """The operator pressed "approve" in the dashboard."""
        if self.pending is None:
            return 0
        n = self.adapter.sync(self.pending.target, prices, ts=ts)
        if n:
            self.guards.register_trade(n)
        self.pending = None
        return n

    def reject_pending(self) -> None:
        """The operator pressed "reject" in the dashboard."""
        self.pending = None
