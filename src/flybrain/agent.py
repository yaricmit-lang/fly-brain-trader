"""
Agent: brain + wallet + reward.

A single class drives both training and backtesting, so both run exactly the
same code path. The only difference is the `learn` flag.

Methodology v2 adds three things:

1. Decision stride. Features arrive on hourly candles, but the fly decides
   once per decision_timeframe (e.g. 4h). The position is held between
   decisions and equity is marked every hour, so the equity curve stays
   honest while the number of trades drops several-fold.

2. Delayed reward. A decision at time t executes at the price of t+1, and its
   reward (the return up to the price at t+1+H bars) is only known then. The
   plasticity-rule token waits in a queue and receives dopamine once the
   reward matures. There is no look-ahead: a reward is applied no earlier
   than the price it depends on is actually known.

3. Per-coin reward: BTC and SOL each get their own.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from .backtest import metrics as M
from .brain.model import FlyBrain
from .brain.plasticity import Plasticity
from .config import Config
from .trading.baseline import baseline_positions, realized_vol
from .trading.portfolio import Portfolio, clip_target
from .trading.reward import CoinRewardTracker


@dataclass
class RunResult:
    """Outcome of a run over a slice of history."""
    metrics: M.Metrics
    equity: list[float]
    rewards: list[float]
    weights_trace: list[np.ndarray] = field(default_factory=list)
    activations: list[np.ndarray] = field(default_factory=list)
    act_bars: list[int] = field(default_factory=list)
    n_decisions: int = 0


@dataclass
class _Pending:
    """A decision waiting for its reward."""
    t: int                       # decision candle
    token: object                # what the plasticity rule remembered
    positions: np.ndarray        # signed weights after execution
    exec_prices: np.ndarray      # execution prices
    turnover: np.ndarray         # |Δweight| per coin
    base_positions: np.ndarray | None = None   # baseline-formula position (for reward v3)


class Agent:
    """Runs the brain over history, tracks the money, and learns when asked."""

    def __init__(self, brain: FlyBrain, cfg: Config, plasticity: Plasticity | None = None):
        self.brain = brain
        self.cfg = cfg
        self.plasticity = plasticity
        self.coins = [s.split("/")[0] for s in cfg.market.symbols]

    def _new_portfolio(self) -> Portfolio:
        return Portfolio(
            coins=self.coins,
            initial_capital=self.cfg.trading.initial_capital,
            fee_rate=self.cfg.trading.fee_rate,
            slippage=self.cfg.trading.slippage,
            allow_short=self.cfg.trading.allow_short,
        )

    def run(
        self,
        X: np.ndarray,
        prices: np.ndarray,
        learn: bool = False,
        record_activations: bool = False,
        seed: int | None = None,
        progress_every: int = 0,
        warm_X: np.ndarray | None = None,
    ) -> RunResult:
        """
        X      — (T, C) input channels on hourly candles
        prices — (T, n_coins) close prices on the same candles
        warm_X — (Tw, C) optional warm-up BEFORE X: the brain is stepped over
                 it without trading or learning so that the readout
                 calibration (_ro_mean/_ro_var/_frac_ema) settles before the
                 metrics/reward of this run start counting. Otherwise the
                 first ~30 decisions after brain.reset() are swamped by the
                 sigmoid starting at var=1e-2.

        In exposure mode the fly does not pick a side: direction and size
        come from the baseline risk-targeting formula, and the fly's output
        is a multiplier on top of it. A multiplier of 1.0 reproduces the
        formula exactly.
        """
        cfg = self.cfg
        base = baseline_positions(cfg, prices) if cfg.trading.position_mode == "exposure" else None
        pf = self._new_portfolio()
        # Shadow portfolio for the baseline formula. The control variate used
        # to compare the fly's REALIZED position (after deadband/gross
        # normalization) with the formula's NOMINAL position (without them),
        # so the difference reflected incomparable execution rather than the
        # fly. The formula now gets its own portfolio with the same
        # min_change/fees, created only when needed (learn + exposure mode)
        # to avoid wasted work in validation/backtest, where
        # relative_to_baseline is unused.
        pf_base = self._new_portfolio() if (learn and base is not None) else None
        crt = CoinRewardTracker(cfg, self.coins)
        gen = torch.Generator(device="cpu").manual_seed(seed if seed is not None else cfg.project.seed)

        stride = cfg.decision_stride
        horizon = max(1, cfg.reward.horizon_hours)     # in hourly candles, independent of stride
        noise = self.plasticity.exploration_noise if (learn and self.plasticity) else 0.0
        min_change = cfg.trading.guards.min_position_change
        max_frac = cfg.trading.guards.max_fraction_per_coin
        allow_short = cfg.trading.allow_short
        rec_every = max(1, cfg.viz.record_every_n_bars // stride)

        # Ex-ante vol for the reward: the same realized_vol as the formula,
        # KNOWN AT DECISION TIME and scaled to the reward horizon, not
        # derived from future forward returns (see reward.py)
        sigma = None
        if learn and self.plasticity:
            sigma = realized_vol(prices, cfg.trading.baseline.vol_window) * np.sqrt(float(horizon))

        self.brain.reset()
        if warm_X is not None and len(warm_X):
            for tw in range(0, len(warm_X), stride):
                self.brain.step(torch.from_numpy(warm_X[tw]), noise=0.0, generator=gen)
        if learn and self.plasticity:
            self.plasticity.reset()
            crt.reset()

        rewards: list[float] = []
        activations: list[np.ndarray] = []
        act_bars: list[int] = []
        pending: deque[_Pending] = deque()
        turnover_total = 0.0
        n_decisions = 0

        T = len(X)
        last_price_idx = T - 1
        for t in range(0, T - 1, stride):
            exec_idx = t + 1
            x = torch.from_numpy(X[t])
            decision, a = self.brain.step(x, noise=noise, generator=gen)

            # What is actually executed: the raw network output (Hebbian) or
            # the policy's sampled action (REINFORCE); see Plasticity.act
            exec_fr = self.plasticity.act(decision) if (learn and self.plasticity) else decision.fractions

            # Simplified guards: per-coin cap and no leverage. The full set
            # lives in guards.py and applies in live mode.
            out = np.array([exec_fr[c] for c in self.coins])
            if base is not None:
                # the baseline sets side and size; the fly scales it by a multiplier
                target = base[t] * out
            else:
                target = out
            target = np.clip(target, -max_frac if allow_short else 0.0, max_frac)

            p_exec = prices[exec_idx]
            before = pf.weights(p_exec)
            pf.rebalance(target, p_exec, min_change=min_change)
            after = pf.weights(p_exec)
            turn = np.abs(after - before)
            turnover_total += float(turn.sum())
            n_decisions += 1

            # mark equity every hour until the next decision
            for k in range(exec_idx, min(t + stride, last_price_idx) + 1):
                equity = pf.mark(prices[k])

            if learn and self.plasticity:
                # fractions_t, not fractions: rule B needs the live
                # computation graph so the gradient can reach the weights
                token = self.plasticity.observe(a, decision.fractions_t)
                base_pos = None
                if pf_base is not None:
                    base_target = np.clip(base[t], -max_frac if allow_short else 0.0, max_frac)
                    pf_base.rebalance(base_target, p_exec, min_change=min_change)
                    base_pos = pf_base.weights(p_exec)
                pending.append(_Pending(t, token, after.copy(), p_exec.copy(), turn.copy(),
                                        base_pos.copy() if base_pos is not None else None))

                # The reward for decision d is known once the price at
                # d+1+horizon exists, and not before; earlier would be
                # look-ahead
                while pending and pending[0].t + 1 + horizon <= exec_idx:
                    pd = pending.popleft()
                    p_later = prices[pd.t + 1 + horizon]
                    fwd = np.log(np.maximum(p_later, 1e-12) / np.maximum(pd.exec_prices, 1e-12))
                    sig = sigma[pd.t] if sigma is not None else np.full(len(self.coins), np.nan)

                    def _frac_reward(frac_vals: dict[str, float], _fwd=fwd, _sig=sig, _pd=pd) -> dict[str, float]:
                        # Counterfactual reward for a hypothetical weight
                        # (ensemble, B6): same clip/gross-normalization
                        # arithmetic as the main loop, but WITHOUT deadband
                        # or execution fees. It estimates "what if this
                        # member alone had decided", not a real trade.
                        arr = np.array([frac_vals[c] for c in self.coins])
                        tgt = base[_pd.t] * arr if base is not None else arr
                        tgt = np.clip(tgt, -max_frac if allow_short else 0.0, max_frac)
                        pos = clip_target(tgt, allow_short)
                        return crt(pos, _fwd, _sig, _pd.turnover, _pd.base_positions)

                    rw = crt(pd.positions, fwd, sig, pd.turnover, pd.base_positions)
                    self.plasticity.learn(rw, pd.token, reward_fn=_frac_reward)
                    rewards.append(float(np.mean(list(rw.values()))))

            if record_activations and n_decisions % rec_every == 0:
                # uint8 instead of float32: 4x smaller on disk
                activations.append(((np.clip(a.detach().numpy(), -1, 1) + 1) * 127.5).astype(np.uint8))
                act_bars.append(t)

            if progress_every and n_decisions % progress_every == 0:
                print(f"    decision {n_decisions:>6,}  candle {t:>6,}/{T:,}  equity {equity:>10,.0f}", end="\r")

        if progress_every:
            print()

        s = pf.summary()
        met = M.compute(pf.equity_curve, s["n_trades"], turnover_total, s["fees_paid"])
        return RunResult(
            metrics=met,
            equity=pf.equity_curve,
            rewards=rewards,
            activations=activations,
            act_bars=act_bars,
            n_decisions=n_decisions,
        )

