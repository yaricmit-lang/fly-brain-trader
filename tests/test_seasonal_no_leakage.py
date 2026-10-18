"""
Regression test explaining the architecture of cycle 6 (`train_seasonal.py`):
concatenating several non-overlapping periods into one array and feeding it
to `Agent.run()` in a single call is a silent bug. `run()` marks equity and
computes the forward reward return over ADJACENT indices of the SLICE
(agent.py: the equity-marking loop `for k in range(exec_idx, ...)`, the
reward `prices[pd.t + 1 + horizon]`), so at the seam between two periods a
price from one period leaks into the other.

The test builds synthetic data with a huge price gap between "block A" and
"block B" (as if they had been concatenated). A single run() call on the
concatenated array must show an anomaly at the seam; that is the bug. Two
separate calls (as in train_seasonal.py) must not.
"""

from __future__ import annotations

import numpy as np
import torch

from flybrain.agent import Agent
from flybrain.brain.model import Decision
from flybrain.config import load_config


class _ConstantBrain:
    """Stub brain: constant position fraction, no activation learning."""

    def __init__(self, coins: list[str], value: float):
        self.coins = coins
        self.value = value
        self.n = 1

    def reset(self) -> None:
        pass

    def step(self, x, noise: float = 0.0, generator=None):
        fr = {c: float(self.value) for c in self.coins}
        fr_t = {c: torch.tensor(self.value) for c in self.coins}
        d = Decision(fractions=fr, activations=np.zeros(1), approach={}, avoid={}, fractions_t=fr_t)
        return d, torch.zeros(1)


class _NoOpPlasticity:
    """Stub rule: learns nothing, it just lets agent.run() compute and
    return rewards (which is what the test checks)."""

    exploration_noise = 0.0

    def act(self, decision):
        return decision.fractions

    def observe(self, activations, fractions):
        return None

    def learn(self, rewards, token, reward_fn=None):
        return 0.0

    def reset(self):
        pass


def _cfg(n_coins: int = 1):
    cfg = load_config()
    cfg.market.symbols = ["BTC/USDT"][:n_coins] or ["BTC/USDT"]
    cfg.trading.position_mode = "direction"   # no baseline formula: simpler and does not affect the point
    cfg.reward.horizon_hours = 8   # was horizon_bars=2 (2 * decision_timeframe of 4h)
    return cfg


def _mild_block(n: int, level: float, seed: int) -> np.ndarray:
    """A calm random-walk price block with no jumps inside it."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.002, n)
    return (level * np.exp(np.cumsum(r))).reshape(-1, 1)


def _max_equity_jump_ratio(equity: list[float]) -> float:
    """Largest equity jump between adjacent hourly marks. Measured on the RAW
    equity curve, not through the reward (whose volatility normalization
    masks the signal), where the effect is undamped: hold a nonzero position
    and equity jumps together with the price."""
    eq = np.array(equity, dtype=np.float64)
    ratios = np.abs(np.log(eq[1:] / np.maximum(eq[:-1], 1e-12)))
    return float(ratios.max()) if len(ratios) else 0.0


def test_concatenated_single_call_leaks_across_boundary():
    """
    The bug: a single run() call on concatenated blocks. With a nonzero
    position, equity at the seam (where the price "teleports" to another
    period) jumps along with the price. We compare the raw equity curve with
    a continuous (non-concatenated) series of the same length and noise,
    which has no such jump because there is no seam.
    """
    cfg = _cfg()
    block_a = _mild_block(600, level=100.0, seed=1)
    block_b_jump = _mild_block(600, level=100_000.0, seed=2)   # x1000 jump at the seam
    block_b_smooth = _mild_block(600, level=100.0, seed=2)     # same noise, NO jump

    coins = [s.split("/")[0] for s in cfg.market.symbols]

    concat_jump = np.concatenate([block_a, block_b_jump], axis=0)
    res_jump = Agent(_ConstantBrain(coins, 0.3), cfg, _NoOpPlasticity()).run(concat_jump, concat_jump, learn=True)

    concat_smooth = np.concatenate([block_a, block_b_smooth], axis=0)
    res_smooth = Agent(_ConstantBrain(coins, 0.3), cfg, _NoOpPlasticity()).run(concat_smooth, concat_smooth, learn=True)

    jump_ratio = _max_equity_jump_ratio(res_jump.equity)
    smooth_ratio = _max_equity_jump_ratio(res_smooth.equity)
    assert jump_ratio > smooth_ratio * 5, (
        f"concatenation with a price jump at the seam must produce a clear jump in the raw equity curve: "
        f"log-jump {jump_ratio:.2f} vs {smooth_ratio:.2f} on the continuous series "
        f"(this is the bug that is why train_seasonal.py does NOT concatenate periods)"
    )


def test_two_separate_calls_do_not_leak():
    """The correct path: the same brain/plasticity, but run() is called on
    EACH block separately (as in train_seasonal.py). There is no anomaly at
    the seam because the array run() sees contains no seam."""
    cfg = _cfg()
    block_a = _mild_block(200, level=100.0, seed=1)
    block_b = _mild_block(200, level=100_000.0, seed=2)

    coins = [s.split("/")[0] for s in cfg.market.symbols]
    brain = _ConstantBrain(coins, 0.5)
    plast = _NoOpPlasticity()
    agent = Agent(brain, cfg, plast)

    res_a = agent.run(block_a, block_a, learn=True)
    res_b = agent.run(block_b, block_b, learn=True)

    for res in (res_a, res_b):
        assert not res.rewards or max(abs(r) for r in res.rewards) < 4.99, (
            "within a single block (no concatenation) the reward must not hit the clipping ceiling"
        )
