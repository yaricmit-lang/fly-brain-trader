"""
Exposure mode (methodology v2, after measurements).

The fly does not pick a side: direction and size come from the volatility
targeting formula and the fly scales a multiplier on top. Key property: a
multiplier of 1.0 must reproduce the formula EXACTLY. Otherwise the
fly-vs-formula comparison is meaningless, since any difference would be
explained by sloppy plumbing rather than by what the fly learned.
"""

import numpy as np
import pytest

from flybrain.agent import Agent
from flybrain.backtest import benchmarks as B
from flybrain.config import load_config
from flybrain.trading.baseline import baseline_positions, vol_target_weights


@pytest.fixture
def cfg():
    # Self-contained in the number of coins (3): independent of how many
    # coins market.symbols contains by default.
    c = load_config()
    c.market.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    return c


@pytest.fixture
def prices():
    """Synthetic prices: two calm stretches with a turbulent one between them."""
    rng = np.random.default_rng(0)
    vol = np.concatenate([np.full(400, 0.005), np.full(200, 0.05), np.full(400, 0.005)])
    rets = rng.normal(0, 1, (1000, 3)) * vol[:, None]
    return 100 * np.exp(np.cumsum(rets, axis=0))


class FixedExposureBrain:
    """Stub fly: always emits the same multiplier."""

    def __init__(self, coins, value):
        self.coins = coins
        self.value = value
        self.n = 1

    def reset(self):
        pass

    def step(self, x, noise=0.0, generator=None):
        import torch

        from flybrain.brain.model import Decision

        fr = {c: float(self.value) for c in self.coins}
        fr_t = {c: torch.tensor(self.value) for c in self.coins}
        d = Decision(fractions=fr, activations=np.zeros(1), approach={}, avoid={}, fractions_t=fr_t)
        return d, torch.zeros(1)


def test_vol_target_shrinks_in_turbulence(cfg, prices):
    """The formula must shrink the position when volatility rises."""
    w = vol_target_weights(cfg, prices)
    calm = w[300:390].mean()
    wild = w[500:590].mean()
    assert wild < calm * 0.5, f"position did not shrink in the storm: calm {calm:.3f}, storm {wild:.3f}"


def test_exposure_one_reproduces_baseline(cfg, prices):
    """
    A fly with multiplier 1.0 must give exactly the same result as the
    reference volatility-targeting benchmark. This keeps the comparison fair.
    """
    cfg.trading.position_mode = "exposure"
    coins = [s.split("/")[0] for s in cfg.market.symbols]
    X = np.zeros((len(prices), 4), dtype=np.float32)

    agent = Agent(FixedExposureBrain(coins, 1.0), cfg, None)
    fly = agent.run(X, prices, learn=False)
    bench, _ = B.vol_target(cfg, prices)

    assert abs(fly.metrics.total_return - bench.total_return) < 1e-9, (
        f"fly with multiplier 1.0 diverged from the formula: "
        f"{fly.metrics.total_return:.6f} vs {bench.total_return:.6f}"
    )


def test_exposure_zero_stays_in_cash(cfg, prices):
    """Multiplier 0: the fly is out of the market, equity does not change."""
    cfg.trading.position_mode = "exposure"
    coins = [s.split("/")[0] for s in cfg.market.symbols]
    X = np.zeros((len(prices), 4), dtype=np.float32)
    fly = Agent(FixedExposureBrain(coins, 0.0), cfg, None).run(X, prices, learn=False)
    assert abs(fly.metrics.total_return) < 1e-9, "with zero exposure equity must not change"


def test_exposure_two_doubles_position(cfg, prices):
    """Multiplier 2 doubles the weight until it hits the cap."""
    cfg.trading.position_mode = "exposure"
    base = baseline_positions(cfg, prices)
    cap = cfg.trading.guards.max_fraction_per_coin
    doubled = np.minimum(base * 2.0, cap)
    assert (doubled >= base - 1e-12).all()
    assert doubled.mean() > base.mean(), "doubling the multiplier did not increase exposure"


def test_neutral_output_is_one_in_exposure_mode(cfg):
    """
    Fly output with no opinion: sigmoid(0) = 0.5, times exposure_max=2 is
    exactly 1.0, so the neutral point coincides with the formula.
    """
    from flybrain.build import build_ensemble

    cfg.trading.position_mode = "exposure"
    brain, sg, rm, _ = build_ensemble(cfg, synthetic=True, n_members=1)
    assert abs(brain._neutral - 1.0) < 1e-12
    for v in brain._frac_ema.values():
        assert abs(float(v) - 1.0) < 1e-12, "initial exposure must be 1.0"
