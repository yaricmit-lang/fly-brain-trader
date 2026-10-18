"""
Short mechanics (methodology v2).

A short is negative units: borrow the coin, sell it, owe it back. A price
fall earns a profit, a rise a loss. There is no leverage: the sum of
|weights| is at most 1. A sign error here would not crash any run, it would
just produce beautiful or terrible numbers, so we test it explicitly.
"""

import numpy as np
import pytest

from flybrain.trading.portfolio import Portfolio


def make_pf(short=True, fee=0.0, slip=0.0):
    return Portfolio(coins=["BTC", "ETH"], initial_capital=10_000.0,
                     fee_rate=fee, slippage=slip, allow_short=short)


def test_short_profits_when_price_falls():
    pf = make_pf()
    p0 = np.array([100.0, 50.0])
    pf.rebalance(np.array([-0.5, 0.0]), p0)          # short half of capital
    assert pf.units[0] < 0, "short was not opened"
    assert np.isclose(pf.equity(p0), 10_000.0), "opening a short must not change equity"

    p1 = np.array([80.0, 50.0])                       # price fell 20%
    eq = pf.equity(p1)
    assert eq > 10_000.0, f"short did not profit from the fall: {eq}"
    assert np.isclose(eq, 11_000.0), f"expected +1000 (50% position x 20%), got {eq - 10_000:.1f}"


def test_short_loses_when_price_rises():
    pf = make_pf()
    p0 = np.array([100.0, 50.0])
    pf.rebalance(np.array([-0.5, 0.0]), p0)
    p1 = np.array([120.0, 50.0])
    assert np.isclose(pf.equity(p1), 9_000.0), "a short must lose on a rise"


def test_long_and_short_together_no_leverage():
    """Long one coin, short another: total gross exposure at most 1."""
    pf = make_pf()
    p = np.array([100.0, 50.0])
    pf.rebalance(np.array([0.8, -0.8]), p)
    w = pf.weights(p)
    assert np.abs(w).sum() <= 1.0 + 1e-6, f"leverage appeared: {np.abs(w).sum():.3f}"


def test_flip_long_to_short():
    """Flipping from long to short passes through zero and does not break equity."""
    pf = make_pf()
    p = np.array([100.0, 50.0])
    pf.rebalance(np.array([0.5, 0.0]), p)
    eq_before = pf.equity(p)
    pf.rebalance(np.array([-0.5, 0.0]), p)
    assert pf.units[0] < 0
    assert np.isclose(pf.equity(p), eq_before), "a fee-free flip must not change equity"
    assert np.isclose(pf.weights(p)[0], -0.5, atol=1e-6)


def test_shorts_disabled_clips_to_zero():
    pf = make_pf(short=False)
    p = np.array([100.0, 50.0])
    pf.rebalance(np.array([-0.5, 0.0]), p)
    assert pf.units[0] == 0.0, "with allow_short=False a short must not open"


def test_fees_charged_on_short_open():
    pf = make_pf(fee=0.001)
    p = np.array([100.0, 50.0])
    pf.rebalance(np.array([-0.5, 0.0]), p)
    assert pf.fees_paid > 0, "fee for opening the short was not charged"
    assert np.isclose(pf.equity(p), 10_000.0 - pf.fees_paid)


@pytest.mark.parametrize("pos", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_reward_sign_matches_pnl(pos):
    """
    The per-coin reward must match the sign of the position's real P&L.
    Otherwise the fly would learn exactly the opposite.
    """
    from flybrain.config import load_config
    from flybrain.trading.reward import CoinRewardTracker

    cfg = load_config()
    cfg.trading.fee_rate = 0.0
    cfg.trading.slippage = 0.0
    crt = CoinRewardTracker(cfg, ["BTC"])
    sigma = np.array([0.01])   # volatility is ex-ante and passed explicitly

    up = crt(np.array([pos]), np.array([0.05]), sigma, np.array([0.0]))["BTC"]
    down = crt(np.array([pos]), np.array([-0.05]), sigma, np.array([0.0]))["BTC"]

    if pos > 0:
        assert up > 0 and down < 0, "long: a rise must be rewarded, a fall punished"
    elif pos < 0:
        assert up < 0 and down > 0, "short: a fall must be rewarded, a rise punished"
    else:
        assert abs(up) < 1e-9 and abs(down) < 1e-9, "no position, no reward"
