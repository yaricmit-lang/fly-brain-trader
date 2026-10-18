"""Portfolio: fees are actually charged and leverage never appears on its own."""

import numpy as np

from flybrain.trading.portfolio import Portfolio


def make_pf(fee=0.001, slip=0.0):
    return Portfolio(coins=["BTC", "ETH"], initial_capital=10_000.0,
                     fee_rate=fee, slippage=slip)


def test_buy_costs_fee():
    pf = make_pf()
    prices = np.array([100.0, 50.0])
    pf.rebalance(np.array([0.5, 0.0]), prices)
    # spent 5000 plus a 0.1% fee
    assert pf.fees_paid > 0
    assert pf.equity(prices) < 10_000.0
    assert np.isclose(pf.equity(prices), 10_000 - pf.fees_paid, rtol=1e-6)


def test_no_leverage():
    """Ask for 80% + 80%. Must be scaled to a sum of 1.0, not borrow money."""
    pf = make_pf(fee=0.0)
    prices = np.array([100.0, 50.0])
    pf.rebalance(np.array([0.8, 0.8]), prices)
    w = pf.weights(prices)
    assert w.sum() <= 1.0 + 1e-6
    assert pf.cash >= -1e-9


def test_min_change_blocks_small_moves():
    """The change threshold stops noise from jittering the position."""
    pf = make_pf(fee=0.0)
    prices = np.array([100.0, 50.0])
    pf.rebalance(np.array([0.5, 0.0]), prices, min_change=0.0)
    n_before = len(pf.trades)
    pf.rebalance(np.array([0.51, 0.0]), prices, min_change=0.05)
    assert len(pf.trades) == n_before, "a move below the threshold must not trade"


def test_liquidate_returns_to_cash():
    pf = make_pf(fee=0.0)
    prices = np.array([100.0, 50.0])
    pf.rebalance(np.array([0.5, 0.5]), prices)
    pf.liquidate(prices)
    assert np.allclose(pf.units, 0.0, atol=1e-9)
    assert np.isclose(pf.equity(prices), pf.cash)


def test_zero_diff_with_zero_min_change_does_not_trade():
    """
    Regression: executor.py ALWAYS calls rebalance(..., min_change=0.0)
    because the deadband is already applied in guards.py (for coins the fly
    chose not to touch, guards sets target=current, so diff is exactly 0).
    With min_change=0.0 the condition `abs(diff) >= min_change` is true even
    for diff==0, and the portfolio executed a phantom trade: 0 coins, 0
    cash, but a real history entry ("sold 0.000000 ZEC").
    """
    pf = make_pf(fee=0.0)
    prices = np.array([100.0, 50.0])
    pf.rebalance(np.array([0.5, 0.0]), prices, min_change=0.0)
    n_before = len(pf.trades)

    # same target allocation: in production guards would send
    # target=current for both coins, so the diff is exactly 0
    current = pf.weights(prices)
    pf.rebalance(current, prices, min_change=0.0)
    assert len(pf.trades) == n_before, "a zero diff must not create a trade record"
