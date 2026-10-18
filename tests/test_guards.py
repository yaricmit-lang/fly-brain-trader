"""Guards must trigger, otherwise they are pointless."""

import numpy as np
import pytest

from flybrain.config import load_config
from flybrain.trading.guards import Guards


@pytest.fixture
def cfg():
    # Self-contained in the number of coins (3): independent of how many
    # coins market.symbols contains by default.
    c = load_config()
    c.market.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    return c


def test_caps_fraction_per_coin(cfg):
    g = Guards(cfg)
    cur = np.zeros(len(cfg.market.symbols))
    res = g.check(np.array([0.9, 0.0, 0.0]), cur, equity=10_000, day="2026-01-01")
    assert res.target[0] <= cfg.trading.guards.max_fraction_per_coin + 1e-9
    assert any("capped" in r for r in res.reasons)


def test_no_leverage_after_guards(cfg):
    g = Guards(cfg)
    cur = np.zeros(3)
    res = g.check(np.array([0.4, 0.4, 0.4]), cur, equity=10_000, day="2026-01-01")
    assert res.target.sum() <= 1.0 + 1e-9


def test_daily_stop_blocks_everything(cfg):
    g = Guards(cfg)
    cur = np.zeros(3)
    g.check(np.array([0.3, 0.3, 0.3]), cur, equity=10_000, day="2026-01-01")
    # same day, equity fell 15% against a 10% limit
    res = g.check(np.array([0.3, 0.3, 0.3]), cur, equity=8_500, day="2026-01-01")
    assert res.blocked
    assert np.allclose(res.target, 0.0)
    assert g.killed


def test_kill_switch_needs_manual_reset(cfg):
    g = Guards(cfg)
    cur = np.zeros(3)
    g.check(np.array([0.3, 0.3, 0.3]), cur, equity=10_000, day="2026-01-01")
    g.check(np.array([0.3, 0.3, 0.3]), cur, equity=8_000, day="2026-01-01")
    assert g.check(np.array([0.1, 0, 0]), cur, 8_000, day="2026-01-02").blocked
    g.reset_kill_switch()
    assert not g.check(np.array([0.1, 0, 0]), cur, 8_000, day="2026-01-02").blocked


def test_trade_rate_limit(cfg):
    g = Guards(cfg)
    cur = np.zeros(3)
    g.approved_count = 999            # disable manual approval
    g.register_trade(cfg.trading.guards.max_trades_per_hour)
    res = g.check(np.array([0.3, 0.0, 0.0]), cur, equity=10_000, day="2026-01-01")
    assert res.blocked
    assert any("limit" in r for r in res.reasons)


def test_first_trades_need_approval(cfg):
    g = Guards(cfg)
    cur = np.zeros(3)
    res = g.check(np.array([0.3, 0.0, 0.0]), cur, equity=10_000, day="2026-01-01")
    assert res.needs_approval
