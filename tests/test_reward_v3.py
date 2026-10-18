"""
Reward v3: risk-penalised utility minus the baseline formula's utility.

Key property: at E[r] = 0 a linear reward cannot tell risk levels apart,
while a quadratic one penalises a large position ahead of a volatility
rise. That is what should push the fly to learn risk timing instead of
sitting on a multiplier of 1.

sigma (volatility) is supplied by the caller (ex-ante, known at decision
time; agent.py computes it with the same formula as baseline.py) rather
than accumulated from the tracker's own forward-return history. The tests
pass a fixed sigma explicitly instead of warming up internal history.
"""

import numpy as np

from flybrain.config import load_config
from flybrain.trading.reward import CoinRewardTracker

SIGMA = np.array([0.01])


def _tracker(lam: float, rel: bool, turnover_penalty: float = 0.0):
    cfg = load_config()
    cfg.reward.risk_aversion = lam
    cfg.reward.relative_to_baseline = rel
    cfg.reward.turnover_penalty = turnover_penalty
    cfg.trading.fee_rate = 0.0     # real trade cost disabled:
    cfg.trading.slippage = 0.0     # these tests check the utility part in isolation
    return CoinRewardTracker(cfg, ["BTC"])


def test_multiplier_one_gives_zero_reward_when_relative():
    tr = _tracker(0.5, True)
    rng = np.random.default_rng(0)
    for _ in range(50):
        fwd = rng.normal(0, 0.01, 1)
        base = np.array([0.3])
        r = tr(base.copy(), fwd, SIGMA, np.zeros(1), base)     # position == baseline
        assert abs(r["BTC"]) < 1e-9


def test_quadratic_penalises_big_exposure_when_return_is_pure_noise():
    """
    Symmetric noise: half the steps +z, half -z. The linear reward for a
    doubled position sums to zero; the quadratic one is strictly negative.
    """
    lin = _tracker(0.0, True)
    quad = _tracker(0.5, True)
    base = np.array([0.3])
    double = np.array([0.6])
    s_lin = s_quad = 0.0
    for sign in [1, -1] * 100:
        fwd = np.array([sign * 0.02])
        s_lin += lin(double, fwd, SIGMA, np.zeros(1), base)["BTC"]
        s_quad += quad(double, fwd, SIGMA, np.zeros(1), base)["BTC"]
    # linear: exactly zero (symmetric noise, fixed sigma);
    # quadratic: clearly negative
    assert abs(s_lin) < 0.5
    assert s_quad < -1.0 and s_quad < s_lin - 1.0


def test_quadratic_rewards_cutting_exposure_before_vol_spike():
    """Ahead of a volatility spike (large |r|) a reduced position must earn a
    positive relative reward, whichever way the price moves."""
    quad = _tracker(0.5, True)
    base = np.array([0.4])
    half = np.array([0.2])
    for sign in (1, -1):
        r = quad(half, np.array([sign * 0.05]), SIGMA, np.zeros(1), base)["BTC"]
        assert r > 0, f"halving the position ahead of a {sign * 5}% move should pay off"


def test_v3_features_exist_and_are_causal():
    """The new features compute, and seasonality does not look ahead: changing
    the last candle does not alter values on earlier ones."""
    import pandas as pd
    from flybrain.features.price import PER_COIN_V3, per_coin_features

    rng = np.random.default_rng(3)
    n = 24 * 7 * 12
    dt = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({
        "dt": dt, "open": close * (1 + rng.normal(0, 0.002, n)), "close": close,
        "high": close * (1 + abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - abs(rng.normal(0, 0.005, n))),
        "volume": rng.uniform(1, 2, n),
    })
    f1 = per_coin_features(df, PER_COIN_V3)
    assert f1.shape[1] == len(PER_COIN_V3)
    assert f1.iloc[-1].notna().all(), "all features must be computed at the end of the series"

    df2 = df.copy()
    df2.loc[df2.index[-1], ["close", "high", "low"]] *= 1.5   # shock in the last candle
    f2 = per_coin_features(df2, PER_COIN_V3)
    a, b = f1.iloc[:-1].to_numpy(), f2.iloc[:-1].to_numpy()
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b)), "a feature looked into the future"


def test_real_trade_cost_is_charged_unconditionally():
    """
    The real trade cost (fee_rate + slippage) is subtracted from the reward
    UNCONDITIONALLY, not via a toy turnover_penalty.
    """
    cfg = load_config()
    cfg.reward.turnover_penalty = 0.0
    cfg.trading.fee_rate = 0.0005
    cfg.trading.slippage = 0.0002
    tr = CoinRewardTracker(cfg, ["BTC"])
    base = np.array([0.3])
    no_turn = tr(base, np.array([0.0]), SIGMA, np.zeros(1), base)["BTC"]
    with_turn = tr(base, np.array([0.0]), SIGMA, np.array([0.5]), base)["BTC"]
    assert with_turn < no_turn, "turnover must cost real money in the reward"


def test_dopamine_gain_asymmetry_and_activity_bonus():
    """Gain/loss asymmetry and the activity bonus as RAW quantities in
    CoinRewardTracker (before the EMA baseline is subtracted in
    Plasticity._delta, where their mean is cancelled; see config/dopamine.yaml)."""
    tr = _tracker(0.0, True)
    tr.cfg.gain_scale, tr.cfg.loss_scale, tr.cfg.activity_bonus = 2.0, 1.0, 0.0
    base, more = np.array([0.3]), np.array([0.5])
    up = tr(more, np.array([0.02]), SIGMA, np.zeros(1), base)["BTC"]
    dn = tr(more, np.array([-0.02]), SIGMA, np.zeros(1), base)["BTC"]
    assert up > 0 > dn and abs(up) > abs(dn) * 1.5, "a gain should be about twice as bright as a loss"

    tr.cfg.activity_bonus = 0.15
    still = tr(base.copy(), np.array([0.01]), SIGMA, np.zeros(1), base)["BTC"]
    assert abs(still) < 1e-9, "at multiplier 1.0 with no deviation there must be no dopamine"
    moved = tr(more, np.array([0.0]), SIGMA, np.zeros(1), base)["BTC"]
    assert moved > 0, "deviating from the baseline pays dopamine by itself (action-dependent bonus)"
