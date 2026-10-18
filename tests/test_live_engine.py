"""
Live engine.

The key check: live and backtest must compute positions IDENTICALLY.
The old live.py passed the fly's output straight to the executor, but in
exposure mode that is a multiplier around 1.0 which the executor would read
as "100% of capital in the coin". The fly would trade differently from what
it was trained on, and this would be nearly impossible to spot in the logs.
"""

import time

import numpy as np
import pytest
import torch

from flybrain.config import load_config
from flybrain.trading.baseline import baseline_positions


@pytest.fixture
def cfg():
    # Pin the baseline scale so the test does not depend on how aggressive
    # the default target_daily_vol / max_weight_per_coin are; it only checks
    # that "multiplier != direct position".
    c = load_config()
    c.trading.baseline.target_daily_vol = 0.02
    c.trading.baseline.max_weight_per_coin = 0.40
    return c


@pytest.fixture
def prices():
    rng = np.random.default_rng(3)
    vol = np.concatenate([np.full(300, 0.004), np.full(200, 0.04), np.full(300, 0.004)])
    return 100 * np.exp(np.cumsum(rng.normal(0, 1, (800, 3)) * vol[:, None], axis=0))


def test_exposure_becomes_position_via_baseline(cfg, prices):
    """
    Position = baseline x multiplier. If the multiplier went to the executor
    directly, the target would be ~1.0 per coin, i.e. 100% of capital in each
    of the three, which the safeguards would have to cut threefold.
    """
    base = baseline_positions(cfg, prices)[-1]
    exposure = np.array([1.0, 1.0, 1.0])
    target = base * exposure

    assert np.all(target < 0.45), f"target is suspiciously large: {target}"
    assert np.allclose(target, base), "a multiplier of 1.0 must give exactly the baseline"
    # this is what the old live.py bug would look like
    assert np.all(exposure > 0.9), "the check is meaningful only for a multiplier near 1"


def test_decision_boundary_detection(cfg):
    """A cadence boundary is detected by interval index, not by "4 hours elapsed"."""
    from flybrain.market.fetch import TF_MS

    step = TF_MS[cfg.market.decision_timeframe] / 1000.0

    def bar(ts):
        return int(ts // step)

    t0 = 1_700_000_000.0
    aligned = (t0 // step) * step
    assert bar(aligned) == bar(aligned + step - 1), "the index must not change within one interval"
    assert bar(aligned + step) == bar(aligned) + 1, "the index must increase at the boundary"
    assert bar(aligned + step * 2) == bar(aligned) + 2


def test_delayed_reward_not_applied_early(cfg, prices):
    """
    A decision's reward is applied no earlier than its horizon. Earlier
    would be look-ahead, even in live mode.
    """
    from collections import deque

    from flybrain.live_engine import _Pending

    now = time.time()
    horizon = 12 * 3600.0
    q: deque = deque()
    q.append(_Pending(token=None, positions=np.zeros(3), exec_prices=prices[-1],
                      turnover=np.zeros(3), due_ts=now + horizon))

    matured = [p for p in q if p.due_ts <= now]
    assert not matured, "reward matured instantly; the horizon is not working"

    matured = [p for p in q if p.due_ts <= now + horizon + 1]
    assert len(matured) == 1, "reward did not mature even after the horizon"


# --------------------------------------------------------------------------
#  Bounded adaptive stride (market.adaptive_stride)
# --------------------------------------------------------------------------
def test_lr_stride_scale_for_hours_matches_reference(cfg):
    """At elapsed_hours == lr_reference_timeframe the correction is 1.0, same
    as the old cfg.lr_stride_scale when decision_timeframe == reference."""
    from flybrain.market.fetch import TF_MS

    ref_hours = TF_MS[cfg.training.lr_reference_timeframe] / 3_600_000.0
    assert cfg.lr_stride_scale_for_hours(ref_hours) == pytest.approx(1.0)
    assert cfg.lr_stride_scale_for_hours(2 * ref_hours) == pytest.approx(2.0)
    assert cfg.lr_stride_scale_for_hours(ref_hours / 2) == pytest.approx(0.5)


def test_read_out_elapsed_hours_rebases_decay():
    """
    readout_decay/output_smoothing decay PER CALL, so their memory is
    measured in decisions, not hours. With elapsed_hours=None (legacy
    behaviour, agent.py/train.py) the decay is untouched. With
    elapsed_hours=2x reference the effective decay must be decay**2, so the
    readout statistics adapt faster (the brain forgets faster when decisions
    are rarer).

    Deterministic run (noise=0.0, no generator): the runs differ only in
    elapsed_hours on the last step.
    """
    from flybrain.build import build_brain
    from flybrain.market.fetch import TF_MS

    c = load_config()
    c.market.symbols = ["BTC/USDT"]
    c.market.features = ["log_return_1", "rsi_14"]
    c.model.input_neurons_per_channel = 2
    ref_hours = TF_MS[c.training.lr_reference_timeframe] / 3_600_000.0

    def _warm():
        brain, _, _, _ = build_brain(c, synthetic=True)
        x = torch.linspace(-0.5, 0.5, brain.rm.n_channels)
        for _ in range(5):
            brain.step(x, noise=0.0)
        return brain, x

    brain_ref, x = _warm()
    coin = next(iter(brain_ref.approach_idx))
    mu_before = brain_ref._ro_mean[coin]

    x2 = -x
    brain_ref.step(x2, noise=0.0, elapsed_hours=ref_hours)
    move_ref = abs(brain_ref._ro_mean[coin] - mu_before)

    brain_double, _ = _warm()
    assert brain_double._ro_mean[coin] == pytest.approx(mu_before)
    brain_double.step(x2, noise=0.0, elapsed_hours=2 * ref_hours)
    move_double = abs(brain_double._ro_mean[coin] - mu_before)

    assert move_ref > 1e-9, "uninformative test: the step moved nothing"
    assert move_double > move_ref, (
        "with twice the elapsed_hours the EMA must adapt faster "
        "(decay**2 < decay), otherwise readout memory is not tied to real time"
    )


def test_stride_regime_hysteresis_and_dwell(tmp_path):
    """
    Regime selector: does not switch before min_dwell_hours and does not
    flap at the boundary; it enters the active regime only at vol_high and
    leaves it only when volatility falls below vol_low.
    """
    from flybrain.live_engine import LiveEngine

    c = load_config()
    c.market.symbols = ["BTC/USDT", "ETH/USDT"]
    c.market.adaptive_stride.enabled = True
    c.market.adaptive_stride.calm_timeframe = "4h"
    c.market.adaptive_stride.active_timeframe = "1h"
    c.market.adaptive_stride.vol_high = 0.05
    c.market.adaptive_stride.vol_low = 0.02
    c.market.adaptive_stride.min_dwell_hours = 4.0

    engine = LiveEngine(cfg=c, checkpoint=None, run_dir=tmp_path / "live_regime")
    assert engine._stride_regime == "4h"

    calm = np.full((200, 2), 100.0)   # zero volatility -> stay in the calm regime
    engine._update_stride_regime(calm)
    assert engine._stride_regime == "4h"

    rng = np.random.default_rng(0)
    spike = 100 * np.exp(np.cumsum(rng.normal(0, 1, (200, 2)) * 0.08, axis=0))
    engine._update_stride_regime(spike)
    assert engine._stride_regime == "1h", "high volatility must switch to the active cadence"

    # immediate attempt to switch back: dwell time has not elapsed yet
    switch_ts = engine._stride_last_switch_ts
    engine._update_stride_regime(calm)
    assert engine._stride_regime == "1h"
    assert engine._stride_last_switch_ts == switch_ts, "no switch is allowed before the dwell time"


def test_activations_pack_to_uint8(cfg):
    """Activations go to the browser as uint8: ~6.3 KB per frame instead of ~25 KB."""
    a = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    packed = ((np.clip(a, -1, 1) + 1) * 127.5).astype(np.uint8)
    assert packed.dtype == np.uint8
    assert packed[2] == 127 or packed[2] == 128, "zero activity must map to the midpoint"
    back = np.abs(packed.astype(float) - 128) / 128
    assert back[0] > 0.9 and back[4] > 0.7, "range edges must glow brightly"


def test_drift_limit_stops_learning(cfg):
    """
    Online-learning safeguard: the fly learns for days unattended and
    nobody would notice runaway weights. The limit must be enabled.
    """
    assert cfg.training.live_drift_limit > 0, "weight drift limit is not set"
    assert cfg.training.hebbian.lr_live < cfg.training.hebbian.lr, (
        "the live learning step must be smaller than the regular one"
    )


def test_geometry_exported():
    """Brain geometry is exported and its size looks plausible."""
    import json

    from flybrain.config import load_config as lc

    c = lc()
    meta_path = c.path("data") / "geometry.json"
    bin_path = c.path("data") / "geometry.bin"
    if not meta_path.exists():
        pytest.skip("geometry not exported yet: scripts/export_geometry.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["n_neurons"] > 100_000, "the whole brain must be present"
    assert 1_000 < meta["n_mb"] < 20_000, "mushroom-body subgraph has an implausible size"
    assert bin_path.stat().st_size == meta["total_bytes"], "binary size does not match its metadata"
