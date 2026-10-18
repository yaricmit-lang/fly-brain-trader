"""
Live learning.

Previously the live loop did not truly learn with either rule:
  - REINFORCE: trade() never called plast.act(), so self._dist stayed empty,
    observation returned an empty token, and learn() silently exited with
    touched=False and returned 0.0. Weight drift was always exactly 0 while
    the dashboard reported "learning";
  - Hebbian rule: the forward pass in think() ran without exploration noise
    (noise=0.0), while the rule by construction amplifies random deviations
    on the MBONs, so there was nothing to amplify.

These tests run a minimal synthetic LiveEngine.trade()/_apply_matured() on a
toy connectome, without network access or a real warmup(). They verify that
weights REALLY move for both rules, and that no weight step is applied while
learning is off, even for rewards already sitting in the queue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from flybrain.build import build_brain
from flybrain.config import load_config
from flybrain.live_engine import LiveEngine


def _cfg(rule: str):
    c = load_config()
    c.market.symbols = ["BTC/USDT", "ETH/USDT"]
    c.market.features = ["log_return_1", "rsi_14"]
    c.market.cross_coin = False
    c.market.baseline_feature = False
    c.market.external_features = []
    c.market.coin_features = {}
    c.model.input_neurons_per_channel = 2
    c.training.rule = rule
    c.trading.executor.adapter = "dry_run"
    c.reward.horizon_hours = 1        # short horizon so the test need not wait hours
    # guards.max_trades_per_hour counts in REAL wall-clock hours; the test
    # makes dozens of decisions within milliseconds and would hit the limit
    # at the 6th trade, leaving the portfolio stuck at zero through no fault
    # of the code under test
    c.trading.guards.max_trades_per_hour = 100_000
    return c


def _synthetic_ctx(cfg, n=260, seed=0):
    rng = np.random.default_rng(seed)
    n_ch = len(cfg.market.features) * len(cfg.market.symbols)
    X = np.clip(rng.normal(0, 0.3, (n, n_ch)).astype(np.float32), -1, 1)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, len(cfg.market.symbols))), axis=0))
    dt = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return {"X": X, "prices": prices, "dt": dt}


def _bare_engine(cfg, tmp_path, rule: str):
    """LiveEngine with a synthetic brain and NO real warmup() (no network, no
    real connectome): only what trade()/_apply_matured() need."""
    engine = LiveEngine(cfg=cfg, checkpoint=None, online_learning=True,
                        auto_approve=True, run_dir=tmp_path / f"live_{rule}")
    names = [f"{c.split('/')[0]}:{f}" for c in cfg.market.symbols for f in cfg.market.features]
    brain, sg, rm, plast = build_brain(cfg, synthetic=True, with_plasticity=True, channel_names=names)
    engine.brain, engine.sg, engine.rm, engine.plast = brain, sg, rm, plast
    from flybrain.trading.portfolio import Portfolio
    engine.pf_base = Portfolio(coins=engine.coins, initial_capital=cfg.trading.initial_capital,
                               fee_rate=cfg.trading.fee_rate, slippage=cfg.trading.slippage)
    engine._w0 = brain.w_mag.detach().clone()
    return engine


@pytest.mark.parametrize("rule", ["hebbian", "reinforce"])
def test_live_trade_actually_moves_weights(tmp_path, rule):
    """Key regression: REINFORCE used to not move weights at all in live mode."""
    cfg = _cfg(rule)
    engine = _bare_engine(cfg, tmp_path, rule)
    ctx = _synthetic_ctx(cfg)

    w_before = engine.brain.w_mag.detach().clone()
    for t in range(0, len(ctx["X"]) - 1, 4):
        sub = {"X": ctx["X"][:t + 2], "prices": ctx["prices"][:t + 2], "dt": ctx["dt"][:t + 2]}
        engine.trade(sub)
        engine._apply_matured(sub["prices"], sub["dt"][-1].isoformat())
    # Drain the queue tail: the horizon is short (1h) but decision_timeframe
    # may be larger, so force maturation regardless of the clock
    import time as _time
    for pd_ in list(engine.pending):
        pd_.due_ts = 0.0
    engine._apply_matured(ctx["prices"], ctx["dt"][-1].isoformat())

    moved = (engine.brain.w_mag.detach() - w_before).abs().sum().item()
    assert moved > 0, f"{rule}: weights did not move, live learning is not working"


def test_save_and_load_state_roundtrip(tmp_path):
    """
    Key regression for live trading with accumulated experience:
    save_checkpoint() was never called in the live loop, so any restart
    rolled the brain back to the cycle checkpoint and lost the dopamine
    baselines, portfolio, guards and reward queue. This runs a full
    save_state()/_load_state() round trip on a synthetic brain.
    """
    cfg = _cfg("hebbian")
    engine = _bare_engine(cfg, tmp_path, "hebbian")
    engine.checkpoint = tmp_path / "fake_base.pt"   # base_checkpoint, compared against in _load_state
    ctx = _synthetic_ctx(cfg)

    for t in range(40, 120, 8):
        sub = {"X": ctx["X"][:t + 2], "prices": ctx["prices"][:t + 2], "dt": ctx["dt"][:t + 2]}
        engine.trade(sub)
    for pd_ in engine.pending:
        pd_.due_ts = 0.0
    engine._apply_matured(ctx["prices"], ctx["dt"][-1].isoformat())
    # leave one immature token in the queue; it must survive a restart
    sub = {"X": ctx["X"][:122], "prices": ctx["prices"][:122], "dt": ctx["dt"][:122]}
    engine.trade(sub)
    assert len(engine.pending) >= 1

    w_saved = engine.brain.w_mag.detach().clone()
    n_decisions_saved = engine.state.n_decisions
    baseline_saved = dict(engine.plast.baseline)
    pf_saved = getattr(engine.executor.adapter, "pf", None)
    cash_saved, units_saved = pf_saved.cash, pf_saved.units.copy()
    pending_saved = len(engine.pending)
    engine.save_state()

    # "restart": brand-new brain/plasticity/engine, as after a process loss
    engine2 = _bare_engine(cfg, tmp_path, "hebbian")
    engine2.checkpoint = engine.checkpoint
    engine2.run_dir = engine.run_dir
    engine2.state_path = engine.state_path
    resumed = engine2._load_state()

    assert resumed, "state should have been restored"
    assert torch.equal(engine2.brain.w_mag.detach(), w_saved), "weights were not restored"
    assert engine2.state.n_decisions == n_decisions_saved
    assert engine2.plast.baseline == baseline_saved, "dopamine baselines were lost"
    pf2 = getattr(engine2.executor.adapter, "pf", None)
    assert pf2.cash == cash_saved and np.array_equal(pf2.units, units_saved), (
        "paper portfolio did not survive the restart"
    )
    assert len(engine2.pending) == pending_saved, "delayed-reward queue did not survive the restart"


def test_load_state_refuses_foreign_base_checkpoint(tmp_path):
    """After the production checkpoint is switched (switch_checkpoint), live
    state saved against a DIFFERENT checkpoint must not be applied; otherwise
    experience accumulated on one set of weights would be overlaid on another."""
    cfg = _cfg("hebbian")
    engine = _bare_engine(cfg, tmp_path, "hebbian")
    engine.checkpoint = tmp_path / "cycle_a.pt"
    engine.save_state()

    engine.checkpoint = tmp_path / "cycle_b.pt"   # a different production checkpoint
    w_before = engine.brain.w_mag.detach().clone()
    resumed = engine._load_state()

    assert not resumed
    assert torch.equal(engine.brain.w_mag.detach(), w_before), "weights must not have changed"


def test_drift_measured_from_base_not_last_restart(tmp_path):
    """
    _w0 used to be taken at the end of warmup(), so drift measurement
    restarted after EVERY restart, and drifting N x the limit over N restarts
    went unnoticed. _w0 is now a snapshot of the BASE checkpoint; restoring
    live state does not touch it, so drift after a "restart" is measured from
    the same point as before it.
    """
    cfg = _cfg("hebbian")
    engine = _bare_engine(cfg, tmp_path, "hebbian")
    engine.checkpoint = tmp_path / "fake_base.pt"
    base_w0 = engine._w0.clone()

    ctx = _synthetic_ctx(cfg)
    for t in range(40, 120, 8):
        sub = {"X": ctx["X"][:t + 2], "prices": ctx["prices"][:t + 2], "dt": ctx["dt"][:t + 2]}
        engine.trade(sub)
    for pd_ in engine.pending:
        pd_.due_ts = 0.0
    engine._apply_matured(ctx["prices"], ctx["dt"][-1].isoformat())
    drift_before_restart = engine.state.weight_drift
    assert drift_before_restart > 0, "weights should have moved away from the base"
    engine.save_state()

    # "restart": new engine; _w0 must remain a snapshot of the BASE (as if
    # freshly warmed up from the checkpoint), not the weights restored from
    # live state
    engine2 = _bare_engine(cfg, tmp_path, "hebbian")
    engine2.checkpoint = engine.checkpoint
    engine2._w0 = base_w0.clone()   # what a fresh load_checkpoint() in warmup() would give
    engine2.run_dir, engine2.state_path = engine.run_dir, engine.state_path
    engine2._load_state()

    drift_after_restart = float((engine2.brain.w_mag.detach() - engine2._w0).abs().mean())
    assert abs(drift_after_restart - drift_before_restart) < 1e-6, (
        "drift after a restart must be measured from the same base as before it"
    )


def test_learning_off_does_not_apply_queued_weight_step(tmp_path):
    """Pausing learning must stop weight updates immediately, even for tokens
    already queued from when learning was on."""
    cfg = _cfg("hebbian")
    engine = _bare_engine(cfg, tmp_path, "hebbian")
    ctx = _synthetic_ctx(cfg)

    sub = {"X": ctx["X"][:10], "prices": ctx["prices"][:10], "dt": ctx["dt"][:10]}
    engine.trade(sub)          # learning on -> the token is queued
    assert len(engine.pending) == 1

    engine.online_learning = False   # "Pause learning"
    w_before = engine.brain.w_mag.detach().clone()
    for pd_ in engine.pending:
        pd_.due_ts = 0.0          # matured right now
    engine._apply_matured(ctx["prices"], ctx["dt"][-1].isoformat())

    assert len(engine.pending) == 0, "the matured token must leave the queue"
    assert torch.equal(engine.brain.w_mag.detach(), w_before), (
        "no weight step may be applied while learning is off"
    )


def test_ensemble_hebbian_tokens_survive_save_load(tmp_path):
    """
    save_state() used to check `isinstance(token, torch.Tensor)`, but for an
    ensemble (n_members>1, the default with 5 members) the token is ALWAYS a
    list [(rule_token, frac_vals), ...] (EnsemblePlasticity.observe), even
    for the fully serializable Hebbian rule. The check was always False for
    an ensemble, so EVERY token was silently flagged as "unserializable
    REINFORCE" and dropped, yet real live cycles train precisely as an
    ensemble of 5 Hebbian brains.
    """
    from flybrain.build import build_ensemble

    cfg = _cfg("hebbian")
    engine = LiveEngine(cfg=cfg, checkpoint=None, online_learning=True,
                        auto_approve=True, run_dir=tmp_path / "live_ens")
    names = [f"{c.split('/')[0]}:{f}" for c in cfg.market.symbols for f in cfg.market.features]
    brain, sg, rm, plast = build_ensemble(cfg, synthetic=True, with_plasticity=True,
                                          n_members=3, channel_names=names)
    engine.brain, engine.sg, engine.rm, engine.plast = brain, sg, rm, plast
    from flybrain.trading.portfolio import Portfolio
    engine.pf_base = Portfolio(coins=engine.coins, initial_capital=cfg.trading.initial_capital,
                               fee_rate=cfg.trading.fee_rate, slippage=cfg.trading.slippage)
    engine._w0 = brain.members[0].w_mag.detach().clone()
    engine.checkpoint = tmp_path / "fake_base.pt"

    ctx = _synthetic_ctx(cfg)
    sub = {"X": ctx["X"][:50], "prices": ctx["prices"][:50], "dt": ctx["dt"][:50]}
    engine.trade(sub)
    assert len(engine.pending) == 1
    engine.save_state()

    saved = torch.load(engine.state_path, weights_only=False)
    assert len(saved["pending"]) == 1, "the ensemble token should have been serialized, not dropped"

    engine2 = LiveEngine(cfg=cfg, checkpoint=None, online_learning=True,
                         auto_approve=True, run_dir=tmp_path / "live_ens")
    brain2, sg2, rm2, plast2 = build_ensemble(cfg, synthetic=True, with_plasticity=True,
                                              n_members=3, channel_names=names)
    engine2.brain, engine2.sg, engine2.rm, engine2.plast = brain2, sg2, rm2, plast2
    engine2._w0 = brain2.members[0].w_mag.detach().clone()
    engine2.checkpoint = engine.checkpoint
    engine2.run_dir, engine2.state_path = engine.run_dir, engine.state_path

    resumed = engine2._load_state()
    assert resumed
    assert len(engine2.pending) == 1, "the ensemble token should have been restored from the saved state"


def test_checkpoint_pointer_survives_restart(tmp_path):
    """
    "Make production" on the dashboard used to change the checkpoint only in
    process memory, so any server restart silently forgot the choice and
    rolled the live brain back to None (the connectome's initial weights).
    The on-disk pointer must survive this.
    """
    from flybrain.live_engine import read_checkpoint_pointer

    cfg = _cfg("hebbian")
    fake_ckpt = tmp_path / "cycle7" / "best.pt"
    fake_ckpt.parent.mkdir(parents=True)
    fake_ckpt.write_text("stub")

    engine = LiveEngine(cfg=cfg, checkpoint=fake_ckpt, online_learning=False,
                        run_dir=tmp_path / "live_ptr")
    assert read_checkpoint_pointer(engine.run_dir) == fake_ckpt

    # "restart": remembers nothing except what is on disk
    assert read_checkpoint_pointer(tmp_path / "live_ptr") == fake_ckpt


def test_checkpoint_pointer_missing_file_is_none(tmp_path):
    from flybrain.live_engine import read_checkpoint_pointer
    assert read_checkpoint_pointer(tmp_path / "nope") is None
