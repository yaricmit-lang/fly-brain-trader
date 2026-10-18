"""
Training cycles (only 6 and 7 are active now; see cycles.py) and the dynamic
"last N months" window.
"""

import numpy as np
import pandas as pd
import pytest

from flybrain.cycles import CYCLE_BY_ID, CYCLES, build_train_args
from flybrain.market.split import make_recent_split


def test_cycles_distinct():
    """Cycles are substantively different, not copies of one run."""
    assert len(CYCLES) >= 2
    assert len({c.id for c in CYCLES}) == len(CYCLES)
    runs = [tuple(build_train_args(c, "/tmp/x")) for c in CYCLES]
    assert len(set(runs)) == len(CYCLES), "cycles must not be identical runs"


def test_build_train_args_includes_run_dir_and_recent_window():
    spec = CYCLE_BY_ID["cycle7_dopamine"]
    args = build_train_args(spec, "/tmp/somewhere")
    assert "--run-dir" in args
    assert args[args.index("--run-dir") + 1] == "/tmp/somewhere"
    assert "--recent-months" in args
    assert "--rule" in args and "hebbian" in args


def _hourly_index(n: int, start: str = "2024-01-01"):
    return pd.DatetimeIndex(pd.date_range(start, periods=n, freq="h", tz="UTC"))


def test_recent_split_ends_at_last_bar():
    dt = _hourly_index(24 * 400)   # ~400 days of hourly candles
    split = make_recent_split(dt, train_months=10, val_months=2)
    assert len(split.train) > 0 and len(split.val) > 0
    assert len(split.test) == 0, "a recent split must have no future period"
    # val ends on the very last candle
    assert dt[split.val[-1]] == dt[-1]
    # train comes before val
    assert dt[split.train[-1]] < dt[split.val[0]]


def test_recent_split_train_precedes_val_no_overlap():
    dt = _hourly_index(24 * 400)
    split = make_recent_split(dt, train_months=10, val_months=2)
    assert set(split.train.tolist()).isdisjoint(set(split.val.tolist()))
    assert split.train.max() < split.val.min()


def test_recent_split_warns_on_insufficient_history(capsys):
    dt = _hourly_index(24 * 10)   # only 10 days, far less than the requested window
    split = make_recent_split(dt, train_months=10, val_months=2)
    out = capsys.readouterr().out
    assert len(split.train) == 0 or "warning" in out


def test_cycles_have_time_estimates_and_pause():
    from flybrain.cycles import PAUSE_SEC
    for c in CYCLES:
        assert c.est_minutes > 0
    args = build_train_args(CYCLES[0], "/tmp/x")
    assert args[args.index("--pause-sec") + 1] == str(PAUSE_SEC)


def test_teacher_cache_key_ignores_row_count():
    """The teacher cache key must not depend on the candle count, otherwise
    live would refit 162 models on every hourly bar."""
    from flybrain.config import load_config
    from flybrain.teacher.predict import _cache_key
    cfg = load_config()
    k1 = _cache_key(cfg)
    k2 = _cache_key(cfg)
    assert k1 == k2
    cfg.teacher.train_months += 1
    assert _cache_key(cfg) != k1


def test_parse_ps_etime_formats():
    """macOS `ps -o etime=` format [[dd-]hh:]mm:ss, not GNU etimes."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from server.app import _parse_ps_etime
    assert _parse_ps_etime("10:31") == 631
    assert _parse_ps_etime(" 1:02:03 ") == 3723
    assert _parse_ps_etime("2-01:02:03") == 2 * 86400 + 3723
    assert _parse_ps_etime("garbage") is None


def test_cycle6_uses_own_entrypoint_and_simpler_cli():
    spec = CYCLE_BY_ID["cycle6_seasonal"]
    assert spec.entrypoint == "flybrain.train_seasonal"
    args = build_train_args(spec, "/tmp/x")
    # train_seasonal.py has no --rule/--recent-months/--val-months/--select-by
    for flag in ("--rule", "--recent-months", "--val-months", "--select-by", "--val-start"):
        assert flag not in args, f"train_seasonal.py does not understand {flag}"
    assert "--run-dir" in args and "--epochs" in args and "--members" in args


def test_cycle7_uses_dopamine_config():
    spec = CYCLE_BY_ID["cycle7_dopamine"]
    args = build_train_args(spec, "/tmp/x")
    assert args[args.index("--config") + 1] == "config/dopamine.yaml"
    assert args[args.index("--select-by") + 1] == "edge"
    # cycle 6 (different entrypoint) must not have --config
    assert "--config" not in build_train_args(CYCLE_BY_ID["cycle6_seasonal"], "/tmp/x")


def test_only_two_active_cycles():
    """2026-09-16: of the eight cycles run, only 6 and 7 remain live candidates
    (channel manifest on the current code + positive edge); the rest were
    removed from the active list so the dashboard does not offer them."""
    assert len(CYCLES) == 2
    assert {c.id for c in CYCLES} == {"cycle6_seasonal", "cycle7_dopamine"}


def test_lr_is_calibrated_per_wall_clock_not_per_decision():
    """
    2026-09-16: lr was set PER DECISION, so with decision_timeframe=1h the
    same history produced four times as many weight updates at the same step,
    and the brain silently overtrained four times harder per unit of
    wall-clock time. The effective step must depend on decision frequency so
    that learning per unit of WALL-CLOCK time is independent of it.
    """
    from flybrain.config import load_config

    cfg = load_config()
    assert cfg.market.decision_timeframe == cfg.training.lr_reference_timeframe
    assert cfg.lr_stride_scale == 1.0, "no correction expected at the reference frequency"

    cfg.market.decision_timeframe = "1h"
    assert cfg.decision_stride == 1
    assert cfg.lr_stride_scale == 0.25, "hourly decisions: step four times smaller"

    cfg.market.decision_timeframe = "1d"    # 8h is not in TF_MS, daily is
    assert cfg.lr_stride_scale == 6.0, "daily decisions: step 6 times larger"

    # net result: the learning step per unit of wall-clock time is the same at any frequency
    for tf in ("1h", "4h", "1d"):
        cfg.market.decision_timeframe = tf
        per_hour = cfg.lr_stride_scale / cfg.decision_stride
        assert abs(per_hour - 0.25) < 1e-12, f"{tf}: per-hour learning step drifted"
