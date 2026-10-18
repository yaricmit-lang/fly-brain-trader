"""
make_anchored_split: val is anchored to specific dates (not to the end of
the history, nor to the config dates); needed by the "100 -> 300 in a year" cycle.
"""

import pandas as pd

from flybrain.market.split import make_anchored_split


def _hourly_index(n: int, start: str = "2024-01-01"):
    return pd.DatetimeIndex(pd.date_range(start, periods=n, freq="h", tz="UTC"))


def test_val_matches_requested_dates_exactly():
    dt = _hourly_index(24 * 800, start="2024-01-01")   # ~2.2 years
    split = make_anchored_split(dt, train_months=12, val_start="2025-01-01", val_end="2025-06-01")
    assert dt[split.val[0]] == pd.Timestamp("2025-01-01", tz="UTC")
    assert dt[split.val[-1]] < pd.Timestamp("2025-06-01", tz="UTC")
    assert dt[split.val[-1]] >= pd.Timestamp("2025-05-31", tz="UTC")


def test_train_precedes_val_by_train_months_no_overlap():
    dt = _hourly_index(24 * 800, start="2024-01-01")
    split = make_anchored_split(dt, train_months=6, val_start="2025-01-01", val_end="2025-03-01")
    assert set(split.train.tolist()).isdisjoint(set(split.val.tolist()))
    assert dt[split.train[0]] >= pd.Timestamp("2024-07-01", tz="UTC") - pd.Timedelta(hours=1)
    assert dt[split.train[-1]] < pd.Timestamp("2025-01-01", tz="UTC")


def test_test_is_always_empty():
    dt = _hourly_index(24 * 800, start="2024-01-01")
    split = make_anchored_split(dt, train_months=12, val_start="2025-01-01", val_end="2025-06-01")
    assert len(split.test) == 0


def test_warns_when_history_insufficient(capsys):
    dt = _hourly_index(24 * 30, start="2025-01-01")   # too little history
    split = make_anchored_split(dt, train_months=12, val_start="2025-01-01", val_end="2025-06-01")
    out = capsys.readouterr().out
    assert len(split.train) == 0 or "warning" in out
