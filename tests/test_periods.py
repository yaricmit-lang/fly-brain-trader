"""
Selection of 6 seasonal periods (cycle 6): non-overlapping calendar blocks
classified by volatility.
"""

import numpy as np
import pandas as pd
import pytest

from flybrain.market.periods import select_seasonal_blocks


def _synthetic(n_months=24, seed=0):
    """
    An hourly series spanning n_months months with CLEARLY different
    volatility in alternating 2-month chunks, so the calm/active
    classification is unambiguously checkable rather than borderline noise.
    """
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2024-09-16", periods=n_months * 30 * 24, freq="h", tz="UTC")
    n = len(dt)
    # alternate calm/turbulent two-month chunks with an explicit pattern
    vol = np.ones(n) * 0.001
    two_months = 60 * 24
    for i in range(0, n, two_months):
        block_idx = i // two_months
        vol[i:i + two_months] = 0.0005 if block_idx % 2 == 0 else 0.02
    rets = rng.normal(0, 1, (n, 2)) * vol[:, None]
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return dt, prices


def test_returns_requested_counts_and_labels():
    dt, prices = _synthetic()
    blocks = select_seasonal_blocks(dt, prices, lookback_months=24, block_months=2,
                                    n_calm=2, n_active=2, n_medium=2)
    assert len(blocks) == 6
    labels = [b.label for b in blocks]
    assert labels.count("calm") == 2
    assert labels.count("active") == 2
    assert labels.count("medium") == 2


def test_blocks_do_not_overlap():
    dt, prices = _synthetic()
    blocks = select_seasonal_blocks(dt, prices)
    all_idx = np.concatenate([b.idx for b in blocks])
    assert len(all_idx) == len(set(all_idx.tolist())), "block indices overlap"


def test_calm_blocks_have_lower_volatility_than_active():
    dt, prices = _synthetic()
    blocks = select_seasonal_blocks(dt, prices)
    calm_vol = max(b.mean_vol for b in blocks if b.label == "calm")
    active_vol = min(b.mean_vol for b in blocks if b.label == "active")
    assert calm_vol < active_vol


def test_raises_when_not_enough_history():
    dt, prices = _synthetic(n_months=3)   # not enough for 6 blocks of 2 months
    with pytest.raises(ValueError):
        select_seasonal_blocks(dt, prices, lookback_months=3, block_months=2)
