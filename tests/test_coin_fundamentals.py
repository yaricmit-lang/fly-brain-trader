"""
Per-coin "fundamental" rules: Elliott waves for ETH, trend for BTC,
mean reversion for gold, liquidity for ZEC. All are ADDITIONAL inputs to
the fly (market.coin_features), not separate strategies.
"""

import numpy as np
import pandas as pd
import pytest

from flybrain.features.price import (
    ELLIOTT_FEATURES,
    LIQUIDITY_FEATURES,
    MEAN_REVERSION_FEATURES,
    TREND_FEATURES,
    _causal_zigzag,
    per_coin_features,
)


def _synthetic_ohlcv(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "dt": dt,
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "close": close,
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "volume": rng.uniform(1, 2, n) * (1 + np.abs(rng.normal(0, 0.3, n))),
    })


def test_causal_zigzag_actually_confirms_reversals():
    """
    Regression: the first version of _causal_zigzag never confirmed a single
    reversal (while the direction was undecided, two consecutive `if`
    branches ran and the second overwrote the extreme_price set by the
    first), so `direction` stayed 0 on any data. Explicitly check that many
    legs and both direction signs accumulate on a long random walk.
    """
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 5000))))
    direction, dist, legs, retr = _causal_zigzag(close, pct=0.03)

    assert legs.max() > 10, "zigzag barely confirms reversals: that is the bug"
    assert (direction == 1).sum() > 100
    assert (direction == -1).sum() > 100
    assert np.std(dist) > 0.01
    assert np.std(retr) > 0.01


def test_causal_zigzag_is_causal():
    """A shock in the last candle must not change the zigzag on earlier bars."""
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000))))
    d1, dist1, legs1, retr1 = _causal_zigzag(close, pct=0.03)

    shocked = close.copy()
    shocked.iloc[-1] *= 1.5
    d2, dist2, legs2, retr2 = _causal_zigzag(shocked, pct=0.03)

    assert np.array_equal(d1[:-1], d2[:-1])
    assert np.array_equal(legs1[:-1], legs2[:-1])
    assert np.allclose(dist1[:-1], dist2[:-1])
    assert np.allclose(retr1[:-1], retr2[:-1])


@pytest.mark.parametrize("names", [ELLIOTT_FEATURES, TREND_FEATURES, MEAN_REVERSION_FEATURES, LIQUIDITY_FEATURES])
def test_coin_feature_sets_compute_without_error_and_vary(names):
    df = _synthetic_ohlcv()
    out = per_coin_features(df, names)
    assert list(out.columns) == names
    assert out.iloc[-1].notna().all(), "feature was not computed to the end of the series"
    for n in names:
        assert out[n].dropna().std() > 1e-6, f"{n} is degenerate (zero variance)"


@pytest.mark.parametrize("names", [ELLIOTT_FEATURES, TREND_FEATURES, MEAN_REVERSION_FEATURES, LIQUIDITY_FEATURES])
def test_coin_feature_sets_are_causal(names):
    """A shock in the last candle must not change feature values on earlier bars."""
    df = _synthetic_ohlcv()
    out1 = per_coin_features(df, names)

    df2 = df.copy()
    df2.loc[df2.index[-1], ["close", "high", "low"]] *= 1.5
    out2 = per_coin_features(df2, names)

    a = out1.iloc[:-1].to_numpy()
    b = out2.iloc[:-1].to_numpy()
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b)), "a feature looked into the future"


def test_store_merges_common_and_coin_specific_features(monkeypatch, tmp_path):
    """
    market.coin_features adds extra channels ONLY to the listed coin; other
    coins do not get them. Tests the merge mechanism in features/store.py
    without network access (external sources disabled).
    """
    from flybrain.config import load_config
    from flybrain.features.store import build_inputs

    cfg = load_config()
    cfg.market.symbols = ["BTC/USDT", "ETH/USDT"]
    cfg.market.features = ["log_return_1", "rsi_14"]
    cfg.market.coin_features = {"ETH": ["elliott_dir", "elliott_wave_no"]}
    cfg.market.cross_coin = False
    cfg.market.baseline_feature = False
    cfg.market.external_features = []
    cfg.paths.data = str(tmp_path)

    n = 2000
    dt = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)

    def _fake_get_all(_cfg):
        out = {}
        for s in cfg.market.symbols:
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
            out[s] = pd.DataFrame({
                "dt": dt, "close": close,
                "open": close, "high": close * 1.01, "low": close * 0.99,
                "volume": rng.uniform(1, 2, n),
            })
        return out

    data = _fake_get_all(cfg)
    X, dt_out, names, prices = build_inputs(cfg, data, use_cache=False)

    assert "ETH:elliott_dir" in names and "ETH:elliott_wave_no" in names
    assert "BTC:elliott_dir" not in names, "ETH fundamentals leaked into BTC"
    assert "BTC:log_return_1" in names and "ETH:log_return_1" in names, "common features must be present for every coin"


def test_coin_feature_boost_amplifies_only_tagged_channels():
    """coin_feature_boost amplifies only the channels from coin_features and
    leaves the common ones untouched; otherwise it would override the
    decision rather than amplify an input."""
    from flybrain.config import load_config
    from flybrain.features.store import build_inputs

    n = 2000
    dt = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)

    def _data(cfg):
        out = {}
        for s in cfg.market.symbols:
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
            out[s] = pd.DataFrame({
                "dt": dt, "close": close,
                "open": close, "high": close * 1.01, "low": close * 0.99,
                "volume": rng.uniform(1, 2, n),
            })
        return out

    def _cfg(boost, tmp):
        c = load_config()
        c.market.symbols = ["BTC/USDT", "ETH/USDT"]
        c.market.features = ["log_return_1", "rsi_14"]
        c.market.coin_features = {"ETH": ["elliott_dir"]}
        c.market.coin_feature_boost = boost
        c.market.cross_coin = False
        c.market.baseline_feature = False
        c.market.external_features = []
        c.paths.data = str(tmp)
        return c

    import tempfile
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        cfg1 = _cfg(1.0, tmp1)
        X1, _, names, _ = build_inputs(cfg1, _data(cfg1), use_cache=False)
        rng = np.random.default_rng(0)   # same series again
        cfg2 = _cfg(2.0, tmp2)
        X2, _, names2, _ = build_inputs(cfg2, _data(cfg2), use_cache=False)

    assert names == names2
    i_boosted = names.index("ETH:elliott_dir")
    i_common = names.index("BTC:log_return_1")
    assert not np.allclose(X1[:, i_boosted], X2[:, i_boosted]), "the boosted channel must change"
    assert np.allclose(X1[:, i_common], X2[:, i_common]), "a common channel must not depend on the boost"
