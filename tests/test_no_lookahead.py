"""
The most important test in the project.

If a feature at candle t depends on what happened after t, the backtest will
show a handsome profit that will not exist in reality. We check it directly:
change the future and verify that the past did not move.
"""

import numpy as np
import pandas as pd
import pytest

from flybrain.market.features import build_one


def make_candles(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "ts": np.arange(n) * 3_600_000,
        "dt": pd.to_datetime(np.arange(n) * 3_600_000, unit="ms", utc=True),
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": rng.uniform(1, 100, n),
    })


FEATURES = [
    "log_return_1", "log_return_4", "log_return_24",
    "price_zscore_168", "volume_zscore_168",
    "rsi_14", "ema_spread_12_48", "volatility_24",
]


def test_future_does_not_leak_into_past():
    """Corrupt the last 100 candles. The first 400 must stay unchanged."""
    df = make_candles(500)
    a = build_one(df, FEATURES)

    tampered = df.copy()
    tampered.loc[400:, "close"] *= 3.0
    tampered.loc[400:, "volume"] *= 17.0
    b = build_one(tampered, FEATURES)

    for col in FEATURES:
        x, y = a[col].to_numpy()[:400], b[col].to_numpy()[:400]
        both_nan = np.isnan(x) & np.isnan(y)
        assert np.allclose(x[~both_nan], y[~both_nan], equal_nan=True), (
            f"feature {col} looks ahead: changing candles after 400 "
            f"changed values before 400"
        )


def test_features_are_bounded_after_squash():
    """
    After squashing all channels lie in [-1, 1], otherwise one would drown out
    the rest. Outliers beyond the window are clipped on purpose: one wild
    candle must not outweigh all the other inputs.
    """
    from flybrain.market.features import squash

    x = np.array([-1e6, -3.0, -1.0, 0.0, 1.0, 3.0, 1e6], dtype=np.float32)
    y = squash(x)
    assert np.all(np.abs(y) <= 1.0)
    # order is preserved within the working range
    assert y[1] < y[2] < y[3] < y[4] < y[5]
    # beyond the range values saturate instead of growing further
    assert np.isclose(y[0], y[1]) and np.isclose(y[6], y[5])


@pytest.mark.parametrize("n", [1, 4, 24])
def test_log_return_shifts_correctly(n):
    """An n-candle return must not use a future price."""
    from flybrain.market.features import log_return

    close = pd.Series(2.0 ** np.arange(40, dtype=float))
    r = log_return(close, n)
    assert np.isnan(r.iloc[:n]).all(), "the first n values must be empty"
    expected = np.log(close.iloc[n] / close.iloc[0])
    assert np.isclose(r.iloc[n], expected)
