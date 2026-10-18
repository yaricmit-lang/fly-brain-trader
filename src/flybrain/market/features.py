"""
Market features: the "odors" the fly smells.

Main rule: no look-ahead. Each feature at candle t is computed from data up
to and including t only. Normalization is rolling rather than over the whole
history, otherwise test-set statistics would leak into training.

The leakage check lives in tests/test_no_lookahead.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, load_config

EPS = 1e-9


# --------------------------------------------------------------------------
#  Individual features
# --------------------------------------------------------------------------
def log_return(close: pd.Series, n: int) -> pd.Series:
    """Log return over n candles."""
    return np.log(close / close.shift(n))


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    """Rolling-window z-score: (x - mean) / std."""
    m = x.rolling(window, min_periods=window // 2).mean()
    s = x.rolling(window, min_periods=window // 2).std()
    return (x - m) / (s + EPS)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative strength index with Wilder smoothing.
    Returned in [-1, 1] rather than [0, 100]: a symmetric range suits the network better.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + EPS)
    return (100 - 100 / (1 + rs)) / 50.0 - 1.0


def ema_spread(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Fast-minus-slow moving average gap as a fraction of price: a trend proxy."""
    f = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return (f - s) / (close + EPS)


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Realized volatility: std of returns within the window."""
    return np.log(close / close.shift(1)).rolling(window, min_periods=window // 2).std()


# --------------------------------------------------------------------------
#  Assembly
# --------------------------------------------------------------------------
def build_one(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute all requested features for one coin."""
    close, volume, dt = df["close"], df["volume"], df["dt"]
    hour = dt.dt.hour

    calc = {
        "log_return_1": lambda: log_return(close, 1),
        "log_return_4": lambda: log_return(close, 4),
        "log_return_24": lambda: log_return(close, 24),
        "price_zscore_168": lambda: rolling_zscore(close, 168),
        "volume_zscore_168": lambda: rolling_zscore(volume, 168),
        "rsi_14": lambda: rsi(close, 14),
        "ema_spread_12_48": lambda: ema_spread(close, 12, 48),
        "volatility_24": lambda: realized_volatility(close, 24),
        "hour_sin": lambda: np.sin(2 * np.pi * hour / 24),
        "hour_cos": lambda: np.cos(2 * np.pi * hour / 24),
    }

    unknown = [f for f in features if f not in calc]
    if unknown:
        raise ValueError(f"unknown features: {unknown}. Available: {sorted(calc)}")

    out = pd.DataFrame({"ts": df["ts"], "dt": dt})
    for name in features:
        out[name] = calc[name]()
    return out


def squash(x: np.ndarray, limit: float = 3.0) -> np.ndarray:
    """
    Squash everything into [-1, 1]. Without this a single outlier channel
    drowns out the others, since each neuron's input passes through tanh anyway.
    """
    return np.tanh(np.clip(x, -limit, limit) / limit * 1.5)


def build_matrix(
    data: dict[str, pd.DataFrame],
    cfg: Config | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    Build the input matrix (T, C): time x channels.
    Channel order matches roles.build_channel_names exactly: coin x feature.

    Returns (X, timestamps, channel names).
    """
    cfg = cfg or load_config()
    feats = cfg.market.features

    frames: list[pd.DataFrame] = []
    names: list[str] = []
    for sym in cfg.market.symbols:
        f = build_one(data[sym], feats)
        coin = sym.split("/")[0]
        frames.append(f[feats])
        names.extend(f"{coin}:{c}" for c in feats)

    X = pd.concat(frames, axis=1).to_numpy(dtype=np.float32)
    dt = pd.DatetimeIndex(data[cfg.market.symbols[0]]["dt"])

    # the first rows contain NaN (windows need warm-up), so trim them
    valid = ~np.isnan(X).any(axis=1)
    first = int(np.argmax(valid)) if valid.any() else len(X)
    X, dt = X[first:], dt[first:]

    # fill residual NaN (gaps in volume) with zero, meaning "no signal"
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return squash(X), dt, names


def prices_matrix(data: dict[str, pd.DataFrame], cfg: Config, n_skip: int) -> np.ndarray:
    """Close prices (T, n_coins), aligned with the feature matrix."""
    cols = [data[s]["close"].to_numpy(dtype=np.float64)[n_skip:] for s in cfg.market.symbols]
    return np.stack(cols, axis=1)
