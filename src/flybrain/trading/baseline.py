"""
Baseline position: the layer the fly operates on top of.

Measurements on our data showed that price direction is unpredictable
(IC ~0.01 at every horizon) while volatility is highly predictable
(IC +0.61). So a simple formula decides direction, and the fly learns
what the data actually supports: how much risk to take.

The baseline formula is volatility targeting: the capital fraction is
inversely proportional to recent volatility. Checked on three periods:
max drawdown is 5-21 pp lower in all of them, Sharpe is better in two
of three.

Final position = baseline x exposure multiplier from the fly.
A fly with no signal outputs a multiplier of 1.0 and reproduces the
formula exactly, so any difference in results is the fly's own
contribution rather than a property of the baseline. This is the fair
reference for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def realized_vol(prices: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation of hourly log returns."""
    lr = np.vstack([np.zeros((1, prices.shape[1])), np.log(prices[1:] / np.maximum(prices[:-1], 1e-12))])
    return pd.DataFrame(lr).rolling(window, min_periods=window // 4).std().to_numpy()


def vol_target_weights(cfg: Config, prices: np.ndarray) -> np.ndarray:
    """
    Capital fractions from the volatility-targeting formula, shape (T, n_coins).

    target_daily_vol / (hourly vol * sqrt(24)) is the factor by which to scale
    the position so that daily risk stays roughly constant. The result is
    divided by the number of coins and capped to avoid leverage.
    """
    b = cfg.trading.baseline
    vol = realized_vol(prices, b.vol_window)
    daily = vol * np.sqrt(24.0)
    n = prices.shape[1]
    w = b.target_daily_vol / (daily + 1e-9) / n
    w = np.clip(np.nan_to_num(w, nan=0.0), 0.0, b.max_weight_per_coin)

    # Cap on gross exposure (sum of |w|). Without it the formula routinely
    # used >100% of capital (mean sum(w) = 1.43 at target_daily_vol=0.04), and
    # clip_target rescaled the whole vector back, so the fly's multiplier was
    # cancelled by the same factor and changed nothing. Keeping the formula
    # deliberately "not full" leaves the fly real headroom.
    gross = w.sum(axis=1, keepdims=True)
    scale = np.minimum(1.0, b.max_gross / np.maximum(gross, 1e-9))
    w = w * scale
    return w


def trend_sign(prices: np.ndarray, window: int) -> np.ndarray:
    """Slow trend: +1 if price is above its moving average, otherwise -1."""
    sma = pd.DataFrame(prices).rolling(window, min_periods=window // 4).mean().to_numpy()
    s = np.sign(np.nan_to_num(prices - sma))
    s[s == 0] = 1.0
    return s


def baseline_positions(cfg: Config, prices: np.ndarray) -> np.ndarray:
    """
    Baseline on top of which the fly scales exposure.

    kind=vol_target  - long-only, risk-based sizing (the reference from our measurements)
    kind=vol_trend   - same sizing, sign taken from a slow trend
                       (adds shorts during prolonged declines)
    kind=equal       - equal weights, i.e. buy-and-hold
    """
    b = cfg.trading.baseline
    if b.kind == "equal":
        return np.full(prices.shape, 1.0 / prices.shape[1])
    w = vol_target_weights(cfg, prices)
    if b.kind == "vol_trend":
        w = w * trend_sign(prices, b.trend_window)
    return w
