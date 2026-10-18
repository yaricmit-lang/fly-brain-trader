"""
Price features (methodology v2): the original 10 plus multi-scale and cross-coin ones.

Everything is computed on hourly candles and from the past only; the fly
then reads them once every 4 hours. Column names are "COIN:name", or
"MKT:name" for market-wide ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..market.features import (
    EPS,
    ema_spread,
    log_return,
    realized_volatility,
    rolling_zscore,
    rsi,
)


def _causal_zigzag(close: pd.Series, pct: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Causal zigzag approximating Elliott waves: until price reverses by at
    least `pct` from the current running extreme, that extreme is not
    confirmed, which is exactly how a trader would see the zigzag in real
    time, with no look-ahead.

    Returns per bar: direction of the current leg (+1/-1/0), normalized
    distance from the last confirmed pivot, leg number (count of confirmed
    reversals, an approximation of the "wave number") and the retracement of
    the current move relative to the SIZE of the previous leg (the classic
    Elliott/Fibonacci relation: a correction usually retraces 38-62% of the
    prior impulse).
    """
    c = close.to_numpy(dtype=np.float64)
    n = len(c)
    direction = np.zeros(n)
    dist = np.zeros(n)
    legs = np.zeros(n)
    retr = np.zeros(n)
    if n == 0:
        return direction, dist, legs, retr

    last_pivot_price = c[0]
    extreme_price = c[0]
    run_max, run_min = c[0], c[0]      # while direction is undecided (cur_dir=0)
    prev_leg_size = 0.0
    cur_dir = 0
    leg_count = 0
    for i in range(n):
        p = c[i]
        if cur_dir == 0:
            # BEFORE the first confirmed reversal both the running max and
            # the running min must be tracked at once, since it is not yet
            # known which way the first leg will be confirmed. This used to
            # be two independent consecutive "if"s, and the second
            # immediately overwrote the extreme_price set by the first, so a
            # reversal was NEVER confirmed and the whole signal stayed silent.
            if p > run_max:
                run_max = p
            if p < run_min:
                run_min = p
            if p <= run_max * (1 - pct):
                prev_leg_size = run_max - last_pivot_price
                last_pivot_price, cur_dir, extreme_price = run_max, -1, p
                leg_count += 1
            elif p >= run_min * (1 + pct):
                prev_leg_size = run_min - last_pivot_price
                last_pivot_price, cur_dir, extreme_price = run_min, 1, p
                leg_count += 1
        elif cur_dir == 1:
            if p > extreme_price:
                extreme_price = p
            if p <= extreme_price * (1 - pct):
                prev_leg_size = extreme_price - last_pivot_price
                last_pivot_price, cur_dir, extreme_price = extreme_price, -1, p
                leg_count += 1
        else:
            if p < extreme_price:
                extreme_price = p
            if p >= extreme_price * (1 + pct):
                prev_leg_size = extreme_price - last_pivot_price
                last_pivot_price, cur_dir, extreme_price = extreme_price, 1, p
                leg_count += 1
        direction[i] = cur_dir
        dist[i] = (p - last_pivot_price) / (abs(last_pivot_price) + EPS)
        legs[i] = leg_count
        retr[i] = (p - last_pivot_price) / (abs(prev_leg_size) + EPS) if prev_leg_size else 0.0
    return direction, dist, legs, retr

# per-coin features: name -> function(df) -> Series
def _per_coin_calcs(df: pd.DataFrame) -> dict[str, "callable"]:
    close, high, low, volume, dt = df["close"], df["high"], df["low"], df["volume"], df["dt"]
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    opn = df["open"]
    # --- v3: everything about future volatility (hard cycle) ---
    # The risk-targeting formula uses the std of hourly returns over 168h.
    # That is slow and blind to what is known in advance: candle-range
    # estimators (Parkinson / Garman-Klass) estimate volatility 5-7x more
    # efficiently from the same data, and crypto has strong seasonality:
    # nights/weekends are quiet, the US session is loud. All causal: past
    # candles only.
    hl2 = np.log((high + EPS) / (low + EPS)) ** 2
    co2 = np.log((close + EPS) / (opn + EPS)) ** 2
    park_24 = np.sqrt(hl2.rolling(24, min_periods=12).mean() / (4 * np.log(2)))
    gk_24 = np.sqrt((0.5 * hl2 - (2 * np.log(2) - 1) * co2).rolling(24, min_periods=12).mean().clip(lower=0))
    ar = log_return(close, 1).abs()
    how = dow * 24 + hour                                   # hour of week 0..167
    # seasonal volatility of this hour of the week: mean |r| over the 8
    # previous identical hours (shift(1) within the group excludes the
    # current candle)
    seas = ar.groupby(how).transform(lambda x: x.shift(1).rolling(8, min_periods=4).mean())
    level = ar.shift(1).rolling(168 * 8, min_periods=168 * 2).mean()
    seas_idx = seas / (level + EPS)
    # expected seasonal volatility for the NEXT 12 hours: slot how+k last
    # occurred 168-k hours ago, and its estimate at that time is past-only
    seas_next12 = sum(seas_idx.shift(168 - k) for k in range(1, 13)) / 12.0

    # --- fundamentals (2026-09-16, per coin): ETH uses Elliott waves,
    # BTC trend, gold mean reversion for frequent small trades ---
    zz_dir, zz_dist, zz_legs, zz_retr = _causal_zigzag(close, pct=0.03)
    ramp = np.arange(len(close), dtype=np.float64)
    logc = np.log(close.clip(lower=EPS))

    def _rolling_r2(window: int) -> pd.Series:
        # trend strength and cleanliness: R^2 of a linear regression of log-price on time
        r = pd.Series(ramp, index=close.index).rolling(window).corr(logc)
        return (r ** 2).fillna(0.0)

    def _rolling_autocorr(window: int) -> pd.Series:
        # negative = mean-reverting regime, what frequent small trades need;
        # positive = momentum-like movement
        r = log_return(close, 1)
        return r.rolling(window).apply(
            lambda x: pd.Series(x).autocorr(1) if np.std(x) > 0 else 0.0, raw=False
        ).fillna(0.0)

    return {
        # --- base (as in v1) ---
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
        # --- multi-scale (v2) ---
        "log_return_12": lambda: log_return(close, 12),
        "log_return_72": lambda: log_return(close, 72),
        "log_return_168": lambda: log_return(close, 168),
        "volatility_168": lambda: realized_volatility(close, 168),
        "vol_of_vol": lambda: realized_volatility(close, 24).rolling(168, min_periods=48).std(),
        "atr_14_rel": lambda: tr.rolling(14).mean() / (close + EPS),
        "drawdown_30d": lambda: close / close.rolling(720, min_periods=168).max() - 1.0,
        "dist_sma_7d": lambda: close / close.rolling(168, min_periods=84).mean() - 1.0,
        "dist_sma_30d": lambda: close / close.rolling(720, min_periods=168).mean() - 1.0,
        "rsi_14_4h": lambda: rsi(close.iloc[::4], 14).reindex(close.index).ffill(),
        "dow_sin": lambda: np.sin(2 * np.pi * dow / 7),
        "dow_cos": lambda: np.cos(2 * np.pi * dow / 7),
        # --- v3 ---
        "parkinson_24": lambda: park_24,
        "gk_24": lambda: gk_24,
        "vol_ratio_24_168": lambda: realized_volatility(close, 24) / (realized_volatility(close, 168) + EPS),
        "park_ratio_24_168": lambda: park_24 / (np.sqrt(hl2.rolling(168, min_periods=84).mean() / (4 * np.log(2))) + EPS),
        "seasonal_vol_now": lambda: seas_idx,
        "seasonal_vol_next12": lambda: seas_next12,
        "range_4h_rel": lambda: (high.rolling(4).max() - low.rolling(4).min()) / (close + EPS),
        "weekend": lambda: (dow >= 5).astype(float),
        "abs_ret_24_mean": lambda: ar.rolling(24, min_periods=12).mean(),
        # --- ETH: Elliott wave approximation (causal zigzag + Fibonacci) ---
        "elliott_dir": lambda: pd.Series(zz_dir, index=close.index),
        "elliott_dist_from_pivot": lambda: pd.Series(np.tanh(zz_dist * 5), index=close.index),
        "elliott_wave_no": lambda: pd.Series(zz_legs % 5, index=close.index) / 5.0 * pd.Series(zz_dir, index=close.index),
        "elliott_fib_retrace": lambda: pd.Series(np.clip(zz_retr, -2, 2), index=close.index),
        # --- BTC: trend strength and regime ---
        "trend_r2_168": lambda: _rolling_r2(168),
        "trend_ema_slope_200": lambda: close.ewm(span=200, min_periods=100).mean().pct_change(24).fillna(0.0),
        "donchian_pos_55": lambda: (
            (close - low.rolling(55, min_periods=28).min())
            / (high.rolling(55, min_periods=28).max() - low.rolling(55, min_periods=28).min() + EPS)
        ),
        # --- gold: mean reversion for frequent small trades ---
        "mr_zscore_12": lambda: rolling_zscore(close, 12),
        "mr_bb_pctb_20": lambda: (
            (close - close.rolling(20, min_periods=10).mean())
            / (2 * close.rolling(20, min_periods=10).std() + EPS) * 0.5 + 0.5
        ),
        "mr_autocorr_24": lambda: _rolling_autocorr(24),
        "mr_range_tightness": lambda: (
            (high.rolling(24, min_periods=12).max() - low.rolling(24, min_periods=12).min())
            / (high.rolling(168, min_periods=84).max() - low.rolling(168, min_periods=84).min() + EPS)
        ),
        # --- ZEC: no honest free history of on-chain metrics (share of
        # shielded transactions, etc.) was found (web research 2026-09-16:
        # Zchain/Solana Beach are down, the rest have either a paid API or
        # no history). Instead of faking on-chain data, use what is truly
        # measurable from our own data: market liquidity. ZEC trades thinner
        # than BTC/ETH, which is itself a fundamental factor for its price
        # and slippage.
        "liq_amihud": lambda: (
            log_return(close, 1).abs() / (volume * close + EPS)
        ).rolling(24, min_periods=12).mean(),   # Amihud: |return| / dollar volume
        "liq_volume_z": lambda: rolling_zscore(volume, 168),
        "liq_volume_trend": lambda: (
            volume.rolling(24, min_periods=12).mean() / (volume.rolling(168, min_periods=84).mean() + EPS) - 1.0
        ),
    }


PER_COIN_V2 = [
    "log_return_1", "log_return_4", "log_return_12", "log_return_24", "log_return_72",
    "log_return_168", "price_zscore_168", "volume_zscore_168", "rsi_14", "rsi_14_4h",
    "ema_spread_12_48", "volatility_24", "volatility_168", "vol_of_vol", "atr_14_rel",
    "drawdown_30d", "dist_sma_7d", "dist_sma_30d", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


PER_COIN_V3 = PER_COIN_V2 + [
    "parkinson_24", "gk_24", "vol_ratio_24_168", "park_ratio_24_168",
    "seasonal_vol_now", "seasonal_vol_next12", "range_4h_rel", "weekend", "abs_ret_24_mean",
]

# --- per-coin "fundamental" sets (2026-09-16): extra channels on top of the
# shared list, not a replacement; the fly learns how far to trust them.
# ETH is traded by Elliott waves, BTC by trend, gold by frequent small
# mean-reversion trades; see market.coin_features in the config.
ELLIOTT_FEATURES = ["elliott_dir", "elliott_dist_from_pivot", "elliott_wave_no", "elliott_fib_retrace"]
TREND_FEATURES = ["trend_r2_168", "trend_ema_slope_200", "donchian_pos_55"]
MEAN_REVERSION_FEATURES = ["mr_zscore_12", "mr_bb_pctb_20", "mr_autocorr_24", "mr_range_tightness"]
LIQUIDITY_FEATURES = ["liq_amihud", "liq_volume_z", "liq_volume_trend"]   # ZEC


def baseline_weight_features(data: dict, cfg: Config) -> pd.DataFrame:
    """
    The risk-targeting formula's own weight as an input to the fly:
    COIN:base_weight. The fly applies a multiplier on top of the formula, so
    it should know how much the formula has already taken. Computed by the
    same function as in backtest/live.
    """
    from ..trading.baseline import vol_target_weights  # noqa: PLC0415

    syms = cfg.market.symbols
    coins = [s.split("/")[0] for s in syms]
    prices = np.stack([data[s]["close"].to_numpy(dtype=np.float64) for s in syms], axis=1)
    w = vol_target_weights(cfg, prices)
    return pd.DataFrame({f"{c}:base_weight": w[:, i] for i, c in enumerate(coins)},
                        index=data[syms[0]].index)


def per_coin_features(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    calcs = _per_coin_calcs(df)
    unknown = [n for n in names if n not in calcs]
    if unknown:
        raise ValueError(f"unknown price features: {unknown}. Available: {sorted(calcs)}")
    out = pd.DataFrame(index=df.index)
    for n in names:
        out[n] = calcs[n]()
    return out


def cross_coin_features(data: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """
    Cross-coin features: BTC return as a feature for all coins, each coin's
    relative strength against BTC, and BTC "dominance" versus the basket by
    price dynamics (a substitute for unavailable market-cap history).
    """
    syms = cfg.market.symbols
    coins = [s.split("/")[0] for s in syms]
    closes = pd.DataFrame({c: data[s]["close"].values for c, s in zip(coins, syms)})
    out = pd.DataFrame(index=closes.index)
    btc = coins[0]
    r24 = np.log(closes / closes.shift(24))
    out[f"MKT:{btc}_ret24"] = r24[btc]
    basket = r24.mean(axis=1)
    out["MKT:btc_vs_basket_24"] = r24[btc] - basket
    out["MKT:btc_vs_basket_168"] = np.log(closes[btc] / closes[btc].shift(168)) - np.log(closes / closes.shift(168)).mean(axis=1)
    for c in coins[1:]:
        out[f"{c}:rel_strength_btc_24"] = r24[c] - r24[btc]
        out[f"{c}:rel_strength_btc_168"] = np.log(closes[c] / closes[c].shift(168)) - np.log(closes[btc] / closes[btc].shift(168))
    return out
