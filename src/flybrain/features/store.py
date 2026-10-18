"""
Input matrix assembly (methodology v2).

    X, dt, names, prices = build_inputs(cfg, data)

X      - (T, C) hourly rows, channels in [-1, 1]
names  - channel names "COIN:feature" / "MKT:feature"; order = columns of X
prices - (T, n_coins) close prices aligned with X

Normalization: causal rolling z-score (30-day window), then tanh, as in v1
but with a longer window. Cached in parquet, keyed by a hash of the
feature config.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..config import Config
from ..market.features import squash
from . import external, price


def _feature_key(cfg: Config) -> str:
    payload = {
        "symbols": cfg.market.symbols,
        "timeframe": cfg.market.timeframe,
        "per_coin": cfg.market.features,
        "external": cfg.market.external_features,
        "cross_coin": cfg.market.cross_coin,
        "baseline_feature": cfg.market.baseline_feature,
        "coin_features": cfg.market.coin_features,
        "baseline": [cfg.trading.baseline.vol_window, cfg.trading.baseline.target_daily_vol,
                     cfg.trading.baseline.max_weight_per_coin],
        "norm_window": cfg.market.norm_window_hours,
        "raw_features": cfg.market.raw_features,
        "v": 4,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def build_inputs(cfg: Config, data: dict[str, pd.DataFrame], use_cache: bool = True,
                 with_teacher: bool | None = None):
    syms = cfg.market.symbols
    coins = [s.split("/")[0] for s in syms]
    base = data[syms[0]]
    dt_all = pd.DatetimeIndex(base["dt"])
    key = _feature_key(cfg)
    cache = cfg.path("data") / "features" / f"inputs_{key}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    if use_cache and cache.exists():
        cached = pd.read_parquet(cache)
        if len(cached) == len(dt_all) and cached.index[-1] == dt_all[-1]:
            feats = cached
        else:
            feats = None
    else:
        feats = None

    if feats is None:
        for c, s in zip(coins, syms):
            # shared feature list + per-coin "fundamental" extra channels
            # (Elliott waves for ETH, trend for BTC, mean reversion for
            # gold, etc.; see market.coin_features)
            names_for_coin = list(cfg.market.features) + [
                n for n in cfg.market.coin_features.get(c, []) if n not in cfg.market.features
            ]
            f = price.per_coin_features(data[s], names_for_coin)
            f.columns = [f"{c}:{n}" for n in f.columns]
            frames.append(f)
        if cfg.market.cross_coin:
            frames.append(price.cross_coin_features(data, cfg))
        if cfg.market.baseline_feature:
            frames.append(price.baseline_weight_features(data, cfg))
        if "fng" in cfg.market.external_features:
            fng = external.fear_greed_features(cfg, dt_all)
            fng.index = base.index
            frames.append(fng)
        if "sol_defi_tvl" in cfg.market.external_features and "SOL/USDT" in syms:
            tvl = external.solana_tvl_features(cfg, dt_all)
            tvl.index = base.index
            frames.append(tvl)
        feats = pd.concat(frames, axis=1)
        feats.index = dt_all
        feats.to_parquet(cache)

    names = list(feats.columns)
    raw = feats.to_numpy(dtype=np.float64)

    # causal normalization: rolling mean and std from the past only
    w = cfg.market.norm_window_hours
    df = pd.DataFrame(raw)
    mu = df.rolling(w, min_periods=w // 4).mean()
    sd = df.rolling(w, min_periods=w // 4).std()
    # copy=True is required: under copy-on-write pandas returns a read-only
    # array from to_numpy(), and below we write the "raw" channel columns
    # into it ("assignment destination is read-only")
    z = ((df - mu) / (sd + 1e-9)).to_numpy(dtype=np.float32, copy=True)

    # "Raw" channels (2026-09-16, D2): a rolling z-score of "formula weight
    # relative to its own 30-day mean" erases exactly what the fly needs to
    # know, the ABSOLUTE level (whether the formula has hit its cap).
    # base_weight is divided by max_weight_per_coin so the full cap reads as
    # 1.0; the other raw_features are already in a typical ~[-1, 1] range,
    # so they are only clipped.
    raw_names = set(cfg.market.raw_features)
    raw_mask = np.array([nm.split(":")[-1] in raw_names for nm in names])
    if raw_mask.any():
        raw_vals = raw[:, raw_mask].astype(np.float32)
        base_w_cols = np.array([nm.split(":")[-1] == "base_weight" for nm in names])[raw_mask]
        if base_w_cols.any():
            cap = max(cfg.trading.baseline.max_weight_per_coin, 1e-6)
            raw_vals[:, base_w_cols] = raw_vals[:, base_w_cols] / cap
        z[:, raw_mask] = np.clip(np.nan_to_num(raw_vals, nan=0.0), -1.0, 1.0)

    valid = ~np.isnan(z).any(axis=1)
    first = int(np.argmax(valid)) if valid.any() else len(z)
    z, dt = z[first:], dt_all[first:]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    X = squash(z)
    # squash is a shared tanh squash; for raw channels already in [-1, 1]
    # restore their values so tanh does not distort the signal shape
    if raw_mask.any():
        X[:, raw_mask] = np.clip(z[:, raw_mask], -1.0, 1.0)

    # "Stick to the fundamental rules" (2026-09-16, cycle 5v2): the brain
    # still decides on its own whether to listen to the signal (it is an
    # extra input, not a hard rule imposed on the fly), but the channel can be
    # made LOUDER so the fly has more reason to pay attention to it. This is
    # a multiplier on the channel value, not a replacement of the decision;
    # above 1.0 the feature stands out more at the input, below 1.0 it is muted.
    boost = cfg.market.coin_feature_boost
    if boost != 1.0 and cfg.market.coin_features:
        boosted = {f"{c}:{n}" for c, feats in cfg.market.coin_features.items() for n in feats}
        idx = [i for i, nm in enumerate(names) if nm in boosted]
        if idx:
            X[:, idx] = np.clip(X[:, idx] * boost, -1.0, 1.0)

    prices = np.stack([data[s]["close"].to_numpy(dtype=np.float64)[first:] for s in syms], axis=1)

    # --- teacher channels (methodology v2) ---
    # The forecasts are already out-of-fold and causally normalized, so they
    # must not be z-scored again; just append them as they are.
    want_teacher = cfg.teacher.enabled if with_teacher is None else with_teacher
    if want_teacher:
        from ..teacher.predict import load_or_train  # noqa: PLC0415

        ts = load_or_train(cfg, X, prices, dt, names, use_cache=use_cache)
        X = np.concatenate([X, np.clip(ts.values, -1.0, 1.0)], axis=1)
        names = names + ts.names
        if ts.ic_by_coin:
            print("  Teacher IC by coin: " +
                  ", ".join(f"{c} {v:+.3f}" for c, v in ts.ic_by_coin.items()))

    return X, dt, names, prices
