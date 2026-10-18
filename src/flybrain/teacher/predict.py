"""
Teacher forecasts as input channels for the fly.

The model is refit on a rolling window: for each test slice it trains on the
preceding train_months months, with an embargo of one horizon. A row that was
in its own model's training set never gets a prediction, only out-of-fold
ones. Otherwise the fly would receive the answer as input (look-ahead
leakage) and the whole backtest would be meaningless.

Channels per coin:
    COIN:teacher_12h   - 12h return forecast, normalized
    COIN:teacher_24h   - same for 24h
    COIN:teacher_conf  - agreement between the horizons (confidence)

Live mode. The cache key does NOT include the row count; otherwise every new
hourly candle would invalidate the cache and think() would refit 162 models
(27 windows x 3 coins x 2 horizons, about ten minutes) on every tick. Instead,
raw forecasts are cached as-is and the tail (new candles) is appended by a
"tail" model trained on the most recent train_months and refit every
TAIL_REFIT_DAYS days. Normalization is recomputed over the whole series in
milliseconds.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from .train import coin_columns, forward_returns, make_model

TAIL_REFIT_DAYS = 7


@dataclass
class TeacherSignals:
    """Teacher forecasts, aligned with the feature matrix."""
    values: np.ndarray          # (T, n_new_channels)
    names: list[str]
    ic_by_coin: dict[str, float]

    @property
    def n_channels(self) -> int:
        return self.values.shape[1]


def _cache_key(cfg: Config) -> str:
    payload = {
        "symbols": cfg.market.symbols,
        "horizons": cfg.teacher.horizons_hours,
        "train_months": cfg.teacher.train_months,
        "test_months": cfg.teacher.test_months,
        "model": [cfg.teacher.max_iter, cfg.teacher.learning_rate, cfg.teacher.max_depth,
                  cfg.teacher.min_samples_leaf, cfg.teacher.l2],
        "v": 2,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def out_of_fold_predictions(
    cfg: Config, X: np.ndarray, prices: np.ndarray, dt: pd.DatetimeIndex,
    names: list[str], horizon: int, verbose: bool = True,
) -> np.ndarray:
    """
    A forecast for every row, made by a model that never saw that row. The
    first train_months months get no forecast (NaN): there is nothing to
    train on yet.
    """
    coins = [s.split("/")[0] for s in cfg.market.symbols]
    fwd = forward_returns(prices, horizon)
    out = np.full((len(X), len(coins)), np.nan, dtype=np.float64)

    t0 = dt[0] + pd.DateOffset(months=cfg.teacher.train_months)
    n_fits = 0
    while True:
        te_end = t0 + pd.DateOffset(months=cfg.teacher.test_months)
        tr_start = t0 - pd.DateOffset(months=cfg.teacher.train_months)
        tr = np.where((dt >= tr_start) & (dt < t0))[0]
        te = np.where((dt >= t0) & (dt < min(te_end, dt[-1] + pd.Timedelta(hours=1))))[0]
        if len(te) == 0:
            break
        tr = tr[:-horizon] if len(tr) > horizon else tr      # embargo
        if len(tr) >= 500:
            for i, c in enumerate(coins):
                cols = coin_columns(names, c)
                y = fwd[tr, i]
                ok = np.isfinite(y)
                if ok.sum() < 500:
                    continue
                model = make_model(cfg)
                model.fit(X[np.ix_(tr[ok], cols)], y[ok])
                out[te, i] = model.predict(X[np.ix_(te, cols)])
            n_fits += 1
            if verbose and n_fits % 5 == 0:
                print(f"    teacher {horizon}h: window {n_fits}, up to {dt[te[-1]]:%Y-%m}", end="\r")
        t0 = te_end
        if t0 > dt[-1]:
            break
    if verbose:
        print(f"    teacher {horizon}h: trained on {n_fits} windows" + " " * 20)
    return out


def _tail_predictions(
    cfg: Config, X: np.ndarray, prices: np.ndarray, dt: pd.DatetimeIndex,
    names: list[str], horizon: int, start_row: int, model_dir: Path,
) -> np.ndarray:
    """
    Forecast rows start_row..end with the "tail" model. The model lives in
    model_dir and is refit when older than TAIL_REFIT_DAYS. It trains on the
    last train_months BEFORE start_row with a horizon-length embargo, so it
    has not seen the tail rows.
    """
    import joblib  # noqa: PLC0415  (sklearn dependency)

    coins = [s.split("/")[0] for s in cfg.market.symbols]
    fwd = forward_returns(prices, horizon)
    out = np.full((len(X) - start_row, len(coins)), np.nan, dtype=np.float64)
    tr_end = start_row
    tr_start_ts = dt[tr_end - 1] - pd.DateOffset(months=cfg.teacher.train_months)
    tr = np.where((dt >= tr_start_ts))[0]
    tr = tr[tr < tr_end]
    tr = tr[:-horizon] if len(tr) > horizon else tr

    for i, c in enumerate(coins):
        path = model_dir / f"tail_{c}_{horizon}h.joblib"
        model = None
        if path.exists() and (time.time() - path.stat().st_mtime) < TAIL_REFIT_DAYS * 86400:
            try:
                model = joblib.load(path)
            except Exception:  # noqa: BLE001
                model = None
        if model is None:
            cols = coin_columns(names, c)
            y = fwd[tr, i]
            ok = np.isfinite(y)
            if ok.sum() < 500:
                continue
            model = make_model(cfg)
            model.fit(X[np.ix_(tr[ok], cols)], y[ok])
            joblib.dump(model, path)
        cols = coin_columns(names, c)
        out[:, i] = model.predict(X[np.ix_(np.arange(start_row, len(X)), cols)])
    return out


def _scale(raw: dict[int, np.ndarray], coins: list[str], horizons: list[int]) -> tuple[dict[str, np.ndarray], int]:
    """Raw forecasts -> channels in [-1, 1]; causal normalization (rolling std)."""
    cols: dict[str, np.ndarray] = {}
    h_main = horizons[0]
    for i, c in enumerate(coins):
        scaled: list[np.ndarray] = []
        for h in horizons:
            s = pd.Series(raw[h][:, i])
            sd = s.rolling(24 * 30, min_periods=24 * 7).std()
            z = np.tanh((s / (sd + 1e-9)).to_numpy() / 3.0)
            z = np.nan_to_num(z, nan=0.0)
            cols[f"{c}:teacher_{h}h"] = z
            scaled.append(z)
        agree = np.sign(scaled[0]) == np.sign(scaled[-1])
        mag = np.minimum(np.abs(scaled[0]), np.abs(scaled[-1]))
        cols[f"{c}:teacher_conf"] = np.where(agree, mag, -mag)
    return cols, h_main


def load_or_train(
    cfg: Config, X: np.ndarray, prices: np.ndarray, dt: pd.DatetimeIndex,
    names: list[str], use_cache: bool = True,
) -> TeacherSignals:
    """Teacher forecasts as ready-made channels. The cache holds raw forecasts in parquet."""
    coins = [s.split("/")[0] for s in cfg.market.symbols]
    horizons = list(cfg.teacher.horizons_hours)
    key = _cache_key(cfg)
    fdir = cfg.path("data") / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    raw_path = fdir / f"teacher_raw_{key}.parquet"
    ic_path = fdir / f"teacher_raw_{key}.ic.json"
    model_dir = fdir / f"teacher_tail_{key}"
    model_dir.mkdir(parents=True, exist_ok=True)

    raw: dict[int, np.ndarray] = {}
    cached_rows = 0
    if use_cache and raw_path.exists():
        df = pd.read_parquet(raw_path)
        cached_rows = len(df)
        if cached_rows <= len(X):
            for h in horizons:
                arr = np.full((len(X), len(coins)), np.nan, dtype=np.float64)
                arr[:cached_rows] = df[[f"raw_{c}_{h}" for c in coins]].to_numpy()
                raw[h] = arr
        else:
            cached_rows = 0        # fewer rows than cached: recompute from scratch

    if cached_rows == 0:
        for h in horizons:
            raw[h] = out_of_fold_predictions(cfg, X, prices, dt, names, h)
    elif cached_rows < len(X):
        n_new = len(X) - cached_rows
        print(f"    teacher: appending a tail of {n_new} candles with the tail model")
        for h in horizons:
            raw[h][cached_rows:] = _tail_predictions(
                cfg, X, prices, dt, names, h, cached_rows, model_dir,
            )

    if cached_rows < len(X):
        df = pd.DataFrame({f"raw_{c}_{h}": raw[h][:, i]
                           for h in horizons for i, c in enumerate(coins)})
        df.to_parquet(raw_path, index=False)

    cols, h_main = _scale(raw, coins, horizons)

    ic_by_coin: dict[str, float] = {}
    if ic_path.exists() and cached_rows == len(X):
        ic_by_coin = json.loads(ic_path.read_text(encoding="utf-8"))
    else:
        fwd = forward_returns(prices, h_main)
        for i, c in enumerate(coins):
            ok = np.isfinite(raw[h_main][:, i]) & np.isfinite(fwd[:, i])
            if ok.sum() > 100:
                a = pd.Series(raw[h_main][ok, i]).rank().to_numpy()
                b = pd.Series(fwd[ok, i]).rank().to_numpy()
                ic_by_coin[c] = float(np.corrcoef(a, b)[0, 1])
        ic_path.write_text(json.dumps(ic_by_coin, indent=2), encoding="utf-8")

    values = np.stack([cols[n] for n in cols], axis=1).astype(np.float32)
    return TeacherSignals(values, list(cols.keys()), ic_by_coin)
