"""
External data sources.

The store's key rule is the available_at column: when a value actually
became known. A daily index is published for the day after it ends, so a
decision at 14:00 the same day may only use yesterday's value. An as-of join
on available_at enforces this automatically, and a test checks that no
future data leaks.

Sources (no API keys):
- Fear & Greed index (alternative.me): daily, since 2018-02-01.
- BTC dominance: CoinGecko's free tier only serves 365 days of history
  (checked 2026-09-15, error 10012), so it is not used. BTC relative
  strength against a basket is computed from our own prices in price.py.
- Solana DeFi TVL (DefiLlama): daily, no key, full history. Checked
  2026-09-16 (web research): the other candidates for SOL/ZEC
  "fundamentals" (Solana Beach, Zchain, Artemis, Dune) either have no free
  API with history or the service is currently down (Solana Beach returned
  522, Zchain 520). For ZEC no honest free source with history was found at
  all (only web dashboards without an API). Instead of invented on-chain
  data, ZEC gets price-based liquidity features (features/price.py:
  LIQUIDITY_FEATURES, Amihud illiquidity and volume regime), which are
  actually available and honestly called proxies rather than a fake of a
  "real" on-chain signal.
"""

from __future__ import annotations

import time
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import requests

from ..config import Config

FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
SOLANA_TVL_URL = "https://api.llama.fi/v2/historicalChainTvl/solana"


def _cache_dir(cfg: Config) -> Path:
    d = cfg.path("data") / "external"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_fear_greed(cfg: Config, max_age_hours: float = 12.0) -> pd.DataFrame:
    """
    Daily Fear & Greed index. Cached in parquet, refreshed every half day.

    Returns a DataFrame: ts (day, UTC 00:00), value (0..100), available_at
    (the next 00:00 UTC: a day's index is published the following day and
    cannot be known earlier).
    """
    path = _cache_dir(cfg) / "fear_greed.parquet"
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return pd.read_parquet(path)

    try:
        r = requests.get(FNG_URL, timeout=30, verify=certifi.where())
        r.raise_for_status()
        rows = r.json()["data"]
    except Exception as e:  # noqa: BLE001
        if path.exists():
            print(f"  Fear & Greed: network unavailable ({type(e).__name__}), using cache")
            return pd.read_parquet(path)
        raise

    df = pd.DataFrame({
        "ts": pd.to_datetime([int(x["timestamp"]) for x in rows], unit="s", utc=True),
        "value": [float(x["value"]) for x in rows],
    }).sort_values("ts").reset_index(drop=True)
    df["ts"] = df["ts"].dt.floor("D")
    df["available_at"] = df["ts"] + pd.Timedelta(days=1)
    df.to_parquet(path, index=False)
    return df


def fetch_solana_tvl(cfg: Config, max_age_hours: float = 12.0) -> pd.DataFrame:
    """
    Daily Solana DeFi TVL (DefiLlama, no key): a "fundamental" metric for
    SOL showing how much capital is actually locked in its ecosystem, not
    just the price. Cached in parquet like fetch_fear_greed.

    Returns ts (day, UTC 00:00), tvl (USD), available_at (the next day: the
    same conservative rule as for Fear & Greed, since the daily point is
    published/recomputed after the fact).
    """
    path = _cache_dir(cfg) / "solana_tvl.parquet"
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return pd.read_parquet(path)

    try:
        r = requests.get(SOLANA_TVL_URL, timeout=30, verify=certifi.where())
        r.raise_for_status()
        rows = r.json()
    except Exception as e:  # noqa: BLE001
        if path.exists():
            print(f"  Solana TVL: network unavailable ({type(e).__name__}), using cache")
            return pd.read_parquet(path)
        raise

    df = pd.DataFrame({
        "ts": pd.to_datetime([int(x["date"]) for x in rows], unit="s", utc=True),
        "tvl": [float(x["tvl"]) for x in rows],
    }).sort_values("ts").reset_index(drop=True)
    df["ts"] = df["ts"].dt.floor("D")
    df["available_at"] = df["ts"] + pd.Timedelta(days=1)
    df.to_parquet(path, index=False)
    return df


def solana_tvl_features(cfg: Config, base_ts: pd.DatetimeIndex) -> pd.DataFrame:
    """
    SOL:defi_tvl_chg30 - 30-day change in Solana DeFi TVL (capital inflow or
    outflow, invisible in the SOL/USDT price alone).
    SOL:defi_tvl_z90   - z-score of daily TVL over a rolling 90-day window.
    Everything is causal (available_at one day after the point), using the
    same as-of join.
    """
    raw = fetch_solana_tvl(cfg)
    daily = raw.set_index("ts")["tvl"]
    chg30 = np.log(daily / daily.shift(30).replace(0, np.nan))
    z90 = (daily - daily.rolling(90, min_periods=30).mean()) / (daily.rolling(90, min_periods=30).std() + 1e-6)
    aux = pd.DataFrame({"chg30": chg30, "z90": z90}).reset_index()
    aux["available_at"] = aux["ts"] + pd.Timedelta(days=1)
    j = asof_join(base_ts, aux, ["chg30", "z90"])
    return pd.DataFrame({
        "SOL:defi_tvl_chg30": np.tanh(j["chg30"].fillna(0.0).to_numpy()),
        "SOL:defi_tvl_z90": j["z90"].fillna(0.0).to_numpy(),
    }, index=base_ts)


def asof_join(base_ts: pd.DatetimeIndex, table: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """
    For each moment in base_ts take the latest row of `table` with
    available_at <= ts, i.e. "what was known at decision time".
    """
    # pandas 3 keeps the time resolution (ms/us/ns) in the dtype and will not
    # merge different ones, so cast both sides to nanoseconds
    left = pd.DataFrame({"ts": pd.DatetimeIndex(base_ts).as_unit("ns")})
    right = table.sort_values("available_at")[["available_at", *value_cols]].copy()
    right["available_at"] = pd.DatetimeIndex(right["available_at"]).as_unit("ns")
    out = pd.merge_asof(left, right, left_on="ts", right_on="available_at", direction="backward")
    return out.drop(columns=["available_at"]).set_index("ts")


def fear_greed_features(cfg: Config, base_ts: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fear & Greed features on the hourly grid:
      MKT:fng      - level, 0..100 -> [-1, 1]
      MKT:fng_z30  - z-score over a rolling 30-day window
      MKT:fng_chg7 - 7-day change, in index units / 100
    """
    raw = fetch_fear_greed(cfg)
    j = asof_join(base_ts, raw, ["value"])
    v = j["value"]
    daily = raw.set_index("ts")["value"]
    z30 = (daily - daily.rolling(30, min_periods=10).mean()) / (daily.rolling(30, min_periods=10).std() + 1e-6)
    chg7 = (daily - daily.shift(7)) / 100.0
    aux = pd.DataFrame({"z30": z30, "chg7": chg7}).reset_index()
    aux["available_at"] = aux["ts"] + pd.Timedelta(days=1)
    j2 = asof_join(base_ts, aux, ["z30", "chg7"])
    return pd.DataFrame({
        "MKT:fng": (v / 50.0 - 1.0).values,
        "MKT:fng_z30": j2["z30"].values,
        "MKT:fng_chg7": j2["chg7"].values,
    }, index=base_ts)
