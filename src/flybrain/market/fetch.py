"""
Exchange candles.

Prices come from the public mainnet API, not the testnet: the testnet order
book is thin and prices jump around. The testnet is only for executing
orders, not for learning from them.

If the primary exchange is region-blocked (HTTP 451), fall back to the
alternates from the config. Results are stored in parquet; a repeat call
only fetches the tail.

    python -m flybrain.market.fetch
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import Config, load_config

TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def _ts(date_str: str) -> int:
    """'2020-08-01' -> UTC milliseconds."""
    return int(datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _cache_path(cfg: Config, exchange: str, symbol: str) -> Path:
    safe = symbol.replace("/", "").replace(":", "")
    return cfg.path("market") / f"{exchange}_{safe}_{cfg.market.timeframe}.parquet"


def make_exchange(name: str, timeout_ms: int = 20_000):
    """Create a ccxt client. Imported lazily so the module loads without ccxt."""
    import ccxt  # noqa: PLC0415

    cls = getattr(ccxt, name, None)
    if cls is None:
        raise ValueError(f"no such exchange in ccxt: {name}")
    return cls({"enableRateLimit": True, "timeout": timeout_ms})


def fetch_last_prices(cfg: Config) -> dict[str, float] | None:
    """
    Current ticker price, not the hourly candle close, for the live equity
    mark on the dashboard between the fly's decisions. Candles from get_all()
    are cached and refreshed hourly (what the brain DECIDES on must stay
    fixed between candles to keep the backtest honest), while the equity on
    screen should move with real price fluctuations, not just once an hour.

    The timeout is short (5s) with no retries across fallback exchanges: this
    is cosmetic dashboard polish, not part of the decision path. On a network
    failure return None and the caller quietly keeps the last known close.
    """
    try:
        ex = make_exchange(cfg.market.exchange, timeout_ms=5_000)
        if ex.has.get("fetchTickers"):
            tickers = ex.fetch_tickers(cfg.market.symbols)
            out = {s: float(tickers[s]["last"]) for s in cfg.market.symbols
                   if s in tickers and tickers[s].get("last")}
        else:
            out = {}
            for s in cfg.market.symbols:
                t = ex.fetch_ticker(s)
                if t.get("last"):
                    out[s] = float(t["last"])
        return out if len(out) == len(cfg.market.symbols) else None
    except Exception:  # noqa: BLE001
        return None


def fetch_ohlcv(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
) -> pd.DataFrame:
    """
    Fetch candles page by page. The exchange returns at most ~1000 per
    request, so walk in windows and advance a cursor past the last candle received.
    """
    ex = make_exchange(exchange_name)
    step = TF_MS[timeframe]
    until_ms = until_ms or int(time.time() * 1000)

    rows: list[list] = []
    cursor = since_ms
    stall = 0

    while cursor < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        new_cursor = batch[-1][0] + step
        if new_cursor <= cursor:
            stall += 1
            if stall > 2:
                break
            new_cursor = cursor + step * 1000
        else:
            stall = 0
        cursor = new_cursor
        print(
            f"    {symbol} {exchange_name}: {len(rows):>7,} candles, "
            f"up to {datetime.fromtimestamp(batch[-1][0] / 1000, timezone.utc):%Y-%m-%d}",
            end="\r",
        )

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    print()
    return df


def get_candles(cfg: Config, symbol: str, force: bool = False) -> pd.DataFrame:
    """
    Candles for one coin: from cache, topping up the tail. If the primary
    exchange is unavailable, try the alternates and remember which one worked.
    """
    candidates = [cfg.market.exchange, *cfg.market.fallback_exchanges]
    since = _ts(cfg.market.since)

    # look for an existing cache from any of the exchanges
    for name in candidates:
        path = _cache_path(cfg, name, symbol)
        if path.exists() and not force:
            df = pd.read_parquet(path)
            last = int(df["ts"].max())
            fresh_until = int(time.time() * 1000) - TF_MS[cfg.market.timeframe]
            if last >= fresh_until:
                return df
            print(f"  topping up {symbol} from {name}...")
            try:
                tail = fetch_ohlcv(name, symbol, cfg.market.timeframe, last + 1)
                if len(tail):
                    df = (
                        pd.concat([df, tail])
                        .drop_duplicates(subset="ts")
                        .sort_values("ts")
                        .reset_index(drop=True)
                    )
                    df.to_parquet(path, index=False)
                return df
            except Exception as e:  # noqa: BLE001
                print(f"  could not top up from {name}: {type(e).__name__}, using what is cached")
                return df

    # no cache: download from scratch
    last_error: Exception | None = None
    for name in candidates:
        try:
            print(f"  downloading {symbol} from {name} ...")
            df = fetch_ohlcv(name, symbol, cfg.market.timeframe, since)
            if len(df) < 100:
                raise RuntimeError(f"suspiciously few candles: {len(df)}")
            df.to_parquet(_cache_path(cfg, name, symbol), index=False)
            return df
        except Exception as e:  # noqa: BLE001
            last_error = e
            msg = str(e)
            hint = " (geo-blocked)" if "451" in msg else ""
            print(f"  {name} returned no data: {type(e).__name__}{hint}")

    raise RuntimeError(
        f"no exchange returned {symbol}. Last error: {last_error}. "
        f"Run python scripts/check_access.py"
    ) from last_error


def get_all(cfg: Config | None = None, force: bool = False) -> dict[str, pd.DataFrame]:
    """Candles for all coins in the config, aligned to common timestamps."""
    cfg = cfg or load_config()
    out: dict[str, pd.DataFrame] = {}
    for sym in cfg.market.symbols:
        out[sym] = get_candles(cfg, sym, force=force)
    return align(out)


def align(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Keep only the timestamps present for every coin. Otherwise the portfolio
    would be computed over gaps and the metrics would drift.
    """
    if not data:
        return data
    common = None
    for df in data.values():
        s = set(df["ts"])
        common = s if common is None else (common & s)
    keep = sorted(common or [])
    return {
        sym: df[df["ts"].isin(keep)].sort_values("ts").reset_index(drop=True)
        for sym, df in data.items()
    }


if __name__ == "__main__":
    cfg = load_config()
    data = get_all(cfg)
    print(f"\n  {'coin':<12} {'candles':>8}  period")
    print("  " + "-" * 52)
    for sym, df in data.items():
        a, b = df["dt"].iloc[0], df["dt"].iloc[-1]
        print(f"  {sym:<12} {len(df):>8,}  {a:%Y-%m-%d} .. {b:%Y-%m-%d}")
    print()
