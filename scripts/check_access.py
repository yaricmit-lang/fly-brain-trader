#!/usr/bin/env python3
"""
Check which external services are reachable from the current network.

Why: Binance returns HTTP 451 in restricted regions, and it is better to find
that out up front than when the first order is placed. The script installs
nothing and needs only the standard library.

    python scripts/check_access.py
"""

from __future__ import annotations

import concurrent.futures
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TIMEOUT = 8

# (name, url, purpose)
#
# Note: the Binance Futures Testnet may redirect to the regular Binance login
# and require a verified (KYC) main account (testnet.binancefuture.com ->
# demo.binance.com -> accounts.binance.com). Bybit Demo Trading may not be
# available in some regions. The default executor is the built-in paper
# simulator, so none of the exchange endpoints below are required.
ENDPOINTS: list[tuple[str, str, str]] = [
    ("Binance spot (mainnet)",     "https://api.binance.com/api/v3/ping",              "candle history"),
    ("Binance futures (mainnet)",  "https://fapi.binance.com/fapi/v1/ping",            "live prices"),
    ("Bybit demo",                 "https://api-demo.bybit.com/v5/market/time",        "paper account, orders (optional)"),
    ("Bybit (mainnet)",            "https://api.bybit.com/v5/market/time",             "fallback price source"),
    ("Binance FUTURES TESTNET",    "https://testnet.binancefuture.com/fapi/v1/ping",   "optional paper account, may need KYC"),
    ("Binance spot testnet",       "https://testnet.binance.vision/api/v3/ping",       "not used (no UI)"),
    ("OKX",                        "https://www.okx.com/api/v5/public/time",           "fallback price source"),
    ("Kraken",                     "https://api.kraken.com/0/public/Time",             "fallback price source"),
    ("FlyWire Codex",              "https://codex.flywire.ai/",                        "fly connectome"),
    ("Groq API",                   "https://api.groq.com/",                            "news sentiment (optional)"),
    ("GitHub API",                 "https://api.github.com/",                          "source hosting"),
]

def _ssl_context() -> ssl.SSLContext:
    """
    Python from python.org ships without root certificates (it needs
    Install Certificates.command or certifi). Use certifi when available so
    the script works from both a venv and the system Python.
    """
    try:
        import certifi  # noqa: PLC0415
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = _ssl_context()


def probe(name: str, url: str, why: str) -> dict:
    """Request the URL and return status and latency. 451 = region blocked."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "fly-brain-trader/0.1 (access check)"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
            ms = int((time.time() - t0) * 1000)
            return {"name": name, "why": why, "ok": True, "code": r.status, "ms": ms, "note": ""}
    except urllib.error.HTTPError as e:
        ms = int((time.time() - t0) * 1000)
        note = ""
        ok = e.code < 400
        if e.code == 451:
            note = "REGION BLOCKED (a VPN may be needed)"
        elif e.code == 403:
            note = "access denied (possibly geo-blocked)"
        elif e.code in (401, 404):
            # the service is up; this path just needs a key or does not exist
            ok, note = True, "service responds"
        return {"name": name, "why": why, "ok": ok, "code": e.code, "ms": ms, "note": note}
    except Exception as e:  # noqa: BLE001
        ms = int((time.time() - t0) * 1000)
        return {"name": name, "why": why, "ok": False, "code": "—",
                "ms": ms, "note": type(e).__name__}


def main() -> int:
    print("\n  Checking service availability...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ENDPOINTS)) as ex:
        results = list(ex.map(lambda a: probe(*a), ENDPOINTS))

    w = max(len(r["name"]) for r in results)
    w2 = max(len(r["why"]) for r in results)
    print(f"  {'service':<{w}}  {'purpose':<{w2}}  {'code':>5}  {'ms':>6}  note")
    print("  " + "-" * (w + w2 + 32))
    for r in results:
        mark = "OK  " if r["ok"] else "FAIL"
        print(f"  {r['name']:<{w}}  {r['why']:<{w2}}  {str(r['code']):>5}  "
              f"{r['ms']:>6}  {mark} {r['note']}")
    print()

    blocked = [r for r in results if r.get("code") == 451]
    bybit_demo_ok = any(r["ok"] and r["name"] == "Bybit demo" for r in results)

    if blocked:
        print("  Result: some services are blocked in your region. Options:")
        print("    - use a VPN and re-run the check;")
        print("    - switch market.exchange to bybit or okx in config/default.yaml.\n")
    elif bybit_demo_ok:
        print(
            "  Result: Bybit demo is reachable - you can set "
            "trading.executor.adapter: bybit_demo in the config.\n"
        )

    # also show how much disk space is left
    try:
        from flybrain.housekeeping import report  # noqa: PLC0415
        report()
    except Exception:  # noqa: BLE001
        import shutil
        free = shutil.disk_usage(ROOT).free / (1024 ** 3)
        print(f"  Free disk space: {free:.1f} GB\n")

    return 0 if not blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
