"""
Live mode: the fly trades on a demo account at real prices.

    python -m flybrain.live --synthetic                      # dry run
    python -m flybrain.live --checkpoint ... --adapter paper
    python -m flybrain.live --checkpoint ... --adapter binance_futures_testnet

Runs while the terminal is open. State is written to SQLite; on the next
start the missed candles are fetched so the brain does not wake up with an
empty head. Trades missed during downtime are not caught up.
"""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .build import build_brain, load_checkpoint, save_checkpoint
from .config import load_config
from .market.features import build_matrix, prices_matrix
from .market.fetch import TF_MS, get_all
from .trading.executor import Executor

RUNNING = True


def _stop(signum, frame):  # noqa: ARG001
    global RUNNING
    RUNNING = False
    print("\n  stop signal received, finishing the current candle and exiting")


class LiveStore:
    """SQLite store of decisions, orders and equity. Also read by the dashboard."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                ts TEXT PRIMARY KEY, raw TEXT, final TEXT, executed INT,
                equity REAL, blocked INT, needs_approval INT, reasons TEXT
            );
            CREATE TABLE IF NOT EXISTS equity (
                ts TEXT PRIMARY KEY, value REAL
            );
            CREATE TABLE IF NOT EXISTS activations (
                ts TEXT PRIMARY KEY, data BLOB
            );
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
        """)
        self.db.commit()

    def write_decision(self, rec, ts: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?)",
            (ts, json.dumps(rec.raw_target), json.dumps(rec.final_target),
             rec.executed, rec.equity, int(rec.blocked), int(rec.needs_approval),
             json.dumps(rec.reasons, ensure_ascii=False)),
        )
        self.db.execute("INSERT OR REPLACE INTO equity VALUES (?,?)", (ts, rec.equity))
        self.db.commit()

    def write_activations(self, ts: str, a: np.ndarray) -> None:
        """uint8 instead of float32: 4x less space."""
        packed = ((np.clip(a, -1, 1) + 1) * 127.5).astype(np.uint8)
        self.db.execute("INSERT OR REPLACE INTO activations VALUES (?,?)", (ts, packed.tobytes()))
        self.db.commit()

    def set_meta(self, k: str, v: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))
        self.db.commit()


def seconds_to_next_bar(timeframe: str, pad: int = 5) -> float:
    """Seconds until the current candle closes, plus padding for the exchange to publish it."""
    step = TF_MS[timeframe] / 1000
    now = time.time()
    return step - (now % step) + pad


def main() -> None:
    ap = argparse.ArgumentParser(description="Live trading on a demo account")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--adapter", default=None,
                    choices=["dry_run", "paper", "binance_futures_testnet"])
    ap.add_argument("--online-learning", action="store_true",
                    help="keep learning on live data (small step)")
    ap.add_argument("--auto-approve", action="store_true",
                    help="do not wait for approval of the first trades")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.adapter:
        cfg.trading.executor.adapter = args.adapter

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"\n  execution mode: {cfg.trading.executor.adapter}")
    if cfg.trading.executor.adapter == "dry_run":
        print("  (dry run: nothing will be sent to the exchange)")

    brain, sg, rm, plast = build_brain(
        cfg, synthetic=args.synthetic, with_plasticity=args.online_learning
    )
    if args.checkpoint:
        info = load_checkpoint(brain, Path(args.checkpoint))
        print(f"  checkpoint loaded (epoch {info.get('epoch', '?')})")
    print(brain.describe())

    run_dir = cfg.path("runs") / f"live_{datetime.now():%Y-%m-%d_%H%M}"
    run_dir.mkdir(parents=True, exist_ok=True)
    store = LiveStore(run_dir / "live.sqlite")
    store.set_meta("adapter", cfg.trading.executor.adapter)
    store.set_meta("symbols", json.dumps(cfg.market.symbols))

    ex = Executor(cfg)
    coins = [s.split("/")[0] for s in cfg.market.symbols]
    print(f"  trading: {', '.join(coins)}   timeframe {cfg.market.timeframe}")
    print(f"  journal: {run_dir / 'live.sqlite'}\n")

    # --- warm-up: replay recent candles so the brain settles into a regime ---
    print("  warming up the brain on recent history...")
    data = get_all(cfg)
    X, dt, _ = build_matrix(data, cfg)
    warmup = X[-500:]
    brain.reset()
    for row in warmup:
        brain.step(torch.from_numpy(row))
    print(f"  warmed up on {len(warmup)} candles, last one {dt[-1]:%Y-%m-%d %H:%M} UTC\n")

    bar = 0
    while RUNNING:
        try:
            data = get_all(cfg)
            X, dt, _ = build_matrix(data, cfg)
            n_skip = len(data[cfg.market.symbols[0]]) - len(X)
            prices = prices_matrix(data, cfg, n_skip)

            x = torch.from_numpy(X[-1])
            p = prices[-1]
            now = dt[-1]

            decision, a = brain.step(x)
            rec = ex.submit(decision.fractions, p, ts=now, auto_approve=args.auto_approve)

            ts_str = now.isoformat()
            store.write_decision(rec, ts_str)
            if cfg.viz.record_activations and bar % max(1, cfg.viz.record_every_n_bars) == 0:
                store.write_activations(ts_str, decision.activations)

            pos = "  ".join(f"{c} {decision.fractions[c] * 100:.0f}%" for c in coins)
            flag = ""
            if rec.blocked:
                flag = "  [BLOCKED]"
            elif rec.needs_approval:
                flag = "  [AWAITING APPROVAL]"
            print(f"  {now:%Y-%m-%d %H:%M}  wants: {pos}   "
                  f"equity {rec.equity:,.0f}   trades {rec.executed}{flag}")
            if rec.reasons:
                print(f"      safeguards: {'; '.join(rec.reasons)}")

            if args.online_learning and plast:
                # reward is computed from the next candle, so learning is lagged
                pass

            save_checkpoint(brain, run_dir / "live_state.pt", {"bar": bar, "ts": ts_str})
            bar += 1

            if args.once:
                break

            wait = seconds_to_next_bar(cfg.market.timeframe)
            print(f"      next candle in {wait / 60:.0f} min")
            slept = 0.0
            while slept < wait and RUNNING:
                time.sleep(min(2.0, wait - slept))
                slept += 2.0

        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            print(f"  error in loop: {type(e).__name__}: {e}")
            print("  waiting a minute and retrying")
            time.sleep(60)

    print(f"\n  stopped. Candles processed: {bar}. Journal: {run_dir}\n")


if __name__ == "__main__":
    main()
