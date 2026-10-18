#!/usr/bin/env python3
"""
Live tail of ALL fly logs in the terminal, like `tail -f` on several files at
once. Two sources, interleaved in the order they are printed:

  1. live trading: the same SQLite database the dashboard reads - decisions,
     trades, and the moments dopamine was applied;
  2. cycle training: the train.log of every training cycle
     (runs.nosync/cycles/<id>/train.log) - the same lines shown in the
     expanded log on a cycle card in the dashboard, but live and without the
     50-line limit.

The dashboard's "Training" panel shows the exact command with the path to the
current run; copying it from there is more reliable than guessing the path,
which changes on every server start.

    python scripts/tail_logs.py                                  # latest run + all cycles
    python scripts/tail_logs.py runs.nosync/live_.../live.sqlite  # a specific live run

Ctrl+C to stop. It only reads - nothing is written to the database or to any
file - so it can stay open alongside the dashboard for as long as you like.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CYCLES_DIR = ROOT / "runs.nosync" / "cycles"

# Same convention that colors dopamine neurons green for reward and red for
# punishment on the dashboard (REWARD_GOOD/REWARD_BAD in web/index.html);
# a terminal has no pixels, but it does have ANSI colors.
_GREEN, _RED, _RESET, _DIM = "\033[32m", "\033[31m", "\033[0m", "\033[2m"


def _colored(sign: float, text: str) -> str:
    color = _GREEN if sign >= 0 else _RED
    return f"{color}{text}{_RESET}"


def fmt_rewards(rewards: dict) -> str:
    return "  ".join(_colored(v, f"{k} {'+' if v >= 0 else ''}{v:.3f}") for k, v in rewards.items())


class CycleLogTailer:
    """Tail of each cycle's train.log, like `tail -f`, one file per cycle."""

    def __init__(self) -> None:
        self._pos: dict[Path, int] = {}

    def poll(self) -> list[str]:
        out: list[str] = []
        if not CYCLES_DIR.exists():
            return out
        for log_path in sorted(CYCLES_DIR.glob("*/train.log")):
            cycle_id = log_path.parent.name
            try:
                size = log_path.stat().st_size
            except OSError:
                continue
            start = self._pos.get(log_path, 0)
            if size < start:
                start = 0  # file was rewritten (cycle restarted)
            if size == start:
                continue
            try:
                with log_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(start)
                    chunk = f.read()
                    self._pos[log_path] = f.tell()
            except OSError:
                continue
            for line in chunk.replace("\r", "\n").split("\n"):
                line = line.strip()
                if line:
                    out.append(f"[training · {cycle_id}] {line}")
        return out


def find_latest_db() -> Path | None:
    runs = ROOT / "runs.nosync"
    candidates = sorted(runs.glob("live_*/live.sqlite"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def fmt_targets(d: dict) -> str:
    return " ".join(f"{k}={v:+.2f}" for k, v in d.items())


def main() -> None:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_db()
    if db_path is not None and not db_path.exists():
        raise SystemExit(f"file not found: {db_path}")

    cycle_tailer = CycleLogTailer()

    if db_path is None:
        print("\n  no live run found (runs.nosync/live_*/live.sqlite missing) - "
              "following cycle training logs only")
    else:
        print(f"\n  following {db_path}")
    print("  + training logs of all cycles (runs.nosync/cycles/*/train.log)")
    print("  Ctrl+C to stop\n")

    # read-only connection: the server holds its own; this script must not write
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False) if db_path else None

    last_dec_ts = ""
    last_trade_id = 0
    last_w_ts = ""
    last_reward_id = 0

    try:
        while True:
            for line in cycle_tailer.poll():
                print(line)
            if con is None:
                time.sleep(1.0)
                continue
            try:
                cur = con.execute(
                    "SELECT ts, target, executed, equity, blocked, needs_approval, reasons "
                    "FROM decisions WHERE ts > ? ORDER BY ts", (last_dec_ts,),
                )
                for ts, target, executed, equity, blocked, needs, reasons in cur.fetchall():
                    last_dec_ts = ts
                    tgt = fmt_targets(json.loads(target))
                    tag = "TRADE" if executed else ("BLOCKED" if blocked else "decision")
                    r = json.loads(reasons or "[]")
                    print(f"[{ts[:19]}] {tag:<8} target[{tgt}]  equity {equity:>9,.0f}"
                          + (f"   — {'; '.join(r)}" if r else ""))

                cur = con.execute(
                    "SELECT id, ts, coin, side, units, price, fee FROM trades "
                    "WHERE id > ? ORDER BY id", (last_trade_id,),
                )
                for tid, ts, coin, side, units, price, fee in cur.fetchall():
                    last_trade_id = tid
                    print(f"[{ts[:19]}]   -> {side:<4} {abs(units):.6f} {coin} "
                          f"at {price:,.2f}, fee {fee:.2f}")

                cur = con.execute(
                    "SELECT ts, mean, drift FROM weights WHERE ts > ? ORDER BY ts", (last_w_ts,),
                )
                for ts, mean, drift in cur.fetchall():
                    last_w_ts = ts
                    print(f"{_DIM}[{ts[:19]}]   weight drift {drift:.5f} "
                          f"(synapse mean {mean:.4f}){_RESET}")

                cur = con.execute(
                    "SELECT id, ts, sign, rewards FROM rewards WHERE id > ? ORDER BY id",
                    (last_reward_id,),
                )
                for rid, ts, sign, rewards_json in cur.fetchall():
                    last_reward_id = rid
                    rw = json.loads(rewards_json)
                    label = _colored(sign, "REWARD" if sign >= 0 else "PUNISHMENT")
                    print(f"[{ts[:19]}]   dopamine: {label}  {fmt_rewards(rw)}")

            except sqlite3.OperationalError:
                pass  # the server is writing to the database right now; retry on the next loop

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
