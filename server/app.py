"""
Dashboard server: the web UI plus the trading loop in a single process.

Why one process: otherwise two processes would have to be synchronized
through files, and the display would regularly drift from what the fly
actually did. Here LiveEngine runs as a background asyncio task and the
websocket pushes its state to subscribers.

SECURITY NOTE: this server has no authentication and it exposes
trading-control endpoints (pause, approve, shutdown, ...). Browser requests
must be same-origin (see `_require_same_origin`), which stops other websites
from driving it via CSRF, but anyone who can reach the port can still use it.
Bind it to 127.0.0.1 (the default) unless you are on a trusted network.

    python -m flybrain.serve --adapter paper --online-learning
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from urllib.parse import urlparse

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from flybrain.config import load_config          # noqa: E402
from flybrain.cycles import CYCLE_BY_ID, CYCLES, build_train_args  # noqa: E402
from flybrain.live_engine import LiveEngine      # noqa: E402

app = FastAPI(title="FlyBrain Trader")


@app.middleware("http")
async def _require_same_origin(request: Request, call_next):
    """Reject cross-site state-changing requests (CSRF). Non-browser clients
    (curl, scripts) send no Origin header and are let through."""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host", ""):
            return JSONResponse({"ok": False, "reason": "cross-origin request rejected"},
                                status_code=403)
    return await call_next(request)


ENGINE: LiveEngine | None = None
CLIENTS: set[WebSocket] = set()
WEB = ROOT / "web"
DATA = ROOT / "data.nosync"

# State of the three training cycles: id -> {status, proc, started, log}. It
# lives in the server's memory rather than in LiveEngine: cycles are separate
# SUBPROCESSES (python -m flybrain.train), not part of the trading loop, on
# purpose, so heavy training does not share a thread with think()/trade() and
# does not interfere with trading.
CYCLE_STATE: dict[str, dict] = {}
CYCLE_QUEUE: list[str] = []        # ids of cycles waiting in the queue
QUEUE_COOLDOWN_SEC = 120           # cool-down between queued cycles


def attach(engine: LiveEngine) -> None:
    """Attach the engine before the server starts (called by flybrain.serve)."""
    global ENGINE
    ENGINE = engine


# --------------------------------------------------------------------------
#  Static files and data
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((WEB / "index.html").read_text(encoding="utf-8"))


@app.get("/api/geometry")
async def geometry() -> Response:
    """Brain coordinates. Served once; the browser caches them afterwards."""
    path = DATA / "geometry.bin"
    if not path.exists():
        return Response("geometry.bin is missing: run scripts/export_geometry.py",
                        status_code=503, media_type="text/plain; charset=utf-8")
    return FileResponse(path, media_type="application/octet-stream",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/geometry.json")
async def geometry_meta() -> Response:
    path = DATA / "geometry.json"
    if not path.exists():
        return Response('{"error":"geometry.json is missing"}', status_code=503,
                        media_type="application/json")
    return Response(path.read_text(encoding="utf-8"), media_type="application/json")


# --------------------------------------------------------------------------
#  State
# --------------------------------------------------------------------------
@app.get("/api/state")
async def state() -> dict:
    if ENGINE is None:
        return {"error": "engine is not running"}
    s = ENGINE.snapshot()
    s["symbols"] = ENGINE.cfg.market.symbols
    # ENGINE._current_decision_timeframe(): with adaptive_stride.enabled=True
    # this is the CURRENTLY selected regime (1h/4h), not the static config
    # value; otherwise the dashboard would claim the cadence is still 4h while
    # the fly is already deciding every hour
    s["decision_timeframe"] = ENGINE._current_decision_timeframe()
    s["adaptive_stride"] = ENGINE.cfg.market.adaptive_stride.enabled
    s["initial_capital"] = ENGINE.cfg.trading.initial_capital
    return s


@app.get("/api/history")
async def history(n: int = 300, decisions_n: int = 100, trades_n: int = 200) -> dict:
    if ENGINE is None:
        return {"error": "engine is not running"}
    return {
        "equity": ENGINE.store.equity_curve(n),
        "trades": ENGINE.store.recent_trades(trades_n),
        "decisions": ENGINE.store.recent_decisions(decisions_n),
    }


@app.get("/api/events")
async def events(n: int = 100) -> dict:
    """Engine event log (start/stop/checkpoint switch/drift/kill-switch/
    rollback), for the "Event log" panel on the dashboard, see Stage 3.6."""
    if ENGINE is None:
        return {"error": "engine is not running"}
    return {"events": ENGINE.store.recent_events(n)}


@app.get("/api/candles")
async def candles(n: int = 300) -> dict:
    """Candles for the chart, taken from the same cache as the features."""
    if ENGINE is None:
        return {"error": "engine is not running"}
    from flybrain.market.fetch import get_all  # noqa: PLC0415

    data = get_all(ENGINE.cfg)
    out = {}
    for sym in ENGINE.cfg.market.symbols:
        df = data[sym].tail(n)
        coin = sym.split("/")[0]
        out[coin] = [
            {"time": int(r.ts // 1000), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close)}
            for r in df.itertuples()
        ]
    return out


# --------------------------------------------------------------------------
#  Control
# --------------------------------------------------------------------------
@app.post("/api/pause")
async def pause() -> dict:
    if ENGINE:
        ENGINE.paused = True
    return {"paused": True}


@app.post("/api/resume")
async def resume() -> dict:
    if ENGINE:
        ENGINE.paused = False
    return {"paused": False}


@app.post("/api/approve")
async def approve() -> dict:
    """Approve a trade that the guards put on hold."""
    if ENGINE is None or ENGINE.executor.pending is None:
        return {"ok": False, "reason": "nothing to approve"}
    prices = np.array([ENGINE.state.prices[c] for c in ENGINE.coins])
    n = ENGINE.executor.approve_pending(prices)
    return {"ok": True, "executed": n}


@app.post("/api/reject")
async def reject() -> dict:
    if ENGINE:
        ENGINE.executor.reject_pending()
    return {"ok": True}


@app.post("/api/decide")
async def decide() -> dict:
    """Make a decision immediately without waiting for the cadence boundary."""
    if ENGINE is None:
        return {"ok": False}
    await asyncio.to_thread(ENGINE.force_decision)
    return {"ok": True}


@app.post("/api/learning/start")
async def learning_start() -> dict:
    if ENGINE:
        ENGINE.start_learning()
    return {"learning": True}


@app.post("/api/learning/stop")
async def learning_stop() -> dict:
    if ENGINE:
        ENGINE.stop_learning()
    return {"learning": False}


@app.post("/api/shutdown")
async def shutdown() -> dict:
    """Stops the brain permanently, until the process is restarted from the terminal."""
    if ENGINE:
        ENGINE.shutdown()
    return {"stopped": True}


@app.post("/api/rollback")
async def rollback() -> dict:
    """
    Roll back to the base (cycle) checkpoint. This discards accumulated live
    experience (weights, rule statistics, reward queue) if online learning
    drifted to something worse than before (2026-09-16, Stage 3.5, "base +
    rollback"). The portfolio and the guards are left untouched.
    """
    if not ENGINE:
        return {"ok": False, "reason": "engine is not running"}
    try:
        await asyncio.to_thread(ENGINE.rollback_to_base)
    except RuntimeError as e:
        return {"ok": False, "reason": str(e)}
    return {"ok": True}


@app.post("/api/kill_switch/reset")
async def kill_switch_reset() -> dict:
    """Manual reset of the daily stop: only on an explicit human command."""
    if not ENGINE:
        return {"ok": False, "reason": "engine is not running"}
    ENGINE.reset_kill_switch()
    return {"ok": True}


# --------------------------------------------------------------------------
#  Three training cycles before going live
# --------------------------------------------------------------------------
def _cycle_run_dir(cycle_id: str) -> Path:
    base = ENGINE.cfg.path("runs") if ENGINE else (ROOT / "runs.nosync")
    return base / "cycles" / cycle_id


def _read_seasonal_result(run_dir: Path) -> dict | None:
    """Seasonal cycles (6 and 8) write a different shape of history.json/result.json
    (no single val; there are metrics for 6 seasonal blocks), hence a dedicated reader."""
    res_path = run_dir / "result.json"
    if not res_path.exists():
        return None
    try:
        res = json.loads(res_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return {
        "epochs_done": res.get("epochs_done"),
        "best_epoch": res.get("best_epoch"),
        "mean_return": res.get("mean_return"),
        # the fields below are absent in old cycle-6 runs, hence .get()
        "mean_edge": res.get("mean_edge"),
        "select_by": res.get("select_by"),
        "rule": res.get("rule"),
        "decision_timeframe": res.get("decision_timeframe"),
        "goal_return": res.get("goal_return"),
        "hit_goal": res.get("hit_goal"),
        "blocks": res.get("blocks", []),
        "duration_sec": res.get("duration_sec"),
    }


def _read_cycle_result(run_dir: Path, entrypoint: str = "flybrain.train") -> dict | None:
    """
    Cycle metric: validation Sharpe/return at the epoch where training saved
    best.pt. The epoch is chosen by the "score" field in history.json, the same
    metric train.py actually used to save the checkpoint (--select-by: sharpe
    by default, return for aggressive cycles such as "100 -> 300"). Previously
    this was hard-coded to h["val"]["sharpe"], so with select_by=return the
    dashboard showed a different epoch than the one that actually became
    best.pt. history.json files from old runs (before this field existed) fall
    back to validation Sharpe.
    There is no true out-of-sample check on the future; the validation period
    is the best available for a checkpoint that will go on to trade for real.
    """
    if entrypoint != "flybrain.train":
        return _read_seasonal_result(run_dir)

    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return None
    try:
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not hist:
        return None
    best = max(hist, key=lambda h: h.get("score", h["val"]["sharpe"]))
    out = {
        "epochs_done": len(hist),
        "best_epoch": best["epoch"],
        "select_by": best.get("select_by", "sharpe"),
        "val_sharpe": best["val"]["sharpe"],
        "val_return": best["val"]["total_return"],
        "val_trades": best["val"]["n_trades"],
        "val_max_drawdown": best["val"]["max_drawdown"],
    }
    # At the end train.py writes result.json: the same validation window for the
    # risk-targeting formula and for buy-and-hold; without it the fly's Sharpe
    # has nothing to be compared against.
    res_path = run_dir / "result.json"
    if res_path.exists():
        try:
            res = json.loads(res_path.read_text(encoding="utf-8"))
            out.update({
                "baseline_sharpe": res["vol_target"]["sharpe"],
                "baseline_return": res["vol_target"]["total_return"],
                "buyhold_sharpe": res["buy_hold"]["sharpe"],
                "edge_sharpe": res["edge_sharpe"],
                "edge_return": res.get("edge_return"),
                "hit_goal": res.get("hit_goal"),
                "goal_return": res.get("goal_return"),
                "duration_sec": res["duration_sec"],
                "val_start": res["val_start"], "val_end": res["val_end"],
            })
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass
    return out


def _cycle_elapsed_sec(st: dict) -> int | None:
    if not st.get("started"):
        return None
    t0 = datetime.fromisoformat(st["started"])
    t1 = datetime.fromisoformat(st["finished"]) if st.get("finished") else datetime.now(timezone.utc)
    return int((t1 - t0).total_seconds())


@app.get("/api/cycles")
async def cycles_list() -> dict:
    active_path = (ENGINE.checkpoint_info or {}).get("path") if ENGINE else None
    out = []
    for spec in CYCLES:
        run_dir = _cycle_run_dir(spec.id)
        st = CYCLE_STATE.get(spec.id, {})
        best = run_dir / "best.pt"
        out.append({
            "id": spec.id, "name": spec.name, "description": spec.description,
            "requires_teacher": spec.requires_teacher,
            "est_minutes": spec.est_minutes, "est_note": spec.est_note,
            "status": st.get("status", "idle"),
            "started": st.get("started"),
            "finished": st.get("finished"),
            "elapsed_sec": _cycle_elapsed_sec(st),
            "returncode": st.get("returncode"),
            "has_checkpoint": best.exists(),
            "is_live": active_path == str(best),
            "queued": spec.id in CYCLE_QUEUE,
            "result": _read_cycle_result(run_dir, spec.entrypoint),
        })
    running = any(v.get("status") == "running" for v in CYCLE_STATE.values())
    return {"cycles": out, "any_running": running, "queue": list(CYCLE_QUEUE)}


@app.get("/api/cycles/{cycle_id}/log")
async def cycle_log(cycle_id: str, n: int = 60) -> dict:
    """Last lines of the training log, for a quick look without a terminal."""
    if cycle_id not in CYCLE_BY_ID:          # also blocks path traversal via cycle_id
        return {"lines": []}
    run_dir = _cycle_run_dir(cycle_id)
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return {"lines": []}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"lines": []}
    lines = [ln for ln in text.replace("\r", "\n").split("\n") if ln.strip()]
    return {"lines": lines[-n:]}


async def _on_cycle_finished(cycle_id: str, ok: bool) -> None:
    """Shared tail for both our own subprocess and one inherited by PID:
    update the status and continue the queue."""
    st = CYCLE_STATE.setdefault(cycle_id, {})
    st["status"] = "done" if ok else "failed"
    st["finished"] = datetime.now(timezone.utc).isoformat()
    st["returncode"] = 0 if ok else 1
    st["proc"] = None
    # "All three in sequence" queue: the next one starts only after a pause,
    # since the machine also needs to cool down between cycles. A failed cycle
    # does not break the queue: the other two are still needed for comparison.
    if CYCLE_QUEUE:
        await asyncio.sleep(QUEUE_COOLDOWN_SEC)
        nxt = CYCLE_QUEUE.pop(0)
        await _launch_cycle(nxt)


async def _watch_cycle(cycle_id: str, proc: asyncio.subprocess.Process, log_f) -> None:
    """Waits for a training subprocess launched by this server."""
    rc = await proc.wait()
    log_f.close()
    await _on_cycle_finished(cycle_id, ok=(rc == 0))


def _parse_ps_etime(raw: str) -> int | None:
    """
    macOS/BSD `ps -o etime=` (GNU `etimes`, in seconds, does not exist here)
    returns [[dd-]hh:]mm:ss, for example "10:31", "1:02:03", "2-01:02:03".
    """
    raw = raw.strip()
    days = 0
    if "-" in raw:
        d, raw = raw.split("-", 1)
        days = int(d)
    parts = raw.split(":")
    try:
        parts_i = [int(p) for p in parts]
    except ValueError:
        return None
    while len(parts_i) < 3:
        parts_i.insert(0, 0)
    h, m, s = parts_i[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def _pid_started_at(pid: int) -> str | None:
    """Actual process start time (not the moment of reattachment); otherwise,
    after reattaching to an inherited training run, the "X of ~Y min" indicator
    would be wrong, counting from zero."""
    try:
        out = subprocess.run(
            ["ps", "-o", "etime=", "-p", str(pid)], capture_output=True, text=True, timeout=5,
        ).stdout
        age_sec = _parse_ps_etime(out)
    except subprocess.SubprocessError:
        age_sec = None
    if age_sec is None:
        return None
    return (datetime.now(timezone.utc) - timedelta(seconds=age_sec)).isoformat()


def _find_running_pid(cycle_id: str) -> int | None:
    """
    Looks through `ps aux` for a `flybrain.train --tag <cycle_id>` subprocess
    that this server did not start itself. This matters right after a
    dashboard restart: training lives on in a separate process and restarting
    the web server does not touch it, but the asyncio handle to it (and with
    it the auto-queue and the correct status on the dashboard) is lost along
    with the old process.
    """
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return None
    needle = f"--tag {cycle_id}"
    for line in out.splitlines():
        if "flybrain.train" in line and needle in line:
            parts = line.split()
            try:
                return int(parts[1])
            except (IndexError, ValueError):
                continue
    return None


async def _watch_pid(cycle_id: str, pid: int) -> None:
    """Like _watch_cycle, but for a process without an asyncio handle: poll
    liveness by PID instead of await proc.wait()."""
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        await asyncio.sleep(5)
    run_dir = _cycle_run_dir(cycle_id)
    await _on_cycle_finished(cycle_id, ok=(run_dir / "result.json").exists())


def _reattach_running_cycles() -> None:
    """Called once at startup: finds a cycle that is already training in the
    background (survived a server restart) and restores monitoring of it:
    the status on the dashboard and the automatic move to the next queued cycle."""
    order = [c.id for c in CYCLES]
    for i, cycle_id in enumerate(order):
        if (_cycle_run_dir(cycle_id) / "result.json").exists():
            continue  # already finished earlier
        pid = _find_running_pid(cycle_id)
        if pid is None:
            continue
        started = _pid_started_at(pid) or datetime.now(timezone.utc).isoformat()
        CYCLE_STATE[cycle_id] = {
            "status": "running", "proc": None,
            "started": started, "finished": None, "returncode": None,
        }
        print(f"  training \"{cycle_id}\" survived the restart (pid {pid}); reattaching")
        # the remaining not-yet-started cycles from the standard order follow
        CYCLE_QUEUE.clear()
        for later in order[i + 1:]:
            if not (_cycle_run_dir(later) / "result.json").exists():
                CYCLE_QUEUE.append(later)
        asyncio.create_task(_watch_pid(cycle_id, pid))
        return  # subprocesses do not run in parallel, no need to look for a second one


async def _launch_cycle(cycle_id: str) -> dict:
    spec = CYCLE_BY_ID[cycle_id]
    run_dir = _cycle_run_dir(cycle_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    args = build_train_args(spec, str(run_dir))
    log_path = run_dir / "train.log"
    log_f = open(log_path, "wb")  # noqa: SIM115 (closed in _watch_cycle)

    # A separate SUBPROCESS, not asyncio.to_thread inside this interpreter:
    # training is heavy and long (gentle mode itself keeps the machine from
    # overheating), and a crash or hang must not drag the trading loop down.
    # OMP_NUM_THREADS is for the teacher (sklearn); otherwise it would take all 8 cores.
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", spec.entrypoint, *args,
        cwd=str(ROOT), env=env, stdout=log_f, stderr=asyncio.subprocess.STDOUT,
    )
    CYCLE_STATE[cycle_id] = {
        "status": "running", "proc": proc,
        "started": datetime.now(timezone.utc).isoformat(),
        "finished": None, "returncode": None,
    }
    asyncio.create_task(_watch_cycle(cycle_id, proc, log_f))
    return {"ok": True}


def _any_cycle_running() -> bool:
    return any(v.get("status") == "running" for v in CYCLE_STATE.values())


@app.post("/api/cycles/{cycle_id}/start")
async def cycle_start(cycle_id: str) -> dict:
    spec = CYCLE_BY_ID.get(cycle_id)
    if spec is None:
        return {"ok": False, "reason": "unknown cycle"}
    if ENGINE is None:
        return {"ok": False, "reason": "engine is not running"}
    if _any_cycle_running():
        return {"ok": False, "reason": "another cycle is already running; wait for it to finish"}
    if cycle_id in CYCLE_QUEUE:
        CYCLE_QUEUE.remove(cycle_id)
    return await _launch_cycle(cycle_id)


@app.post("/api/cycles/start_all")
async def cycles_start_all() -> dict:
    """All three in sequence: the first immediately, the rest go into the queue."""
    if ENGINE is None:
        return {"ok": False, "reason": "engine is not running"}
    if _any_cycle_running():
        return {"ok": False, "reason": "a cycle is already running; wait for it to finish"}
    ids = [c.id for c in CYCLES]
    CYCLE_QUEUE.clear()
    CYCLE_QUEUE.extend(ids[1:])
    return await _launch_cycle(ids[0])


@app.post("/api/cycles/start_many")
async def cycles_start_many(ids: list[str]) -> dict:
    """Like start_all, but with a custom list, e.g. "the remaining two" after a
    server restart, so an already finished cycle is not rerun."""
    if ENGINE is None:
        return {"ok": False, "reason": "engine is not running"}
    if _any_cycle_running():
        return {"ok": False, "reason": "a cycle is already running; wait for it to finish"}
    ids = [i for i in ids if i in CYCLE_BY_ID]
    if not ids:
        return {"ok": False, "reason": "empty or unknown list of cycles"}
    CYCLE_QUEUE.clear()
    CYCLE_QUEUE.extend(ids[1:])
    return await _launch_cycle(ids[0])


@app.post("/api/cycles/clear_queue")
async def cycles_clear_queue() -> dict:
    """Clear waiting cycles; the current one runs to completion."""
    CYCLE_QUEUE.clear()
    return {"ok": True}


@app.post("/api/cycles/{cycle_id}/use")
async def cycle_use(cycle_id: str) -> dict:
    """\"Make live\": switch the live engine to this cycle's checkpoint."""
    spec = CYCLE_BY_ID.get(cycle_id)
    if spec is None or ENGINE is None:
        return {"ok": False, "reason": "engine is not running"}
    best = _cycle_run_dir(cycle_id) / "best.pt"
    if not best.exists():
        return {"ok": False, "reason": "this cycle has no checkpoint yet; run training first"}
    await asyncio.to_thread(ENGINE.switch_checkpoint, best, spec.requires_teacher, spec.config)
    return {"ok": True}


# --------------------------------------------------------------------------
#  Stream to the browser
# --------------------------------------------------------------------------
@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    origin = sock.headers.get("origin")
    if origin and urlparse(origin).netloc != sock.headers.get("host", ""):
        await sock.close(code=1008)            # cross-site websocket hijacking guard
        return
    await sock.accept()
    CLIENTS.add(sock)
    try:
        while True:
            await sock.receive_text()          # keep the connection open
    except WebSocketDisconnect:
        pass
    finally:
        CLIENTS.discard(sock)


async def _broadcast(snapshot: dict, acts: bytes) -> None:
    """State goes as text and activations as binary, to keep traffic small."""
    dead = []
    payload = json.dumps(snapshot, ensure_ascii=False)
    for c in list(CLIENTS):
        try:
            await c.send_text(payload)
            await c.send_bytes(acts)
        except Exception:  # noqa: BLE001
            dead.append(c)
    for c in dead:
        CLIENTS.discard(c)


async def _loop() -> None:
    """Background loop: the brain thinks, sometimes trades, everything goes to the browser."""
    assert ENGINE is not None
    await asyncio.to_thread(ENGINE.warmup)
    print(f"\n  dashboard: http://{ENGINE.cfg.viz.server_host}:{ENGINE.cfg.viz.server_port}\n")
    while not ENGINE.stopping:
        try:
            await asyncio.to_thread(ENGINE.tick)
            await _broadcast(ENGINE.snapshot(), ENGINE.activations_bytes())
        except Exception as e:  # noqa: BLE001
            print(f"  error in loop: {type(e).__name__}: {e}")
            await asyncio.sleep(30)
        await asyncio.sleep(ENGINE.think_every_sec)
    print("  brain stopped (\"Shut down brain\" button). To resume, restart the process.")


@app.on_event("startup")
async def startup() -> None:
    if ENGINE is not None:
        asyncio.create_task(_loop())
    _reattach_running_cycles()
