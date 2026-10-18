"""
Live trading engine.

Two cadences in one loop, and the split is deliberate:

  "think" - every few seconds. Fetches the latest price, publishes the
            visualisation frame (brain.peek(), not brain.step()!) and the
            current equity. The brain makes NO decisions and does NOT
            learn here. This keeps the dashboard alive; with a 4-hour
            decision timeframe it would otherwise update a few times a day.

  "trade" - on a decision_timeframe boundary (4 hours). The ONLY place
            where brain.step() is called: decision -> baseline x
            multiplier -> safeguards -> executor -> journal. The learning
            token is also taken here.

Calling brain.step() on every think tick (~1800 times per hour instead of
once per decision) let the EMAs inside read_out (motor-output inertia,
adaptive calibration) converge within seconds instead of hours, so the
live brain behaved differently from the one that was validated. Now
decision/act/observe/learn live strictly in trade(); think() does only
peek() + prices + equity.

Also fixed: the old live.py passed decision.fractions straight to the
executor. In exposure mode that value is a MULTIPLIER around 1.0, which the
executor would read as "100% of capital in the coin". The correct target is
baseline_positions(...) * multiplier, exactly as in agent.py.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .build import build_ensemble, check_manifest, load_checkpoint, read_checkpoint_manifest
from .config import Config, load_config
from .features import build_inputs
from .market.fetch import TF_MS, fetch_last_prices, get_all
from .trading.baseline import baseline_positions, realized_vol
from .trading.executor import Executor
from .trading.portfolio import Portfolio, clip_target
from .trading.reward import CoinRewardTracker


def _serialize_token(token: Any) -> tuple[Any, bool]:
    """
    Returns (serializable copy, can_be_saved).

    For an ensemble (n_members > 1, default 5) the token is ALWAYS a list
    [(rule_token, frac_vals), ...] from EnsemblePlasticity.observe(), not a
    bare tensor. A plain `isinstance(token, torch.Tensor)` check would
    therefore reject every ensemble token, including the fully serializable
    Hebbian ones. A single REINFORCE rule's token is a dict
    {coin: (log_prob, entropy)} that holds a live autograd graph and cannot
    be serialized.
    """
    if isinstance(token, torch.Tensor):
        return token.detach().clone(), True
    if isinstance(token, list):   # ensemble
        out = []
        for rule_token, frac_vals in token:
            if not isinstance(rule_token, torch.Tensor):
                return None, False
            out.append((rule_token.detach().clone(), dict(frac_vals)))
        return out, True
    return None, False


def read_checkpoint_pointer(run_dir: Path) -> Path | None:
    """Last checkpoint chosen via the dashboard's "make live" button.

    Read by serve.py on startup when --checkpoint is not given explicitly."""
    p = run_dir / "checkpoint.txt"
    if not p.exists():
        return None
    path = Path(p.read_text(encoding="utf-8").strip())
    return path if path.exists() else None


def _deserialize_token(saved: Any) -> Any:
    if isinstance(saved, list):
        return [(t.clone(), dict(f)) for t, f in saved]
    return saved


@dataclass
class _Pending:
    """A decision waiting for its reward (same logic as in agent.py)."""
    token: Any
    positions: np.ndarray
    exec_prices: np.ndarray
    turnover: np.ndarray
    due_ts: float
    base_positions: np.ndarray | None = None   # REALIZED weights of the shadow
                                                # baseline portfolio (control variate)
    sigma: np.ndarray | None = None   # ex-ante volatility at decision time
    nominal_base: np.ndarray | None = None     # NOMINAL baseline (baseline_positions,
                                                # no deadband/fees), used for the
                                                # ensemble's counterfactual credit
    lr_scale: float | None = None     # overrides cfg.lr_stride_scale for the ACTUAL
                                       # interval before this decision (adaptive
                                       # stride); None means use the config value


@dataclass
class LiveState:
    """Dashboard snapshot: everything the engine knows right now."""
    ts: str = ""
    prices: dict[str, float] = field(default_factory=dict)
    exposure: dict[str, float] = field(default_factory=dict)
    target: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    next_decision_in: float = 0.0
    n_decisions: int = 0
    n_thoughts: int = 0
    mode: str = "dry_run"
    learning: bool = False
    paused: bool = False
    weight_drift: float = 0.0
    last_reasons: list[str] = field(default_factory=list)
    last_reward_ts: float = 0.0     # when the last reward was applied
    last_reward_sign: float = 0.0   # sign sets the DAN flash colour in the frontend


class LiveStore:
    """SQLite store of decisions, equity and events. Also read by the dashboard."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        # WAL: the dashboard reads this DB while the engine writes every tick;
        # without it concurrent access regularly fails with "database is locked".
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                ts TEXT PRIMARY KEY, exposure TEXT, target TEXT, executed INT,
                equity REAL, blocked INT, needs_approval INT, reasons TEXT
            );
            CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, value REAL);
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, coin TEXT,
                side TEXT, units REAL, price REAL, fee REAL
            );
            CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);
            CREATE TABLE IF NOT EXISTS weights (ts TEXT PRIMARY KEY, mean REAL, drift REAL);
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
            CREATE TABLE IF NOT EXISTS rewards (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, sign REAL, rewards TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, detail TEXT
            );
        """)
        self.db.commit()

    def decision(self, ts: str, exposure: dict, target: dict, rec) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?)",
            (ts, json.dumps(exposure), json.dumps(target), rec.executed, rec.equity,
             int(rec.blocked), int(rec.needs_approval),
             json.dumps(rec.reasons, ensure_ascii=False)),
        )
        self.db.execute("INSERT OR REPLACE INTO equity VALUES (?,?)", (ts, rec.equity))
        self.db.commit()

    def trade(self, ts: str, t: dict) -> None:
        self.db.execute(
            "INSERT INTO trades (ts, coin, side, units, price, fee) VALUES (?,?,?,?,?,?)",
            (ts, t["coin"], t["side"], t["units"], t["price"], t["fee"]))
        self.db.commit()

    def weights(self, ts: str, mean: float, drift: float) -> None:
        self.db.execute("INSERT OR REPLACE INTO weights VALUES (?,?,?)", (ts, mean, drift))
        self.db.commit()

    def reward_event(self, ts: str, sign: float, rewards: dict) -> None:
        """Dopamine moment: the same sign that colours DAN neurons green/red
        in the frontend; tail_logs.py uses the same colours."""
        self.db.execute(
            "INSERT INTO rewards (ts, sign, rewards) VALUES (?,?,?)",
            (ts, sign, json.dumps(rewards)),
        )
        self.db.commit()

    def set_meta(self, k: str, v: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))
        self.db.commit()

    def event(self, kind: str, detail: str = "") -> None:
        """
        Event log: start, stop, checkpoint switch, drift/kill-switch
        triggers, rollback. Without it the DB gave no way to tell why the
        fly did not trade for an hour; the only trace was stdout of a
        terminal nobody watches for days.
        """
        self.db.execute(
            "INSERT INTO events (ts, kind, detail) VALUES (?,?,?)",
            (datetime.now(timezone.utc).isoformat(), kind, detail),
        )
        self.db.commit()

    def recent_events(self, n: int = 100) -> list[dict]:
        cur = self.db.execute(
            "SELECT ts, kind, detail FROM events ORDER BY id DESC LIMIT ?", (n,))
        return [{"ts": r[0], "kind": r[1], "detail": r[2]} for r in cur.fetchall()]

    def recent_decisions(self, n: int = 50) -> list[dict]:
        cur = self.db.execute(
            "SELECT ts, exposure, target, executed, equity, blocked, needs_approval, reasons "
            "FROM decisions ORDER BY ts DESC LIMIT ?", (n,))
        return [{
            "ts": r[0], "exposure": json.loads(r[1]), "target": json.loads(r[2]),
            "executed": r[3], "equity": r[4], "blocked": bool(r[5]),
            "needs_approval": bool(r[6]), "reasons": json.loads(r[7]),
        } for r in cur.fetchall()]

    def recent_trades(self, n: int = 100) -> list[dict]:
        cur = self.db.execute(
            "SELECT ts, coin, side, units, price, fee FROM trades ORDER BY id DESC LIMIT ?", (n,))
        return [{"ts": r[0], "coin": r[1], "side": r[2], "units": r[3],
                 "price": r[4], "fee": r[5]} for r in cur.fetchall()]

    def equity_curve(self, n: int = 500) -> list[dict]:
        cur = self.db.execute("SELECT ts, value FROM equity ORDER BY ts DESC LIMIT ?", (n,))
        return [{"ts": r[0], "value": r[1]} for r in reversed(cur.fetchall())]

    def recent_rewards(self, since_id: int = 0, n: int = 200) -> list[dict]:
        cur = self.db.execute(
            "SELECT id, ts, sign, rewards FROM rewards WHERE id > ? ORDER BY id LIMIT ?",
            (since_id, n),
        )
        return [{"id": r[0], "ts": r[1], "sign": r[2], "rewards": json.loads(r[3])}
                for r in cur.fetchall()]


class LiveEngine:
    """
    Brain + wallet + journal, in real time.

    Used from the CLI and from the dashboard server (server/app.py), where
    it runs as a background asyncio task.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        checkpoint: Path | None = None,
        online_learning: bool = False,
        auto_approve: bool = False,
        think_every_sec: float = 2.0,
        run_dir: Path | None = None,
        on_update: Callable[[dict], None] | None = None,
        fresh: bool = False,
    ) -> None:
        self.cfg = cfg or load_config()
        self.coins = [s.split("/")[0] for s in self.cfg.market.symbols]
        self.online_learning = online_learning
        self.auto_approve = auto_approve
        self.think_every_sec = think_every_sec
        self.on_update = on_update
        self.paused = False
        self.stopping = False
        # fresh=True deliberately starts from a clean slate: live_state.pt
        # from a previous run is not loaded. By default accumulated
        # experience is continued across process restarts.
        self.fresh = fresh

        # The brain is built in warmup(), not here: the input layout depends
        # on the ACTUAL channel names, known only to the feature store.
        # Building earlier gives the channel count from the config instead
        # of the real one and breaks the layout.
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.brain = None
        self.sg = None
        self.rm = None
        self.plast = None

        self.executor = Executor(self.cfg)
        self.reward = CoinRewardTracker(self.cfg, self.coins)
        self.pending: deque[_Pending] = deque()

        # Shadow portfolio of the baseline formula: the control variate needs
        # the formula's REALIZED weights under the SAME arithmetic (min_change,
        # fees) as the real portfolio. Otherwise the relative reward compares
        # smoothed reality with an ideal nominal and measures execution
        # differences instead of the fly's decision (see agent.py::pf_base).
        # Recreated in warmup() for the current cfg.
        self.pf_base: Portfolio | None = None

        # Input cache: build_inputs is a full rolling z-score over the whole
        # history (tens of thousands of rows x ~190 channels). Recomputing it
        # on every think() (every ~2 s) is pointless since the hourly candle
        # changes once an hour, so rebuild only when a new candle appears.
        self._feat_cache: tuple | None = None
        self._feat_cache_last_dt = None

        # One persistent run instead of a new folder per launch: a timestamped
        # run_dir piled up many mostly empty folders and every restart began
        # with an empty history. A fixed path means the same SQLite across
        # restarts; --fresh (serve.py) does not change the path, it only skips
        # restoring live_state.pt (see _load_state).
        self.run_dir = run_dir or (self.cfg.path("runs") / "live")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "live_state.pt"
        self.store = LiveStore(self.run_dir / "live.sqlite")
        if self.checkpoint is not None:
            self._save_checkpoint_pointer()
        self.store.set_meta("adapter", self.cfg.trading.executor.adapter)
        self.store.set_meta("symbols", json.dumps(self.cfg.market.symbols))
        self.store.set_meta("started", datetime.now(timezone.utc).isoformat())
        self.store.event("engine_start", f"fresh={self.fresh}")

        self.state = LiveState(
            mode=self.cfg.trading.executor.adapter,
            learning=online_learning,
        )
        self._w0: torch.Tensor | None = None       # weights of the BASE checkpoint, for drift
        self._last_decision_bar: int | None = None
        # Bounded adaptive stride (market.adaptive_stride, off by default):
        # currently selected timeframe, time of the last switch and time of
        # the last decision (for elapsed_hours).
        self._stride_regime: str = self.cfg.market.adaptive_stride.calm_timeframe
        self._stride_last_switch_ts: float = 0.0
        self._last_decision_ts: float | None = None
        self._activations = np.zeros(1, dtype=np.uint8)
        self.checkpoint_info: dict = {}
        self._weight_history: deque = deque(maxlen=200)  # for the drift mini-chart
        self._reward_events: deque = deque(maxlen=30)     # dopamine feed for the learning panel

    # ------------------------------------------------------------------
    @property
    def _first_brain(self):
        return getattr(self.brain, "members", [self.brain])[0]

    def _current_decision_timeframe(self) -> str:
        """
        Timeframe of the current decision cadence: the static
        cfg.market.decision_timeframe, or the currently selected regime
        (_update_stride_regime) when market.adaptive_stride.enabled. With
        adaptive_stride disabled the behaviour is bit-for-bit the static one.
        """
        a = self.cfg.market.adaptive_stride
        return self._stride_regime if a.enabled else self.cfg.market.decision_timeframe

    def _bar_index(self, ts: float) -> int:
        """Decision interval index, used to detect a cadence boundary."""
        step = TF_MS[self._current_decision_timeframe()] / 1000.0
        return int(ts // step)

    def _update_stride_regime(self, prices: np.ndarray) -> None:
        """
        Bounded adaptive stride: switch between calm_timeframe and
        active_timeframe based on realized_vol (the same quantity that feeds
        the reward sigma and the baseline formula; volatility is measurably
        predictable, unlike price direction). Off by default, see
        AdaptiveStrideCfg; enable only after a walk-forward comparison with
        fixed 1h/4h.

        Hysteresis (vol_high to enter the active regime, vol_low < vol_high
        to leave it) and a minimum dwell time (min_dwell_hours) keep the
        cadence from flapping on noise around the threshold.
        """
        a = self.cfg.market.adaptive_stride
        if not a.enabled:
            return
        now = time.time()
        if now - self._stride_last_switch_ts < a.min_dwell_hours * 3600.0:
            return
        # mean over coins: the regime is global for the fly, not per coin
        vol = float(np.mean(realized_vol(prices, self.cfg.trading.baseline.vol_window)[-1]))
        target = self._stride_regime
        if self._stride_regime == a.calm_timeframe and vol >= a.vol_high:
            target = a.active_timeframe
        elif self._stride_regime == a.active_timeframe and vol <= a.vol_low:
            target = a.calm_timeframe
        if target != self._stride_regime:
            old = self._stride_regime
            self._stride_regime = target
            self._stride_last_switch_ts = now
            detail = f"{old} -> {target} (vol={vol:.4f}, thresholds {a.vol_low}/{a.vol_high})"
            self.store.event("stride_switch", detail)
            print(f"  decision cadence switched: {detail}")

    def warmup(self) -> None:
        """
        Build the brain for the actual channels and warm it up on history.
        Without warm-up the network would start from a zero state and behave
        differently from training for the first hours.
        """
        print("  loading market data and features...")
        data = get_all(self.cfg)
        X, dt, names, prices = build_inputs(self.cfg, data)

        print(f"  building the brain for {len(names)} channels...")
        # Plasticity is ALWAYS built, not only when online_learning=True at
        # startup. Otherwise the dashboard's learning on/off buttons could not
        # enable learning later without rebuilding the brain; the toggle goes
        # only through self.online_learning and the structures are ready.
        self.brain, self.sg, self.rm, self.plast = build_ensemble(
            self.cfg, with_plasticity=True, channel_names=names,
        )
        if self.checkpoint:
            # Check the channel manifest BEFORE applying weights. Otherwise a
            # changed market.features silently gives the brain a scrambled
            # input: the state_dict stores w_mag per connectome EDGE, which
            # does not depend on the number of input channels, so the weights
            # load cleanly, but the channel layout over projection neurons
            # (roles.py) is drawn from an rng seeded by n_channels. A different
            # channel count changes the whole RNG stream, including the
            # coin-to-output-neuron binding. Nothing crashes; the brain just
            # reads RSI where it was trained to read volume.
            #
            # Two manifest sources, by priority: the .pt itself (new
            # checkpoints, build.checkpoint_manifest) and channels.json next
            # to it (older checkpoints). A manifest missing from both is a
            # refusal, not a skip: there is nothing to verify against, and a
            # silent load is more dangerous than a crash.
            manifest = read_checkpoint_manifest(self.checkpoint)
            if manifest is None:
                manifest_path = self.checkpoint.parent / "channels.json"
                if manifest_path.exists():
                    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest = {
                        "channel_names": raw.get("names"),
                        "n_members": raw.get("n_members"),
                        "symbols": None,   # legacy channels.json did not store symbols
                    }
            n_members = len(getattr(self.brain, "members", [self.brain]))
            check_manifest(manifest, names, n_members, self.cfg.market.symbols,
                           str(self.checkpoint))

            info = load_checkpoint(self.brain, self.checkpoint)
            # The key is "score" (what train.py writes), not "val_sharpe".
            self.checkpoint_info = {
                "path": str(self.checkpoint),
                "epoch": info.get("epoch"),
                "score": info.get("score"),
            }
            print(f"  checkpoint loaded (epoch {info.get('epoch', '?')}, "
                  f"score {info.get('score', '?')})")
        else:
            print("  WARNING: no checkpoint given, the brain runs on initial connectome weights "
                  "(real FlyWire topology and synapses, but not trained on the market)")
        self._activations = np.zeros(self.sg.n_nodes, dtype=np.uint8)

        # Snapshot of the BASE weights right after loading the checkpoint,
        # BEFORE warm-up and BEFORE any restore of accumulated live
        # experience. Drift is therefore always measured from the cycle
        # checkpoint, not from the last process start, so N restarts cannot
        # silently drift N x the limit.
        self._w0 = self._first_brain.w_mag.detach().clone()

        stride = self.cfg.decision_stride
        warm = X[::stride][-500:]
        self.brain.reset()
        for row in warm:
            self.brain.step(torch.from_numpy(row))
        print(f"  warmed up on {len(warm)} decisions, last candle {dt[-1]:%Y-%m-%d %H:%M} UTC")
        print(f"  neurons: {self.sg.n_nodes:,}, coins: {len(self.coins)}")

        # shadow baseline portfolio: rebuilt for the current cfg
        self.pf_base = Portfolio(
            coins=self.coins,
            initial_capital=self.cfg.trading.initial_capital,
            fee_rate=self.cfg.trading.fee_rate,
            slippage=self.cfg.trading.slippage,
        ) if self.cfg.trading.position_mode == "exposure" else None

        # Seed the feature cache with what was just computed; otherwise the
        # first think() after startup would run build_inputs again for nothing.
        self._feat_cache = (X, dt, names, prices)
        self._feat_cache_last_dt = dt[-1]

        # Restore accumulated live experience AFTER warm-up: if a saved state
        # exists it replaces the weights, the readout calibration, the
        # portfolio, the safeguards and the reward queue. Warm-up on top of
        # fresh BASE weights only matters when there is no live state yet
        # (first run on this checkpoint).
        if not self._load_state():
            self._last_decision_bar = self._bar_index(time.time())

    # ------------------------------------------------------------------
    #  Experience continuity
    # ------------------------------------------------------------------
    def save_state(self) -> None:
        """
        Persist live experience: weights, learning-rule statistics,
        portfolio, safeguards and the queue of delayed rewards. Without it
        any restart (code update, network drop, reboot) would roll the brain
        back to the cycle checkpoint and erase what it accumulated.

        REINFORCE tokens in the queue hold a live autograd graph and cannot
        be serialized, so they are dropped (and noted in the event log).
        This is a known limitation, not a silent loss: weights and all
        HEBBIAN tokens are preserved.
        """
        if self.brain is None:
            return
        pf = getattr(self.executor.adapter, "pf", None)
        portfolio_sd = None
        if pf is not None:
            portfolio_sd = {"cash": pf.cash, "units": pf.units.copy(), "fees_paid": pf.fees_paid}
        g = self.executor.guards
        guards_sd = {
            "trade_times": list(g.trade_times), "approved_count": g.approved_count,
            "killed": g.killed, "kill_reason": g.kill_reason,
            "day_start_equity": g.day_start_equity, "day_key": g.day_key,
        }
        pending_sd, dropped = [], 0
        for p in self.pending:
            token_sd, ok = _serialize_token(p.token)
            if ok:
                pending_sd.append({
                    "token": token_sd, "positions": p.positions,
                    "exec_prices": p.exec_prices, "turnover": p.turnover, "due_ts": p.due_ts,
                    "base_positions": p.base_positions, "sigma": p.sigma,
                    "nominal_base": p.nominal_base, "lr_scale": p.lr_scale,
                })
            else:
                dropped += 1
        payload = {
            "schema": 1,
            "base_checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "brain": self.brain.state_dict(),
            "plasticity": self.plast.state_dict() if self.plast is not None else None,
            "n_decisions": self.state.n_decisions,
            "n_thoughts": self.state.n_thoughts,
            "last_decision_bar": self._last_decision_bar,
            "stride_regime": self._stride_regime,
            "stride_last_switch_ts": self._stride_last_switch_ts,
            "last_decision_ts": self._last_decision_ts,
            "portfolio": portfolio_sd,
            "guards": guards_sd,
            "pending": pending_sd,
        }
        tmp = self.state_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(self.state_path)   # atomic replace: never leave a half-written file
        if dropped:
            self.store.event("state_saved", f"{dropped} queued REINFORCE tokens (live autograd graph) "
                             f"were not serialized")

    def _load_state(self) -> bool:
        """Apply saved live experience on top of the freshly warmed-up base
        brain. Returns True if a state was applied."""
        if self.fresh or not self.state_path.exists():
            return False
        d = torch.load(self.state_path, weights_only=False)
        saved_base = d.get("base_checkpoint")
        cur_base = str(self.checkpoint) if self.checkpoint else None
        if saved_base != cur_base:
            # The saved state belongs to a DIFFERENT base checkpoint (the cycle
            # was switched via "make live"); applying it to the current brain
            # is riskier than starting warmed up from scratch.
            print(f"  live state in {self.state_path} belongs to checkpoint "
                  f"{saved_base!r}, current is {cur_base!r}; ignoring it")
            self.store.event("state_ignored", f"different base checkpoint: {saved_base}")
            return False

        self.brain.load_state_dict(d["brain"])
        if self.plast is not None and d.get("plasticity") is not None:
            self.plast.load_state_dict(d["plasticity"])
        self.state.n_decisions = d.get("n_decisions", 0)
        self.state.n_thoughts = d.get("n_thoughts", 0)
        self._last_decision_bar = d.get("last_decision_bar")
        self._stride_regime = d.get("stride_regime", self.cfg.market.adaptive_stride.calm_timeframe)
        self._stride_last_switch_ts = d.get("stride_last_switch_ts", 0.0)
        self._last_decision_ts = d.get("last_decision_ts")

        pf = getattr(self.executor.adapter, "pf", None)
        p = d.get("portfolio")
        if pf is not None and p is not None:
            pf.cash = p["cash"]
            pf.units = np.asarray(p["units"], dtype=np.float64)
            pf.fees_paid = p["fees_paid"]

        g = self.executor.guards
        gd = d.get("guards")
        if gd is not None:
            g.trade_times = list(gd["trade_times"])
            g.approved_count = gd["approved_count"]
            g.killed = gd["killed"]
            g.kill_reason = gd["kill_reason"]
            g.day_start_equity = gd["day_start_equity"]
            g.day_key = gd["day_key"]

        self.pending = deque()
        for pd_ in d.get("pending", []):
            pd_ = dict(pd_)
            pd_["token"] = _deserialize_token(pd_["token"])
            self.pending.append(_Pending(**pd_))

        drift = float((self._first_brain.w_mag.detach() - self._w0).abs().mean())
        self.state.weight_drift = drift
        print(f"  live state restored: {self.state.n_decisions} decisions, "
              f"{len(self.pending)} rewards queued, drift from base {drift:.4f}"
              + (", KILL-SWITCH ACTIVE" if g.killed else ""))
        self.store.event("resumed", f"decisions={self.state.n_decisions} "
                         f"queue={len(self.pending)} drift={drift:.4f}")
        return True

    def rollback_to_base(self) -> None:
        """
        Roll back to the base (cycle) checkpoint, discarding all accumulated
        live experience: weights, learning-rule statistics, reward queue.
        The portfolio and safeguards are NOT touched: this is about the
        brain, not the money.
        """
        if self.checkpoint is None:
            raise RuntimeError("no base checkpoint to roll back to")
        self.pending.clear()
        self._weight_history.clear()
        self.state.weight_drift = 0.0
        prev_fresh = self.fresh
        self.fresh = True   # do not reload the live_state.pt being rolled back
        try:
            self.warmup()
        finally:
            self.fresh = prev_fresh
        if self.state_path.exists():
            self.state_path.unlink()
        self.store.event("rollback", f"rolled back to base checkpoint {self.checkpoint}")
        print(f"  rollback to base done: {self.checkpoint}")

    def reset_kill_switch(self) -> None:
        """Manual reset of the daily stop, on an explicit command from the dashboard."""
        self.executor.guards.reset_kill_switch()
        self.store.event("kill_switch_reset", "")

    # ------------------------------------------------------------------
    def _cached_inputs(self) -> tuple:
        """
        Returns (X, dt, names, prices), recomputing build_inputs only when a
        NEW candle appears. Otherwise the full rolling z-score over the whole
        history would run on every think(), i.e. every think_every_sec
        seconds instead of once an hour when the data actually changes.
        """
        data = get_all(self.cfg)
        last_dt = data[self.cfg.market.symbols[0]]["dt"].iloc[-1]
        if self._feat_cache is None or self._feat_cache_last_dt != last_dt:
            self._feat_cache = build_inputs(self.cfg, data)
            self._feat_cache_last_dt = last_dt
        return self._feat_cache

    def think(self) -> dict:
        """
        Light tick: fresh prices/equity for the dashboard and a frame for
        the 3D view. The brain does NOT decide here. Calling brain.step()
        every think_every_sec (~1800 times per hour instead of once per
        decision) let the EMAs inside read_out (_frac_ema for motor-output
        inertia, _ro_mean/_ro_var for adaptive calibration) converge within
        seconds, so the inertia and calibration the checkpoint was trained
        with did not work live. brain.step() is now called ONLY on a
        decision boundary, inside trade().

        Makes no trades. Returns the data needed by tick()/trade().
        """
        X, dt, names, prices = self._cached_inputs()
        p_now = prices[-1]

        # The frame comes from peek(): it leaves the brain state untouched and
        # needs no decision. It adds spontaneous noise on top of the last
        # settled pattern so the view does not freeze between decisions.
        x = torch.from_numpy(X[-1])
        viz_a = self.brain.peek(x)
        self._activations = ((np.clip(viz_a.detach().numpy(), -1, 1) + 1) * 127.5).astype(np.uint8)

        # Live ticker price, not the hourly candle close. The price the brain
        # DECIDES on (p_now/prices) is left untouched to stay consistent with
        # the backtest; that candle changes once an hour. Displayed equity,
        # however, must move with real price changes between decisions,
        # otherwise a "live" dashboard stands still for hours.
        live = fetch_last_prices(self.cfg)
        p_disp = p_now
        if live is not None:
            p_disp = np.array([live.get(sym, p_now[i])
                               for i, sym in enumerate(self.cfg.market.symbols)])

        now = time.time()
        step = TF_MS[self._current_decision_timeframe()] / 1000.0
        self.state.ts = dt[-1].isoformat()
        self.state.prices = {c: float(p_disp[i]) for i, c in enumerate(self.coins)}
        # Weights drift continuously with price even without a new decision,
        # so refresh them every think tick; this does not need the brain.
        self.state.weights = {c: float(w) for c, w in
                              zip(self.coins, self.executor.adapter.weights(p_disp))}
        self.state.equity = float(self.executor.adapter.equity(p_disp))
        self.state.next_decision_in = step - (now % step)
        self.state.n_thoughts += 1
        self.state.paused = self.paused
        if self._w0 is not None:
            dw = (self._first_brain.w_mag.detach() - self._w0).abs()
            self.state.weight_drift = float(dw.mean())

        return {"X": X, "prices": prices, "dt": dt}

    # ------------------------------------------------------------------
    def trade(self, ctx: dict) -> dict | None:
        """
        Decision tick: the brain thinks (the only place that calls
        brain.step()), the decision passes the safeguards and is executed.
        Returns the execution record, or None if paused.
        """
        if self.paused:
            return None
        X, prices, dt = ctx["X"], ctx["prices"], ctx["dt"]
        p_now = prices[-1]
        ts = dt[-1]
        ts_str = ts.isoformat()

        learning = self.online_learning and self.plast is not None
        # Exploration noise only while actually learning; the Hebbian rule
        # needs it (noise on MBONs is the signal for the eligibility trace),
        # whereas without learning it would only disturb deterministic trading.
        noise = self.plast.exploration_noise if learning else 0.0
        x = torch.from_numpy(X[-1])

        # Bounded adaptive stride (market.adaptive_stride): elapsed_hours and
        # lr_scale are derived from the ACTUAL interval between two decisions,
        # and only when the mechanism is enabled; otherwise behaviour is
        # unchanged (elapsed_hours=None -> read_out uses the constant decay,
        # lr_scale=None -> learn() uses cfg.lr_stride_scale).
        now_ts = time.time()
        elapsed_hours = None
        lr_scale = None
        if self.cfg.market.adaptive_stride.enabled:
            ref_hours = TF_MS[self.cfg.training.lr_reference_timeframe] / 3_600_000.0
            elapsed_hours = ((now_ts - self._last_decision_ts) / 3600.0
                              if self._last_decision_ts is not None else ref_hours)
            lr_scale = self.cfg.lr_stride_scale_for_hours(elapsed_hours)
        self._last_decision_ts = now_ts

        decision, a = self.brain.step(x, noise=noise, elapsed_hours=elapsed_hours)

        # What is actually executed: the raw network output or the sampled
        # policy action. For REINFORCE without act() the learning token stays
        # empty and the weights never move.
        exec_fr = self.plast.act(decision) if learning else decision.fractions

        max_frac = self.cfg.trading.guards.max_fraction_per_coin
        allow_short = self.cfg.trading.allow_short
        base = None
        if self.cfg.trading.position_mode == "exposure":
            base = baseline_positions(self.cfg, prices)[-1]
        exposure = np.array([exec_fr[c] for c in self.coins])
        target = base * exposure if base is not None else exposure
        target = np.clip(target, -max_frac if allow_short else 0.0, max_frac)

        self.state.exposure = {c: float(exposure[i]) for i, c in enumerate(self.coins)}
        self.state.target = {c: float(target[i]) for i, c in enumerate(self.coins)}

        before = self.executor.adapter.weights(p_now)
        rec = self.executor.submit(
            {c: float(v) for c, v in zip(self.coins, target)},
            p_now, ts=ts, auto_approve=self.auto_approve,
        )
        after = self.executor.adapter.weights(p_now)
        turn = np.abs(after - before)

        self.store.decision(ts_str, self.state.exposure, self.state.target, rec)
        pf = getattr(self.executor.adapter, "pf", None)
        if pf is not None and pf.trades:
            for t in pf.trades[-rec.executed:] if rec.executed else []:
                self.store.trade(ts_str, t)

        self.state.n_decisions += 1
        self.state.last_reasons = rec.reasons

        if learning:
            token = self.plast.observe(a, decision.fractions_t)
            base_pos, nominal_base = None, None
            if base is not None:
                nominal_base = np.asarray(base, dtype=np.float64).copy()
                if self.pf_base is not None:
                    # Shadow baseline portfolio: same rebalance arithmetic
                    # (min_change/fees) as the real account.
                    base_target = np.clip(base, -max_frac if allow_short else 0.0, max_frac)
                    self.pf_base.rebalance(
                        base_target, p_now, ts=ts,
                        min_change=self.cfg.trading.guards.min_position_change,
                    )
                    base_pos = self.pf_base.weights(p_now).copy()
            horizon_hours = self.cfg.reward.horizon_hours
            sigma = realized_vol(prices, self.cfg.trading.baseline.vol_window)[-1] * np.sqrt(horizon_hours)
            self.pending.append(_Pending(token, after.copy(), p_now.copy(), turn.copy(),
                                         time.time() + horizon_hours * 3600.0,
                                         base_pos, sigma, nominal_base, lr_scale))

        # Persist experience on every decision tick (once per
        # decision_timeframe, i.e. hours), not on every think() (seconds).
        self.save_state()
        return {"rec": rec, "ts": ts_str}

    def _apply_matured(self, prices: np.ndarray, ts_str: str) -> None:
        """
        Apply rewards whose horizon has elapsed. Never earlier, as that
        would be look-ahead even in live mode.

        Called from tick() on EVERY tick, not only inside trade() with
        learning enabled. Otherwise the queue would not drain while paused
        or with learning off, and after re-enabling all overdue tokens would
        be applied at once with a forward return up to today's price instead
        of the proper reward.horizon_hours. The reward and the dopamine feed
        are always computed (otherwise the learning panel goes silent), but
        the WEIGHT STEP happens only if learning is ON right now: "pause
        learning" promises an immediate halt of weight updates, not "stop
        new tokens while burning down old ones". Matured tokens drained
        while learning was off are simply discarded, so they neither pile up
        nor get applied later with a foreign horizon.
        """
        if self.plast is None:
            return
        now = time.time()
        p_now = prices[-1]
        max_frac = self.cfg.trading.guards.max_fraction_per_coin
        allow_short = self.cfg.trading.allow_short
        while self.pending and self.pending[0].due_ts <= now:
            pd = self.pending.popleft()
            fwd = np.log(np.maximum(p_now, 1e-12) / np.maximum(pd.exec_prices, 1e-12))
            rw = self.reward(pd.positions, fwd, pd.sigma, pd.turnover, pd.base_positions)

            if self.online_learning:
                def _frac_reward(frac_vals: dict[str, float], _fwd=fwd, _sig=pd.sigma,
                                 _pd=pd) -> dict[str, float]:
                    # Counterfactual reward for a hypothetical ensemble-member
                    # fraction: the same clip/gross-normalisation arithmetic as
                    # the main decision but WITHOUT execution deadband/fees. It
                    # estimates "had this member decided alone", not a real
                    # trade. Without it every member would receive the same
                    # reward for the ensemble's averaged action.
                    arr = np.array([frac_vals[c] for c in self.coins])
                    tgt = _pd.nominal_base * arr if _pd.nominal_base is not None else arr
                    tgt = np.clip(tgt, -max_frac if allow_short else 0.0, max_frac)
                    pos = clip_target(tgt, allow_short)
                    return self.reward(pos, _fwd, _sig, _pd.turnover, _pd.base_positions)

                self.plast.learn(rw, pd.token, reward_fn=_frac_reward, lr_scale=pd.lr_scale)

            # Dopamine moment: the frontend flashes dopaminergic neurons green
            # for a positive reward and red for a negative one; the flash
            # fades within a few seconds.
            mean_r = float(np.mean(list(rw.values()))) if rw else 0.0
            self.state.last_reward_ts = time.time()
            self.state.last_reward_sign = mean_r
            self._reward_events.append({"ts": ts_str, "rewards": rw, "sign": mean_r})
            self.store.reward_event(ts_str, mean_r, rw)

            # safeguard: weights must not drift away unattended
            if self._w0 is not None:
                drift = float((self._first_brain.w_mag.detach() - self._w0).abs().mean())
                self.store.weights(ts_str, float(self._first_brain.w_mag.detach().mean()), drift)
                self._weight_history.append({"ts": ts_str, "drift": drift})
                self.state.weight_drift = drift
                limit = self.cfg.training.live_drift_limit
                if limit > 0 and drift > limit:
                    self.online_learning = False
                    self.state.learning = False
                    msg = (f"online learning stopped: mean weight shift {drift:.4f} "
                           f"exceeded the limit {limit}")
                    self.state.last_reasons = [msg] + self.state.last_reasons
                    print(f"  WARNING: {msg}")
                    self.store.event("drift_limit", msg)
                    self.save_state()   # persist the learning stop immediately, not at the
                                        # next decision tick

    # ------------------------------------------------------------------
    def tick(self) -> dict:
        """
        One step: always thinks, trades on a cadence boundary, and matures
        rewards every time, so reward timing is accurate to think_every_sec
        rather than to decision_timeframe.
        """
        ctx = self.think()
        self._update_stride_regime(ctx["prices"])
        bar = self._bar_index(time.time())
        if self._last_decision_bar is None:
            self._last_decision_bar = bar
        elif bar != self._last_decision_bar:
            self._last_decision_bar = bar
            self.trade(ctx)
        self._apply_matured(ctx["prices"], ctx["dt"][-1].isoformat())
        if self.on_update:
            self.on_update(self.snapshot())
        return ctx

    def force_decision(self) -> None:
        """Make a decision immediately without waiting for a boundary. For debugging."""
        self.trade(self.think())

    # ------------------------------------------------------------------
    #  Dashboard controls
    # ------------------------------------------------------------------
    def start_learning(self) -> None:
        """"Learn" button. The plasticity structures are already built in warmup()."""
        self.online_learning = True
        self.state.learning = True

    def stop_learning(self) -> None:
        """
        "Pause learning" button. Stops ONLY learning; trading continues as
        is. Separate from the global pause (self.paused), which halts
        everything, including trades.
        """
        self.online_learning = False
        self.state.learning = False

    def switch_checkpoint(self, path: Path, requires_teacher: bool = False,
                          config_path: str | Path | None = None) -> None:
        """
        "Make live" button of a training cycle.

        Rebuilds the whole brain via warmup() instead of swapping w_mag in
        place: if the new checkpoint was trained with teacher channels
        (cycle 3) and the current live brain was built without them, the
        number of input channels differs and the input -> projection-neuron
        layout of the old brain does not cover those channels. Rebuilding
        from scratch (seconds) is cheaper than chasing a silent semantic
        mismatch.
        """
        # Cycle config: prefer config_used.json NEXT TO THE CHECKPOINT. It
        # holds EXACTLY what training was launched with, including the initial
        # (pre-lr-decay) hyperparameters. The yaml path from cycles.py is only
        # a fallback: if config/default.yaml was edited after training, using
        # it would silently desync the live brain from the checkpoint.
        manifest = Path(path).parent / "config_used.json"
        new_cfg = None
        if manifest.exists():
            new_cfg = Config(**json.loads(manifest.read_text(encoding="utf-8")))
        elif config_path is not None:
            # The cycle was trained with its own config (different features,
            # different reward - cycle 4). The brain and features must be built
            # from IT, otherwise the channel count will not match the checkpoint.
            new_cfg = load_config(config_path)
        if new_cfg is not None:
            # Keep this process's runtime settings (exchange adapter, live
            # learning step, port); they are not part of the trained model.
            new_cfg.trading.executor = self.cfg.trading.executor
            new_cfg.training.hebbian.lr = self.cfg.training.hebbian.lr
            new_cfg.training.reinforce.lr = self.cfg.training.reinforce.lr
            new_cfg.viz = self.cfg.viz
            self.cfg = new_cfg
            self.reward = CoinRewardTracker(self.cfg, self.coins)
        if requires_teacher != self.cfg.teacher.enabled:
            self.cfg.teacher.enabled = requires_teacher
        old_checkpoint = self.checkpoint
        self.checkpoint = Path(path)
        self.pending.clear()          # old tokens do not fit the new weights
        self._weight_history.clear()  # drift restarts from the new baseline
        self.state.weight_drift = 0.0
        self.store.event("checkpoint_switch", f"{old_checkpoint} -> {self.checkpoint}")
        self._save_checkpoint_pointer()
        self.warmup()

    def _save_checkpoint_pointer(self) -> None:
        """
        Persist a pointer to the last selected checkpoint next to the DB.
        The "make live" button used to change self.checkpoint only in process
        memory, so any server restart silently reset the engine to None
        (initial connectome weights) and the choice was lost. serve.py reads
        this pointer on startup when --checkpoint is not given explicitly.
        """
        if self.checkpoint is None:
            return
        (self.run_dir / "checkpoint.txt").write_text(str(self.checkpoint), encoding="utf-8")

    def shutdown(self) -> None:
        """
        "Shut down brain" button. The loop in server/app.py checks
        `stopping` on every iteration and exits, so no further thoughts or
        trades happen. Irreversible within this process: restart it with
        `python -m flybrain.serve` from a terminal.

        Experience is saved before stopping, so the next run (without
        --fresh) continues from exactly this point.
        """
        self.save_state()
        self.store.event("engine_stop", "")
        self.stopping = True

    def snapshot(self) -> dict:
        s = self.state
        return {
            "ts": s.ts, "prices": s.prices, "exposure": s.exposure, "target": s.target,
            "weights": s.weights, "equity": s.equity,
            "next_decision_in": s.next_decision_in, "n_decisions": s.n_decisions,
            "n_thoughts": s.n_thoughts, "mode": s.mode, "learning": s.learning,
            "paused": s.paused, "stopped": self.stopping, "weight_drift": s.weight_drift,
            "weight_history": list(self._weight_history),
            "reward_ts": s.last_reward_ts, "reward_sign": s.last_reward_sign,
            "reward_events": list(self._reward_events),
            "pending_count": len(self.pending),
            "checkpoint": self.checkpoint_info,
            "run_dir": str(self.run_dir),
            "tail_cmd": self._tail_cmd(),
            "reasons": s.last_reasons,
            "pending_approval": self.executor.pending is not None,
        }

    def _tail_cmd(self) -> str:
        """
        Ready-to-run command for scripts/tail_logs.py, shown on the dashboard.

        sys.executable is EXACTLY the interpreter running the server (the
        venv), not a bare "python": on macOS that command is often absent
        from PATH (only python3 or a venv interpreter exists). Absolute paths
        let the command work from any terminal directory without a prior cd.
        """
        script = self.cfg.root / "scripts" / "tail_logs.py"
        db = self.run_dir / "live.sqlite"
        return f'"{sys.executable}" "{script}" "{db}"'

    def activations_bytes(self) -> bytes:
        """Subgraph activations as uint8, ~6.3 KB per frame."""
        return self._activations.tobytes()
