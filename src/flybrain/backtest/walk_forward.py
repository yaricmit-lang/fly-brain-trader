"""
Walk-forward: the only honest evaluation on a non-stationary market.

A single "train before 2024 / test after" split answers "did the fly learn
the 2021 bull market and how did it survive the 2025 bear market".
Walk-forward answers the real question: if the fly is retrained every two
months on the latest year and trades the following two, how does that look
on average and in the worst window.

    python -m flybrain.backtest.walk_forward                 # full run, heavy
    python -m flybrain.backtest.walk_forward --windows 3 --members 1 --epochs 2   # smoke test

Inside each window: the last val-days of the training period are used to
select the best epoch; the network does not see the test window until the
final run.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .. import gentle
from ..agent import Agent
from ..build import build_ensemble
from ..config import Config, load_config
from ..features import build_inputs
from ..market.fetch import get_all
from . import benchmarks as B
from . import metrics as M
from .report import make_run_dir


def make_windows(dt: pd.DatetimeIndex, start: str, train_m: int, test_m: int, step_m: int):
    """Windows (train_idx, test_idx) split by calendar months."""
    out = []
    t0 = pd.Timestamp(start, tz="UTC")
    idx = np.arange(len(dt))
    while True:
        tr_end = t0 + pd.DateOffset(months=train_m)
        te_end = tr_end + pd.DateOffset(months=test_m)
        if te_end > dt[-1]:
            break
        tr = idx[(dt >= t0) & (dt < tr_end)]
        te = idx[(dt >= tr_end) & (dt < te_end)]
        if len(tr) > 500 and len(te) > 100:
            out.append((tr, te))
        t0 = t0 + pd.DateOffset(months=step_m)
    return out


def _snapshot(brain) -> dict:
    return copy.deepcopy(brain.state_dict())


def run_window(cfg: Config, X, prices, tr, te, val_bars: int, epochs: int, members: int,
               synthetic: bool, n_random: int, seed: int, names: list[str] | None = None) -> dict:
    """Train on tr (selecting the epoch on its tail), run on te, compare."""
    brain, sg, rm, plast = build_ensemble(cfg, synthetic=synthetic, with_plasticity=True,
                                          n_members=members, channel_names=names)
    agent = Agent(brain, cfg, plast)
    fit, val = tr[:-val_bars], tr[-val_bars:]

    hours = cfg.training.warmup_hours
    warm_val = X[max(0, val[0] - hours):val[0]] if hours > 0 and len(val) else None
    warm_te = X[max(0, te[0] - hours):te[0]] if hours > 0 and len(te) else None

    # Baseline formula on the validation tail, used to select the epoch by
    # edge (same logic as train.py): raw Sharpe on a 2-month window is noise,
    # while edge over the formula better reflects what we want to maximise
    vt_val, _ = B.vol_target(cfg, prices[val])

    best_score, best_state, best_epoch = -np.inf, _snapshot(brain), 0
    for ep in range(1, epochs + 1):
        agent.run(X[fit], prices[fit], learn=True, seed=seed + ep)
        v = agent.run(X[val], prices[val], learn=False, seed=seed, warm_X=warm_val).metrics
        edge = v.sharpe - vt_val.sharpe
        if edge > best_score:
            best_score, best_state, best_epoch = edge, _snapshot(brain), ep
        gentle.pause(cfg, f"epoch {ep}")

    brain.load_state_dict(best_state)
    test_agent = Agent(brain, cfg, None)
    fly = test_agent.run(X[te], prices[te], learn=False, seed=seed, warm_X=warm_te)

    p = prices[te]
    bh, _ = B.buy_and_hold(cfg, p)
    vt, _ = B.vol_target(cfg, p)
    sma, _ = B.sma_crossover(cfg, p)
    rnd = B.random_agents(cfg, p, n_agents=n_random, seed=seed)
    pct = B.percentile_of(fly.metrics.total_return, rnd)
    rnd_med = float(np.median([m.total_return for m in rnd]))

    return {
        "best_epoch": best_epoch,
        "val_edge_sharpe": float(best_score),
        "fly": fly.metrics.as_dict(),
        "buy_hold": bh.as_dict(),
        "vol_target": vt.as_dict(),
        "sma": sma.as_dict(),
        "random_median_return": rnd_med,
        "random_pct": pct,
        "n_decisions": fly.n_decisions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward evaluation of the fly")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=2)
    ap.add_argument("--step-months", type=int, default=2)
    ap.add_argument("--val-days", type=int, default=45)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--windows", type=int, default=0, help="limit the number of windows (smoke test)")
    ap.add_argument("--random-agents", type=int, default=30)
    ap.add_argument("--no-gentle", action="store_true")
    ap.add_argument("--pause-sec", type=int, default=None,
                    help="pause between epochs; over 25 windows the default 60 s "
                         "adds almost an hour of idle time, 25-30 is usually enough")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_gentle:
        cfg.training.gentle.enabled = False
    if args.pause_sec is not None:
        cfg.training.gentle.pause_between_epochs_sec = args.pause_sec
    gentle.apply(cfg)
    epochs = args.epochs or cfg.training.epochs
    members = args.members or cfg.training.ensemble.n_members

    print("  loading market data...")
    data = get_all(cfg)
    X, dt, names, prices = build_inputs(cfg, data)
    print(f"  inputs: {X.shape[1]} channels")

    windows = make_windows(dt, args.start, args.train_months, args.test_months, args.step_months)
    if args.windows:
        # evenly spaced windows so the smoke test sees both bull and bear regimes
        pick = np.linspace(0, len(windows) - 1, args.windows).round().astype(int)
        windows = [windows[i] for i in sorted(set(pick))]
    val_bars = args.val_days * 24
    print(f"  windows: {len(windows)}   train {args.train_months} mo -> test {args.test_months} mo, "
          f"step {args.step_months}   epochs {epochs}   ensemble {members}\n")

    run_dir = make_run_dir(cfg, f"walkforward_{args.tag or cfg.training.rule}")
    rows: list[dict] = []
    for i, (tr, te) in enumerate(windows, 1):
        a, b = dt[te[0]], dt[te[-1]]
        print(f"  window {i}/{len(windows)}: test {a:%Y-%m-%d} .. {b:%Y-%m-%d}  "
              f"(train {len(tr):,} candles)")
        r = run_window(cfg, X, prices, tr, te, val_bars, epochs, members,
                       args.synthetic, args.random_agents, cfg.project.seed + 100 * i, names)
        r["test_start"], r["test_end"] = f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"
        rows.append(r)
        f, bh, vt_ = r["fly"], r["buy_hold"], r["vol_target"]
        win = "BETTER" if f["sharpe"] > vt_["sharpe"] else "worse"
        print(f"      fly {f['total_return'] * 100:>+7.1f}%  Sharpe {f['sharpe']:>5.2f}  "
              f"trades {f['n_trades']:>4}  | vol-target {vt_['total_return'] * 100:>+7.1f}% "
              f"Sharpe {vt_['sharpe']:>5.2f} - fly {win}  | b&h {bh['total_return'] * 100:>+7.1f}%"
              f"  | percentile {r['random_pct']:.0f}")
        (run_dir / "windows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        gentle.pause(cfg, "between windows")

    # ---------------- summary ----------------
    fly_s = np.array([r["fly"]["sharpe"] for r in rows])
    fly_r = np.array([r["fly"]["total_return"] for r in rows])
    bh_r = np.array([r["buy_hold"]["total_return"] for r in rows])
    vt_r = np.array([r["vol_target"]["total_return"] for r in rows])
    vt_s = np.array([r["vol_target"]["sharpe"] for r in rows])
    pcts = np.array([r["random_pct"] for r in rows])
    beats_bh = float((fly_r > bh_r).mean() * 100)
    beats_vt = float((fly_s > vt_s).mean() * 100)
    top_q = float((pcts >= 75).mean() * 100)
    rng = np.random.default_rng(0)
    boot = [np.median(rng.choice(fly_s, len(fly_s))) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    summary = {
        "n_windows": len(rows),
        "median_sharpe": float(np.median(fly_s)),
        "median_sharpe_ci95": [float(lo), float(hi)],
        "mean_return": float(fly_r.mean()),
        "median_return": float(np.median(fly_r)),
        "worst_return": float(fly_r.min()),
        "beats_buy_hold_pct": beats_bh,
        "beats_vol_target_sharpe_pct": beats_vt,
        "median_sharpe_vol_target": float(np.median(vt_s)),
        "median_sharpe_edge_over_vol_target": float(np.median(fly_s - vt_s)),
        "windows_above_75pct_random": top_q,
        "mean_random_percentile": float(pcts.mean()),
        "bear_windows": int((bh_r < 0).sum()),
        "beats_bh_in_bear_pct": float((fly_r[bh_r < 0] > bh_r[bh_r < 0]).mean() * 100) if (bh_r < 0).any() else None,
    }

    lines = [
        f"# Walk-forward: {cfg.training.rule}, ensemble {members}, epochs {epochs}",
        f"\nDate: {datetime.now():%Y-%m-%d %H:%M}. Windows: {len(rows)}. "
        f"train {args.train_months} mo -> test {args.test_months} mo, step {args.step_months}.\n",
        "| # | test | fly | fly Sharpe | vol-target | baseline Sharpe | trades | b&h | percentile |",
        "|--:|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for i, r in enumerate(rows, 1):
        f, v = r["fly"], r["vol_target"]
        lines.append(
            f"| {i} | {r['test_start']} … {r['test_end']} | {f['total_return'] * 100:+.1f}% | "
            f"{f['sharpe']:.2f} | {v['total_return'] * 100:+.1f}% | {v['sharpe']:.2f} | "
            f"{f['n_trades']} | {r['buy_hold']['total_return'] * 100:+.1f}% | {r['random_pct']:.0f} |"
        )
    lines += [
        "\n## Summary\n",
        f"- median Sharpe: **{summary['median_sharpe']:.2f}** (95% CI {lo:.2f} … {hi:.2f})",
        f"- **vs. baseline (volatility targeting): higher Sharpe in {beats_vt:.0f}% of windows, "
        f"median edge {summary['median_sharpe_edge_over_vol_target']:+.2f}** "
        f"(baseline Sharpe {summary['median_sharpe_vol_target']:.2f})",
        f"- median window return: {summary['median_return'] * 100:+.1f}%, worst window {summary['worst_return'] * 100:+.1f}%",
        f"- beats buy-and-hold: **{beats_bh:.0f}%** of windows"
        + (f", in bear windows ({summary['bear_windows']}): {summary['beats_bh_in_bear_pct']:.0f}%" if summary["beats_bh_in_bear_pct"] is not None else ""),
        f"- above the 75th percentile of random agents: **{top_q:.0f}%** of windows, mean percentile {pcts.mean():.0f}",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "\n".join(lines[-4:]))
    print(f"\n  report: {run_dir / 'report.md'}\n")


if __name__ == "__main__":
    main()
