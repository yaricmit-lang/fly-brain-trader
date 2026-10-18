"""
Training on seasonal blocks: 6 non-overlapping 2-month periods, namely
the 2 calmest, 2 most active and 2 medium by volatility over the last 2 years.

This script serves TWO cycles (they differ only in --config/--select-by):
  cycle 6 - base: Hebbian rule, decisions every 4h, selection by edge;
  cycle 8 - hybrid 5+6+2: the same seasonal split, but HOURLY decisions and
    boosted fundamental channels (from cycle 5), the REINFORCE rule (from
    cycle 2), epoch selection by return and a x3.2-per-year goal (cycle 5).

Why a separate script instead of a train.py flag: the regular train.py
learns on ONE contiguous slice (`X[tr]`/`prices[tr]`) and passes it to
`Agent.run()` in a single call. If six disjoint periods were concatenated
into one array and fed in, `Agent.run()` would mark equity and compute the
forward reward return across ADJACENT slice indices; at the seam between any
two periods that drags a price from one period into another as if one hour
had passed between them. A silent bug in the equity curve and the reward.

The workaround breaks nothing: `agent.run()` already supports many
independent consecutive calls on the same brain. `FlyBrain.reset()` zeroes
only the current activity (`self.state`) and the readout calibration; the
trained weights (`w_mag`) are untouched. `Plasticity.reset()` likewise clears
only the likelihood trace, not the weights and not the running dopamine
baseline. So `agent.run()` can be called once per BLOCK: everything inside a
block is local and correct, while the weights keep learning from block to
block.

    python -m flybrain.train_seasonal --synthetic --epochs 1   # smoke test
    python -m flybrain.train_seasonal                          # real cycle

The goal is not generalization (unlike walk-forward) but targeted return in
exactly these 6 periods, so there is no separate held-out test: the
checkpoint is selected by the fly's mean return ACROSS ALL 6 blocks
(learn=False).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import gentle
from .agent import Agent
from .backtest import benchmarks as B
from .backtest.report import make_run_dir
from .build import build_ensemble, checkpoint_manifest, save_checkpoint
from .config import load_config
from .features import build_inputs
from .market.fetch import get_all
from .market.periods import select_seasonal_blocks


def main() -> None:
    ap = argparse.ArgumentParser(description="Training on seasonal periods (cycles 6 and 8)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--no-gentle", action="store_true")
    ap.add_argument("--pause-sec", type=int, default=None)
    ap.add_argument("--lookback-months", type=int, default=24)
    ap.add_argument("--block-months", type=int, default=2)
    ap.add_argument("--n-calm", type=int, default=2)
    ap.add_argument("--n-active", type=int, default=2)
    ap.add_argument("--n-medium", type=int, default=2)
    ap.add_argument("--select-by", choices=["edge", "return"], default="edge",
                    help="what to pick the best epoch by: 'edge' is the fly's Sharpe "
                         "minus the formula's Sharpe, averaged over blocks (default, more "
                         "honest); 'return' is mean return over blocks, aggressive, with no "
                         "drawdown penalty (how cycle 5 and hybrid cycle 8 select)")
    ap.add_argument("--goal-return", type=float, default=0.0,
                    help="target for the MEAN block return for result.json[hit_goal], as a "
                         "fraction (0.214 per 2 months = x3.2 per year, the cycle 5 goal). "
                         "Does not affect training, only the report; 0 = no goal")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--tag", default="seasonal")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_gentle:
        cfg.training.gentle.enabled = False
    if args.pause_sec is not None:
        cfg.training.gentle.pause_between_epochs_sec = args.pause_sec
    epochs = args.epochs or cfg.training.epochs
    members = args.members if args.members is not None else cfg.training.ensemble.n_members
    gentle.apply(cfg)
    t_start = time.time()
    # the learning rule comes from the cycle config (training.rule), not a
    # flag: this script has no --rule but does have --config, and
    # build_ensemble below builds the right plasticity via make_plasticity
    print(f"  learning rule: {cfg.training.rule}   "
          f"decision timeframe: {cfg.market.decision_timeframe}   epoch selection: {args.select_by}")
    scale = cfg.lr_stride_scale
    if scale != 1.0:
        base_lr = cfg.training.hebbian.lr if cfg.training.rule == "hebbian" else cfg.training.reinforce.lr
        print(f"  step correction for decision frequency (reference "
              f"{cfg.training.lr_reference_timeframe}): lr {base_lr:g} x {scale:g} = {base_lr * scale:g}")

    print("  loading market data...")
    data = get_all(cfg)
    X, dt, names, prices = build_inputs(cfg, data)
    print(f"  inputs: {X.shape[1]} channels, {len(X):,} hourly candles")

    blocks = select_seasonal_blocks(
        dt, prices, lookback_months=args.lookback_months, block_months=args.block_months,
        n_calm=args.n_calm, n_active=args.n_active, n_medium=args.n_medium,
    )
    print(f"  periods ({len(blocks)}):")
    for b in blocks:
        print(f"    {b.label:<10} {b.start} .. {b.end}  volatility {b.mean_vol:.4f}")

    brain, sg, rm, plast = build_ensemble(
        cfg, synthetic=args.synthetic, with_plasticity=True, n_members=members, channel_names=names,
    )
    print(brain.describe())
    if len(names) != rm.n_channels:
        raise SystemExit(
            f"data has {len(names)} channels, but the brain expects {rm.n_channels}. "
            f"Check market.features and market.symbols in the config."
        )

    agent = Agent(brain, cfg, plast)
    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(cfg, f"train_{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "channels.json").write_text(
        json.dumps({"names": names, "seed": cfg.project.seed, "n_members": members, "schema": 2},
                   indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # readout warm-up before EVERY block (2026-09-16, C2): brain.reset()
    # inside agent.run() resets the calibration on every call, and there are
    # 12 runs per epoch (6 learn + 6 eval); without a warm-up the transient
    # eats a noticeable share of a block's ~365 decisions
    hours = cfg.training.warmup_hours
    warm_by_block = [X[max(0, b.idx[0] - hours):b.idx[0]] if hours > 0 else None for b in blocks]

    best_score, best_epoch = -np.inf, 0
    patience = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        print(f"\n  epoch {epoch}/{epochs}")
        order = np.random.default_rng(cfg.project.seed + epoch).permutation(len(blocks))
        for i in order:
            b = blocks[i]
            res = agent.run(X[b.idx], prices[b.idx], learn=True, seed=cfg.project.seed + epoch,
                            warm_X=warm_by_block[i])
            print(f"    {b.label:<10} {b.start}..{b.end}  return {res.metrics.total_return*100:>+6.1f}%  "
                  f"trades {res.metrics.n_trades:>4}")
            gentle.pause(cfg, f"after block {b.label}")

        # Evaluation: agent.run(learn=False) on EVERY block. The selection
        # criterion is the mean Sharpe EDGE over the formula (2026-09-16, C3),
        # not raw mean return: returns rise in a bull market with no
        # contribution from the fly, while edge is what the cycle exists for.
        block_results = []
        for i, b in enumerate(blocks):
            res = agent.run(X[b.idx], prices[b.idx], learn=False, seed=cfg.project.seed,
                            warm_X=warm_by_block[i])
            vt, _ = B.vol_target(cfg, prices[b.idx])
            block_results.append({
                "label": b.label, "start": b.start, "end": b.end,
                "fly_return": res.metrics.total_return, "fly_sharpe": res.metrics.sharpe,
                "baseline_return": vt.total_return, "baseline_sharpe": vt.sharpe,
            })
        mean_return = float(np.mean([r["fly_return"] for r in block_results]))
        mean_edge = float(np.mean([r["fly_sharpe"] - r["baseline_sharpe"] for r in block_results]))
        score = mean_return if args.select_by == "return" else mean_edge
        print(f"    mean return over 6 blocks: {mean_return*100:+.1f}%   mean Sharpe edge: {mean_edge:+.2f}")

        history.append({"epoch": epoch, "mean_return": mean_return, "mean_edge": mean_edge,
                        "score": score, "select_by": args.select_by, "blocks": block_results})

        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(brain, run_dir / "best.pt", {"epoch": epoch, "score": score,
                                                          "select_by": args.select_by,
                                                          "mean_edge": mean_edge,
                                                          "mean_return": mean_return},
                            manifest=checkpoint_manifest(cfg, brain, names))
            print("    val (block mean) improved, checkpoint saved")
            patience = 0
        else:
            patience += 1
            print(f"    no improvement ({patience}/{cfg.training.early_stopping_patience})")
            if patience >= cfg.training.early_stopping_patience:
                print("    early stopping")
                break

    save_checkpoint(brain, run_dir / "last.pt", {"epoch": len(history)},
                    manifest=checkpoint_manifest(cfg, brain, names))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    best = history[best_epoch - 1] if best_epoch else (history[-1] if history else None)
    result = {
        "rule": cfg.training.rule,
        "select_by": args.select_by,
        "decision_timeframe": cfg.market.decision_timeframe,
        "members": members,
        "epochs_done": len(history),
        "best_epoch": best_epoch or (len(history) if history else None),
        "mean_return": best["mean_return"] if best else None,
        "mean_edge": best["mean_edge"] if best else None,
        # goal for the MEAN block return (cycle 8 inherits it from cycle 5:
        # +21.4% per 2 months = x3.2 per year). 0 = no goal (cycle 6).
        "goal_return": args.goal_return or None,
        "hit_goal": bool(best and args.goal_return and best["mean_return"] >= args.goal_return),
        "blocks": best["blocks"] if best else [],
        "duration_sec": round(time.time() - t_start),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if best:
        print(f"\n  mean return over 6 seasonal blocks: {best['mean_return']*100:+.1f}%"
              f"   mean Sharpe edge: {best['mean_edge']:+.2f}")
        for r in best["blocks"]:
            edge = r["fly_return"] - r["baseline_return"]
            print(f"    {r['label']:<10} {r['start']}..{r['end']}  "
                  f"fly {r['fly_return']*100:>+6.1f}%  formula {r['baseline_return']*100:>+6.1f}%  "
                  f"delta {edge*100:>+6.1f} pp")
        if args.goal_return:
            print(f"  goal (mean block return >= {args.goal_return*100:.1f}%): "
                  f"{'REACHED' if result['hit_goal'] else 'not reached'}")
    print(f"  time: {result['duration_sec'] / 60:.1f} min")
    print(f"\n  done. Run: {run_dir}\n")


if __name__ == "__main__":
    main()
