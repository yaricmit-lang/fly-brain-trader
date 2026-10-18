"""
Run the fly against the benchmarks.

    python -m flybrain.backtest.run --synthetic
    python -m flybrain.backtest.run --checkpoint runs.nosync/.../best.pt

Without a checkpoint an untrained fly is run. That is a useful reference
point too: if the trained fly is no better than the untrained one,
training achieved nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..agent import Agent
from ..build import build_ensemble, load_checkpoint
from ..config import load_config
from ..features import build_inputs
from ..market.fetch import get_all
from ..market.split import make_split
from . import benchmarks as B
from . import metrics as M
from .report import make_run_dir, write_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the fly against benchmarks")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="path to best.pt")
    ap.add_argument("--period", default="test", choices=["train", "val", "test"])
    ap.add_argument("--random-agents", type=int, default=100)
    ap.add_argument("--record-activations", action="store_true")
    ap.add_argument("--members", type=int, default=None, help="ensemble size, same as used in training")
    args = ap.parse_args()

    cfg = load_config(args.config)

    print("  loading market data...")
    data = get_all(cfg)
    X, dt, names, prices = build_inputs(cfg, data)
    split = make_split(dt, cfg)
    idx = getattr(split, args.period)
    print(split.describe(dt))

    p, x, ts = prices[idx], X[idx], dt[idx]

    print("  building the brain...")
    brain, sg, rm, _ = build_ensemble(cfg, synthetic=args.synthetic, n_members=args.members,
                                      channel_names=names)
    first = getattr(brain, "members", [brain])[0]
    state = "untrained"
    if args.checkpoint:
        info = load_checkpoint(brain, Path(args.checkpoint))
        state = f"trained (epoch {info.get('epoch', '?')})"
        print(f"  loaded checkpoint: {args.checkpoint}")
    print(brain.describe())

    fly_name = f"fly, {state}"
    agent = Agent(brain, cfg)
    print(f"  running the fly on period {args.period} ({len(idx):,} candles)...")
    fly = agent.run(x, p, learn=False, record_activations=args.record_activations,
                    progress_every=2000)

    results: dict[str, M.Metrics] = {fly_name: fly.metrics}
    curves: dict[str, list[float]] = {fly_name: fly.equity}

    print("  computing benchmarks...")
    m, c = B.buy_and_hold(cfg, p, ts)
    results["buy and hold"], curves["buy and hold"] = m, c
    m, c = B.vol_target(cfg, p, ts)
    # main reference: the same baseline the fly sits on, with multiplier exactly 1.0
    results["volatility targeting"], curves["volatility targeting"] = m, c
    m, c = B.sma_crossover(cfg, p, ts=ts)
    results["SMA crossover"], curves["SMA crossover"] = m, c

    print(f"  running {args.random_agents} random agents...")
    rnd = B.random_agents(cfg, p, n_agents=args.random_agents, seed=cfg.project.seed)

    print("\n" + M.Metrics.header())
    for name, met in results.items():
        print(met.row(name))
    print(B.describe_random(rnd))

    pct = B.percentile_of(fly.metrics.total_return, rnd)
    print(f"  The fly ranks at percentile {pct:.0f} of random agents.\n")

    run_dir = make_run_dir(cfg, f"backtest_{args.period}")
    extra = (
        f"Brain: {'synthetic stub' if args.synthetic else 'FlyWire connectome'}, "
        f"{first.n:,} neurons, {len(first.w_mag):,} connections, "
        f"{int(first.plastic.sum()):,} plastic, "
        f"ensemble {len(getattr(brain, 'members', [brain]))}.\n\n"
        f"Learning rule: {cfg.training.rule}. State: {state}."
    )
    path = write_report(
        run_dir, cfg, f"{args.period} ({ts[0]:%Y-%m-%d} .. {ts[-1]:%Y-%m-%d})",
        results, curves, random_results=rnd, fly_name=fly_name, extra=extra,
    )

    if args.record_activations and fly.activations:
        acts = np.stack(fly.activations)
        np.savez_compressed(
            run_dir / "activations.npz",
            activations=acts,
            bars=np.array(fly.act_bars),
        )
        mb = acts.nbytes / 1024 / 1024
        print(f"  activations: {acts.shape[0]:,} frames x {acts.shape[1]:,} neurons ({mb:.1f} MB)")

    print(f"  report: {path}\n")


if __name__ == "__main__":
    main()
