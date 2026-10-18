"""
Stage 2: run the benchmarks alone, without the fly.

This is the reference point. Until we know what buy-and-hold returns on the
test period, discussing the network's results is pointless.

    python -m flybrain.backtest.run_benchmarks
    python -m flybrain.backtest.run_benchmarks --period train
"""

from __future__ import annotations

import argparse

from ..config import load_config
from ..market.features import build_matrix, prices_matrix
from ..market.fetch import get_all
from ..market.split import make_split
from . import benchmarks as B
from . import metrics as M
from .report import make_run_dir, write_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark backtest")
    ap.add_argument("--period", default="test", choices=["train", "val", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--random-agents", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)

    print("  loading market data...")
    data = get_all(cfg)
    X, dt, _ = build_matrix(data, cfg)
    n_skip = len(data[cfg.market.symbols[0]]) - len(X)
    prices = prices_matrix(data, cfg, n_skip)

    split = make_split(dt, cfg)
    idx = getattr(split, args.period)
    print(split.describe(dt))
    if len(idx) < 100:
        raise SystemExit(f"period {args.period} is too short: {len(idx)} candles")

    p = prices[idx]
    ts = dt[idx]

    print(f"  computing benchmarks on period {args.period} ({len(idx):,} candles)...")
    results: dict[str, M.Metrics] = {}
    curves: dict[str, list[float]] = {}

    m, c = B.buy_and_hold(cfg, p, ts)
    results["buy and hold"], curves["buy and hold"] = m, c

    m, c = B.sma_crossover(cfg, p, ts=ts)
    results["SMA crossover"], curves["SMA crossover"] = m, c

    print(f"  running {args.random_agents} random agents...")
    rnd = B.random_agents(cfg, p, n_agents=args.random_agents, seed=cfg.project.seed)

    print("\n" + M.Metrics.header())
    for name, met in results.items():
        print(met.row(name))
    print(B.describe_random(rnd))

    run_dir = make_run_dir(cfg, f"benchmarks_{args.period}")
    path = write_report(
        run_dir, cfg, f"{args.period} ({ts[0]:%Y-%m-%d} .. {ts[-1]:%Y-%m-%d})",
        results, curves, random_results=rnd,
        extra="Benchmarks only. The fly is not attached yet: waiting for the connectome.",
    )
    print(f"  report: {path}\n")


if __name__ == "__main__":
    main()
