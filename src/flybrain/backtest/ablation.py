"""
Ablation: does the fly's topology contribute anything?

The central question of the project. If the connectome is replaced by a
random graph of the same size and density and the result does not change,
the Drosophila topology is irrelevant and we merely trained a sparse
network. That is a perfectly valid scientific outcome, but it has to be
reported honestly.

Graph variants:
  connectome - the real FlyWire mushroom-body subgraph
  rewired    - edge rewiring that preserves each neuron's degree
               (configuration model). Keeps "how many connections each
               neuron has", breaks "who connects to whom". The fairest
               comparison
  erdos      - random graph of the same density, degrees not preserved
  shuffled   - same edges, but neuron roles (KC/MBON/PN) are permuted

    python -m flybrain.backtest.ablation --variants connectome rewired --windows 3
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .. import gentle
from ..agent import Agent
from ..brain.model import FlyBrain
from ..config import Config, load_config
from ..connectome.loader import Connectome
from ..connectome.roles import assign
from ..connectome.subgraph import Subgraph
from ..features import build_inputs
from ..market.fetch import get_all
from . import benchmarks as B
from .report import make_run_dir
from .walk_forward import make_windows

VARIANTS = ("connectome", "rewired", "erdos", "shuffled")


def rewire_preserving_degree(conn: Connectome, seed: int) -> Connectome:
    """
    Degree-preserving rewiring: every neuron keeps its number of inputs and
    outputs but gets random partners. Done by permuting the pre- and
    post-synaptic index lists independently.
    """
    rng = np.random.default_rng(seed)
    pre, post = conn.edge_index
    new_pre = rng.permutation(pre)
    new_post = rng.permutation(post)
    keep = new_pre != new_post                       # drop self-loops
    return Connectome(
        node_ids=conn.node_ids,
        edge_index=np.vstack([new_pre[keep], new_post[keep]]),
        syn_count=conn.syn_count[keep],
        sign=conn.sign[keep],                        # the sign travels with the edge (Dale's law)
        meta=conn.meta,
    )


def erdos_renyi(conn: Connectome, seed: int) -> Connectome:
    """Random graph of the same density. Degrees are not preserved."""
    rng = np.random.default_rng(seed)
    n, e = len(conn.node_ids), conn.edge_index.shape[1]
    pre = rng.integers(0, n, e)
    post = rng.integers(0, n, e)
    keep = pre != post
    return Connectome(
        node_ids=conn.node_ids,
        edge_index=np.vstack([pre[keep], post[keep]]),
        syn_count=rng.permutation(conn.syn_count)[keep],
        sign=rng.permutation(conn.sign)[keep],
        meta=conn.meta,
    )


def make_variant(sg: Subgraph, variant: str, seed: int) -> Subgraph:
    """Build the requested subgraph variant. Roles (KC, MBON, ...) are kept except in `shuffled`."""
    if variant == "connectome":
        return sg
    if variant == "rewired":
        return Subgraph(rewire_preserving_degree(sg.conn, seed), sg.local_index, sg.groups)
    if variant == "erdos":
        return Subgraph(erdos_renyi(sg.conn, seed), sg.local_index, sg.groups)
    if variant == "shuffled":
        rng = np.random.default_rng(seed)
        all_idx = np.concatenate(list(sg.groups.values()))
        perm = rng.permutation(all_idx)
        groups, i = {}, 0
        for role, arr in sg.groups.items():
            groups[role] = perm[i:i + len(arr)].astype(np.int32)
            i += len(arr)
        return Subgraph(sg.conn, sg.local_index, groups)
    raise ValueError(f"unknown graph variant: {variant}")


def run_variant(cfg: Config, sg: Subgraph, variant: str, X, prices, windows,
                names, members: int, epochs: int, n_random: int) -> dict:
    """Run one graph variant over all walk-forward windows."""
    from ..brain.ensemble import EnsembleBrain, EnsemblePlasticity  # noqa: PLC0415
    from ..brain.plasticity_rl import make_plasticity  # noqa: PLC0415

    rows = []
    for wi, (tr, te) in enumerate(windows, 1):
        seed = cfg.project.seed + 100 * wi
        sub = make_variant(sg, variant, seed)
        brains, rules = [], []
        for m in range(members):
            s = seed + m
            rm = assign(sub, cfg, seed=s, channel_names=names)
            b = FlyBrain(sub, rm, cfg, seed=s)
            brains.append(b)
            rules.append(make_plasticity(b, cfg))
        brain = brains[0] if members == 1 else EnsembleBrain(brains)
        plast = rules[0] if members == 1 else EnsemblePlasticity(brain, rules)

        agent = Agent(brain, cfg, plast)
        val_bars = 45 * 24
        fit, val = tr[:-val_bars], tr[-val_bars:]
        best, best_state = -np.inf, None
        import copy
        for ep in range(epochs):
            agent.run(X[fit], prices[fit], learn=True, seed=seed + ep)
            v = agent.run(X[val], prices[val], learn=False, seed=seed).metrics.sharpe
            if v > best:
                best, best_state = v, copy.deepcopy(brain.state_dict())
            gentle.pause(cfg, f"{variant} window {wi} epoch {ep + 1}")
        if best_state is not None:
            brain.load_state_dict(best_state)

        fly = Agent(brain, cfg, None).run(X[te], prices[te], learn=False, seed=seed)
        rnd = B.random_agents(cfg, prices[te], n_agents=n_random, seed=seed)
        rows.append({
            "window": wi,
            "return": fly.metrics.total_return,
            "sharpe": fly.metrics.sharpe,
            "trades": fly.metrics.n_trades,
            "random_pct": B.percentile_of(fly.metrics.total_return, rnd),
        })
        print(f"    {variant:<11} window {wi}: {fly.metrics.total_return * 100:>+7.1f}%  "
              f"Sharpe {fly.metrics.sharpe:>5.2f}  percentile {rows[-1]['random_pct']:>3.0f}")
    return {
        "windows": rows,
        "median_return": float(np.median([r["return"] for r in rows])),
        "median_sharpe": float(np.median([r["sharpe"] for r in rows])),
        "mean_percentile": float(np.mean([r["random_pct"] for r in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Connectome vs. random graphs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--variants", nargs="*", default=["connectome", "rewired"], choices=VARIANTS)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--members", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--random-agents", type=int, default=20)
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--no-gentle", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_gentle:
        cfg.training.gentle.enabled = False
    gentle.apply(cfg)

    print("  loading market data and features...")
    data = get_all(cfg)
    X, dt, names, prices = build_inputs(cfg, data, with_teacher=args.teacher or None)

    from ..build import load_graph  # noqa: PLC0415
    from ..connectome.subgraph import extract  # noqa: PLC0415
    sg = extract(cfg, load_graph(cfg))
    print(f"  subgraph: {sg.n_nodes:,} neurons, {sg.n_edges:,} connections\n")

    windows = make_windows(dt, "2022-01-01", 12, 2, 2)
    pick = np.linspace(0, len(windows) - 1, args.windows).round().astype(int)
    windows = [windows[i] for i in sorted(set(pick))]
    print(f"  windows: {len(windows)}, variants: {len(args.variants)}, "
          f"epochs {args.epochs}, ensemble {args.members}\n")

    results = {}
    for v in args.variants:
        print(f"  === {v} ===")
        results[v] = run_variant(cfg, sg, v, X, prices, windows, names,
                                 args.members, args.epochs, args.random_agents)
        r = results[v]
        print(f"    summary: median return {r['median_return'] * 100:+.1f}%, "
              f"Sharpe {r['median_sharpe']:.2f}, mean percentile {r['mean_percentile']:.0f}\n")

    run_dir = make_run_dir(cfg, "ablation")
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Ablation: connectome vs. random graphs", "",
             "| graph | median return | median Sharpe | mean percentile vs. random agents |",
             "|---|--:|--:|--:|"]
    for v, r in results.items():
        lines.append(f"| {v} | {r['median_return'] * 100:+.1f}% | {r['median_sharpe']:.2f} | "
                     f"{r['mean_percentile']:.0f} |")
    if "connectome" in results and len(results) > 1:
        c = results["connectome"]["mean_percentile"]
        others = [r["mean_percentile"] for v, r in results.items() if v != "connectome"]
        diff = c - max(others)
        lines += ["", f"**Connectome vs. best random variant: {diff:+.0f} percentile points.** "
                  + ("The topology contributes something." if diff > 10 else
                     "The difference is within noise: the fly topology plays no role.")]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-2:]))
    print(f"\n  report: {run_dir / 'report.md'}\n")


if __name__ == "__main__":
    main()
