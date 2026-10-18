#!/usr/bin/env python3
"""
Toy connectome for testing the code before the real data is downloaded.

Mimics the fly mushroom-body wiring in miniature:
  PN   -> KC    random and sparse (each fly KC listens to ~6 PNs)
  KC   -> MBON  all-to-all; these are the plastic connections
  KC   -> APL   and APL -> KC, an inhibitory loop that keeps the code sparse
  DAN  -> MBON  modulatory connections, sign zero

It lets you check the model, the learning rules and the full loop without
waiting for the download. On the real connectome the sizes simply get larger.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flybrain.config import load_config          # noqa: E402
from flybrain.connectome.loader import Connectome  # noqa: E402


def build(n_pn=120, n_kc=800, n_mbon=40, n_dan=20, claws=6, seed=42) -> Connectome:
    """Build a graph with a fly-like wiring structure."""
    rng = np.random.default_rng(seed)

    # index layout: PN, KC, MBON, DAN, APL
    pn = np.arange(n_pn)
    kc = np.arange(n_pn, n_pn + n_kc)
    mbon = np.arange(n_pn + n_kc, n_pn + n_kc + n_mbon)
    dan = np.arange(n_pn + n_kc + n_mbon, n_pn + n_kc + n_mbon + n_dan)
    apl = np.array([n_pn + n_kc + n_mbon + n_dan])
    n_total = int(apl[-1]) + 1

    pre: list[np.ndarray] = []
    post: list[np.ndarray] = []
    syn: list[np.ndarray] = []
    sign: list[np.ndarray] = []

    # PN -> KC: each Kenyon cell listens to a few random PNs
    src = rng.choice(pn, size=(n_kc, claws))
    pre.append(src.reshape(-1))
    post.append(np.repeat(kc, claws))
    syn.append(rng.integers(5, 30, n_kc * claws).astype(np.float32))
    sign.append(np.ones(n_kc * claws, dtype=np.float32))       # cholinergic, excitatory

    # KC -> MBON: all-to-all, the future plastic connections
    a = np.repeat(kc, n_mbon)
    b = np.tile(mbon, n_kc)
    pre.append(a)
    post.append(b)
    syn.append(rng.integers(5, 15, len(a)).astype(np.float32))
    sign.append(np.ones(len(a), dtype=np.float32))

    # KC -> APL and APL -> KC: inhibitory feedback
    pre.append(kc)
    post.append(np.full(n_kc, apl[0]))
    syn.append(rng.integers(5, 12, n_kc).astype(np.float32))
    sign.append(np.ones(n_kc, dtype=np.float32))

    pre.append(np.full(n_kc, apl[0]))
    post.append(kc)
    syn.append(rng.integers(5, 12, n_kc).astype(np.float32))
    sign.append(-np.ones(n_kc, dtype=np.float32))              # GABA, inhibitory

    # DAN -> MBON: modulators, no direct drive
    a = np.repeat(dan, n_mbon)
    b = np.tile(mbon, n_dan)
    pre.append(a)
    post.append(b)
    syn.append(rng.integers(5, 20, len(a)).astype(np.float32))
    sign.append(np.zeros(len(a), dtype=np.float32))

    edge_index = np.vstack([
        np.concatenate(pre).astype(np.int32),
        np.concatenate(post).astype(np.int32),
    ])
    syn_count = np.concatenate(syn)
    sign_arr = np.concatenate(sign)

    # node table with types and coordinates (coordinates are a 3D placeholder)
    cell_type = (
        ["PN_synthetic"] * n_pn + ["KCg-m"] * n_kc + ["MBON-01"] * n_mbon
        + ["PAM-01"] * n_dan + ["APL"]
    )
    cell_class = (
        ["ALPN"] * n_pn + ["Kenyon_Cell"] * n_kc + ["MBON"] * n_mbon
        + ["DAN"] * n_dan + ["APL"]
    )
    xyz = rng.normal(0, 1, (n_total, 3)).astype(np.float32) * 100

    meta = pl.DataFrame({
        "root_id": np.arange(n_total, dtype=np.uint64),
        "cell_type": cell_type,
        "cell_class": cell_class,
        "side": ["left"] * n_total,
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
    })

    return Connectome(
        node_ids=np.arange(n_total, dtype=np.uint64),
        edge_index=edge_index,
        syn_count=syn_count,
        sign=sign_arr,
        meta=meta,
    )


def main() -> None:
    cfg = load_config()
    conn = build()
    out = cfg.path("connectome") / "graph_synthetic.npz"
    conn.save(out)
    print(f"\n  toy connectome built: {conn.n_nodes:,} neurons, "
          f"{conn.n_edges:,} connections")
    print(f"  saved: {out}")
    print("  this is a PLACEHOLDER for testing the code, not a real brain\n")


if __name__ == "__main__":
    main()
