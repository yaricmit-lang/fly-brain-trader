#!/usr/bin/env python3
"""
Export brain geometry for the browser dashboard. Run once.

The whole brain is 138,533 neurons, and their coordinates are already in the
connectome cache. We normalize them into the cube [-1, 1] and pack them as
int16: 0.79 MB instead of 1.59 MB as float32, with a position error below
one screen pixel.

Layout of geometry.bin (all little-endian, back to back):
    positions   int16  x 3 x n_all     coordinates of all neurons
    mb_index    int32  x n_mb          indices of mushroom-body neurons
    mb_role     uint8  x n_mb          role: 0 KC, 1 MBON, 2 DAN, 3 PN, 4 APL
    edges       int32  x 2 x n_edges   strongest KC->MBON connections (for lines)

geometry.json describes offsets and lengths so the frontend need not guess.

    python scripts/export_geometry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flybrain.config import load_config          # noqa: E402
from flybrain.connectome.loader import Connectome  # noqa: E402
from flybrain.connectome.subgraph import extract   # noqa: E402

ROLE_CODES = {"kenyon": 0, "output": 1, "dopamine": 2, "input": 3, "inhibit": 4}
MAX_EDGES = 2000


def main() -> None:
    cfg = load_config()
    cache = cfg.path("connectome") / f"graph_v{cfg.connectome.version}_syn{cfg.connectome.min_syn_count}.npz"
    if not cache.exists():
        raise SystemExit(f"connectome cache not found: {cache}\nrun first: python -m flybrain.connectome.loader")

    print("  reading connectome...")
    conn = Connectome.load(cache)
    meta = conn.meta
    xyz = np.stack([meta["x"].to_numpy(), meta["y"].to_numpy(), meta["z"].to_numpy()], axis=1)
    xyz = np.nan_to_num(xyz.astype(np.float64))

    # Center and normalize into the cube [-1, 1] by the longest axis so the
    # brain's proportions are preserved.
    center = (xyz.max(axis=0) + xyz.min(axis=0)) / 2.0
    scale = (xyz.max(axis=0) - xyz.min(axis=0)).max() / 2.0
    norm = (xyz - center) / scale

    # FlyWire's Z axis runs into the imaging slice; swap axes so the brain
    # faces the camera in the usual orientation.
    norm = norm[:, [0, 2, 1]]
    norm[:, 1] *= -1.0

    positions = np.clip(norm * 32767, -32767, 32767).astype("<i2")
    print(f"  neurons: {len(positions):,}   positions: {positions.nbytes / 1024 / 1024:.2f} MB")

    print("  extracting mushroom bodies...")
    sg = extract(cfg, conn)
    mb_index = sg.local_index.astype("<i4")
    role = np.full(len(mb_index), 255, dtype=np.uint8)
    # local_index holds the subgraph nodes' positions in the full graph;
    # groups stores indices WITHIN the subgraph, so assign roles by them directly
    for name, code in ROLE_CODES.items():
        idx = sg.groups.get(name)
        if idx is not None and len(idx):
            role[idx] = code
    counts = {n: int((role == c).sum()) for n, c in ROLE_CODES.items()}
    print(f"  subgraph: {len(mb_index):,} neurons  {counts}")

    # Lines: the strongest Kenyon cell -> output neuron connections.
    # Indices are converted to those of the FULL graph; the frontend draws by them.
    pre, post = sg.conn.edge_index
    kc = np.zeros(sg.n_nodes, dtype=bool)
    kc[sg.group("kenyon")] = True
    out = np.zeros(sg.n_nodes, dtype=bool)
    out[sg.group("output")] = True
    mask = kc[pre] & out[post]
    syn = np.where(mask, sg.conn.syn_count, -1.0)
    top = np.argsort(syn)[::-1][:MAX_EDGES]
    top = top[syn[top] > 0]
    edges = np.vstack([mb_index[pre[top]], mb_index[post[top]]]).T.astype("<i4")
    print(f"  KC->MBON lines: {len(edges):,}")

    out_dir = cfg.path("data")

    # 4-byte alignment is required: browsers refuse to create an Int32Array
    # at an offset that is not a multiple of four. int16 x 3 x 138533
    # positions take 831,198 bytes, which is even but not divisible by 4, and
    # the frontend failed with "start offset of Int32Array should be a
    # multiple of 4".
    chunks: list[bytes] = []
    off = 0
    layout = {}
    for key, arr in (("positions", positions), ("mb_index", mb_index),
                     ("mb_role", role), ("edges", edges)):
        pad = (-off) % 4
        if pad:
            chunks.append(b"\0" * pad)
            off += pad
        layout[key] = {"offset": off, "bytes": arr.nbytes, "count": int(arr.size)}
        chunks.append(arr.tobytes())
        off += arr.nbytes

    (out_dir / "geometry.bin").write_bytes(b"".join(chunks))

    meta_json = {
        "n_neurons": int(len(positions)),
        "n_mb": int(len(mb_index)),
        "n_edges": int(len(edges)),
        "roles": ROLE_CODES,
        "role_counts": counts,
        "position_scale": 32767,
        "layout": layout,
        "total_bytes": off,
    }
    (out_dir / "geometry.json").write_text(json.dumps(meta_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  done: {out_dir / 'geometry.bin'}  ({off / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
