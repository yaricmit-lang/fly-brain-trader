"""
Extract the working subgraph from the full connectome.

The whole brain is ~139k neurons and tens of millions of synapses. Training
that on a laptop-class machine is pointless: with such a sparse input most
neurons would never receive a meaningful signal. We take the mushroom body,
which in the real fly is the learning center driven by reward.

Subgraph composition:
  KC   -- Kenyon cells, a sparse code of the "odor" (here: the market state)
  MBON -- mushroom body output neurons, from which the decision is read
  DAN  -- dopaminergic neurons (PAM, PPL1), carrying the reward signal
  APL  -- inhibitory interneuron that keeps the code sparse
  PN   -- projection neurons, the input (here: market features)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..config import Config, load_config
from .loader import Connectome, load_connectome


@dataclass
class Subgraph:
    """
    The subgraph plus role annotations.

    local_index -- indices of the subgraph nodes in the original full graph
    groups      -- role -> array of LOCAL indices (kenyon, output, dopamine, ...)
    """

    conn: Connectome
    local_index: np.ndarray
    groups: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.conn.n_nodes

    @property
    def n_edges(self) -> int:
        return self.conn.n_edges

    def group(self, name: str) -> np.ndarray:
        if name not in self.groups or len(self.groups[name]) == 0:
            raise KeyError(
                f"group '{name}' is empty. Check the regex in config "
                f"connectome.subgraph.cell_class_patterns and the file "
                f"runs.nosync/connectome_vocab.txt with the actual cell type values."
            )
        return self.groups[name]


def _match_mask(meta: pl.DataFrame, pattern: str) -> np.ndarray:
    """
    Search the regex across all cell-type text columns at once.
    Depending on the release, the type lives in cell_type, hemibrain_type or class.
    """
    rx = re.compile(pattern, re.IGNORECASE)
    n = meta.height
    mask = np.zeros(n, dtype=bool)
    for col in ("cell_type", "cell_class"):
        if col not in meta.columns:
            continue
        vals = meta[col].cast(pl.Utf8).fill_null("").to_list()
        mask |= np.fromiter((bool(rx.search(v)) for v in vals), dtype=bool, count=n)
    return mask


def _side_mask(meta: pl.DataFrame, sides: list[str]) -> np.ndarray:
    """Hemisphere filter. Keeping one side roughly halves the graph."""
    n = meta.height
    if "side" not in meta.columns or not sides or len(sides) >= 2:
        return np.ones(n, dtype=bool)
    wanted = {s.lower()[0] for s in sides}  # 'left' -> 'l'
    vals = meta["side"].cast(pl.Utf8).fill_null("").to_list()
    return np.fromiter(
        (bool(v) and v.lower()[0] in wanted for v in vals), dtype=bool, count=n
    )


def extract(cfg: Config | None = None, conn: Connectome | None = None) -> Subgraph:
    """Build the subgraph according to the rules in the config."""
    cfg = cfg or load_config()
    conn = conn or load_connectome(cfg)
    sub_cfg = cfg.connectome.subgraph
    meta = conn.meta

    if sub_cfg.mode == "threshold_only":
        keep = np.ones(conn.n_nodes, dtype=bool)
        role_masks: dict[str, np.ndarray] = {}
    else:
        role_masks = {
            role: _match_mask(meta, pat)
            for role, pat in sub_cfg.cell_class_patterns.items()
        }
        keep = np.zeros(conn.n_nodes, dtype=bool)
        for m in role_masks.values():
            keep |= m

        if sub_cfg.mode == "neuropil" and sub_cfg.neuropils:
            # also pull in every neuron present in the requested neuropils
            rx = re.compile("|".join(re.escape(n) for n in sub_cfg.neuropils), re.IGNORECASE)
            # the neuropil is stored on edges, so nodes are collected via
            # edges (not implemented here; the pattern is currently unused)
            _ = rx

    keep &= _side_mask(meta, sub_cfg.sides)

    if keep.sum() == 0:
        raise ValueError(
            "subgraph is empty: no neuron matched the patterns.\n"
            "Inspect runs.nosync/connectome_vocab.txt and fix "
            "connectome.subgraph.cell_class_patterns in the config."
        )

    # --- reindexing ---
    local_index = np.flatnonzero(keep).astype(np.int32)
    remap = np.full(conn.n_nodes, -1, dtype=np.int32)
    remap[local_index] = np.arange(len(local_index), dtype=np.int32)

    pre, post = conn.edge_index
    edge_keep = (remap[pre] >= 0) & (remap[post] >= 0)
    new_edges = np.vstack([remap[pre[edge_keep]], remap[post[edge_keep]]])

    sub_conn = Connectome(
        node_ids=conn.node_ids[local_index],
        edge_index=new_edges,
        syn_count=conn.syn_count[edge_keep],
        sign=conn.sign[edge_keep],
        meta=meta[local_index.tolist()],
    )

    groups = {
        role: np.flatnonzero(m[local_index]).astype(np.int32)
        for role, m in role_masks.items()
    }
    return Subgraph(sub_conn, local_index, groups)


def summary(sg: Subgraph) -> str:
    lines = [
        "",
        f"  subgraph: {sg.n_nodes:,} neurons, {sg.n_edges:,} edges",
        "",
        f"  {'role':<12} {'neurons':>10}",
        "  " + "-" * 24,
    ]
    for role, idx in sorted(sg.groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {role:<12} {len(idx):>10,}")

    # density: edges per neuron
    if sg.n_nodes:
        lines += ["", f"  mean degree: {sg.n_edges / sg.n_nodes:.1f} edges per neuron"]

    # number of KC -> MBON edges: the future plastic connections
    if "kenyon" in sg.groups and "output" in sg.groups:
        kc = set(sg.groups["kenyon"].tolist())
        mbon = set(sg.groups["output"].tolist())
        pre, post = sg.conn.edge_index
        n_plastic = sum(
            1 for a, b in zip(pre.tolist(), post.tolist()) if a in kc and b in mbon
        )
        lines.append(f"  plastic KC->MBON connections: {n_plastic:,}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sg = extract()
    print(summary(sg))
