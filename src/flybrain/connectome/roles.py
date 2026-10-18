"""
Role assignment: which subgraph neurons serve as inputs and outputs.

The idea is biological: in the fly a single odor activates not one neuron but
its own pattern of projection neurons (PNs). Here a single market channel (for
example "BTC RSI") activates its own group of PNs. The decision is read from
the output MBONs, split into "approach" and "avoid" groups, one pair per coin.

The assignment is fixed by the seed: for a given seed it is reproducible,
otherwise comparing experiments would be meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config, load_config
from .subgraph import Subgraph


@dataclass
class RoleMap:
    """
    channel_to_input -- (C, k) PN indices for each input channel
    channel_names    -- human-readable channel names, in order
    coin_approach    -- coin -> indices of the "buy" MBON group
    coin_avoid       -- coin -> indices of the "sell" MBON group
    dopamine         -- indices of dopaminergic neurons (highlighted in 3D)
    plastic_mask     -- (E,) bool: which edges are trainable at all
    """

    channel_to_input: np.ndarray
    channel_names: list[str]
    coin_approach: dict[str, np.ndarray]
    coin_avoid: dict[str, np.ndarray]
    dopamine: np.ndarray
    plastic_mask: np.ndarray

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)


def build_channel_names(cfg: Config) -> list[str]:
    """Channel names: coin x feature. The order matters; it is also the feature column order."""
    return [
        f"{sym.split('/')[0]}:{feat}"
        for sym in cfg.market.symbols
        for feat in cfg.market.features
    ]


def assign(
    sg: Subgraph,
    cfg: Config | None = None,
    seed: int | None = None,
    channel_names: list[str] | None = None,
) -> RoleMap:
    """
    Map channels onto input neurons and coins onto output neurons.

    seed overrides the config seed. The ensemble needs it: each member gets
    its own input mapping and its own MBON split, otherwise they would be
    identical.

    channel_names are the real channel names from the feature store (it also
    contains market-wide "MKT:*" channels, not only coin x feature). Without
    them the legacy scheme from the config is used.
    """
    cfg = cfg or load_config()
    rng = np.random.default_rng(cfg.project.seed if seed is None else seed)

    names = list(channel_names) if channel_names else build_channel_names(cfg)
    n_ch = len(names)

    # ---- input: each channel gets its own group of projection neurons ----
    pn = sg.group("input")
    k = min(cfg.model.input_neurons_per_channel, len(pn))
    if k < cfg.model.input_neurons_per_channel:
        print(
            f"  warning: only {len(pn)} input neurons available, "
            f"using {k} per channel instead of {cfg.model.input_neurons_per_channel}"
        )
    # groups overlap, as real glomeruli do for similar odors
    channel_to_input = np.stack([rng.choice(pn, size=k, replace=False) for _ in range(n_ch)])

    # ---- output: each coin gets a pair of MBON groups ----
    mbon = sg.group("output").copy()
    rng.shuffle(mbon)
    coins = [s.split("/")[0] for s in cfg.market.symbols]
    n_coins = len(coins)
    if len(mbon) < 2 * n_coins:
        raise ValueError(
            f"{len(mbon)} output neurons available, but at least {2 * n_coins} "
            f"are required (one pair of groups per coin). Widen the 'output' pattern in the config."
        )
    chunks = np.array_split(mbon, 2 * n_coins)
    coin_approach = {c: chunks[2 * i] for i, c in enumerate(coins)}
    coin_avoid = {c: chunks[2 * i + 1] for i, c in enumerate(coins)}

    dopamine = sg.groups.get("dopamine", np.array([], dtype=np.int32))

    # ---- which edges are plastic ----
    pre, post = sg.conn.edge_index
    if cfg.model.plastic_edges == "all":
        plastic = np.ones(sg.n_edges, dtype=bool)
    else:
        kc = np.zeros(sg.n_nodes, dtype=bool)
        kc[sg.group("kenyon")] = True
        out = np.zeros(sg.n_nodes, dtype=bool)
        out[sg.group("output")] = True
        plastic = kc[pre] & out[post]

    return RoleMap(channel_to_input, names, coin_approach, coin_avoid, dopamine, plastic)


def summary(rm: RoleMap, sg: Subgraph) -> str:
    lines = [
        "",
        f"  input channels: {rm.n_channels} "
        f"({rm.channel_to_input.shape[1]} neurons per channel)",
        f"  output coins: {len(rm.coin_approach)}",
        "",
        f"  {'coin':<8} {'approach':>9} {'avoid':>9}",
        "  " + "-" * 28,
    ]
    for coin in rm.coin_approach:
        lines.append(
            f"  {coin:<8} {len(rm.coin_approach[coin]):>9} {len(rm.coin_avoid[coin]):>9}"
        )
    lines += [
        "",
        f"  dopaminergic neurons: {len(rm.dopamine):,}",
        f"  plastic connections: {int(rm.plastic_mask.sum()):,} "
        f"of {sg.n_edges:,} ({100 * rm.plastic_mask.mean():.1f}%)",
    ]
    return "\n".join(lines) + "\n"
