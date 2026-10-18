"""
Load the FlyWire connectome into a sparse graph.

Codex renamed columns between releases, so column names are not hard-coded
but resolved from the table contents. All cell_class / cell_type values found
are written to runs.nosync/connectome_vocab.txt, which is then used to refine
the subgraph regexes in the config.

    python -m flybrain.connectome.loader
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config, load_config
from .download import check as check_files

# Possible column names: (canonical name, candidates in order of preference)
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "pre": ("pre_root_id", "pre_pt_root_id", "pre_id", "source"),
    "post": ("post_root_id", "post_pt_root_id", "post_id", "target"),
    "syn_count": ("syn_count", "synapses", "count", "weight", "n_syn"),
    "neuropil": ("neuropil", "region", "brain_region"),
    "nt_type": ("nt_type", "neurotransmitter", "nt", "top_nt"),
    "root_id": ("root_id", "pt_root_id", "id", "neuron_id"),
    # primary_type comes from the Codex Cell Types table
    # (consolidated_cell_types.csv): values such as KCab, MBON01, PAM01,
    # PPL101, DA1_lPN. It is the main source for carving out the mushroom
    # body and is more precise than cell_class
    "cell_type": ("primary_type", "cell_type", "hemibrain_type", "type", "cell_sub_class"),
    "cell_class": ("class", "cell_class", "super_class", "cell_super_class"),
    "side": ("side", "hemisphere", "soma_side"),
    "x": ("pos_x", "x", "soma_x", "soma_position_x", "centroid_x"),
    "y": ("pos_y", "y", "soma_y", "soma_position_y", "centroid_y"),
    "z": ("pos_z", "z", "soma_z", "soma_position_z", "centroid_z"),
}


def resolve(columns: list[str], canonical: str) -> str | None:
    """Find the column matching a canonical name."""
    lower = {c.lower(): c for c in columns}
    for cand in COLUMN_ALIASES[canonical]:
        if cand in lower:
            return lower[cand]
    # fallback: substring match
    for cand in COLUMN_ALIASES[canonical]:
        for c_low, c_orig in lower.items():
            if cand in c_low:
                return c_orig
    return None


def scan_columns(path: Path) -> list[str]:
    """Read only the header, without loading the whole file into memory."""
    return pl.scan_csv(path, infer_schema_length=0).collect_schema().names()


@dataclass
class Connectome:
    """
    The graph after filtering.

    node_ids   -- neuron root_ids; array index = neuron index in the graph
    edge_index -- (2, E) edge indices: row 0 presynaptic, row 1 postsynaptic
    syn_count  -- (E,) synapse counts
    sign       -- (E,) sign from the presynaptic neurotransmitter (Dale's law)
    meta       -- node table: cell_type, cell_class, side, coordinates
    """

    node_ids: np.ndarray
    edge_index: np.ndarray
    syn_count: np.ndarray
    sign: np.ndarray
    meta: pl.DataFrame

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            node_ids=self.node_ids,
            edge_index=self.edge_index,
            syn_count=self.syn_count,
            sign=self.sign,
        )
        self.meta.write_parquet(path.with_suffix(".meta.parquet"))

    @classmethod
    def load(cls, path: Path) -> "Connectome":
        z = np.load(path)
        meta = pl.read_parquet(path.with_suffix(".meta.parquet"))
        return cls(z["node_ids"], z["edge_index"], z["syn_count"], z["sign"], meta)


def _nt_sign_map(cfg: Config) -> dict[str, float]:
    """Neurotransmitter -> sign. Keys are lowercased and mapped to short codes."""
    base = {k.lower(): v for k, v in cfg.connectome.neurotransmitter_sign.items()}
    short = {
        "ach": base.get("acetylcholine", 1.0),
        "glut": base.get("glutamate", -1.0),
        "gaba": base.get("gaba", -1.0),
        "da": base.get("dopamine", 0.0),
        "oct": base.get("octopamine", 0.0),
        "ser": base.get("serotonin", 0.0),
        "5ht": base.get("serotonin", 0.0),
        "unk": base.get("unknown", 1.0),
    }
    return {**base, **short}


def write_vocab(meta: pl.DataFrame, out: Path) -> None:
    """
    Dump the unique cell type values.
    Without this it is impossible to write exact regexes for the mushroom body.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for col in ("cell_class", "cell_type", "side"):
        if col not in meta.columns:
            continue
        vals = (
            meta.select(pl.col(col).cast(pl.Utf8))
            .drop_nulls()
            .to_series()
            .value_counts(sort=True)
        )
        lines.append(f"\n=== {col}: {vals.height} unique values ===")
        for row in vals.head(400).iter_rows():
            lines.append(f"{row[1]:>8}  {row[0]}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  cell type vocabulary: {out}")


def load_connectome(cfg: Config | None = None, force: bool = False) -> Connectome:
    """
    Read the CSVs, filter by synapse threshold, build indices and cache the result.
    A repeated call uses the cache if present (unless force=True).
    """
    cfg = cfg or load_config()
    cache = cfg.path("connectome") / f"graph_v{cfg.connectome.version}_syn{cfg.connectome.min_syn_count}.npz"

    if cache.exists() and not force:
        print(f"  using cache: {cache.name}")
        return Connectome.load(cache)

    files = check_files(verbose=False)

    # ---------------- nodes ----------------
    print("  reading neuron table...")
    neurons = pl.read_csv(files["neurons"], infer_schema_length=10_000)
    n_root = resolve(neurons.columns, "root_id")
    if n_root is None:
        raise ValueError(f"root_id column not found in neurons.csv; available: {neurons.columns}")
    meta = neurons.select(pl.col(n_root).cast(pl.UInt64).alias("root_id"))

    # classification is a separate table, joined on root_id
    if "classification" in files:
        print("  reading cell classification...")
        cls = pl.read_csv(files["classification"], infer_schema_length=10_000)
        c_root = resolve(cls.columns, "root_id")
        pick: dict[str, str] = {}
        for canon in ("cell_type", "cell_class", "side"):
            col = resolve(cls.columns, canon)
            if col:
                pick[canon] = col
        if c_root and pick:
            sel = [pl.col(c_root).cast(pl.UInt64).alias("root_id")] + [
                pl.col(src).cast(pl.Utf8).alias(dst) for dst, src in pick.items()
            ]
            meta = meta.join(cls.select(sel), on="root_id", how="left")

    # Detailed cell type is a separate Codex table (Cell Types) that provides
    # granular names such as KCab / MBON01 / PAM01 / PPL101 / DA1_lPN. The
    # mushroom body subgraph is carved out by these; cell_class from the
    # classification table (Kenyon_Cell/MBON/DAN) is only a coarse fallback.
    if "cell_types" in files:
        print("  reading detailed cell types...")
        ct = pl.read_csv(files["cell_types"], infer_schema_length=10_000)
        t_root = resolve(ct.columns, "root_id")
        t_type = resolve(ct.columns, "cell_type")
        if t_root and t_type:
            sel = [
                pl.col(t_root).cast(pl.UInt64).alias("root_id"),
                pl.col(t_type).cast(pl.Utf8).alias("cell_type"),
            ]
            joined = ct.select(sel).unique(subset="root_id")
            if "cell_type" in meta.columns:
                meta = meta.drop("cell_type")
            meta = meta.join(joined, on="root_id", how="left")

    # Coordinates for the 3D view. Codex provides them as a single string
    # column "position" of the form "[352484 175164 229040]" (nanometers,
    # space-separated) rather than three separate x/y/z columns, so we parse
    # it manually.
    if "coordinates" in files:
        print("  reading coordinates...")
        coords = pl.read_csv(files["coordinates"], infer_schema_length=10_000)
        k_root = resolve(coords.columns, "root_id")
        xyz = {c: resolve(coords.columns, c) for c in ("x", "y", "z")}

        if k_root and all(xyz.values()):
            sel = [pl.col(k_root).cast(pl.UInt64).alias("root_id")] + [
                pl.col(src).cast(pl.Float32, strict=False).alias(dst)
                for dst, src in xyz.items()
            ]
            joined = coords.select(sel).unique(subset="root_id")
        elif k_root and "position" in coords.columns:
            # Codex pads the numbers inside the brackets with variable-width
            # spaces ("[631424 216180  55880]"), so split_exact on a single
            # space does not work: split first, then drop the empty pieces
            # produced by double spaces.
            parsed = coords.select([
                pl.col(k_root).cast(pl.UInt64).alias("root_id"),
                pl.col("position").str.strip_chars("[]").alias("_pos"),
            ]).with_columns(
                pl.col("_pos").str.split(" ").list.eval(
                    pl.element().filter(pl.element() != "")
                ).alias("_parts")
            )
            joined = parsed.select([
                pl.col("root_id"),
                pl.col("_parts").list.get(0).cast(pl.Float32, strict=False).alias("x"),
                pl.col("_parts").list.get(1).cast(pl.Float32, strict=False).alias("y"),
                pl.col("_parts").list.get(2).cast(pl.Float32, strict=False).alias("z"),
            ]).unique(subset="root_id")
        else:
            joined = None

        if joined is not None:
            meta = meta.join(joined, on="root_id", how="left")

    write_vocab(meta, cfg.path("runs") / "connectome_vocab.txt")

    # ---------------- edges ----------------
    print("  reading connections (the heaviest file)...")
    cols = scan_columns(files["connections"])
    c_pre, c_post = resolve(cols, "pre"), resolve(cols, "post")
    c_syn = resolve(cols, "syn_count")
    c_np, c_nt = resolve(cols, "neuropil"), resolve(cols, "nt_type")
    if not (c_pre and c_post and c_syn):
        raise ValueError(f"pre/post/syn_count not found in connections; available: {cols}")

    want = [c_pre, c_post, c_syn] + [c for c in (c_np, c_nt) if c]
    edges = (
        pl.scan_csv(files["connections"], infer_schema_length=10_000)
        .select(want)
        .filter(pl.col(c_syn).cast(pl.Int64) >= cfg.connectome.min_syn_count)
        .collect()
    )
    rename = {c_pre: "pre", c_post: "post", c_syn: "syn_count"}
    if c_np:
        rename[c_np] = "neuropil"
    if c_nt:
        rename[c_nt] = "nt_type"
    edges = edges.rename(rename)
    print(f"  edges after syn_count >= {cfg.connectome.min_syn_count} threshold: {edges.height:,}")

    # ---------------- indexing ----------------
    used = pl.concat([
        edges.select(pl.col("pre").cast(pl.UInt64).alias("root_id")),
        edges.select(pl.col("post").cast(pl.UInt64).alias("root_id")),
    ]).unique().sort("root_id")

    meta = used.join(meta, on="root_id", how="left")
    node_ids = meta["root_id"].to_numpy().astype(np.uint64)
    index = {int(v): i for i, v in enumerate(node_ids)}

    pre_idx = np.fromiter((index[int(v)] for v in edges["pre"]), dtype=np.int32, count=edges.height)
    post_idx = np.fromiter((index[int(v)] for v in edges["post"]), dtype=np.int32, count=edges.height)
    edge_index = np.vstack([pre_idx, post_idx])
    syn = edges["syn_count"].to_numpy().astype(np.float32)

    # edge sign = sign of the PREsynaptic neuron's neurotransmitter (Dale's law)
    sign = np.ones(edges.height, dtype=np.float32)
    if "nt_type" in edges.columns:
        smap = _nt_sign_map(cfg)
        nt = edges["nt_type"].cast(pl.Utf8).fill_null("unknown").to_list()
        sign = np.fromiter(
            (smap.get(str(v).strip().lower(), 1.0) for v in nt),
            dtype=np.float32, count=len(nt),
        )

    conn = Connectome(node_ids, edge_index, syn, sign, meta)
    conn.save(cache)
    print(f"  cache written: {cache.name}")

    if not cfg.housekeeping.keep_raw_connectome:
        print("  (raw CSVs can be deleted: set keep_raw_connectome=false in the config)")

    return conn


def summary(conn: Connectome) -> str:
    lines = [
        "",
        f"  neurons:   {conn.n_nodes:,}",
        f"  edges:     {conn.n_edges:,}",
        f"  synapses:  {int(conn.syn_count.sum()):,}",
        f"  excitatory edges: {int((conn.sign > 0).sum()):,}",
        f"  inhibitory edges: {int((conn.sign < 0).sum()):,}",
        f"  modulatory (sign 0): {int((conn.sign == 0).sum()):,}",
    ]
    if "x" in conn.meta.columns:
        have = conn.meta.select(pl.col("x").is_not_null().sum()).item()
        lines.append(f"  with coordinates: {have:,} of {conn.n_nodes:,}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    c = load_connectome()
    print(summary(c))
