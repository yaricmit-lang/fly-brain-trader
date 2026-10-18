"""
Check the connectome files and print manual download instructions.

Codex requires a Google login, so the files cannot be fetched automatically.
This module reports what is missing and shows the exact steps.

    python -m flybrain.connectome.download
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import load_config


@dataclass(frozen=True)
class Expected:
    """An expected file: base name, why it is needed, whether it is required."""
    stem: str
    why: str
    required: bool = True
    # accepted alternative names (Codex renamed files between releases)
    aliases: tuple[str, ...] = ()


EXPECTED: tuple[Expected, ...] = (
    Expected(
        "connections",
        "graph edges: pre -> post, synapse count, neuropil, neurotransmitter",
        # Codex calls the filtered version "Connections (Filtered)"; the file
        # on disk is connections_princeton.csv.gz. We do not use the
        # connections_princeton_no_threshold version (Unfiltered, 277 MB).
        aliases=("connections_princeton", "connections_no_threshold"),
    ),
    Expected(
        "classification",
        "hierarchy: super_class, class (Kenyon_Cell/MBON/DAN/ALPN), side",
    ),
    Expected(
        "cell_types",
        "detailed cell type: KCab, MBON01, PAM01, PPL101, DA1_lPN, etc. "
        "-- mainly used to carve out the mushroom body",
        aliases=("consolidated_cell_types",),
    ),
    Expected(
        "neurons",
        "predicted neurotransmitter per neuron (ACH/GLUT/GABA/DA/OCT/SER), "
        "for reference only -- the edge sign is taken directly from connections",
        required=False,
    ),
    Expected(
        "coordinates",
        "soma coordinates for the 3D view. Codex format: a position column "
        "of the form '[x y z]' in nanometers, parsed separately from plain x/y/z",
        required=False,
        aliases=("neuron_coordinates", "cell_coordinates"),
    ),
    Expected(
        "cell_stats",
        "neuron length/area, optional",
        required=False,
    ),
)

INSTRUCTIONS = """
  How to download the connectome (one-time, manual)

  1. Open https://codex.flywire.ai and sign in with Google.
     Registration is free; the data is licensed under CC-BY.

  2. Go to Downloads and choose the FAFB (brain) dataset, version {version}.

  3. Download these tables (csv or csv.gz, either works):
{files}

  4. Put the files into this folder:
     {folder}

  5. Run the check again:
     python -m flybrain.connectome.download
"""


def find_file(folder: Path, exp: Expected) -> Path | None:
    """Find a file by its primary name or aliases, with any supported extension."""
    names = (exp.stem, *exp.aliases)
    for name in names:
        for ext in (".csv", ".csv.gz", ".feather", ".parquet"):
            p = folder / f"{name}{ext}"
            if p.exists():
                return p
    # fallback: anything that merely starts with the expected name
    for name in names:
        hits = sorted(folder.glob(f"{name}*"))
        if hits:
            return hits[0]
    return None


def ungzip_if_needed(path: Path) -> Path:
    """
    Decompress a .csv.gz next to the original.
    The raw gz can be deleted afterwards; the loader works with the csv.
    """
    if path.suffix != ".gz":
        return path
    out = path.with_suffix("")
    if out.exists() and out.stat().st_size > 0:
        return out
    print(f"  decompressing {path.name} ...")
    with gzip.open(path, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return out


def sha1_head(path: Path, nbytes: int = 1 << 20) -> str:
    """Hash of the first megabyte: a quick fingerprint to detect a swapped file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:12]


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(x) < 1024 or unit == "GB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} GB"


def check(verbose: bool = True) -> dict[str, Path]:
    """
    Return a {stem: path} dict for the files found.
    If a required file is missing, print the instructions and raise.
    """
    cfg = load_config()
    folder = cfg.path("connectome")

    found: dict[str, Path] = {}
    missing: list[Expected] = []

    if verbose:
        print(f"\n  Folder: {folder}\n")
        print(f"  {'table':<16} {'status':<10} {'size':>10}  purpose")
        print("  " + "-" * 78)

    for exp in EXPECTED:
        p = find_file(folder, exp)
        if p is None:
            missing.append(exp)
            status = "MISSING" if exp.required else "missing (opt.)"
            if verbose:
                print(f"  {exp.stem:<16} {status:<10} {'—':>10}  {exp.why}")
            continue
        p = ungzip_if_needed(p)
        found[exp.stem] = p
        if verbose:
            print(f"  {exp.stem:<16} {'found':<10} {human(p.stat().st_size):>10}  {exp.why}")

    required_missing = [e for e in missing if e.required]
    if verbose:
        print()

    if required_missing:
        files = "\n".join(
            f"       - {e.stem}.csv   ({e.why})"
            for e in EXPECTED
            if e.required or e.stem == "coordinates"
        )
        print(INSTRUCTIONS.format(version=cfg.connectome.version, files=files, folder=folder))
        raise FileNotFoundError(
            "missing connectome tables: "
            + ", ".join(e.stem for e in required_missing)
        )

    if verbose:
        print("  All files present. Next step: python -m flybrain.connectome.loader\n")
    return found


if __name__ == "__main__":
    try:
        check()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
