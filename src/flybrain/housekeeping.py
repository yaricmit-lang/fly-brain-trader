"""
Disk housekeeping.

Activation dumps and checkpoints grow quietly. This module reports what
takes how much space and prunes old runs, skipping runs marked with a KEEP
file inside the run folder.

    python -m flybrain.housekeeping --report
    python -m flybrain.housekeeping --clean
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .config import load_config, project_root

GB = 1024 ** 3


def dir_size(path: Path) -> int:
    """Folder size in bytes. Symlinks are skipped to avoid double counting."""
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def free_space_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / GB


def report(cfg=None) -> None:
    """Print a table of sizes and the remaining free disk space."""
    cfg = cfg or load_config()
    root = project_root()

    rows: list[tuple[str, int]] = []
    for name in (".venv", "data.nosync", "runs.nosync", "web/node_modules", "src", "config"):
        rows.append((name, dir_size(root / name)))

    width = max(len(r[0]) for r in rows)
    print("\n  Project disk usage")
    print("  " + "-" * (width + 14))
    for name, size in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<{width}}  {human(size):>10}")
    print("  " + "-" * (width + 14))
    print(f"  {'total':<{width}}  {human(sum(r[1] for r in rows)):>10}")

    free = free_space_gb(root)
    budget = cfg.housekeeping.runs_budget_gb
    runs = dir_size(root / cfg.paths.runs) / GB
    print(f"\n  Free disk space: {free:.1f} GB")
    print(f"  Runs: {runs:.2f} GB of {budget:.1f} GB budget", end="")
    print("  <- time to clean up" if runs > budget else "")
    if free < 5:
        print("  WARNING: less than 5 GB free, training may fail while writing a checkpoint")
    print()


def clean(cfg=None, dry: bool = False) -> None:
    """
    Delete the oldest runs until the total fits the budget.
    Runs containing a KEEP file are never touched.
    """
    cfg = cfg or load_config()
    runs_dir = project_root() / cfg.paths.runs
    if not runs_dir.exists():
        print("  no runs folder yet, nothing to clean")
        return

    budget_bytes = int(cfg.housekeeping.runs_budget_gb * GB)
    runs = [d for d in runs_dir.iterdir() if d.is_dir()]
    runs.sort(key=lambda d: d.stat().st_mtime)  # oldest first

    total = dir_size(runs_dir)
    freed = 0
    for d in runs:
        if total - freed <= budget_bytes:
            break
        if (d / "KEEP").exists():
            print(f"  skipping {d.name} (marked KEEP)")
            continue
        size = dir_size(d)
        print(f"  {'would delete' if dry else 'deleting'} {d.name}  {human(size)}")
        if not dry:
            shutil.rmtree(d)
        freed += size

    if freed == 0:
        print(f"  all good: {human(total)} within budget {human(budget_bytes)}")
    else:
        print(f"  freed {human(freed)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Disk usage report and cleanup of old runs")
    ap.add_argument("--report", action="store_true", help="show what takes how much space")
    ap.add_argument("--clean", action="store_true", help="delete old runs beyond the budget")
    ap.add_argument("--dry", action="store_true", help="only show what would be deleted")
    args = ap.parse_args()

    if args.clean:
        clean(dry=args.dry)
    else:
        report()


if __name__ == "__main__":
    main()
