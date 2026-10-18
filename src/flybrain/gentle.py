"""
Gentle mode for fanless laptops (e.g. MacBook Air).

Two back-to-back training runs at 100% of a single core once overheated a
fanless machine into an emergency reboot. This module lowers process
priority, caps torch threads, pauses between epochs and checks whether
macOS is thermally throttling the CPU.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

from .config import Config


def apply(cfg: Config) -> None:
    """Call once at the start of a heavy script."""
    g = cfg.training.gentle
    if not g.enabled:
        return
    try:
        os.nice(19)
    except OSError:
        pass
    # Not only torch: the teacher (sklearn HistGradientBoosting) parallelises
    # via OpenMP and grabs ALL cores by default, which is exactly the
    # scenario that caused the overheating reboot on a fanless machine.
    n = str(max(1, g.torch_threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, n)
    try:
        import torch  # noqa: PLC0415
        torch.set_num_threads(max(1, g.torch_threads))
    except Exception:  # noqa: BLE001
        pass
    print(f"  gentle mode: nice 19, torch threads {g.torch_threads}, "
          f"{g.pause_between_epochs_sec} s pause between epochs")


def cpu_speed_limit() -> int | None:
    """
    macOS: `pmset -g therm` prints CPU_Speed_Limit = 100 when healthy and
    less than 100 under thermal throttling. Returns None on other OSes.
    """
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    return int(m.group(1)) if m else None


def pause(cfg: Config, label: str = "") -> None:
    """Pause between epochs; if the CPU is throttling, wait until it recovers."""
    g = cfg.training.gentle
    if not g.enabled:
        return
    secs = g.pause_between_epochs_sec
    if secs > 0:
        print(f"  pausing {secs} s to cool down{(' (' + label + ')') if label else ''}")
        time.sleep(secs)
    if g.check_thermal:
        waited = 0
        while True:
            lim = cpu_speed_limit()
            if lim is None or lim >= 100:
                break
            print(f"  CPU is throttling (limit {lim}%), waiting another 60 s")
            time.sleep(60)
            waited += 60
            if waited >= 600:
                print("  10 minutes of continuous throttling; continuing, but stopping is advisable")
                break
