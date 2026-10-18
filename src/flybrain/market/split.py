"""
Time-based splitting.

No shuffling: train on the earlier period, validate on the later one.
Shuffle and you get nice numbers that mean nothing, because the network
would see the future during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config, load_config


@dataclass(frozen=True)
class Split:
    """Row indices for each period."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def describe(self, dt: pd.DatetimeIndex) -> str:
        lines = ["", f"  {'period':<8} {'candles':>8}  dates", "  " + "-" * 44]
        for name in ("train", "val", "test"):
            idx = getattr(self, name)
            if len(idx) == 0:
                lines.append(f"  {name:<8} {0:>8}  empty")
                continue
            lines.append(
                f"  {name:<8} {len(idx):>8,}  "
                f"{dt[idx[0]]:%Y-%m-%d} .. {dt[idx[-1]]:%Y-%m-%d}"
            )
        return "\n".join(lines) + "\n"


def make_split(dt: pd.DatetimeIndex, cfg: Config | None = None) -> Split:
    """Split by the dates in the config."""
    cfg = cfg or load_config()
    train_end = pd.Timestamp(cfg.market.split.train_end, tz="UTC")
    val_end = pd.Timestamp(cfg.market.split.val_end, tz="UTC")

    idx = np.arange(len(dt))
    train = idx[dt < train_end]
    val = idx[(dt >= train_end) & (dt < val_end)]
    test = idx[dt >= val_end]

    for name, part in (("train", train), ("val", val), ("test", test)):
        if len(part) == 0:
            print(
                f"  warning: period {name} is empty. "
                f"Check market.split and market.since in the config."
            )
    return Split(train, val, test)


def make_recent_split(dt: pd.DatetimeIndex, train_months: int, val_months: int) -> Split:
    """
    Split anchored at the end of the history rather than at fixed config
    dates: train is the train_months months BEFORE val, val is the last
    val_months months UP TO THE VERY END of the data. test is always empty:
    there is no future yet, and this is exactly the point from which the fly
    will trade for real.

    Needed for the training cycles run before going live: the fly must know
    the latest market regime, not only the 2020-2024 one the first
    checkpoint was trained on.
    """
    val_end = dt[-1] + pd.Timedelta(hours=1)     # include the last candle
    train_end = val_end - pd.DateOffset(months=val_months)
    train_start = train_end - pd.DateOffset(months=train_months)

    idx = np.arange(len(dt))
    train = idx[(dt >= train_start) & (dt < train_end)]
    val = idx[(dt >= train_end) & (dt < val_end)]
    test = idx[0:0]

    for name, part in (("train", train), ("val", val)):
        if len(part) == 0:
            print(f"  warning: period {name} is empty; check train_months/val_months")
    return Split(train, val, test)


def make_anchored_split(
    dt: pd.DatetimeIndex, train_months: int, val_start: str, val_end: str,
) -> Split:
    """
    Split with val anchored to SPECIFIC dates, not to the end of the history
    (`make_recent_split`) nor to the config dates (`make_split`). Needed for
    targeted cycles such as "triple the capital over August 2025 - August
    2026": val must be exactly that year, not "the last two months".

    train is the train_months months BEFORE val_start. test is always empty.
    """
    val_start_ts = pd.Timestamp(val_start, tz="UTC")
    val_end_ts = pd.Timestamp(val_end, tz="UTC")
    train_start_ts = val_start_ts - pd.DateOffset(months=train_months)

    idx = np.arange(len(dt))
    train = idx[(dt >= train_start_ts) & (dt < val_start_ts)]
    val = idx[(dt >= val_start_ts) & (dt < val_end_ts)]
    test = idx[0:0]

    for name, part in (("train", train), ("val", val)):
        if len(part) == 0:
            print(f"  warning: period {name} is empty; check val_start/val_end/train_months")
    return Split(train, val, test)


def walk_forward_windows(
    dt: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward windows: train on train_days, trade test_days, then roll
    forward. More honest than a single split: it shows whether the model
    survives regime changes.
    """
    step_days = step_days or test_days
    out: list[tuple[np.ndarray, np.ndarray]] = []
    idx = np.arange(len(dt))
    start = dt[0]

    while True:
        tr_end = start + pd.Timedelta(days=train_days)
        te_end = tr_end + pd.Timedelta(days=test_days)
        if te_end > dt[-1]:
            break
        tr = idx[(dt >= start) & (dt < tr_end)]
        te = idx[(dt >= tr_end) & (dt < te_end)]
        if len(tr) and len(te):
            out.append((tr, te))
        start = start + pd.Timedelta(days=step_days)
    return out
