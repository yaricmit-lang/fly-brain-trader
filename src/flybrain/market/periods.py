"""
Selection of non-overlapping calendar blocks by volatility (cycle 6).

The idea: instead of a general walk-forward ("on average"), train on specific
historical market REGIMES: calm, turbulent and medium. The last
lookback_months months are tiled with non-overlapping blocks of
block_months; for each block we compute the mean realized volatility (across
all coins at once, so "calm" means calm for every coin), then simply sort and
take the extremes and the middle.

Blocks do NOT overlap by construction (a calendar tiling of one range), so
no extra deduplication is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PeriodBlock:
    label: str            # "calm" | "active" | "medium"
    start: str             # ISO date, for the report/dashboard
    end: str
    idx: np.ndarray         # row indices into dt/X/prices
    mean_vol: float         # mean realized volatility of the block


def _block_volatility(prices_block: np.ndarray) -> float:
    """Mean (across coins) realized volatility of hourly log returns inside
    the block: one number per block, used for ranking."""
    r = np.log(prices_block[1:] / np.maximum(prices_block[:-1], 1e-12))
    return float(np.nanstd(r, axis=0).mean())


def select_seasonal_blocks(
    dt: pd.DatetimeIndex, prices: np.ndarray,
    lookback_months: int = 24, block_months: int = 2,
    n_calm: int = 2, n_active: int = 2, n_medium: int = 2,
) -> list[PeriodBlock]:
    """
    Tile the last lookback_months months with non-overlapping blocks of
    block_months (12 blocks over 24 months by default), rank them by mean
    volatility and return the n_calm quietest + n_active most turbulent +
    n_medium closest to the median among the remainder.
    """
    end = dt[-1] + pd.Timedelta(hours=1)
    start = end - pd.DateOffset(months=lookback_months)

    candidates: list[tuple[str, str, np.ndarray, float]] = []
    t0 = start
    idx_all = np.arange(len(dt))
    while t0 + pd.DateOffset(months=block_months) <= end:
        t1 = t0 + pd.DateOffset(months=block_months)
        idx = idx_all[(dt >= t0) & (dt < t1)]
        if len(idx) > 200:   # the block must be substantial, not a stub
            vol = _block_volatility(prices[idx])
            candidates.append((f"{t0:%Y-%m-%d}", f"{t1:%Y-%m-%d}", idx, vol))
        t0 = t1

    n_need = n_calm + n_active + n_medium
    if len(candidates) < n_need:
        raise ValueError(
            f"only {len(candidates)} blocks of {block_months} months fit into "
            f"{lookback_months} months of history, but {n_need} are needed; "
            f"increase lookback_months or decrease block_months"
        )

    ranked = sorted(candidates, key=lambda c: c[3])   # ascending volatility
    out: list[PeriodBlock] = []
    for s, e, idx, vol in ranked[:n_calm]:
        out.append(PeriodBlock("calm", s, e, idx, vol))
    for s, e, idx, vol in ranked[-n_active:]:
        out.append(PeriodBlock("active", s, e, idx, vol))

    remaining = ranked[n_calm:len(ranked) - n_active]
    median_vol = float(np.median([c[3] for c in ranked]))
    remaining.sort(key=lambda c: abs(c[3] - median_vol))
    for s, e, idx, vol in remaining[:n_medium]:
        out.append(PeriodBlock("medium", s, e, idx, vol))

    out.sort(key=lambda b: b.start)
    return out
