"""
Run metrics.

Return alone is not enough: a strategy that made 200% with an 80% drawdown
along the way is worse than one that made a steady 60%. So we also measure
risk, drawdown and turnover.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# hourly candles per year: 24 * 365
BARS_PER_YEAR = 24 * 365


@dataclass
class Metrics:
    total_return: float      # total return, as a fraction
    cagr: float              # compound annual growth rate
    sharpe: float            # return per unit of risk (annualised)
    sortino: float           # same, but only downside deviation counts as risk
    max_drawdown: float      # maximum peak-to-trough drawdown, as a fraction
    volatility: float        # annualised volatility
    n_trades: int
    turnover: float          # total turnover, in units of capital
    fees_paid: float
    final_equity: float

    def as_dict(self) -> dict:
        return asdict(self)

    def row(self, name: str) -> str:
        return (
            f"  {name:<22} {self.total_return * 100:>9.1f}% "
            f"{self.cagr * 100:>8.1f}% {self.sharpe:>7.2f} "
            f"{self.max_drawdown * 100:>8.1f}% {self.n_trades:>8,}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"  {'strategy':<22} {'return':>10} {'CAGR':>9} "
            f"{'Sharpe':>7} {'max DD':>9} {'trades':>9}\n"
            "  " + "-" * 70
        )


def max_drawdown(equity: np.ndarray) -> float:
    """Worst decline from the running peak."""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(-dd.min())


def compute(
    equity: list[float] | np.ndarray,
    n_trades: int = 0,
    turnover: float = 0.0,
    fees_paid: float = 0.0,
    bars_per_year: int = BARS_PER_YEAR,
) -> Metrics:
    """Compute all metrics from the equity curve."""
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2 or eq[0] <= 0:
        return Metrics(0, 0, 0, 0, 0, 0, n_trades, turnover, fees_paid,
                       float(eq[-1]) if len(eq) else 0.0)

    rets = np.diff(np.log(np.maximum(eq, 1e-9)))
    total = float(eq[-1] / eq[0] - 1)
    years = len(eq) / bars_per_year
    cagr = float((eq[-1] / eq[0]) ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else 0.0

    vol = float(rets.std() * np.sqrt(bars_per_year))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year)) if rets.std() > 0 else 0.0

    downside = rets[rets < 0]
    sortino = (
        float(rets.mean() / downside.std() * np.sqrt(bars_per_year))
        if len(downside) > 1 and downside.std() > 0 else 0.0
    )

    return Metrics(
        total_return=total,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown(eq),
        volatility=vol,
        n_trades=n_trades,
        turnover=turnover,
        fees_paid=fees_paid,
        final_equity=float(eq[-1]),
    )
