"""
Run report: markdown plus plots.

Each run goes into its own folder under runs.nosync with a timestamp in the
name. To protect a run from cleanup, put an empty file named KEEP inside.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import Config
from . import metrics as M
from .benchmarks import describe_random, percentile_of


def make_run_dir(cfg: Config, tag: str) -> Path:
    """Run folder: runs.nosync/2026-09-14_1712_<tag>/"""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    d = cfg.path("runs") / f"{stamp}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_equity(curves: dict[str, list[float]], out: Path, title: str) -> None:
    """Equity curves and drawdown in one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for name, eq in curves.items():
        e = np.asarray(eq, dtype=float)
        if len(e) < 2:
            continue
        norm = e / e[0]
        lw = 2.2 if "fly" in name.lower() else 1.3
        ax1.plot(norm, label=name, linewidth=lw)
        peak = np.maximum.accumulate(norm)
        ax2.fill_between(range(len(norm)), (norm - peak) / peak * 100, 0, alpha=0.25)

    ax1.set_title(title)
    ax1.set_ylabel("equity, multiple of start")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.axhline(1.0, color="black", linewidth=0.8, linestyle=":")

    ax2.set_ylabel("drawdown, %")
    ax2.set_xlabel("candles")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_random_distribution(
    random_results: list[M.Metrics], fly_return: float | None, out: Path
) -> None:
    """Histogram of random agents and where the fly falls on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rets = np.array([m.total_return for m in random_results]) * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(rets, bins=30, alpha=0.75, edgecolor="white")
    ax.axvline(np.median(rets), linestyle="--", linewidth=1.5,
               label=f"random median: {np.median(rets):.1f}%")
    if fly_return is not None:
        ax.axvline(fly_return * 100, color="crimson", linewidth=2.5,
                   label=f"fly: {fly_return * 100:.1f}%")
    ax.set_xlabel("total return, %")
    ax.set_ylabel("number of agents")
    ax.set_title("Random agents vs. the fly")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def write_report(
    run_dir: Path,
    cfg: Config,
    period_name: str,
    results: dict[str, M.Metrics],
    curves: dict[str, list[float]],
    random_results: list[M.Metrics] | None = None,
    fly_name: str | None = None,
    extra: str = "",
) -> Path:
    """Assemble the markdown report and plots."""
    lines: list[str] = [
        f"# Backtest: {period_name}",
        "",
        f"Run date: {datetime.now():%Y-%m-%d %H:%M}",
        f"Coins: {', '.join(cfg.market.symbols)}, timeframe {cfg.market.timeframe}",
        f"Capital: {cfg.trading.initial_capital:,.0f} USDT, "
        f"fee {cfg.trading.fee_rate * 100:.3f}%, "
        f"slippage {cfg.trading.slippage * 100:.3f}%",
        "",
        "## Results",
        "",
        "| strategy | return | CAGR | Sharpe | Sortino | max drawdown | trades | fees |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m.total_return * 100:.1f}% | {m.cagr * 100:.1f}% | "
            f"{m.sharpe:.2f} | {m.sortino:.2f} | {m.max_drawdown * 100:.1f}% | "
            f"{m.n_trades:,} | {m.fees_paid:,.0f} |"
        )

    if random_results:
        lines += ["", "## Random-agent check", "", "```",
                  describe_random(random_results).strip(), "```"]
        if fly_name and fly_name in results:
            pct = percentile_of(results[fly_name].total_return, random_results)
            verdict = (
                "the fly is clearly better than chance" if pct >= 95 else
                "the fly beats most random agents, but not convincingly" if pct >= 75 else
                "the fly is indistinguishable from a random agent" if pct >= 25 else
                "the fly is WORSE than a random agent"
            )
            lines += ["", f"**The fly ranks at percentile {pct:.0f} of random agents: {verdict}.**"]

    if extra:
        lines += ["", "## Notes", "", extra]

    lines += ["", "## Plots", "", "![Equity curves](equity.png)"]
    if random_results:
        lines.append("![Random agents](random.png)")

    # output files
    plot_equity(curves, run_dir / "equity.png", f"Equity, period {period_name}")
    if random_results:
        fly_ret = results[fly_name].total_return if fly_name and fly_name in results else None
        plot_random_distribution(random_results, fly_ret, run_dir / "random.png")

    (run_dir / "metrics.json").write_text(
        json.dumps({k: v.as_dict() for k, v in results.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
