"""
What to compare the fly against.

Without these baselines any return figure is meaningless.

buy_and_hold  - buy and hold. Hard to beat in a rising market.
sma_crossover - classic moving-average crossover. A check that the fly does
                something more sophisticated than a textbook rule.
random_agents - a population of random agents (100 by default). The key
                test: if the fly lands in the middle of their distribution
                it has learned nothing, and a nice curve is just luck.
vol_target    - volatility targeting, the main reference of methodology v2.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..trading.portfolio import Portfolio
from . import metrics as M


def _run(cfg: Config, prices: np.ndarray, targets: np.ndarray, ts=None) -> tuple[M.Metrics, list[float]]:
    """
    Shared benchmark runner.

    targets - (T, n_coins), desired weights for every hour.

    Important: decisions are made at the same cadence as the fly's
    (decision_timeframe, every 4 hours by default) while equity is marked
    every hour. Otherwise the benchmark would trade four times as often and
    pay different fees, so the fly-vs-formula comparison would reflect a
    difference in trading frequency rather than decision quality.

    Fills use the next candle's price to avoid look-ahead.
    """
    coins = [s.split("/")[0] for s in cfg.market.symbols]
    pf = Portfolio(
        coins=coins,
        initial_capital=cfg.trading.initial_capital,
        fee_rate=cfg.trading.fee_rate,
        slippage=cfg.trading.slippage,
        allow_short=cfg.trading.allow_short,
    )
    min_change = cfg.trading.guards.min_position_change
    stride = cfg.decision_stride
    turnover = 0.0
    T = len(prices)
    last = T - 1

    for t in range(0, T - 1, stride):
        exec_idx = t + 1
        p_exec = prices[exec_idx]
        before = pf.weights(p_exec)
        pf.rebalance(targets[t], p_exec, ts=None if ts is None else ts[exec_idx],
                     min_change=min_change)
        turnover += float(np.abs(pf.weights(p_exec) - before).sum())
        # mark equity every hour until the next decision
        for k in range(exec_idx, min(t + stride, last) + 1):
            pf.mark(prices[k])

    s = pf.summary()
    return M.compute(pf.equity_curve, s["n_trades"], turnover, s["fees_paid"]), pf.equity_curve


def buy_and_hold(cfg: Config, prices: np.ndarray, ts=None):
    """Equal weights across all coins, held to the end."""
    n = prices.shape[1]
    targets = np.full((len(prices), n), 1.0 / n)
    return _run(cfg, prices, targets, ts)


def vol_target(cfg: Config, prices: np.ndarray, ts=None):
    """
    Volatility targeting: the weight is inversely proportional to volatility.

    The main reference of methodology v2. Price direction is unpredictable
    while volatility is predictable (IC +0.61), so this formula captures the
    only thing the data actually contains. The fly must beat it, otherwise it
    has learned nothing beyond arithmetic.
    """
    from ..trading.baseline import baseline_positions  # noqa: PLC0415

    return _run(cfg, prices, baseline_positions(cfg, prices), ts)


def sma_crossover(cfg: Config, prices: np.ndarray, fast: int = 24, slow: int = 168, ts=None):
    """Hold the coin while the fast average is above the slow one; otherwise stay in cash."""
    T, n = prices.shape
    targets = np.zeros((T, n))
    for i in range(n):
        p = prices[:, i]
        f = np.convolve(p, np.ones(fast) / fast, mode="full")[:T]
        s = np.convolve(p, np.ones(slow) / slow, mode="full")[:T]
        f[:fast], s[:slow] = np.nan, np.nan
        targets[:, i] = np.where(np.nan_to_num(f) > np.nan_to_num(s), 1.0 / n, 0.0)
    return _run(cfg, prices, targets, ts)


def random_agents(cfg: Config, prices: np.ndarray, n_agents: int = 100, seed: int = 0):
    """
    Random-agent baseline (100 agents by default). Returns their metrics,
    the distribution the fly is fairly compared against.

    Positions change rarely (once a day); otherwise fees would kill every
    agent and the comparison would be unfairly easy.
    """
    rng = np.random.default_rng(seed)
    T, n = prices.shape
    out: list[M.Metrics] = []
    hold = 24

    for _ in range(n_agents):
        if cfg.trading.allow_short:
            # same action space as the fly: weights in [-1, 1] with
            # sum(|w|) <= 1; otherwise the comparison is unfair
            steps = rng.uniform(-1.0, 1.0, (T // hold + 1, n))
            steps = steps / np.maximum(np.abs(steps).sum(axis=1, keepdims=True), 1.0)
        else:
            steps = rng.random((T // hold + 1, n))
            steps = steps / np.maximum(steps.sum(axis=1, keepdims=True), 1.0)
        targets = np.repeat(steps, hold, axis=0)[:T]
        m, _ = _run(cfg, prices, targets)
        out.append(m)
    return out


def describe_random(results: list[M.Metrics]) -> str:
    """Random-agent distribution: median and tails."""
    rets = np.array([m.total_return for m in results])
    sharpes = np.array([m.sharpe for m in results])
    q = np.percentile(rets, [5, 25, 50, 75, 95]) * 100
    return (
        f"\n  Random agents ({len(results)}), total return:\n"
        f"    5%: {q[0]:.1f}%   25%: {q[1]:.1f}%   median: {q[2]:.1f}%   "
        f"75%: {q[3]:.1f}%   95%: {q[4]:.1f}%\n"
        f"    median Sharpe: {np.median(sharpes):.2f}\n"
    )


def percentile_of(value: float, results: list[M.Metrics], key: str = "total_return") -> float:
    """Percentile of the random agents that the fly beats. Below 50 means it has not learned."""
    vals = np.array([getattr(m, key) for m in results])
    return float((vals < value).mean() * 100)
