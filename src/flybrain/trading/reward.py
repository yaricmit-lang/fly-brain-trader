"""
Reward: the fly's "dopamine" signal.

We use return divided by recent volatility rather than raw dollar P&L.
Reason: in a calm period +1% is a success, in a storm it is noise. Without
normalisation the learning rule would react most strongly to the wildest
candles, i.e. learn from noise.

The reward's moving average is then subtracted from it; this is done by the
learning rule itself, see Plasticity._delta. Biologically this is dopamine:
it encodes the difference between received and expected reward, not the
reward itself.

Methodology v2 makes two fundamental changes:

1. Per-coin reward. Previously the whole portfolio produced one number, so
   the SOL MBONs got dopamine for BTC's move. Now each coin has its own
   reward: coin position x its forward return.

2. Horizon. The reward for a decision at time t is the return over the next
   H decision steps (default 3 x 4h = 12h), known only at t+H. The agent
   stores the trace taken at time t and applies the reward to it once it
   becomes known. A one-hour reward was almost pure noise.
"""

from __future__ import annotations

import numpy as np

from ..config import Config


class RewardTracker:
    """Portfolio-level single-step reward (kept for logs and legacy runs)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg.reward
        self.window = self.cfg.vol_window
        self.returns: list[float] = []

    def __call__(self, equity_prev: float, equity_now: float, turnover: float = 0.0) -> float:
        if equity_prev <= 0 or equity_now <= 0:
            return -5.0 if equity_now <= 0 < equity_prev else 0.0

        r = float(np.log(equity_now / equity_prev))
        self.returns.append(r)
        if len(self.returns) > self.window * 4:
            self.returns = self.returns[-self.window * 2:]

        vol = float(np.std(self.returns[-self.window:])) if len(self.returns) >= 3 else 0.0
        scaled = r / (vol + 1e-6) if vol > 0 else r * 100.0

        if self.cfg.turnover_penalty:
            scaled -= self.cfg.turnover_penalty * turnover

        return float(np.clip(scaled, -5.0, 5.0))

    def reset(self) -> None:
        self.returns.clear()


class CoinRewardTracker:
    """
    Per-coin reward with a horizon.

        r_c = pos_c * fwd_ret_c / sigma_c - trade cost

    pos_c      - signed capital fraction in the coin after the decision is executed
    fwd_ret_c  - log return of the coin from the execution price to the price H steps later
    sigma_c    - volatility KNOWN AT DECISION TIME (ex-ante), supplied by the
                 caller (agent.py: the same realized_vol as the baseline formula,
                 taken BEFORE the decision and stretched over the reward horizon).

    sigma_c must not be estimated from the forward returns themselves
    (including the current one): that would draw on the future window
    [t+1, t+1+H] of earlier decisions. The reward delta would then be
    rescaled by exactly the volatility the reward is supposed to teach the
    fly to predict (the only measured edge in the data, IC +0.61), making
    learning blind to volatility timing by construction. With sigma known
    before t, as in the baseline formula, the quadratic risk penalty
    (risk_aversion) actually penalises realized volatility exceeding the
    forecast while the position is large, rather than risk in general.
    """

    def __init__(self, cfg: Config, coins: list[str]) -> None:
        self.cfg = cfg.reward
        self.coins = coins
        # real trade cost (fee + slippage) rather than an arbitrary
        # turnover_penalty; see __call__
        self.trade_cost = cfg.trading.fee_rate + cfg.trading.slippage

    def _utility(self, a: float) -> float:
        """
        Utility of the vol-normalised return A = pos * r / vol.

        With risk_aversion = 0 this is just A (methodology v2). With
        lambda > 0 it is A - lambda*A^2: the same quadratic (mean-variance)
        utility that Sharpe comes from. The key difference: when price
        direction is unpredictable (E[r] = 0, our case), a linear reward is
        indifferent to how much risk is taken, since the expectation is zero
        either way. The quadratic penalty punishes a large position ahead of
        a volatility rise and rewards a small one, which is exactly what the
        fly should learn to predict.
        """
        lam = self.cfg.risk_aversion
        return a - lam * a * a if lam > 0 else a

    def __call__(
        self,
        positions: np.ndarray,
        fwd_returns: np.ndarray,
        sigma: np.ndarray,
        turnover: np.ndarray,
        base_positions: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        positions, fwd_returns, sigma, turnover are arrays of length n_coins.
        sigma is ex-ante volatility (known at decision time, see class docstring).
        base_positions is the baseline position (multiplier 1.0) for the same
        decision; with relative_to_baseline the reward is U(fly) - U(baseline).
        Returns {coin: reward}.
        """
        out: dict[str, float] = {}
        rel = self.cfg.relative_to_baseline and base_positions is not None
        for i, c in enumerate(self.coins):
            fr = float(fwd_returns[i])
            vol = float(sigma[i])
            # `vol > 0` also rejects NaN (comparison with NaN is always False):
            # on the first ~42 bars of a run realized_vol is not yet defined
            # (rolling with min_periods), so sigma is NaN there. The scale is
            # kept as ONE multiplier so that return and trade cost below are
            # expressed in the same units in both branches. Dividing the cost
            # by vol separately and without this check would turn the reward
            # into NaN at the start of a run, poison the moving dopamine
            # baseline in Plasticity._delta, and the rest of the run would
            # learn from garbage.
            inv_vol = 1.0 / vol if vol > 0 else 100.0
            z = fr * inv_vol
            scaled = self._utility(float(positions[i]) * z)
            if rel:
                # control variate: market noise shared with the baseline is
                # subtracted, so multiplier 1.0 gives exactly zero; dopamine
                # is paid only for the deviation
                scaled -= self._utility(float(base_positions[i]) * z)
            # dopamine protocol: asymmetry, gains are brighter than pain
            scaled *= self.cfg.gain_scale if scaled > 0 else self.cfg.loss_scale
            # ...plus dopamine for acting at all, i.e. deviating from the baseline
            if self.cfg.activity_bonus and base_positions is not None:
                scaled += self.cfg.activity_bonus * abs(float(positions[i]) - float(base_positions[i]))
            # legacy penalty (in reward units, configurable); kept for
            # backward compatibility, default 0.0
            if self.cfg.turnover_penalty:
                scaled -= self.cfg.turnover_penalty * float(turnover[i])
            # real trade cost (fee + slippage), converted to reward units by
            # the same inv_vol; applied unconditionally, not a toy penalty
            if self.trade_cost:
                scaled -= self.trade_cost * float(turnover[i]) * inv_vol
            # clip tails: a single wild outlier must not flip all the weights
            out[c] = float(np.clip(scaled, -5.0, 5.0))
        return out

    def reset(self) -> None:
        pass
