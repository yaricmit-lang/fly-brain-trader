"""
Common interface for learning rules.

The point: train.py and agent.py should not need to know whether the network
learns with a biologically inspired Hebbian rule or with an exact gradient.
They simply call observe() after every decision and learn() once the reward
arrives.

Delayed per-coin reward:

    token = rule.observe(activations, fractions)   # capture the trace at time t
    ...                                             # H steps pass
    rule.learn({"BTC": r_btc, ...}, token)          # dopamine for time t

The token is whatever the rule needs to remember about the moment of the
decision: for the Hebbian rule it is a snapshot of the eligibility trace, for
REINFORCE it is the action log-probabilities. The agent keeps the tokens in a
queue and hands them back once the reward is known.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch

from ..config import Config
from .model import FlyBrain


class Plasticity(ABC):
    """A synaptic weight update rule."""

    def __init__(self, brain: FlyBrain, cfg: Config) -> None:
        self.brain = brain
        self.cfg = cfg
        self.coins = list(brain.approach_idx.keys())
        # Separate baseline per coin: the "usual" reward differs between BTC
        # and SOL, and the surprise is computed relative to the coin's own.
        self.baseline: dict[str, float] = {c: 0.0 for c in self.coins}
        # Running per-coin reward variance. Without it delta has a different
        # scale in calm and stormy market regimes, and the same lr would mean
        # a learning step of different strength depending on when the cycle
        # started. We normalize delta by its own recent spread.
        self.baseline_var: dict[str, float] = {c: 1.0 for c in self.coins}
        self.n_updates = 0

    @property
    @abstractmethod
    def exploration_noise(self) -> float:
        """How much noise to add at the output during the forward pass."""

    def act(self, decision: Any) -> dict[str, float]:
        """
        The action actually executed in the portfolio for this decision.

        By default this is the network output as is (Hebbian rule: exploration
        is already done by the noise on the MBONs inside forward(), so no
        action sampling is needed). REINFORCE overrides it: it samples the
        action from a Beta distribution and that is the action the exchange
        must execute; otherwise the gradient would be computed for an action
        that was never traded (the earlier behavior).
        """
        return decision.fractions

    @abstractmethod
    def observe(self, activations: torch.Tensor, fractions: dict[str, torch.Tensor]) -> Any:
        """Record the moment of decision. Returns a token for learn()."""

    @abstractmethod
    def learn(self, rewards: dict[str, float], token: Any, reward_fn: Any = None,
              lr_scale: float | None = None) -> float:
        """
        Apply per-coin rewards to the token. Returns the magnitude of the update.

        reward_fn is an optional callback {coin: fraction} -> {coin: reward}
        for counterfactual credit assignment in the ensemble (see
        EnsemblePlasticity): each member learns from the reward for its OWN
        action rather than for the averaged one. Single rules
        (Hebbian/REINFORCE) ignore it, since `rewards` already holds the
        reward for the action that was actually executed.

        lr_scale overrides cfg.lr_stride_scale with a concrete number
        (bounded-adaptive decision cadence, live_engine.py): None keeps the
        old behavior and takes the value from the config.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the traces between epochs."""

    def state_dict(self) -> dict:
        """
        Running dopamine statistics are part of the accumulated live
        experience. Without them, after a restart the first few dozen live
        rewards would be computed from a zero baseline and baseline_var=1, so
        delta would be noisy for a while as if learning had just begun,
        although the weights already carry a day of experience.
        """
        return {"baseline": dict(self.baseline), "baseline_var": dict(self.baseline_var)}

    def load_state_dict(self, d: dict) -> None:
        self.baseline.update(d.get("baseline", {}))
        self.baseline_var.update(d.get("baseline_var", {}))

    def _delta(self, coin: str, reward: float, decay: float) -> float:
        """
        Per-coin reward prediction error: how much better than usual.
        This is what dopamine encodes in the brain: not the reward itself but
        the surprise.

        Normalized by the running std of the coin's reward: without it the
        same lr means a learning step of different strength depending on the
        market regime (calm/storm). With it delta has a std of roughly one,
        and lr becomes interpretable.
        """
        # Keep NaN out of the running statistics: a single NaN reward turns
        # the baseline into NaN forever, and then Python's
        # max(-3, min(3, nan)) silently returns 3.0, so every following step
        # would receive the MAXIMUM positive dopamine and the run would learn
        # from garbage, betraying itself only through a "mean reward nan"
        # line in the log.
        if not math.isfinite(reward):
            return 0.0
        mean = self.baseline[coin]
        d = reward - mean
        self.baseline[coin] = decay * mean + (1 - decay) * reward
        var = self.baseline_var[coin]
        self.baseline_var[coin] = decay * var + (1 - decay) * (reward - mean) ** 2
        dn = d / (var ** 0.5 + 1e-3)
        return float(max(-3.0, min(3.0, dn)))

    def _coin_edge_masks(self) -> dict[str, torch.Tensor]:
        """
        Per-coin edge masks: an edge belongs to a coin if its postsynaptic
        neuron is one of that coin's MBONs ("approach" or "avoid"). This way
        dopamine for Bitcoin changes only the synapses onto Bitcoin's MBONs.
        """
        post = self.brain.edge_index[1]
        masks: dict[str, torch.Tensor] = {}
        for c in self.coins:
            owned = torch.zeros(self.brain.n, dtype=torch.bool, device=self.brain.device)
            owned[self.brain.approach_idx[c]] = True
            owned[self.brain.avoid_idx[c]] = True
            masks[c] = (owned[post] & self.brain.plastic).to(torch.float32)
        return masks
