"""
Ensemble of brains.

N copies of the network with different seeds: each has its own random mapping
of inputs onto projection neurons, its own split of MBONs into
"approach"/"avoid" groups, and its own noise. The decision is the mean over
members. A single brain gives high variance across seeds; five cut it
substantially at a cost of 5 x 10 ms per candle.

From the outside the ensemble looks like a single FlyBrain to the agent
(reset/step) and like a single learning rule (observe/learn); the agent does
not know about the ensemble.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from .model import Decision, FlyBrain
from .plasticity import Plasticity


class EnsembleBrain:
    """Several FlyBrain instances behind the interface of one."""

    def __init__(self, members: list[FlyBrain]) -> None:
        assert members, "ensemble has no members"
        self.members = members
        self.cfg: Config = members[0].cfg
        self.n = members[0].n
        self.approach_idx = members[0].approach_idx   # for compatibility with Plasticity
        self.avoid_idx = members[0].avoid_idx
        self._last: list[tuple[Decision, torch.Tensor]] = []

    def reset(self) -> None:
        for m in self.members:
            m.reset()
        self._last = []

    def step(
        self,
        x: torch.Tensor,
        noise: float = 0.0,
        generator: torch.Generator | None = None,
        elapsed_hours: float | None = None,
    ) -> tuple[Decision, torch.Tensor]:
        self._last = [m.step(x, noise=noise, generator=generator, elapsed_hours=elapsed_hours)
                      for m in self.members]
        coins = list(self._last[0][0].fractions.keys())
        n = len(self.members)
        fr = {c: float(np.mean([d.fractions[c] for d, _ in self._last])) for c in coins}
        fr_t = {c: torch.stack([d.fractions_t[c] for d, _ in self._last]).mean() for c in coins}
        app = {c: float(np.mean([d.approach[c] for d, _ in self._last])) for c in coins}
        avo = {c: float(np.mean([d.avoid[c] for d, _ in self._last])) for c in coins}
        d0, a0 = self._last[0]
        decision = Decision(
            fractions=fr, activations=d0.activations, approach=app, avoid=avo, fractions_t=fr_t
        )
        return decision, a0

    def peek(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Activations for visualization, taken from the first member only
        (that is the one the dashboard shows); see FlyBrain.peek."""
        return self.members[0].peek(x, **kwargs)

    # --- serialization ---
    def state_dict(self) -> dict:
        return {"members": [m.state_dict() for m in self.members]}

    def load_state_dict(self, d: dict) -> None:
        saved = d["members"]
        if len(saved) != len(self.members):
            raise ValueError(
                f"checkpoint was saved with {len(saved)} ensemble members, "
                f"but {len(self.members)} were built; a silent zip() used to "
                f"quietly truncate the extras. Check training.ensemble.n_members."
            )
        for m, sd in zip(self.members, saved):
            m.load_state_dict(sd)

    def describe(self) -> str:
        return f"\n  ensemble of {len(self.members)} brains, seeds {[m.seed for m in self.members]}" \
               + self.members[0].describe()


class EnsemblePlasticity(Plasticity):
    """One learning rule per member; exposed to the outside as a single rule."""

    def __init__(self, ensemble: EnsembleBrain, rules: list[Plasticity]) -> None:
        self.ensemble = ensemble
        self.rules = rules
        self.brain = ensemble.members[0]
        self.cfg = ensemble.cfg
        self.coins = rules[0].coins
        self.baseline = rules[0].baseline
        self.baseline_var = rules[0].baseline_var
        self.n_updates = 0
        self._last_acts: list[dict[str, float]] | None = None

    @property
    def exploration_noise(self) -> float:
        return self.rules[0].exploration_noise

    def act(self, decision) -> dict[str, float]:
        # Average the EXECUTED action of each member (sampled for REINFORCE,
        # raw for Hebbian): not the mean readout but the mean of decisions
        # already taken. Per-member acts are stored because observe() uses
        # them as frac_vals for counterfactual credit in learn(): the reward
        # must be computed for the action the member ACTUALLY submitted (the
        # Beta sample for REINFORCE), not for the raw network output.
        acts = [rule.act(d) for rule, (d, _a) in zip(self.rules, self.ensemble._last)]
        self._last_acts = acts
        coins = acts[0].keys()
        return {c: float(np.mean([ac[c] for ac in acts])) for c in coins}

    def observe(self, activations: torch.Tensor, fractions: dict) -> list:
        # The agent passes averaged values; each rule needs ITS OWN
        # activations and ITS OWN fractions, taken from the last ensemble step.
        # The token also carries a snapshot of the member's own EXECUTED
        # fraction (frac_vals, from act(), not the raw d.fractions), needed for
        # counterfactual credit in learn(): each member is rewarded for what IT
        # decided, not for the averaged action of the whole ensemble.
        tokens = []
        for i, (rule, (d, a)) in enumerate(zip(self.rules, self.ensemble._last)):
            tok = rule.observe(a, d.fractions_t)
            frac_vals = self._last_acts[i] if self._last_acts is not None else dict(d.fractions)
            tokens.append((tok, frac_vals))
        return tokens

    def learn(self, rewards: dict[str, float], token: list, reward_fn=None,
              lr_scale: float | None = None) -> float:
        """
        reward_fn(frac_dict) -> {coin: reward} is the counterfactual reward for
        a hypothetical fraction of one specific member. Without it (or when
        agent.py does not pass it) we fall back to the old behavior: the same
        reward for the averaged action goes to every member, so a right and a
        wrong member received identical dopamine, which is what reward_fn fixes.
        """
        total = 0.0
        for rule, (tok, frac_vals) in zip(self.rules, token):
            rw = reward_fn(frac_vals) if reward_fn is not None else rewards
            total += rule.learn(rw, tok, lr_scale=lr_scale)
        self.n_updates += 1
        return total

    def reset(self) -> None:
        for r in self.rules:
            r.reset()

    def state_dict(self) -> dict:
        return {"rules": [r.state_dict() for r in self.rules]}

    def load_state_dict(self, d: dict) -> None:
        saved = d.get("rules", [])
        if saved and len(saved) != len(self.rules):
            raise ValueError(
                f"saved learning state has {len(saved)} members, "
                f"but {len(self.rules)} were built; did training.ensemble.n_members change?"
            )
        for r, sd in zip(self.rules, saved):
            r.load_state_dict(sd)
