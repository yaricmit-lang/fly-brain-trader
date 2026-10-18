"""
Rule A: reward-modulated Hebbian learning (three-factor rule).

How it works in the fly. When a Kenyon cell and an output neuron fire
together, a chemical trace (eligibility trace) remains in the synapse between
them and lasts for seconds. If dopamine arrives within that window, the
synapse changes. If it does not, the trace dissolves and nothing happens.

Three factors: presynaptic activity, postsynaptic activity, and a reward
signal. Plain Hebb ("neurons that fire together wire together") is not enough:
it cannot tell whether a decision was good or bad.

Nothing is learned without exploration noise: if the network always outputs
the same thing, the reward carries no information about what could have been
done better. The noise is injected only on the MBONs (see model.forward), and
the rule reinforces the random deviations that were followed by improvement.

Design:
- the trace is captured at decision time (the token) and receives dopamine H
  steps later, when the reward is known, as in a real fly where dopamine
  arrives seconds after the odor;
- dopamine is per coin: synapses onto the Bitcoin MBONs change only from the
  Bitcoin reward.
"""

from __future__ import annotations

import torch

from ..config import Config
from .model import FlyBrain
from .plasticity import Plasticity


class HebbianPlasticity(Plasticity):
    def __init__(self, brain: FlyBrain, cfg: Config) -> None:
        super().__init__(brain, cfg)
        self.p = cfg.training.hebbian

        # The trace and the per-coin masks are indexed by PLASTIC edges only
        # (kc_to_mbon, usually ~20% of all edges) rather than by a dense (E,)
        # tensor, 80% of which would be multiplied by 0 anyway. This is faster
        # and uses 4-5x less memory per token in the queue.
        self.plastic_idx = torch.nonzero(brain.plastic, as_tuple=True)[0]
        n_plastic = len(self.plastic_idx)
        self.trace = torch.zeros(n_plastic, device=brain.device)
        # Running mean of each neuron's activity: the reference for "unusualness".
        self.mean_act = torch.zeros(brain.n, device=brain.device)
        self._act_decay = 0.99
        full_masks = self._coin_edge_masks()
        self.coin_mask = {c: m[self.plastic_idx] for c, m in full_masks.items()}

    @property
    def exploration_noise(self) -> float:
        return self.p.exploration_noise

    def observe(self, activations: torch.Tensor, fractions: dict) -> torch.Tensor:
        """
        Accumulate the trace: presynaptic activity times the deviation of the
        postsynaptic neuron from its usual level.

        Subtracting the mean is essential. Without it the trace grows on all
        connections alike, and the rule can no longer tell who is responsible
        for the reward.

        Returns a snapshot of the trace, i.e. what the synapse "remembered" at
        decision time. The reward arrives later and is applied to this snapshot.
        """
        a = activations.detach()
        self.mean_act.mul_(self._act_decay).add_(a, alpha=1 - self._act_decay)

        pre_idx, post_idx = self.brain.edge_index[:, self.plastic_idx]
        pre_a = a[pre_idx]
        post_dev = (a - self.mean_act)[post_idx]

        self.trace.mul_(self.p.eligibility_decay).add_(pre_a * post_dev)
        return self.trace.clone()

    def learn(self, rewards: dict[str, float], token: torch.Tensor, reward_fn=None,
              lr_scale: float | None = None) -> float:
        """
        Dopamine arrives, a separate signal for each coin. Weights change in
        proportion to the trace snapshot and the reward surprise, only on
        edges onto that coin's MBONs.
        """
        # lr is the step per decision at the reference cadence (4h by
        # default); at other cadences we rescale so that learning per unit of
        # REAL time does not depend on how often the fly decides.
        # lr_scale is an external override (bounded-adaptive cadence,
        # live_engine.py): there decision_timeframe is not constant over a
        # run, and the right scale comes from the ACTUAL interval between two
        # specific decisions, not from cfg.lr_stride_scale.
        lr = self.p.lr * (lr_scale if lr_scale is not None else self.cfg.lr_stride_scale)
        dw = torch.zeros_like(self.trace)
        for coin, r in rewards.items():
            delta = self._delta(coin, r, self.p.reward_baseline_decay)
            if delta != 0.0:
                dw.add_(token * self.coin_mask[coin], alpha=lr * delta)

        self.brain.w_mag[self.plastic_idx] += dw
        self.brain.clamp_weights()
        self.brain.apply_homeostasis()
        self.n_updates += 1
        return float(dw.abs().sum())

    def reset(self) -> None:
        self.trace.zero_()
        self.mean_act.zero_()

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["trace"] = self.trace.detach().clone()
        d["mean_act"] = self.mean_act.detach().clone()
        return d

    def load_state_dict(self, d: dict) -> None:
        super().load_state_dict(d)
        if "trace" in d:
            self.trace = d["trace"].to(self.brain.device)
        if "mean_act" in d:
            self.mean_act = d["mean_act"].to(self.brain.device)
