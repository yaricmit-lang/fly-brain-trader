"""
Rule B: REINFORCE, an exact policy gradient.

Less biologically plausible (a real synapse cannot compute a derivative
through the whole network) but noticeably more reliable in convergence. It
serves as a reference point: if the Hebbian rule loses to this one on the same
graph, the cause is the rule and not the topology.

Policy: the position fraction is drawn from a Beta distribution centered on
the network output (rescaled to that range for [-1, 1]). This provides
exploration and a correct action log-probability.

Delayed per-coin reward. The token stores the action log-probabilities for
time t together with their computation graph; the reward arrives H steps
later. Subtlety: if the optimizer modifies the weights in place between
observe and learn, the old graph breaks ("modified by an inplace operation").
So the weights are updated functionally, as a new tensor instead of an edit of
the old one, which keeps the old graphs intact. Adam is implemented by hand,
in three lines.
"""

from __future__ import annotations

import torch

from ..config import Config
from .model import FlyBrain
from .plasticity import Plasticity


class ReinforcePlasticity(Plasticity):
    def __init__(self, brain: FlyBrain, cfg: Config) -> None:
        super().__init__(brain, cfg)
        self.p = cfg.training.reinforce

        # Only plastic weights are learned; the rest are frozen by the gradient mask.
        self.brain.w_mag = self.brain.w_mag.detach().clone().requires_grad_(True)
        self.brain.invalidate()

        # Hand-written Adam: moment state
        self._m = torch.zeros_like(self.brain.w_mag)
        self._v = torch.zeros_like(self.brain.w_mag)
        self._t = 0
        self.betas = (0.9, 0.999)

        self.concentration = 4.0   # how narrow the distribution is around the network output
        self.allow_short = cfg.trading.allow_short
        self.exposure_mode = cfg.trading.position_mode == "exposure"
        self.exposure_max = cfg.trading.exposure_max
        self._dist: dict[str, tuple[torch.distributions.Beta, torch.Tensor]] = {}

    @property
    def exploration_noise(self) -> float:
        return 0.0   # exploration comes from sampling the policy

    def act(self, decision) -> dict[str, float]:
        """
        Sample the action from a Beta distribution around the network output
        and remember (distribution, sample) for observe(). The action executed
        on the exchange and the one used for log_prob are now THE SAME (before,
        the sampled action was discarded for the log-probability and the
        exchange always traded the raw network output, which amounted to
        off-policy REINFORCE without correction).

        In exposure mode the multiplier lies in [0, exposure_max] (neutral
        1.0, see model.read_out), so mean01 = frac/exposure_max rather than
        (frac+1)/2. That formula was valid only for direction mode; in
        exposure mode it was almost always clamped to 0.999 (the fly sits
        near 1.0, and (1+1)/2=1.0), making the policy degenerate.
        """
        self._dist = {}
        out: dict[str, float] = {}
        for coin, frac in decision.fractions_t.items():
            if self.exposure_mode:
                mean01 = (frac / self.exposure_max).clamp(1e-3, 1 - 1e-3)
            elif self.allow_short:
                mean01 = ((frac + 1.0) / 2.0).clamp(1e-3, 1 - 1e-3)
            else:
                mean01 = frac.clamp(1e-3, 1 - 1e-3)
            a = mean01 * self.concentration
            b = (1 - mean01) * self.concentration
            dist = torch.distributions.Beta(a, b)
            action01 = dist.sample()
            self._dist[coin] = (dist, action01)
            if self.exposure_mode:
                out[coin] = float(action01 * self.exposure_max)
            elif self.allow_short:
                out[coin] = float(2.0 * action01 - 1.0)
            else:
                out[coin] = float(action01)
        return out

    def observe(self, activations: torch.Tensor, fractions: dict) -> dict:
        """
        Log-probability and entropy of exactly the action that act() has just
        handed to the exchange (self._dist, a live autograd graph).
        """
        token: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for coin, (dist, action01) in self._dist.items():
            token[coin] = (dist.log_prob(action01), dist.entropy())
        return token

    def learn(self, rewards: dict[str, float], token: dict, reward_fn=None,
              lr_scale: float | None = None) -> float:
        """
        Gradient step: raise the probability of the actions on a coin that were
        followed by improvement on that coin.
        """
        loss = torch.zeros((), dtype=torch.float32)
        touched = False
        for coin, r in rewards.items():
            if coin not in token:
                continue
            logp, ent = token[coin]
            delta = self._delta(coin, r, self.p.reward_baseline_decay)
            loss = loss - delta * logp - self.p.entropy_bonus * ent
            touched = True
        if not touched:
            return 0.0

        grad, = torch.autograd.grad(loss, self.brain.w_mag, allow_unused=True)
        if grad is None:
            return 0.0
        grad = grad * self.brain.plastic          # the gradient flows only through plastic weights

        # --- hand-written Adam, functional update ---
        self._t += 1
        b1, b2 = self.betas
        self._m.mul_(b1).add_(grad, alpha=1 - b1)
        self._v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
        m_hat = self._m / (1 - b1 ** self._t)
        v_hat = self._v / (1 - b2 ** self._t)
        # The same decision-frequency correction as in the Hebbian rule: Adam
        # normalizes the gradient MAGNITUDE but not the NUMBER of steps, and
        # with decision_timeframe=1h there are four times as many steps over
        # the same history.
        scale = lr_scale if lr_scale is not None else self.cfg.lr_stride_scale
        step = self.p.lr * scale * m_hat / (v_hat.sqrt() + 1e-8)

        new_w = (self.brain.w_mag.detach() - step).clamp(0.0, self.cfg.model.w_max)
        self.brain.w_mag = new_w.requires_grad_(True)
        self.brain.invalidate()
        self.n_updates += 1
        return float(grad.abs().sum())

    def reset(self) -> None:
        pass

    def state_dict(self) -> dict:
        d = super().state_dict()
        d.update({"m": self._m.detach().clone(), "v": self._v.detach().clone(), "t": self._t})
        return d

    def load_state_dict(self, d: dict) -> None:
        super().load_state_dict(d)
        if "m" in d:
            self._m = d["m"].to(self.brain.device)
        if "v" in d:
            self._v = d["v"].to(self.brain.device)
        if "t" in d:
            self._t = d["t"]


def make_plasticity(brain: FlyBrain, cfg: Config) -> Plasticity:
    """Factory driven by the training.rule flag in the config."""
    from .plasticity_hebb import HebbianPlasticity  # noqa: PLC0415

    if cfg.training.rule == "hebbian":
        return HebbianPlasticity(brain, cfg)
    if cfg.training.rule == "reinforce":
        return ReinforcePlasticity(brain, cfg)
    raise ValueError(f"unknown learning rule: {cfg.training.rule}")
