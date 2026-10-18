"""
The fly brain as a computational network.

Nodes are the neurons of the connectome subgraph; edges are synapses with a
learnable weight. Activity propagates through the graph for several steps,
which acts as the "depth of deliberation" for a single candle.

Dale's law: all outgoing synapses of a real neuron share one sign, and a
neuron cannot switch from excitatory to inhibitory. We therefore store the
weight magnitude (learned) separately from the sign (derived from the
neurotransmitter type and frozen).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# Torch warns that it does not validate sparse tensor invariants. We build the
# tensors ourselves from connectome indices, so they are valid by construction,
# and checking them on every step would only cost time.
torch.sparse.check_sparse_tensor_invariants.disable()

from ..config import Config, load_config
from ..connectome.roles import RoleMap
from ..connectome.subgraph import Subgraph


@dataclass
class Decision:
    """The fly's decision for one candle."""
    fractions: dict[str, float]          # coin -> fraction of capital, 0..1
    activations: np.ndarray              # state of all neurons (for 3D rendering)
    approach: dict[str, float]           # mean activity of the "approach" (buy) MBON group
    avoid: dict[str, float]              # mean activity of the "avoid" (sell) MBON group
    # The same fractions as live tensors still attached to the autograd graph.
    # REINFORCE needs them so that the gradient flows back to the synaptic
    # weights. Passing plain floats here makes backward() fail with
    # "does not require grad".
    fractions_t: dict[str, "torch.Tensor"] = field(default_factory=dict)


class FlyBrain:
    """
    Network built on the connectome topology.

    Forward pass for one candle:
        a <- (1-leak) * a + leak * f(W @ a + B @ x)
    repeated `steps` times, after which the output is read from the MBONs.
    """

    def __init__(
        self,
        sg: Subgraph,
        rm: RoleMap,
        cfg: Config | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.sg = sg
        self.rm = rm
        self.n = sg.n_nodes
        self.device = torch.device(self.cfg.project.device)
        self.seed = self.cfg.project.seed if seed is None else seed
        g = torch.Generator(device="cpu").manual_seed(self.seed)

        # --- neuron groups used by the forward pass ---
        # Kenyon cells: k-winner-take-all (a stand-in for APL inhibition);
        # MBONs: the only neurons that receive exploration noise
        kc = sg.groups.get("kenyon")
        self.kc_idx = (
            torch.from_numpy(kc.astype(np.int64)).to(self.device)
            if kc is not None and len(kc) else None
        )
        mbon_all = np.concatenate(
            [*rm.coin_approach.values(), *rm.coin_avoid.values()]
        ).astype(np.int64)
        self.mbon_idx = torch.from_numpy(np.unique(mbon_all)).to(self.device)
        self.k_active = (
            max(1, int(round(self.cfg.model.kc_sparsity * len(self.kc_idx))))
            if self.kc_idx is not None else 0
        )

        # --- edges ---
        ei = torch.from_numpy(sg.conn.edge_index.astype(np.int64))
        self.edge_index = ei.to(self.device)
        self.sign = torch.from_numpy(sg.conn.sign).to(self.device)

        # initial weight magnitude: log(1 + synapse count), normalized
        mag = torch.log1p(torch.from_numpy(sg.conn.syn_count))
        if self.cfg.model.normalize_incoming:
            mag = self._normalize_incoming(mag)
        self.w_mag = mag.to(self.device).clone()

        self.plastic = torch.from_numpy(rm.plastic_mask).to(self.device)

        # --- homeostasis ---
        # Target: the sum of each neuron's incoming PLASTIC weights as it was
        # at initialization. Without it, a learning rate large enough to move
        # the weights inflates the overall readout scale indefinitely instead
        # of relearning the DISTRIBUTION of weight across Kenyon cells.
        # See apply_homeostasis().
        if self.cfg.model.homeostatic:
            post_p = self.edge_index[1][self.plastic]
            self._homeo_target = torch.zeros(self.n, device=self.device).index_add_(
                0, post_p, self.w_mag[self.plastic]
            )
            # Weight floor: a fraction of the mean INITIAL plastic weight of
            # this particular graph (see w_min_frac in config.py).
            plastic_mean = float(self.w_mag[self.plastic].mean()) if bool(self.plastic.any()) else 0.0
            self._w_floor = self.cfg.model.w_min_frac * plastic_mean
        else:
            self._homeo_target = None
            self._w_floor = 0.0

        # --- input: channel -> group of projection neurons ---
        # B is stored as a sparse (N, C) matrix with fixed weights
        c2i = rm.channel_to_input                       # (C, k)
        n_ch, k = c2i.shape
        rows = torch.from_numpy(c2i.reshape(-1).astype(np.int64))
        cols = torch.arange(n_ch).repeat_interleave(k)
        vals = torch.rand(n_ch * k, generator=g) * 0.5 + 0.75   # spread in glomerulus strength
        self.B = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, (self.n, n_ch)
        ).coalesce().to(self.device)

        # --- output ---
        self.approach_idx = {
            c: torch.from_numpy(v.astype(np.int64)).to(self.device)
            for c, v in rm.coin_approach.items()
        }
        self.avoid_idx = {
            c: torch.from_numpy(v.astype(np.int64)).to(self.device)
            for c, v in rm.coin_avoid.items()
        }

        self.state = torch.zeros(self.n, device=self.device)
        self._W_cache: torch.Tensor | None = None

        # Running statistics of the MBON difference for the self-calibrating
        # readout. The variance starts at 1e-2 rather than 1e-4: with 1e-4 the
        # z-scores of the first decisions after reset() were ~100x too large
        # and the sigmoid saturated. This is used together with the warm-up
        # (Agent.run(warm_X=...)) and does not replace it: the warm-up decides
        # WHEN the calibration has settled, this decides HOW wildly it starts.
        self._ro_mean: dict[str, float] = {c: 0.0 for c in self.approach_idx}
        self._ro_var: dict[str, float] = {c: 1e-2 for c in self.approach_idx}
        # Smoothed output: a "motor" signal with inertia, see read_out.
        # The neutral value depends on the position mode: in "exposure" it is a
        # multiplier of 1.0 (a fly with no opinion reproduces the baseline
        # formula exactly), in "direction" it is 0.0.
        self._neutral = 1.0 if self.cfg.trading.position_mode == "exposure" else 0.0
        self._frac_ema: dict[str, torch.Tensor] = {
            c: torch.tensor(self._neutral) for c in self.approach_idx
        }

    # ------------------------------------------------------------------
    #  Weights
    # ------------------------------------------------------------------
    def _normalize_incoming(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Rescale each neuron's summed incoming weights to input_scale.

        Why: a mushroom body output neuron receives thousands of inputs from
        Kenyon cells. Without normalization the drive onto it is huge, tanh
        saturates, and the neuron stops responding to input changes (and stops
        learning, because the derivative is nearly zero there).

        Checked on a toy graph: scale 1 suppresses the signal (|MBON| 0.18),
        scale 5 gives 53% saturated neurons, scale 2 keeps |MBON| around 0.47
        with no saturated neurons.
        """
        pre = torch.from_numpy(self.sg.conn.edge_index[0].astype(np.int64))
        post = torch.from_numpy(self.sg.conn.edge_index[1].astype(np.int64))
        total = torch.zeros(self.n).index_add_(0, post, mag)
        mag = mag / total.clamp(min=1e-6)[post] * self.cfg.model.input_scale

        # Sparsity compensation: after k-WTA only ~5% of Kenyon cells are
        # active, and without boosting their outputs the drive onto MBONs
        # drops 20-fold (measured: |MBON| 0.035 instead of ~0.4). In the fly
        # the sparse KC code works precisely because KC->MBON synapses are
        # strong, so we amplify all edges FROM Kenyon cells.
        if self.kc_idx is not None and self.cfg.model.kc_sparsity > 0:
            is_kc = torch.zeros(self.n, dtype=torch.bool)
            is_kc[self.kc_idx.cpu()] = True
            mag = torch.where(is_kc[pre], mag * self.cfg.model.kc_output_gain, mag)
        return mag

    @property
    def W(self) -> torch.Tensor:
        """
        Signed sparse weight matrix.

        Mind the index order. In edge_index row 0 is the presynaptic neuron and
        row 1 the postsynaptic one. For W @ a we need
        (W @ a)[post] = sum over pre of W[post, pre] * a[pre], i.e. the input
        to a neuron is gathered from the neurons that fire onto it.

        So the matrix is built as [post, pre], not [pre, post]. With the
        reverse order activity would flow backwards through the graph, from
        output neurons to input neurons, and the MBONs would stay at zero.
        """
        if self._W_cache is None:
            vals = self.sign * self.w_mag
            idx = torch.stack([self.edge_index[1], self.edge_index[0]])
            self._W_cache = torch.sparse_coo_tensor(
                idx, vals, (self.n, self.n)
            ).coalesce()
        return self._W_cache

    def invalidate(self) -> None:
        """Signal that the weights changed and the matrix must be rebuilt."""
        self._W_cache = None

    def clamp_weights(self) -> None:
        """
        After a learning step: keep the weight magnitude non-negative and
        bounded above. The sign is never touched; that is Dale's law.
        """
        self.w_mag.clamp_(min=0.0, max=self.cfg.model.w_max)
        self.invalidate()

    def apply_homeostasis(self) -> None:
        """
        Synaptic scaling of plastic weights.

        After a learning step, return each neuron's summed incoming PLASTIC
        weights to their initial value. This way the network learns the
        distribution of weight across Kenyon cells rather than the overall
        readout scale, which would otherwise grow or shrink without bound
        under a learning rate large enough to move the weights. _w_floor (a
        w_min_frac share of the mean initial plastic weight of THIS graph) is
        a floor: a synapse pushed to zero must not die permanently, since
        learning is an lr * delta * trace inflow onto weights near zero, not
        onto weights that no longer exist.
        """
        m = self.cfg.model
        if not m.homeostatic or self._homeo_target is None:
            return
        idx = self.plastic
        post_p = self.edge_index[1][idx]
        cur = torch.zeros(self.n, device=self.device).index_add_(0, post_p, self.w_mag[idx])
        scale = torch.where(
            cur > 1e-6, self._homeo_target / cur.clamp(min=1e-6), torch.ones_like(cur)
        ).clamp(0.2, 5.0)
        self.w_mag[idx] = (self.w_mag[idx] * scale[post_p]).clamp(min=self._w_floor, max=m.w_max)
        self.invalidate()

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero the activity and the readout calibration. Call between epochs and runs."""
        self.state = torch.zeros(self.n, device=self.device)
        self._ro_mean = {c: 0.0 for c in self.approach_idx}
        self._ro_var = {c: 1e-2 for c in self.approach_idx}
        self._frac_ema = {c: torch.tensor(self._neutral) for c in self.approach_idx}

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) if self.cfg.model.nonlinearity == "relu" else torch.tanh(x)

    def forward(
        self,
        x: torch.Tensor,
        noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Run one feature vector through the graph.

        x     -- (C,) values of the input channels for this candle
        noise -- exploration noise on the output neurons; without it the
                 Hebbian rule cannot tell which action led to the reward
        """
        m = self.cfg.model
        a = self.state if m.recurrent_across_bars else torch.zeros(self.n, device=self.device)

        drive = torch.sparse.mm(self.B, x.unsqueeze(1)).squeeze(1)
        if m.input_normalize:
            # Gain control in the antennal lobe: the overall "odor" intensity
            # is constant and only the pattern varies. Otherwise in a storm all
            # inputs are loud and the Kenyon cells saturate, while in a calm
            # all inputs are quiet and the fly "cannot sense" the market.
            nz = drive[drive != 0]
            if nz.numel():
                rms = torch.sqrt((nz * nz).mean())
                drive = drive * (m.input_drive_rms / (rms + 1e-6))

        W = self.W
        for _ in range(m.steps):
            inflow = torch.sparse.mm(W, a.unsqueeze(1)).squeeze(1)
            a = (1 - m.leak) * a + m.leak * self._act(inflow + drive)
            a = self._kc_winner_take_all(a)

        if noise > 0:
            if m.noise_target == "mbon":
                # Noise on the action only, not on perception. If the whole
                # network were noisy, the eligibility trace would record a
                # corrupted "odor" and the rule could not tell whose deviation
                # led to the reward.
                eps = torch.zeros(self.n, device=self.device)
                eps[self.mbon_idx] = torch.randn(
                    len(self.mbon_idx), generator=generator, device=self.device
                ) * noise
            else:
                eps = torch.randn(self.n, generator=generator, device=self.device) * noise
            a = a + eps

        self.state = a.detach() if a.requires_grad else a
        return a

    def _kc_winner_take_all(self, a: torch.Tensor) -> torch.Tensor:
        """
        Sparse Kenyon cell code: keep the top-k most active cells and silence
        the rest. In the fly this is done by the inhibitory APL neuron, and it
        is this sparseness (~5% active) that separates similar odors. Without
        it all MBONs see nearly the same input and no "state -> reward"
        association can form.

        Not in-place: the mask is applied by multiplication so that the
        REINFORCE gradient flows through the surviving cells.
        """
        if self.kc_idx is None or self.k_active <= 0 or self.k_active >= len(self.kc_idx):
            return a
        kc_act = a[self.kc_idx]
        kth = torch.topk(kc_act, self.k_active).values[-1]
        keep = torch.ones(self.n, device=self.device)
        keep[self.kc_idx] = (kc_act >= kth).to(a.dtype)
        return a * keep

    # ------------------------------------------------------------------
    #  Reading out the decision
    # ------------------------------------------------------------------
    def read_out(
        self, a: torch.Tensor, elapsed_hours: float | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict, dict]:
        """
        Convert MBON activity into a capital fraction for each coin.

        In the fly, MBONs split into "approach" and "avoid" groups, and
        behavior is set by their difference. Same here: an excess of
        "approach" activity means buy.

        elapsed_hours: real time since the previous read_out call for this
        brain. readout_decay and output_smoothing are per-CALL decays by
        construction, so their memory is measured in decisions, not hours.
        While the decision cadence is constant (a regular backtest, agent.py)
        the two are equivalent up to scale and elapsed_hours can be omitted
        with no change in behavior. With a bounded-adaptive cadence (1h at
        times, 4h at others) this would be incorrect: the EMA memory would
        jump fourfold in real time with unchanged constants. Passing
        elapsed_hours rebases the decay onto elapsed time:
        decay ** (elapsed_hours / reference_hours), where reference_hours is
        training.lr_reference_timeframe (the same reference frequency that
        already calibrates lr_stride_scale).
        """
        fr: dict[str, torch.Tensor] = {}
        app: dict[str, float] = {}
        avo: dict[str, float] = {}
        m = self.cfg.model
        decay = m.readout_decay
        smoothing = m.output_smoothing
        if elapsed_hours is not None:
            from ..market.fetch import TF_MS  # noqa: PLC0415

            ref_hours = TF_MS[self.cfg.training.lr_reference_timeframe] / 3_600_000.0
            power = max(0.0, elapsed_hours) / max(1e-9, ref_hours)
            decay = decay ** power
            smoothing = smoothing ** power

        for coin in self.approach_idx:
            pa = a[self.approach_idx[coin]].mean()
            pv = a[self.avoid_idx[coin]].mean()
            diff = pa - pv

            if m.adaptive_readout:
                # Divide the difference by its own recent spread. Otherwise
                # the position fraction depends on graph size: in a large
                # network the difference between the groups is always small
                # and the sigmoid sticks at 0.5 forever. The statistics use
                # only the past, so nothing leaks from the future.
                d = float(diff)
                mu, var = self._ro_mean[coin], self._ro_var[coin]
                z = (diff - mu) / (var ** 0.5 + 1e-6)
                self._ro_mean[coin] = decay * mu + (1 - decay) * d
                self._ro_var[coin] = decay * var + (1 - decay) * (d - mu) ** 2
                raw = torch.sigmoid(m.output_gain * z)
            else:
                raw = torch.sigmoid(m.output_gain * diff)

            if self.cfg.trading.position_mode == "exposure":
                # Exposure multiplier: [0,1] -> [0, exposure_max], with the
                # NEUTRAL point at exactly 1.0 for any exposure_max. The old
                # formula `raw * exposure_max` put neutral at exposure_max/2,
                # so with exposure_max=4.0 a fly with no opinion traded twice
                # as aggressively as the baseline formula, and the whole
                # control-variate construction (relative_to_baseline,
                # activity_bonus) rested on a broken assumption. Instead we
                # use a piecewise-linear map:
                # sigmoid <= 0.5 -> [0, 1], sigmoid > 0.5 -> [1, exposure_max].
                exp_max = self.cfg.trading.exposure_max
                if float(raw) <= 0.5:
                    raw = 2.0 * raw
                else:
                    raw = 1.0 + (exp_max - 1.0) * (2.0 * raw - 1.0)
            elif self.cfg.trading.allow_short:
                # long/cash/short: [0,1] -> [-1,1]. Neutral = 0 = flat (cash)
                raw = 2.0 * raw - 1.0

            # Smooth the position fraction with an exponential moving average:
            # a "motor output" with inertia rather than a jump on every candle.
            #
            # Without it, on the real connectome the fraction moved by more
            # than 5% on 81.5% of candles (measured on the test period): the
            # raw z-score of the MBON difference is noisy every hour, and the
            # fly paid fees on almost every candle instead of accumulating
            # conviction. A real insect's motor command does not jump either;
            # behavior builds up rather than switching instantly.
            #
            # The previous value is detached: like self.state, the inertia is
            # part of the decision dynamics, not something REINFORCE should
            # backpropagate through to earlier steps.
            prev = self._frac_ema[coin].detach()
            smoothed = smoothing * prev + (1 - smoothing) * raw
            self._frac_ema[coin] = smoothed
            fr[coin] = smoothed

            app[coin] = float(pa)
            avo[coin] = float(pv)
        return fr, app, avo

    def step(
        self,
        x: torch.Tensor,
        noise: float = 0.0,
        generator: torch.Generator | None = None,
        elapsed_hours: float | None = None,
    ) -> tuple[Decision, torch.Tensor]:
        """Candle -> decision. The second return value is the activations (needed by the learning rules)."""
        a = self.forward(x, noise=noise, generator=generator)
        fr, app, avo = self.read_out(a, elapsed_hours=elapsed_hours)
        d = Decision(
            fractions={c: float(v) for c, v in fr.items()},
            activations=a.detach().cpu().numpy(),
            approach=app,
            avoid=avo,
            fractions_t=fr,
        )
        return d, a

    def peek(self, x: torch.Tensor, jitter: float = 0.06,
              spike_frac: float = 0.03, spike_mag: float = 0.9,
              generator: torch.Generator | None = None) -> torch.Tensor:
        """
        Activations for the dashboard visualization ONLY, with no side effects
        on the fly's decision state.

        While an hourly candle is still open, think() feeds the same x every
        few seconds. The recurrent state (self.state), with leak 0.3, quickly
        settles into a fixed point, and then `a` is BIT-FOR-BIT identical from
        tick to tick: nothing on screen lights up anew even though the brain
        is formally "thinking".

        The first version added uniform Gaussian noise across the whole
        network. Activations did change (verified), but the amplitude (+-0.06
        on top of an almost-zero settled signal) was not visually readable:
        6283 points trembling in sync look like a static picture. Real
        neurons do not tremble in unison; they emit sporadic, sparse, sharp
        single spikes. Here we do the same: a small random subset of neurons
        (3% by default, ~190 of 6283) flashes to the activation limit on each
        tick, while the rest get a light background jitter. On screen this
        appears as scattered bright points that go out and light up in new
        places every few seconds. It is noticeable, and a fair metaphor
        (spontaneous activity of real neurons) rather than decoration.

        self.state is rolled back right after the pass: the trading decision
        (step) is computed separately and deterministically; this function
        neither sees nor affects it.
        """
        saved_state = self.state
        a = self.forward(x)
        self.state = saved_state
        a = a.clone()
        n_spike = max(1, int(round(self.n * spike_frac)))
        idx = torch.randperm(self.n, generator=generator, device=self.device)[:n_spike]
        signs = torch.randint(0, 2, (n_spike,), generator=generator, device=self.device).to(a.dtype) * 2 - 1
        a[idx] = signs * spike_mag
        eps = torch.randn(self.n, generator=generator, device=self.device) * jitter
        return a + eps

    # ------------------------------------------------------------------
    #  Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "w_mag": self.w_mag.cpu(),
            "state": self.state.cpu(),
            # Readout calibration: it used to be omitted, so a live brain
            # warmed up from zero every time instead of resuming from the
            # calibration it had at the end of training.
            "ro_mean": dict(self._ro_mean),
            "ro_var": dict(self._ro_var),
            # Inertia of the motor output. Without it a restored brain would
            # jump the exposure on its first decision instead of continuing
            # from its accumulated conviction.
            "frac_ema": {c: v.cpu() for c, v in self._frac_ema.items()},
        }

    def load_state_dict(self, d: dict) -> None:
        self.w_mag = d["w_mag"].to(self.device)
        self.state = d["state"].to(self.device)
        if "ro_mean" in d:
            self._ro_mean = dict(d["ro_mean"])
        if "ro_var" in d:
            self._ro_var = dict(d["ro_var"])
        if "frac_ema" in d:
            self._frac_ema = {c: v.to(self.device) for c, v in d["frac_ema"].items()}
        self.invalidate()

    def describe(self) -> str:
        w = self.w_mag
        m = self.cfg.model
        kc = f"{self.k_active} of {len(self.kc_idx)}" if self.kc_idx is not None else "none"
        return (
            f"\n  neurons: {self.n:,}   connections: {len(self.w_mag):,}\n"
            f"  plastic connections: {int(self.plastic.sum()):,}\n"
            f"  steps per candle: {m.steps}   leak: {m.leak}\n"
            f"  active KC (k-WTA): {kc}   noise: {m.noise_target}   "
            f"input: {'normalized' if m.input_normalize else 'raw'}\n"
            f"  output: {'[-1, 1] long/short' if self.cfg.trading.allow_short else '[0, 1] long/cash'}\n"
            f"  weight: mean {w.mean():.4f}, max {w.max():.4f}\n"
        )
