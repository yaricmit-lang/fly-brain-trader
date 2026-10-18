"""
Brain tests.

The key one concerns the direction of propagation. At one point the weight
matrix was built as W[pre, post] instead of W[post, pre], and activity flowed
backwards through the graph, from output neurons to input neurons. The outputs
stayed at zero forever, the fly always produced exactly 0.5, and the result
matched buy-and-hold to the hundredth. The bug is silent, so we test for it
explicitly.
"""

import numpy as np
import polars as pl
import pytest
import torch

from flybrain.brain.model import FlyBrain
from flybrain.config import load_config
from flybrain.connectome.loader import Connectome
from flybrain.connectome.roles import assign
from flybrain.connectome.subgraph import Subgraph


@pytest.fixture
def cfg():
    c = load_config()
    c.market.symbols = ["BTC/USDT"]
    c.market.features = ["log_return_1", "rsi_14"]
    c.model.input_neurons_per_channel = 2
    return c


@pytest.fixture
def chain_brain(cfg):
    """
    Minimal chain: 2 input -> 2 Kenyon -> 4 output neurons.
    All connections are excitatory so the signal is clearly visible.
    """
    n = 8
    pn, kc, mbon = [0, 1], [2, 3], [4, 5, 6, 7]
    pre, post = [], []
    for a in pn:
        for b in kc:
            pre.append(a); post.append(b)
    for a in kc:
        for b in mbon:
            pre.append(a); post.append(b)

    e = len(pre)
    conn = Connectome(
        node_ids=np.arange(n, dtype=np.uint64),
        edge_index=np.vstack([np.array(pre, np.int32), np.array(post, np.int32)]),
        syn_count=np.full(e, 10.0, np.float32),
        sign=np.ones(e, np.float32),
        meta=pl.DataFrame({
            "root_id": np.arange(n, dtype=np.uint64),
            "cell_type": ["PN", "PN", "KCg-m", "KCg-m", "MBON-01", "MBON-02", "MBON-03", "MBON-04"],
            "cell_class": ["ALPN"] * 2 + ["Kenyon_Cell"] * 2 + ["MBON"] * 4,
            "side": ["left"] * n,
        }),
    )
    sg = Subgraph(conn, np.arange(n, dtype=np.int32), {
        "input": np.array(pn, np.int32),
        "kenyon": np.array(kc, np.int32),
        "output": np.array(mbon, np.int32),
    })
    return FlyBrain(sg, assign(sg, cfg), cfg), sg


def test_signal_reaches_output_neurons(chain_brain):
    """
    Feed a strong input. The output neurons must become active.
    If they stay at zero, activity is flowing backwards through the graph.
    """
    brain, sg = chain_brain
    brain.reset()
    x = torch.ones(brain.rm.n_channels if hasattr(brain, "rm") else 2)
    a = brain.forward(torch.ones(2))

    out = a[sg.group("output")].abs().max().item()
    kc = a[sg.group("kenyon")].abs().max().item()
    assert kc > 1e-6, "signal did not reach the Kenyon cells"
    assert out > 1e-6, (
        "output neurons stayed at zero: activity is flowing backwards through the graph. "
        "Check the index order when building W in model.py"
    )


def test_flow_direction_is_forward(chain_brain):
    """
    Excite ONLY the output neurons directly. The input neurons must stay
    quiet: the PN -> KC -> MBON chain has no feedback connections.
    """
    brain, sg = chain_brain
    brain.reset()
    a = torch.zeros(brain.n)
    a[sg.group("output")] = 1.0
    brain.state = a

    inflow = torch.sparse.mm(brain.W, a.unsqueeze(1)).squeeze(1)
    assert inflow[sg.group("input")].abs().max().item() < 1e-9, (
        "excitation of the output neurons reached the input neurons: the graph is reversed"
    )
    assert inflow[sg.group("kenyon")].abs().max().item() < 1e-9


def test_dale_law_sign_never_flips(cfg):
    """After learning, the sign of every connection must be unchanged."""
    from flybrain.build import build_brain
    from flybrain.brain.plasticity_hebb import HebbianPlasticity

    c = load_config()
    brain, sg, rm, _ = build_brain(c, synthetic=True)
    sign_before = brain.sign.clone()

    plast = HebbianPlasticity(brain, c)
    gen = torch.Generator().manual_seed(0)
    coins = [s.split("/")[0] for s in c.market.symbols]
    for _ in range(20):
        x = torch.randn(rm.n_channels, generator=gen) * 0.5
        _, a = brain.step(x, noise=0.2, generator=gen)
        token = plast.observe(a, {})
        plast.learn({k: float(torch.randn(1, generator=gen)) for k in coins}, token)

    assert torch.equal(sign_before, brain.sign), "synapse sign changed"
    assert (brain.w_mag >= 0).all(), "weight magnitude went negative"
    assert (brain.w_mag <= c.model.w_max + 1e-6).all(), "weight exceeded the ceiling"


def test_frozen_edges_do_not_change(cfg):
    """Non-plastic connections must remain exactly the same."""
    from flybrain.build import build_brain
    from flybrain.brain.plasticity_hebb import HebbianPlasticity

    c = load_config()
    brain, sg, rm, _ = build_brain(c, synthetic=True)
    frozen = ~brain.plastic
    before = brain.w_mag[frozen].clone()

    plast = HebbianPlasticity(brain, c)
    gen = torch.Generator().manual_seed(1)
    coins = [s.split("/")[0] for s in c.market.symbols]
    for _ in range(20):
        x = torch.randn(rm.n_channels, generator=gen) * 0.5
        _, a = brain.step(x, noise=0.2, generator=gen)
        token = plast.observe(a, {})
        plast.learn({k: 1.0 for k in coins}, token)

    assert torch.allclose(before, brain.w_mag[frozen]), (
        "connections outside the plasticity mask changed"
    )


def test_reward_credit_is_per_coin(cfg):
    """
    Dopamine for Bitcoin changes only the synapses onto Bitcoin's MBONs.
    Reward a single coin: edges onto the other coins' MBONs must not move.
    """
    from flybrain.build import build_brain
    from flybrain.brain.plasticity_hebb import HebbianPlasticity

    c = load_config()
    brain, sg, rm, _ = build_brain(c, synthetic=True)
    plast = HebbianPlasticity(brain, c)
    gen = torch.Generator().manual_seed(3)
    coins = list(brain.approach_idx.keys())
    target, others = coins[0], coins[1:]

    for _ in range(5):
        x = torch.randn(rm.n_channels, generator=gen) * 0.5
        _, a = brain.step(x, noise=0.2, generator=gen)
        token = plast.observe(a, {})
    before = brain.w_mag.clone()
    # first reward: the baseline is still zero, so the surprise equals the reward itself
    plast.learn({target: 2.0, **{o: 0.0 for o in others}}, token)
    # The trace and per-coin masks are indexed by PLASTIC edges only
    # (see plasticity_hebb.py), so compare in the same index space
    changed = (brain.w_mag[plast.plastic_idx] - before[plast.plastic_idx]).abs() > 1e-12

    assert changed[plast.coin_mask[target].bool()].any(), "edges of the target coin did not change"
    for o in others:
        assert not changed[plast.coin_mask[o].bool()].any(), (
            f"reward for {target} changed edges onto the MBONs of {o}"
        )


def test_kc_code_is_sparse(cfg):
    """k-WTA: exactly k Kenyon cells are active, the rest are zero."""
    from flybrain.build import build_brain

    c = load_config()
    brain, sg, rm, _ = build_brain(c, synthetic=True)
    brain.reset()
    gen = torch.Generator().manual_seed(4)
    kc = sg.group("kenyon")
    for _ in range(5):
        x = torch.randn(rm.n_channels, generator=gen) * 0.7
        _, a = brain.step(x)
        active = int((a[kc] != 0).sum())
        assert active == brain.k_active, f"{active} active KC, expected {brain.k_active}"


def test_noise_only_on_mbon(cfg):
    """Exploration noise touches nothing except the MBONs."""
    from flybrain.build import build_brain

    c = load_config()
    c.model.noise_target = "mbon"
    brain, sg, rm, _ = build_brain(c, synthetic=True)
    brain.reset()
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(rm.n_channels, generator=gen) * 0.5
    brain.step(x)
    st = brain.state.clone()
    a0 = brain.forward(x).clone()
    brain.state = st
    a1 = brain.forward(x, noise=0.3, generator=gen)
    non = torch.ones(brain.n, dtype=torch.bool)
    non[brain.mbon_idx] = False
    assert torch.equal(a0[non], a1[non]), "noise leaked onto neurons outside the MBONs"
    assert not torch.equal(a0[brain.mbon_idx], a1[brain.mbon_idx])


def test_output_is_not_stuck_at_neutral(cfg):
    """
    The fly must respond to different inputs. If the position fraction is
    always exactly 0.5, the output is saturated or the signal is not getting through.
    """
    from flybrain.build import build_brain

    c = load_config()
    brain, sg, rm, _ = build_brain(c, synthetic=True)
    brain.reset()
    gen = torch.Generator().manual_seed(2)

    fractions = []
    for _ in range(300):
        x = torch.randn(rm.n_channels, generator=gen) * 0.7
        d, _ = brain.step(x)
        fractions.append(list(d.fractions.values()))

    arr = np.array(fractions)
    assert arr.std() > 0.01, (
        f"position fraction barely changes (std {arr.std():.5f}): "
        f"the fly's output carries no information"
    )


def test_peek_varies_between_calls_but_does_not_touch_state(cfg):
    """
    The live dashboard feeds the same x every few seconds while the hourly
    candle is still open. After convergence, step() with an unchanged x
    returns BIT-FOR-BIT identical activations, so nothing on screen would
    light up anew. peek() must (1) differ on every call (spontaneous noise)
    and (2) leave self.state untouched, since the real trading decision
    depends on it.
    """
    from flybrain.build import build_brain

    c = load_config()
    brain, sg, rm, _ = build_brain(c, synthetic=True)
    brain.reset()
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(rm.n_channels, generator=gen) * 0.5
    for _ in range(5):
        brain.step(x)

    # same x from the same state: step() is deterministic (which is why the
    # live dashboard picture froze without peek())
    saved_state = brain.state.clone()
    _, a1 = brain.step(x)
    brain.state = saved_state.clone()
    _, a2 = brain.step(x)
    assert torch.equal(a1, a2), "step() with an unchanged x must be deterministic"

    # peek() varies from call to call and does not corrupt the decision state
    state_snapshot = brain.state.clone()
    v1 = brain.peek(x, generator=torch.Generator().manual_seed(1))
    v2 = brain.peek(x, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(v1, v2), "peek() must vary from tick to tick"
    assert torch.equal(brain.state, state_snapshot), "peek() must not touch the decision state"

    # step() from the same state gives the same result as before peek(),
    # so peek() left nothing behind in the state
    _, a3 = brain.step(x)
    brain.state = state_snapshot.clone()
    _, a4 = brain.step(x)
    assert torch.equal(a3, a4), "peek() left a trace in the decision state"
