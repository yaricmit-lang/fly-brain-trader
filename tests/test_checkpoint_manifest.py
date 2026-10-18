"""
Checkpoint manifest (live loop).

Without it, loading a checkpoint with a different number of channels does not
fail: the state_dict stores w_mag per connectome EDGE, which does not depend
on the number of input channels. The channel-to-projection-neuron layout
(roles.py, drawn from an rng seeded by n_channels) would silently break, and
the brain would read RSI where it was trained to read volume. A missing
manifest must therefore be a refusal, not a silent skip, and that is what
these tests check.
"""

from __future__ import annotations

import pytest

from flybrain.build import build_brain, checkpoint_manifest, check_manifest, save_checkpoint, load_checkpoint
from flybrain.config import load_config


def _cfg():
    c = load_config()
    c.market.symbols = ["BTC/USDT", "ETH/USDT"]
    c.market.features = ["log_return_1", "rsi_14"]
    return c


def test_missing_manifest_is_refused_not_skipped():
    """Core property: a missing manifest raises RuntimeError instead of being skipped silently."""
    with pytest.raises(RuntimeError, match="has no channel manifest"):
        check_manifest(None, ["BTC:log_return_1"], 5, ["BTC/USDT"], "some/best.pt")


def test_channel_count_mismatch_is_refused():
    manifest = {"channel_names": ["BTC:a", "BTC:b", "ETH:a"], "n_members": 5, "symbols": ["BTC/USDT", "ETH/USDT"]}
    with pytest.raises(RuntimeError, match="input channels"):
        check_manifest(manifest, ["BTC:a", "BTC:b"], 5, ["BTC/USDT", "ETH/USDT"], "x.pt")


def test_member_count_mismatch_is_refused():
    manifest = {"channel_names": ["BTC:a"], "n_members": 5, "symbols": ["BTC/USDT"]}
    with pytest.raises(RuntimeError, match="ensemble members"):
        check_manifest(manifest, ["BTC:a"], 3, ["BTC/USDT"], "x.pt")


def test_symbol_mismatch_is_refused():
    manifest = {"channel_names": ["BTC:a"], "n_members": 1, "symbols": ["BTC/USDT", "ETH/USDT"]}
    with pytest.raises(RuntimeError, match="trained on coins"):
        check_manifest(manifest, ["BTC:a"], 1, ["BTC/USDT"], "x.pt")


def test_matching_manifest_passes():
    manifest = {"channel_names": ["BTC:a", "ETH:a"], "n_members": 1, "symbols": ["BTC/USDT", "ETH/USDT"]}
    check_manifest(manifest, ["BTC:a", "ETH:a"], 1, ["BTC/USDT", "ETH/USDT"], "x.pt")  # must not raise


def test_save_load_roundtrip_carries_manifest(tmp_path):
    """A manifest stored at save time can be read without applying the weights,
    which is what allows rejecting a checkpoint BEFORE it touches the live brain."""
    from flybrain.build import read_checkpoint_manifest

    cfg = _cfg()
    brain, sg, rm, _ = build_brain(cfg, synthetic=True)
    names = ["BTC:log_return_1", "BTC:rsi_14", "ETH:log_return_1", "ETH:rsi_14"]

    path = tmp_path / "best.pt"
    save_checkpoint(brain, path, {"epoch": 1}, manifest=checkpoint_manifest(cfg, brain, names))

    manifest = read_checkpoint_manifest(path)
    assert manifest["channel_names"] == names
    assert manifest["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert manifest["n_members"] == 1

    # and a full load still works
    brain2, *_ = build_brain(cfg, synthetic=True)
    info = load_checkpoint(brain2, path)
    assert info["epoch"] == 1
