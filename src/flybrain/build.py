"""
Fly assembly: graph -> subgraph -> roles -> model.

A separate module because training, backtest, live and the dashboard
server all build the brain the same way.
"""

from __future__ import annotations

from pathlib import Path

from .brain.model import FlyBrain
from .brain.plasticity import Plasticity
from .brain.plasticity_rl import make_plasticity
from .config import Config, load_config
from .connectome.loader import Connectome, load_connectome
from .connectome.roles import RoleMap, assign
from .connectome.subgraph import Subgraph, extract


def load_graph(cfg: Config, synthetic: bool = False) -> Connectome:
    """The real connectome, or a synthetic stand-in for smoke-testing the code."""
    if synthetic:
        path = cfg.path("connectome") / "graph_synthetic.npz"
        if not path.exists():
            raise FileNotFoundError(
                "synthetic connectome not found. Build it with: "
                "python scripts/synthetic_connectome.py"
            )
        return Connectome.load(path)
    return load_connectome(cfg)


def build_brain(
    cfg: Config | None = None,
    synthetic: bool = False,
    with_plasticity: bool = False,
    channel_names: list[str] | None = None,
) -> tuple[FlyBrain, Subgraph, RoleMap, Plasticity | None]:
    """Full build from files to a ready network."""
    cfg = cfg or load_config()
    conn = load_graph(cfg, synthetic=synthetic)
    sg = extract(cfg, conn)
    rm = assign(sg, cfg, channel_names=channel_names)
    brain = FlyBrain(sg, rm, cfg)
    plast = make_plasticity(brain, cfg) if with_plasticity else None
    return brain, sg, rm, plast


def build_ensemble(
    cfg: Config | None = None,
    synthetic: bool = False,
    with_plasticity: bool = False,
    n_members: int | None = None,
    channel_names: list[str] | None = None,
):
    """
    Ensemble of n brains with seeds seed, seed+1, ...; each has its own
    input layout and MBON split. n_members=1 returns a plain FlyBrain, so
    the training code is the same either way.
    """
    from .brain.ensemble import EnsembleBrain, EnsemblePlasticity  # noqa: PLC0415

    cfg = cfg or load_config()
    n = n_members if n_members is not None else cfg.training.ensemble.n_members
    conn = load_graph(cfg, synthetic=synthetic)
    sg = extract(cfg, conn)

    members: list[FlyBrain] = []
    rules: list[Plasticity] = []
    for i in range(max(1, n)):
        seed = cfg.project.seed + i
        rm = assign(sg, cfg, seed=seed, channel_names=channel_names)
        brain = FlyBrain(sg, rm, cfg, seed=seed)
        members.append(brain)
        if with_plasticity:
            rules.append(make_plasticity(brain, cfg))

    if n <= 1:
        return members[0], sg, members[0].rm, (rules[0] if rules else None)

    ens = EnsembleBrain(members)
    plast = EnsemblePlasticity(ens, rules) if with_plasticity else None
    return ens, sg, members[0].rm, plast


CHECKPOINT_SCHEMA = 2


def checkpoint_manifest(cfg: Config, brain, channel_names: list[str] | None) -> dict:
    """
    Checkpoint manifest: the input layout the checkpoint was trained on.

    Stored INSIDE the .pt (2026-09-16). Previously the only safeguard was a
    separate channels.json next to the file, which got lost along with the
    folder, and the checkpoint then loaded silently. Silent loading is
    dangerous: state_dict stores w_mag per connectome EDGE, which does not
    depend on the number of input channels, so weights from a 76-channel
    brain load cleanly into a 190-channel one. What breaks is the mapping of
    channels onto projection neurons (roles.py draws it from an rng seeded
    by n_channels), so the brain would read RSI where it was trained to
    read volume.
    """
    return {
        "schema": CHECKPOINT_SCHEMA,
        "channel_names": list(channel_names) if channel_names is not None else None,
        "n_channels": len(channel_names) if channel_names is not None else None,
        "symbols": list(cfg.market.symbols),
        "n_coins": len(cfg.market.symbols),
        "seed": cfg.project.seed,
        "n_members": len(getattr(brain, "members", [brain])),
    }


def check_manifest(manifest: dict | None, names: list[str], n_members: int,
                   symbols: list[str], where: str) -> None:
    """
    Compare the manifest with what the brain is built for right now. Raises
    RuntimeError with a readable explanation; a mismatch must never pass
    silently (see the docstring above).
    """
    if not manifest:
        raise RuntimeError(
            f"checkpoint {where} has no channel manifest. Such checkpoints were "
            f"trained before 2026-09-16 and most likely for a different input "
            f"set; there is nothing to verify against, and a silent load would "
            f"mix up inputs and outputs. Retrain the cycle with the current config."
        )
    got = manifest.get("channel_names")
    if got is not None and list(got) != list(names):
        raise RuntimeError(
            f"checkpoint {where} was trained on {len(got)} input channels, but the "
            f"current config yields {len(names)}. The input layout would not match; "
            f"retrain the cycle or restore the previous market.features."
        )
    if manifest.get("n_members") not in (None, n_members):
        raise RuntimeError(
            f"checkpoint {where} was trained with {manifest['n_members']} ensemble members, "
            f"but {n_members} are built now (training.ensemble.n_members)."
        )
    if manifest.get("symbols") is not None and list(manifest["symbols"]) != list(symbols):
        raise RuntimeError(
            f"checkpoint {where} was trained on coins {manifest['symbols']}, "
            f"but market.symbols is now {symbols}. The coin-to-output-neuron "
            f"mapping would not match."
        )


def save_checkpoint(brain, path: Path, extra: dict | None = None,
                    manifest: dict | None = None) -> None:
    import torch  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"brain": brain.state_dict(), **(extra or {})}
    if manifest is not None:
        payload["manifest"] = manifest
    torch.save(payload, path)


def load_checkpoint(brain: FlyBrain, path: Path) -> dict:
    import torch  # noqa: PLC0415

    d = torch.load(path, weights_only=False)
    brain.load_state_dict(d["brain"])
    return d


def read_checkpoint_manifest(path: Path) -> dict | None:
    """
    Peek at the checkpoint manifest WITHOUT applying weights, so an
    incompatible checkpoint can be rejected before it touches the live brain
    (live_engine.warmup checks BEFORE load_checkpoint).
    """
    import torch  # noqa: PLC0415

    d = torch.load(path, weights_only=False)
    return d.get("manifest")
