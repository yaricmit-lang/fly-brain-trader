"""
Training cycles run before going live.

2026-09-16: of eight cycles evaluated, only 6 and 7 remain in the active
list. A review of the live loop showed they are the only two that carry an
input-channel manifest (channels.json / the passport embedded in the .pt) on
the current code and show a positive edge: cycle6_seasonal (mean_edge +0.301,
beats the formula in 5 of 6 blocks) and cycle7_dopamine (edge_sharpe +0.566,
the best of all eight). The others (1, 2, 3, 4, 5, 8) are either worse than
the formula, incompatible with the old channel set, or not ready for live use,
so they were removed to keep the dashboard from offering them as candidates.
Their run folders under runs.nosync/cycles/ are untouched; restoring a spec is
enough to bring one back.

Expectation check: NONE of the cycles currently shorts. The baseline formula
(kind: vol_target, trading/baseline.py) always yields non-negative weights,
and the fly's multiplier in exposure mode lies in [0, exposure_max], also
non-negative (see model.py::read_out; neutral 1.0 reproduces the formula
exactly). Cycles 6 and 7 differ not in position direction but in the training
PROTOCOL: 6 applies the Hebbian rule over 6 seasonal blocks of differing
volatility; 7 applies the same rule with an elevated dopamine dose on a single
window. Both are long-only/cash with different exposure aggressiveness, not
long/short.

Common to both: the baseline risk-targeting formula (volatility IC +0.61 is
predictable, price-direction IC ~0.01 is not), so the decision is not
"which way" but "how much".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CycleSpec:
    id: str
    name: str
    description: str
    cli_args: list[str] = field(default_factory=list)
    requires_teacher: bool = False   # the live engine must enable the same channels
    # Runtime estimate in gentle mode on a MacBook Air M1 (2 torch threads,
    # ensemble of 5, 3 epochs, 30 s pauses). Measured 2026-09-15: Hebbian
    # 52 ms/decision, REINFORCE 387 ms/decision; the teacher fits 162 models
    # on the first run (27 windows x 3 coins x 2 horizons), then uses the cache.
    est_minutes: int = 0
    est_note: str = ""
    config: str | None = None        # own yaml overlay on top of default (cycles 4, 5)
    recent_months: int | None = None # history window if not the shared one; for
                                      # val_start cycles: train_months BEFORE val
    epochs: int | None = None        # number of epochs if not the shared one
    val_start: str | None = None     # cycle on a specific calendar period
    val_end: str | None = None       # (not "last N months") - cycle 5
    select_by: str | None = None     # "return" | "sharpe" (train.py default)
    goal_return: float | None = None  # return target for result.json[hit_goal]
    entrypoint: str = "flybrain.train"  # cycle6 uses a different module


CYCLES: list[CycleSpec] = [
    CycleSpec(
        id="cycle6_seasonal",
        name="Cycle 6 - seasons (calm / active / medium)",
        description=(
            "6 non-overlapping 2-month periods from the last 2 years: "
            "the 2 calmest, the 2 most active and 2 medium by volatility. "
            "The goal is to earn as much as possible in exactly these; this "
            "is not a generalization test (unlike walk-forward) but a "
            "targeted adaptation to specific historical market regimes."
        ),
        entrypoint="flybrain.train_seasonal",
        est_minutes=18,
        est_note="6 blocks of 2 months = ~1 year of data per epoch, own script (not flybrain.train)",
    ),
    CycleSpec(
        id="cycle7_dopamine",
        name="Cycle 7 - dopamine: elevated dose and tolerance",
        description=(
            "Same data and window as cycle 1; only the dopamine protocol "
            "differs: a higher dose (learning step) than baseline with "
            "per-epoch tolerance, a longer eligibility trace, more probes "
            "and faster expectation adaptation. The \"greed\" / \"action "
            "bonus\" asymmetries were removed on 2026-09-16: measurements "
            "showed the EMA baseline absorbs their mean, so they carried "
            "no real signal, only noise."
        ),
        cli_args=["--rule", "hebbian"],
        config="config/dopamine.yaml",
        epochs=8,
        select_by="edge",
        est_minutes=25,
        est_note="same window as cycle 1 (14 mo.), 8 epochs, compare one-to-one with cycle 1",
    ),
]

CYCLE_BY_ID: dict[str, CycleSpec] = {c.id: c for c in CYCLES}

# window and training-volume parameters shared by all cycles
RECENT_MONTHS = 14
VAL_MONTHS = 2
# 2026-09-16: was 3. Measured: with 3 epochs weight_mean moved only
# 0.03-0.56% over the whole cycle (see training history of cycles 1/4/7); with
# the renormalized lr (config.py, HebbianCfg) and homeostasis (model.py) more
# passes are needed for the weights to actually converge rather than merely
# not having had time to drift from initialization.
EPOCHS = 6
MEMBERS = 5
PAUSE_SEC = 30      # between epochs; the config's 60 is a safety margin for a
                    # fully loaded core, here 2 threads at nice 19 need only half


def build_train_args(spec: CycleSpec, run_dir: str) -> list[str]:
    """
    Full argument list for the cycle subprocess. For most cycles this is
    `python -m flybrain.train ...`; a spec's entrypoint may point to another
    module (cycle 6: `flybrain.train_seasonal`) with a deliberately simpler
    CLI (no --recent-months/--rule/--val-months: there is no single window
    and no rule choice between two).
    """
    if spec.entrypoint != "flybrain.train":
        args = [
            "--epochs", str(spec.epochs or EPOCHS),
            "--members", str(MEMBERS),
            "--pause-sec", str(PAUSE_SEC),
            "--tag", spec.id,
            "--run-dir", run_dir,
        ]
        # 2026-09-16 (F2): cycle 6 used to be unable to take its own config
        # overlay: entrypoint != flybrain.train ignored spec.config entirely,
        # although train_seasonal.py has accepted --config from the start.
        if spec.config:
            args += ["--config", spec.config]
        # train_seasonal.py has its own defaults for these flags (per-block
        # edge / mean block return), so emit them only when the spec asks
        # explicitly; cycle 6 stays on the script's defaults
        if spec.select_by:
            args += ["--select-by", spec.select_by]
        if spec.goal_return is not None:
            args += ["--goal-return", str(spec.goal_return)]
        return args

    args = list(spec.cli_args)
    if spec.val_start:
        args += ["--val-start", spec.val_start, "--val-end", spec.val_end,
                  "--recent-months", str(spec.recent_months or RECENT_MONTHS)]
    else:
        args += ["--recent-months", str(spec.recent_months or RECENT_MONTHS),
                  "--val-months", str(VAL_MONTHS)]
    if spec.select_by:
        args += ["--select-by", spec.select_by]
    if spec.goal_return is not None:
        args += ["--goal-return", str(spec.goal_return)]
    args += [
        "--epochs", str(spec.epochs or EPOCHS),
        "--members", str(MEMBERS),
        "--pause-sec", str(PAUSE_SEC),
        "--tag", spec.id,
        "--run-dir", run_dir,
    ]
    if spec.config:
        args += ["--config", spec.config]
    return args
