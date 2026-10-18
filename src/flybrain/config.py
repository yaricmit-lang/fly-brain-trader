"""
Config schema and loader.

The whole project reads its settings from here only. No magic numbers in
code: anything worth tuning belongs in config/*.yaml. Pydantic catches typos
in the YAML at startup rather than three hours into a training run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
#  Config sections
# --------------------------------------------------------------------------
class ProjectCfg(BaseModel):
    name: str = "fly-brain-trader"
    seed: int = 42
    device: str = "cpu"


class PathsCfg(BaseModel):
    data: str = "data.nosync"
    runs: str = "runs.nosync"
    connectome: str = "data.nosync/connectome"
    market: str = "data.nosync/market"
    skeletons: str = "data.nosync/skeletons"


class HousekeepingCfg(BaseModel):
    runs_budget_gb: float = 5.0
    keep_checkpoints: int = 3
    keep_raw_connectome: bool = False
    storage_backend: Literal["local", "hf", "s3"] = "local"


class SubgraphCfg(BaseModel):
    mode: Literal["mushroom_body", "neuropil", "threshold_only"] = "mushroom_body"
    neuropils: list[str] = Field(default_factory=list)
    cell_class_patterns: dict[str, str] = Field(default_factory=dict)
    sides: list[str] = Field(default_factory=lambda: ["left", "right"])


class ConnectomeCfg(BaseModel):
    version: str = "783"
    min_syn_count: int = 5
    subgraph: SubgraphCfg = Field(default_factory=SubgraphCfg)
    neurotransmitter_sign: dict[str, float] = Field(default_factory=dict)


class ModelCfg(BaseModel):
    steps: int = 4
    leak: float = 0.3
    nonlinearity: Literal["tanh", "relu"] = "tanh"
    recurrent_across_bars: bool = True
    weight_init: str = "log_syn_count"
    input_neurons_per_channel: int = 24
    output_gain: float = 1.5
    input_scale: float = 2.0
    adaptive_readout: bool = True
    readout_decay: float = 0.97
    output_smoothing: float = 0.6
    dale_law: bool = True
    plastic_edges: Literal["kc_to_mbon", "all"] = "kc_to_mbon"
    w_max: float = 20.0
    normalize_incoming: bool = True
    # methodology v2
    kc_sparsity: float = 0.05
    noise_target: Literal["mbon", "all"] = "mbon"
    input_normalize: bool = True
    input_drive_rms: float = 0.5
    kc_output_gain: float = 7.0
    # --- homeostasis (2026-09-16): without it, a learning rate large enough
    # to actually move the weights would inflate each MBON's total incoming
    # weight without bound. After every learn() call the summed plastic
    # incoming weight of each MBON is rescaled back to its initial sum, so
    # what is learned is the DISTRIBUTION of weight across KCs, not the
    # overall scale. w_min_frac is a floor: a synapse at zero does not die
    # permanently (it can regrow at the next rescaling).
    homeostatic: bool = True
    # The weight floor is a FRACTION of the mean initial plastic weight, not
    # an absolute value: on the real connectome the KC->MBON weights reach
    # ~14 after kc_output_gain, while on the toy graph (--synthetic) the
    # whole range is below 0.5. An absolute floor there would lift every
    # weight at once and break the per-coin addressing of dopamine.
    w_min_frac: float = 0.05


class HebbianCfg(BaseModel):
    # 2026-09-16: was 0.002. Measured over all completed cycles,
    # weight_mean moved by only 0.03-0.56% over the ENTIRE training cycle
    # (history.json), i.e. the dopamine signal clip(r - EMA(r)) with E~0
    # produced a random walk rather than learning. The final lr is to be
    # chosen by a walk-forward sweep (see the plan, verification step 4);
    # 0.05 is the starting point of that sweep.
    lr: float = 0.05
    lr_live: float = 0.0002
    eligibility_decay: float = 0.8
    exploration_noise: float = 0.15
    reward_baseline_decay: float = 0.99
    # "Dopamine protocol" (cycle 7): per-epoch dose and noise. 1.0 = unchanged.
    # <1 models tolerance: the first epoch runs on a large dose, then it is
    # reduced so the weights do not blow up once the coarse structure has
    # already been learned.
    lr_epoch_decay: float = 1.0
    noise_epoch_decay: float = 1.0


class EnsembleCfg(BaseModel):
    n_members: int = 5


class GentleCfg(BaseModel):
    enabled: bool = True
    torch_threads: int = 4
    pause_between_epochs_sec: int = 60
    check_thermal: bool = True


class ReinforceCfg(BaseModel):
    lr: float = 0.0005
    # Same caution factor as hebbian.lr_live (0.0002/0.002 = 0.1): the reward
    # arrives once per reward.horizon_hours so there is no hurry, and a
    # mistake costs more in live trading, where there is no data to retrain on.
    lr_live: float = 0.00005
    entropy_bonus: float = 0.01
    reward_baseline_decay: float = 0.99


class TrainingCfg(BaseModel):
    rule: Literal["hebbian", "reinforce"] = "hebbian"
    epochs: int = 6
    early_stopping_patience: int = 3
    live_drift_limit: float = 0.05
    # The decision step that hebbian.lr / reinforce.lr are calibrated for.
    # 2026-09-16: lr used to be defined PER DECISION, so with
    # decision_timeframe=1h (cycle 5) the same history produced four times as
    # many weight updates at the same step size: per unit of real time the
    # brain silently retrained four times as hard. Measured: cycle 5 rewrote
    # 96% of the weights in 3 epochs versus 100% in 8 epochs for cycle 7, and
    # turnover rose to 37 versus 2.5 for the baseline formula. The effective
    # step is now multiplied by decision_stride / reference_stride, so lr
    # means "step per 4 hours of history" regardless of decision frequency.
    lr_reference_timeframe: str = "4h"
    # Warm-up of the readout calibration (_ro_mean/_ro_var/_frac_ema) before
    # every evaluated/traded segment: without it the first ~30 decisions after
    # brain.reset() are lost in a sigmoid saturated by var=1e-4, which is
    # ~27% of a 2-month validation window. Hours BEFORE the segment starts;
    # they do not count toward its metrics.
    warmup_hours: int = 240
    ensemble: EnsembleCfg = Field(default_factory=EnsembleCfg)
    gentle: GentleCfg = Field(default_factory=GentleCfg)
    hebbian: HebbianCfg = Field(default_factory=HebbianCfg)
    reinforce: ReinforceCfg = Field(default_factory=ReinforceCfg)


class SplitCfg(BaseModel):
    train_end: str
    val_end: str


class WalkForwardCfg(BaseModel):
    enabled: bool = False
    refit_every_days: int = 90


class AdaptiveStrideCfg(BaseModel):
    """
    Bounded-adaptive decision cadence: instead of a single constant
    decision_timeframe, pick from a small fixed set according to the
    volatility regime. This is not "an LLM decides when to think"
    (unverifiable, and contradicted by what the project measured: price
    direction has IC~0.01, i.e. is unpredictable). It is a switch between
    1h/4h driven by realized_vol, a quantity the project already knows how
    to measure and already uses (trading/baseline.py, IC +0.61).

    OFF by default: the mechanism has not been validated walk-forward on
    historical data (see the plan "Bounded-adaptive decision cadence");
    enable it only after comparing against fixed 1h and 4h.
    """
    enabled: bool = False
    calm_timeframe: str = "4h"
    active_timeframe: str = "1h"
    # Realized volatility (see trading.baseline.realized_vol) with hysteresis
    # to avoid chattering at the boundary: switch to the active regime only at
    # vol_high, and back to the calm regime only once it falls below
    # vol_low < vol_high.
    vol_high: float = 0.06
    vol_low: float = 0.04
    # Minimum time in a regime before switching again is allowed; without
    # it, noisy volatility near the threshold could change the cadence every hour.
    min_dwell_hours: float = 4.0


class MarketCfg(BaseModel):
    exchange: str = "binance"
    fallback_exchanges: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    timeframe: str = "1h"
    decision_timeframe: str = "4h"
    since: str = "2020-08-01"
    features: list[str] = Field(default_factory=list)
    cross_coin: bool = True
    baseline_feature: bool = False    # COIN:base_weight: formula weight as an input (v3)
    coin_features: dict[str, list[str]] = Field(default_factory=dict)
    coin_feature_boost: float = 1.0   # gain of the fundamental channels, see features/store.py
    # Extra per-coin features ON TOP of the shared `features` list:
    # "fundamental" sets (Elliott waves for ETH, trend for BTC,
    # mean reversion for gold), see features/price.py
    external_features: list[str] = Field(default_factory=list)
    norm_window_hours: int = 720
    # Channels that are NOT rolling z-scored (clipped to +-1 only): the brain
    # must see their ABSOLUTE level rather than "relative to its own 30-day
    # mean". Chiefly base_weight: otherwise the fly cannot tell that the
    # formula has already hit its ceiling.
    # elliott_dir is deliberately NOT included: it is a categorical flag
    # {-1,0,1}, and coin_feature_boost becomes a no-op on it (boost*(+-1) is
    # clipped back to the same +-1), which would defeat the point of the boost.
    raw_features: list[str] = Field(default_factory=lambda: [
        "base_weight", "weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ])
    split: SplitCfg
    walk_forward: WalkForwardCfg = Field(default_factory=WalkForwardCfg)
    adaptive_stride: AdaptiveStrideCfg = Field(default_factory=AdaptiveStrideCfg)


class GuardsCfg(BaseModel):
    max_fraction_per_coin: float = 0.40
    min_position_change: float = 0.08
    max_trades_per_hour: int = 6
    daily_drawdown_stop: float = 0.10
    manual_approval_first_n: int = 5


class ExecutorCfg(BaseModel):
    adapter: Literal["dry_run", "paper", "binance_futures_testnet", "bybit_demo"] = "dry_run"
    testnet_base_url: str = "https://testnet.binancefuture.com"


class BaselineCfg(BaseModel):
    kind: Literal["vol_target", "vol_trend", "equal"] = "vol_target"
    vol_window: int = 168
    target_daily_vol: float = 0.02
    max_weight_per_coin: float = 0.40
    trend_window: int = 720
    # Cap on the SUM of the formula's |weights| (2026-09-16): without it, after
    # target_daily_vol/max_weight_per_coin were raised, the formula's gross
    # exposure regularly exceeded 1.0 and Portfolio.clip_target divided the
    # whole vector back by the sum; the fly's multiplier shrank by the same
    # factor and stopped changing anything. The formula now uses at most
    # max_gross of capital, leaving the fly real headroom to go to cash or
    # double up to 100%.
    max_gross: float = 0.5


class TradingCfg(BaseModel):
    initial_capital: float = 10_000.0
    allow_short: bool = True
    position_mode: Literal["exposure", "direction"] = "exposure"
    exposure_max: float = 2.0
    baseline: BaselineCfg = Field(default_factory=BaselineCfg)
    leverage: float = 1.0
    fee_rate: float = 0.0005
    slippage: float = 0.0002
    guards: GuardsCfg = Field(default_factory=GuardsCfg)
    executor: ExecutorCfg = Field(default_factory=ExecutorCfg)


class RewardCfg(BaseModel):
    kind: str = "vol_scaled_net_return"
    vol_window: int = 24
    baseline_decay: float = 0.99
    # Legacy turnover penalty: a quantity in reward units, not tied to the
    # real fee (0.1 against fee+slippage of 0.07% is a penalty hundreds of
    # times larger than the actual trade cost). Kept with a default of 0.0 for
    # the tests; the REAL cost is now always computed in CoinRewardTracker
    # from trading.fee_rate/slippage.
    turnover_penalty: float = 0.0
    per_coin: bool = True
    # Reward horizon IN HOURS (it used to be in decision steps of
    # decision_timeframe, so cycle 5 with decision_timeframe=1h silently got a
    # 3h horizon instead of 12h). It is now independent of decision_timeframe.
    horizon_hours: int = 12
    # --- v3 (hard cycle) ---
    # risk_aversion > 0: reward = A - lambda*A^2, where A is return/vol. A
    # linear reward with E[r]=0 cannot distinguish "take more risk" from
    # "take less"; on average it is indifferent, yet Sharpe improves precisely
    # through risk timing.
    risk_aversion: float = 0.0
    # "Dopamine protocol" (cycle 7):
    # gain_scale/loss_scale: dopamine asymmetry, where profit feels brighter
    #   than the pain of a loss (greed). Symmetric 1/1 = unchanged. To be
    #   honest: with E[r]=0 a convex reward pushes toward maximum exposure;
    #   that is the intended "risk addiction".
    # activity_bonus: dopamine for the act of acting itself, |position - formula|.
    #   With relative_to_baseline, "sitting at multiplier 1.0" yields exactly
    #   zero dopamine, a lazy fixed point; the bonus breaks it.
    gain_scale: float = 1.0
    loss_scale: float = 1.0
    activity_bonus: float = 0.0
    # relative_to_baseline: the same utility computed for the formula's
    # position (multiplier 1.0) is subtracted from the reward. Common market
    # noise cancels out, leaving only the fly's contribution.
    relative_to_baseline: bool = False


class VizCfg(BaseModel):
    record_activations: bool = True
    record_every_n_bars: int = 50
    ws_fps: int = 10
    server_host: str = "127.0.0.1"
    server_port: int = 8000


class TeacherCfg(BaseModel):
    enabled: bool = False
    horizons_hours: list[int] = Field(default_factory=lambda: [12, 24])
    train_months: int = 18
    test_months: int = 2
    max_iter: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    min_samples_leaf: int = 200
    l2: float = 1.0
    min_ic: float = 0.03


class SentimentCfg(BaseModel):
    enabled: bool = False
    provider: str = "groq"
    model: str = "openai/gpt-oss-20b"
    batch_size: int = 20
    cache: bool = True


class AnalystCfg(BaseModel):
    enabled: bool = False
    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 4000


class Config(BaseModel):
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    housekeeping: HousekeepingCfg = Field(default_factory=HousekeepingCfg)
    connectome: ConnectomeCfg = Field(default_factory=ConnectomeCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    training: TrainingCfg = Field(default_factory=TrainingCfg)
    market: MarketCfg
    trading: TradingCfg = Field(default_factory=TradingCfg)
    reward: RewardCfg = Field(default_factory=RewardCfg)
    viz: VizCfg = Field(default_factory=VizCfg)
    teacher: TeacherCfg = Field(default_factory=TeacherCfg)
    sentiment: SentimentCfg = Field(default_factory=SentimentCfg)
    analyst: AnalystCfg = Field(default_factory=AnalystCfg)

    # --- convenience derived values ---
    @property
    def root(self) -> Path:
        """Project root (the directory that contains config/)."""
        return project_root()

    def path(self, key: str) -> Path:
        """Absolute path for a key from the paths section; creates the directory."""
        p = self.root / getattr(self.paths, key)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def n_channels(self) -> int:
        """Total number of input channels: features x coins."""
        return len(self.market.features) * len(self.market.symbols)

    @property
    def decision_stride(self) -> int:
        """Number of feature candles between two fly decisions (4h / 1h = 4)."""
        from .market.fetch import TF_MS  # noqa: PLC0415

        step = TF_MS[self.market.decision_timeframe] // TF_MS[self.market.timeframe]
        return max(1, int(step))

    @property
    def lr_stride_scale(self) -> float:
        """
        Learning-step correction for the actual decision frequency.

        The lr in the config means the step per ONE decision at the frequency
        training.lr_reference_timeframe (4h by default). If the fly decides
        more often, there are proportionally more decisions over the same
        history, and without this correction the total weight movement per
        unit of real time grows in proportion to the frequency (see cycle 5).

        Decision frequency adds no independent observations to the reward:
        with a 12h horizon and a 1h step, 12 consecutive decisions share
        nearly the same future window, so their dopamine is nearly identical.
        The right invariance is therefore over time, not over the number of
        decisions.
        """
        from .market.fetch import TF_MS  # noqa: PLC0415

        ref = TF_MS[self.training.lr_reference_timeframe] // TF_MS[self.market.timeframe]
        return self.decision_stride / max(1, int(ref))

    def lr_stride_scale_for_hours(self, elapsed_hours: float) -> float:
        """
        Same as lr_stride_scale, but based on the ACTUAL time between two
        decisions rather than the run's static decision_stride.

        Needed for the bounded-adaptive cadence (market.adaptive_stride):
        there decision_timeframe is not a per-run constant, and "step per 4h
        of history" must be computed from the real interval between the two
        specific decisions rather than from a single config value.
        """
        from .market.fetch import TF_MS  # noqa: PLC0415

        ref_hours = TF_MS[self.training.lr_reference_timeframe] / 3_600_000.0
        return elapsed_hours / max(1e-9, ref_hours)


# --------------------------------------------------------------------------
#  Loading
# --------------------------------------------------------------------------
def project_root() -> Path:
    """Walk up from this file to find the project root (where config/ lives)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "default.yaml").exists():
            return parent
    return here.parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay `override` on `base` (for config/experiments/*.yaml)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> Config:
    """
    Read config/default.yaml and, if given, overlay an experiment file on top.

    Example:
        cfg = load_config()                                  # defaults only
        cfg = load_config("config/experiments/rl.yaml")      # defaults + override
    """
    root = project_root()
    with open(root / "config" / "default.yaml", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = root / p
        with open(p, encoding="utf-8") as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})

    return Config(**data)


def env(name: str, default: str = "") -> str:
    """Fetch a secret from .env / the environment. Secrets NEVER live in YAML."""
    root = project_root()
    dotenv = root / ".env"
    if dotenv.exists() and name not in os.environ:
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
    return os.environ.get(name, default)
