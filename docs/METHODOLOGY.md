*Translated from the original Russian working notes; numbers and conclusions are unchanged.*

# Fly training methodology v2

> The original project plan (structure, stages 0-10, known pitfalls) lives in
> the repository as [DESIGN_LOG.md](DESIGN_LOG.md). This document is the plan for the next step:
> how to train the fly so that the result stops being "slightly better than
> random".
>
> Status note: this is a working document that mixes measured results with
> plans. Items that were planned but never built (the cloud hyper-parameter
> search `flybrain.search`, `pack_for_cloud.py`, the `flybrain.features.build`
> entry point) do not exist in the repository.

## Context

Done so far: the FlyWire v783 connectome is loaded, the mushroom-body subgraph
is extracted (6,283 neurons, 21,275 plastic KC→MBON connections), the market
data covers 3 coins × 1h bars since 2020, and there is a portfolio simulator,
benchmarks, two learning rules, and a live mode running on an internal paper
account. The first honest training run on the real brain put the fly at the
**71st percentile** of 100 random agents, with −36% versus −20% for
buy-and-hold on the 2025–2026 test period. Better than random, worse than the
market.

Design decisions for v2: fly + ML teacher; decisions every 4h, features at 1h;
shorts allowed; heavy searches in a free cloud (Kaggle/Colab); external data
are the fear and greed index, BTC dominance, and news via LLM sentiment; the
LLM acts as a sensor and a weekly analyst (no automatic edits); a mandatory
"connectome versus random graph" ablation.

Goal for the next 3–4 weeks: the fly consistently at or above the 75th
percentile of random agents in walk-forward evaluation, beating buy-and-hold at
least in bearish windows, and an honest answer to what the fly's topology
itself contributes. No profit is promised; the result is framed as "the fly as
a research agent", not as a strategy.

---

## Part 1. How the fly learned at the very beginning (v1, historical version)

> This section describes the **first** version of the brain, before all the
> measurements and rework below. It no longer works the way it is written here
> (3 coins, a linear reward on the whole portfolio, long-only). It is kept
> as is: an honest baseline from which the measurements in parts 1.5+ started.
> **How the fly thinks right now is in Part 3, at the very bottom of this
> document.**

One step = one hourly candle `t`.

1. **Features** (`market/features.py`): 10 per coin: log-returns over 1/4/24
   candles, z-score of price and volume over a 168 window, RSI14, the EMA12−EMA48
   difference, 24-candle volatility, sin/cos of the hour. Clip ±3 → tanh. In
   total 30 channels in [−0.9, 0.9]. Everything uses only the past (there is a
   leakage test).
2. **Input** (`connectome/roles.py`, `brain/model.py`): each channel →
   24 random projection neurons (PN), fixed seed, weights U(0.75, 1.25). The
   matrix `B` (6283×30) is sparse.
3. **Forward pass** (`FlyBrain.forward`): the state `a` is not reset between
   candles. 4 steps of `a ← 0.7·a + 0.3·tanh(W·a + B·x)`. `W`: 103,647
   edges, sign set by neurotransmitter (Dale's law), magnitude
   `log1p(syn)/Σincoming·2`.
4. **Exploration noise**: `N(0, 0.15)` is added to **all 6,283** neurons after
   the 4 steps (training only).
5. **Readout** (`read_out`): for a coin, `diff = mean(a[16 "approach" MBONs]) −
   mean(a[16 "avoid" MBONs])`; z-score using running statistics (decay 0.995);
   `raw = sigmoid(1.5·z)`; `frac = 0.85·prev + 0.15·raw`. The position
   fraction is in [0, 1], long-only.
6. **Execution** (`agent.py`, `trading/portfolio.py`): cap of 0.4 per coin,
   Σ ≤ 1; rebalance only if |Δ| ≥ 0.08; price is the close of the next
   candle; fee 0.05% + slippage 0.02%.
7. **Reward** (`trading/reward.py`): `r = log(eq_t/eq_{t−1}) / std_24(r)
   − 0.1·turnover`, clip ±5. **One number for the whole portfolio.**
8. **Plasticity A** (`plasticity_hebb.py`): trace `e ← 0.9·e +
   a_pre·(a_post − ā_post)` on KC→MBON only (ā is an EMA with 0.99). Surprise
   `δ = r − r̄` (EMA 0.99). `Δw = 0.002·δ·e`, clip [0, 5], the sign never
   changes. Updated on every candle.
9. **Training** (`train.py`): 3 consecutive epochs over 33,251 train candles
   (2020-08 … 2024-05); after each, a validation run (2024-06 … 2024-12)
   without noise and without learning; checkpoint by best val Sharpe;
   patience 2.
10. **Evaluation** (`backtest/run.py`): a single test run 2025-01 … 2026-09
    against buy-and-hold, an SMA crossover, and 100 random agents.

### Why it was weak, in descending order of harm

- **A. Signal-to-noise.** Hourly crypto returns are almost white noise; the
  reward spans 1 candle; the trace memory is about 10 candles (0.9^10 ≈ 0.35).
  The rule searches for a link "smell of the last 10 hours → next-hour
  return" that is practically absent from the data. This is the main reason.
- **B. No sparse code.** The mean |activity| of KC is 0.23, which is dense.
  In a real fly about 5% of KCs are active at once (the inhibitory APL neuron
  enforces this). A dense code means all MBONs see nearly the same thing →
  market-state patterns are not separated → the "state → reward" association
  cannot be built.
- **C. Noise in the wrong place.** Exploration noise on KC and PN corrupts the
  state representation itself, not just the action. The trace remembers a
  noisy "smell", so it is unclear who should receive credit.
- **D. Cross-coin credit assignment is broken.** All MBONs receive the same
  portfolio δ: BTC rose, SOL fell, and the "approach" MBONs of SOL get
  rewarded for BTC.
- **E. Long-only in a bearish test.** The ceiling of the result is "lose less
  than the market".
- **F. Features without an edge.** Only price derivatives; at 1h they have
  roughly zero predictive power. The fly knows nothing the market does not
  already know.
- **G. One split, one seed.** Choosing the checkpoint by val is itself a form
  of fitting; the spread between seeds is unknown; walk-forward is off.
- **H. Regime change.** Train is the mega-bullish 2020–2024 (SOL ×50), test is
  the bearish 2025–2026. The fly learned "buy dips" and applied it where dips
  are not bought back.

What was already done correctly and remains unchanged to this day: Dale's law,
plasticity only on KC→MBON, output smoothing, honest benchmarks with random
agents, a ban on looking into the future, and a separation of brain and
execution. Items A–D are addressed below (parts 1.5–1.9); E is lifted (shorts
are allowed); F, G, H are discussed in parts 1.5–1.8 and are a property of the
data, not a forgotten improvement.

---

## Part 1.5. What the measurements showed (2026-09-15)

Before improving the learning rule, I put a gradient-boosting model next to the
fly as a "teacher" and measured honestly whether the data contain any
predictability at all. The check is walk-forward: train on 18 months, test on
the next two, roll forward, 27 windows. The metric is the rank correlation of
the forecast with the outcome (IC). The significance threshold is 0.03.

### Price direction cannot be predicted

| Horizon | Mean IC | Sign hit rate | Conclusion |
|---|---:|---:|---|
| 12 hours | +0.011 | 50–51% | noise |
| 24 hours | −0.007 | 49–50% | noise |
| 1 week | −0.035 | 47–49% | noise |
| 2 weeks | +0.002 | 49–50% | noise |

76 price features plus the fear index give zero edge at every horizon from 12
hours to two weeks. **This is the ceiling of the task, not a shortcoming of the
fly.** No learning rule can extract a signal that is not there.

A trap I nearly fell into: on the full history at once, drawdown from the
monthly high showed an IC of +0.087 at two weeks, and the fear index +0.080.
It looked like a discovery. But at a 336-hour horizon neighbouring hourly
observations share 335/336 of the future window, so there are only 53,000 /
336 ≈ 158 independent points, and the standard error of the IC is about 0.08.
That means +0.087 is one sigma from zero. Walk-forward confirmed it: out of
sample the signal disappeared. Overlapping labels are the main way to fool
yourself on this kind of data.

### Volatility can be predicted, and very well

| Horizon | IC of past volatility with future volatility | Independent windows |
|---|---:|---:|
| 12 hours | **+0.612** | 4,421 |
| 24 hours | **+0.621** | 2,210 |
| 3 days | **+0.602** | 736 |
| 1 week | **+0.608** | 315 |

Sixty times stronger than anything about direction, and on thousands of
independent observations. This is a well-known market asymmetry: volatility
clustering is persistent, while direction is close to a random walk.

### This edge can be monetized

Simple risk targeting, with the capital fraction inversely proportional to
recent volatility and no direction prediction at all:

| Period | Sharpe: hold → risk target | Drawdown |
|---|---|---:|
| train 2020-08 … 2024-05 | 1.07 → **1.18** | −21.0 pp |
| val 2024-06 … 2024-12 | 0.38 → 0.24 | −5.4 pp |
| test 2025-01 … 2026-09 | −0.25 → **−0.13** | −12.8 pp |

Return in a bull market is lower (less time in the position during the
rally), but drawdown is consistently smaller in all three periods, and Sharpe
is better in two of three. This is a real, reproducible effect.

### What this means for the fly

The task changes from "predict where the price will go" (impossible) to
**"decide how much risk to take right now"** (possible and confirmed).

Biologically this is even more faithful. The mushroom-body output neurons do
not only say "approach or avoid"; they also modulate the strength of behaviour,
the readiness to act. A fly that decides "how energetically" rather than "in
which direction" is still a fly.

In practice:
- direction comes from a simple baseline rule (a long bias like buy-and-hold,
  or a slow trend); this is beta, and nobody promises to improve on it;
- the fly learns an exposure multiplier: when to pull back and when to go all
  out, looking at the whole market state at once;
- the reward is already normalized by volatility, so the fly honestly receives
  dopamine for stepping away from risk in bad periods;
- the reference for comparison is not buy-and-hold but static risk targeting.
  Beating it would mean the connectome learned something beyond the formula.

The research question does not change: the "connectome versus random graph"
ablation is just as meaningful on this task, and it now has a measurable goal
that can in principle be reached.

## Part 1.6. Outcome of the exposure mode (2026-09-15, 22 walk-forward windows)

The fly in exposure mode was run honestly: 22 windows, training on 12 months,
trading the next 2, an ensemble of 3 brains. The reference is the same
risk-targeting formula that sits underneath it, with the multiplier exactly
1.0.

| Metric | Value |
|---|---:|
| Beats the baseline on Sharpe | **45% of windows** |
| Median advantage over the baseline | **−0.06** |
| Median fly Sharpe | 1.22 (95% CI −1.32 … 2.73) |
| Median window return | +7.1% |
| Worst window | −24.9% |
| Beats buy-and-hold | 23% of windows, 40% in bearish ones |
| Mean percentile among random agents | 69 |

**Conclusion: the fly does not beat the formula.** A 45% win rate with a
slightly negative median advantage is zero. The Sharpe confidence interval
from −1.32 to +2.73 covers anything.

This is exactly what the IC measurements predicted. Direction is
unpredictable, so the exposure multiplier has nothing to build an advantage
on: it mixes noise into the formula. The good news is that it also does little
harm.

Separately, it is worth noting a methodological trap that I avoided only
because of the full run. On a sample of 6 windows the result was a 50% win rate
and an advantage of +0.09, which is easy to mistake for "almost works". On 22
windows it became 45% and −0.06. The difference between these two pictures is
pure window-sampling chance.

### What this means for the project

The trading result is exhausted on the available data: the ceiling is set by
the market, not by the architecture. There are exactly two honest paths
forward.

1. **Data with a documented edge.** Funding, open interest, long/short ratio,
   order-book imbalance, liquidations. These are positioning and
   microstructure signals, not price derivatives. The public Binance Futures
   API serves them without a key. This is the only thing that could actually
   move the ceiling.
2. **The research question.** Does Drosophila topology contribute anything?
   The data allow a confident answer here, because flies are compared with each
   other on the same task, not a fly with the market. This is the ablation.

Profit is no longer promised for the project in any formulation.

## Part 1.7. Ablation: does the fly's topology contribute anything? (2026-09-15)

The central research question of the project. Three graph variants, the same
data, the same training code, 6 walk-forward windows:

- **connectome** is the real FlyWire mushroom-body subgraph;
- **rewired** is an edge rewiring that preserves each neuron's degree (each
  neuron keeps the same number of connections, but who it connects to is
  random);
- **erdos** is an Erdős–Rényi random graph of the same density, with degrees
  not preserved at all.

| Graph | Median return | Median Sharpe | Mean percentile among random agents |
|---|--:|--:|--:|
| connectome | +13.8% | 2.23 | 74 |
| rewired | +14.9% | 2.50 | 79 |
| erdos | +13.8% | 2.20 | 74 |

The connectome does not merely fail to win; it is **worse** than both random
variants, though within noise (−4 percentiles from the best). The per-window
breakdown is more telling: the three graphs run practically neck and neck in
every single window.

| Window | connectome | rewired | erdos |
|---|--:|--:|--:|
| 1 | +26.0% | +28.9% | +24.8% |
| 2 | +29.6% | +23.4% | +33.1% |
| 3 | +4.2% | +6.5% | +2.9% |
| 4 | −1.1% | −1.7% | −0.1% |
| 5 | −21.2% | −18.2% | −16.5% |
| 6 | +23.4% | +24.6% | +24.7% |

The result in each window is determined by the market regime (the data are the
same for all three), not by the graph structure. The spread between
connectome/rewired/erdos within one window is an order of magnitude smaller
than the spread between windows.

**Honest answer: the topology of the Drosophila connectome gives no measurable
trading advantage over a random graph of the same size and density.** Neither
over the rewired one (same degrees, random partners) nor over the Erdős–Rényi
graph (no structure at all).

This does not mean that computing on a graph is useless in itself: all three
variants clearly beat random agents (percentile 74–79). Sparse coding,
inhibition via k-WTA, and reward-modulated plasticity work and give a
structure-agnostic advantage. But the **evolved wiring specific to Drosophila**,
which is what distinguishes the real connectome from a random graph with the
same parameters, played no role.

A plausible explanation: the fly's mushroom bodies are optimized for olfactory
associative memory (a sparse odour code, a few valence classes), not for
multidimensional numeric time series of a financial market. A structure that is
useful for one task need not transfer to another. This is an expected and
informative negative result, not a failed experiment.

## Part 1.8. Three cycles on fresh data and the "hard" cycle 4 (2026-09-15)

Three cycles (Hebbian, REINFORCE, Hebbian + teacher) on the last 14 months,
with val being the last 2 months (15.07–15.09.2026), the same window for all:

| Cycle | Fly Sharpe | Formula Sharpe | Δ |
|---|--:|--:|--:|
| 1 Hebbian | 3.41 | 3.49 | −0.08 |
| 2 REINFORCE | 3.51 | 3.49 | +0.02 |
| 3 Hebbian + teacher | 3.44 | 3.49 | −0.05 |

A spread of ±0.08 on identical data is noise; the fly sits on the formula. This
agrees with walk-forward on 22 windows (median Δ −0.06).

### Why the fly cannot beat the formula with the v2 reward

The v2 reward is `pos · r / vol`, linear in the position. The direction of `r`
is unpredictable (IC ≈ 0), so **E[reward] = pos · E[r]/vol ≈ 0 for any pos**.
On average the learning rule is indifferent to whether the fly took a lot of
risk or a little: the reward simply contains no signal about what to give
dopamine for. Meanwhile Sharpe grows precisely from risk timing (taking less
before a vol spike), which is the only thing in the data that is predictable
(vol IC +0.61). The fly could not learn what it was not rewarded for.

### Cycle 4: what was changed and why

1. **Reward = risk-penalized utility minus the formula's utility.**
   `U(A) = A − λ·A²`, `A = pos·r/vol`, λ = 0.5 (log utility to first order).
   The per-coin reward is `U(fly) − U(formula)`. The quadratic term: a large
   position before a large |r| is a penalty, a small one is a gain, for either
   sign of r. Subtracting the formula (a control variate): a multiplier of 1.0
   gives exactly zero, so common market noise is removed from the dopamine
   signal. Test: `tests/test_reward_v3.py`.
2. **Inputs about future vol** (`features/price.py`, v3): Parkinson and
   Garman-Klass over 24h (range estimators are 5–7 times more efficient than
   std on the same candles), the ratio of fast to slow vol, hour-of-week
   seasonality, both current and expected for the next 12h (for crypto,
   weekends and nights are quiet and the US session is loud; a formula with a
   168h window averages this out), a weekend flag, and **the weight of the
   formula itself** (`COIN:base_weight`), because the fly needs to know what it
   is scaling on top of. Everything is causal (test: "a shock in the last
   candle does not change the past").
3. **More data and epochs:** 36 months, 6 epochs, patience 3. Turnover penalty
   0.15 (the fly paid 1.5 times more in fees than the formula).

A separate config, `config/hard.yaml`, sits on top of default, so old
checkpoints (with a different number of channels) remain usable; the "Make
live" button for cycle 4 rebuilds the live brain from its config.

The honest check is not the val of a single window but walk-forward with the
same config:
`python -m flybrain.backtest.walk_forward --config config/hard.yaml`.
The result is below, once it finishes computing.

## Part 1.9. Five coins, aggressive defaults, per-coin fundamentals, goals 5-6 (2026-09-16)

After cycle 4, two more fundamentally different cycles were added (not "beat
the formula on Sharpe" but concrete targets), and the market was expanded.

### Coins: 3 → 5

`market.symbols`: BTC, ETH, SOL → + **PAXG** (gold, tokenized 1:1 to the spot
price) + **ZEC**. CRCL (the Circle stock on NYSE) was also considered and
rejected: `align()` in `market/fetch.py` intersects the hourly timestamps of
ALL coins, and the stock exchange trades on weekdays 9:30–16:00 ET, so adding
CRCL would cut the history of all five coins down to the NYSE hours. BTC stays
first in the list on purpose: `cross_coin_features` (`features/price.py`)
takes `coins[0]` as the market benchmark.

### Aggressive defaults, and their real cost

Two levers in `config/default.yaml` were deliberately raised (they take effect
immediately for all cycles and for the live brain):
- `trading.exposure_max`: 2.0 → **4.0** (ceiling on the fly's multiplier over
  the formula);
- `trading.baseline.target_daily_vol`: 0.02 → **0.04** (target daily risk of
  the formula itself).

The first run of cycle 5 ("100 → 300 in a year", see below) showed that
`exposure_max` decides almost nothing by itself: `guards.max_fraction_per_coin`
and `baseline.max_weight_per_coin` (both set to 0.40) cut the actual position
**before** the fly's multiplier, and the multiplier ran idle against that
ceiling. They were raised together to 0.75 → cycle 5 return did not grow but
dropped (+20% → +6.6%), which looks like training noise (Hebbian rule, 3
epochs) rather than a systematic effect of the limits. A rollback to the
midpoint, **0.50**, is the current value. This is an open question, not a
closed result: the relationship between the position ceiling and final return
on this data is non-monotonic and not fully explained.

Technically, no extra leverage arises in any case: `guards.py` separately
normalizes the sum of portfolio fractions to 1.0 if it exceeds it, because a
per-coin ceiling is not the same as a ceiling on the whole portfolio.

### Per-coin fundamental rules: as extra inputs, not separate strategies

Given an explicit choice between "let the fly decide how much to trust them"
and "the rules trade bypassing the fly", the first option was chosen.
`market.coin_features` (`config.py`, `features/store.py`) adds channels ON TOP
of the shared `market.features` list, for the specified coin only:

| Coin | Set | Meaning |
|---|---|---|
| ETH | `ELLIOTT_FEATURES` | causal zigzag (a pivot is confirmed only after a 3% reversal, with no look-ahead) + leg number + Fibonacci retracement of the current move relative to the previous one |
| BTC | `TREND_FEATURES` | R² of a linear regression of log-price on time (trend strength/cleanliness), EMA200 slope, position in the Donchian channel |
| PAXG (gold) | `MEAN_REVERSION_FEATURES` | z-score, Bollinger %B, return autocorrelation, range tightness, aimed at frequent small trades |
| ZEC | `LIQUIDITY_FEATURES` | Amihud illiquidity, volume z-score and trend. **Not on-chain data:** no honest free history for ZEC (share of shielded transactions, etc.) was found (see below), so this is a transparently labelled proxy |
| SOL | `external_features: sol_defi_tvl` | a genuine on-chain metric: Solana DeFi TVL, DefiLlama API, no key, full history |

For SOL and ZEC, an honest free source was found for SOL (DefiLlama). For ZEC
there was none: Zchain and Solana Beach were not responding at the time of the
search (520/522), and the others either have no free API or no history. The
decision was not to fake an "on-chain" signal but to use what can actually be
measured: liquidity.

Along the way a bug was found and fixed in `_causal_zigzag`: in the "direction
not yet decided" state, two consecutive `if` statements meant the second
overwrote the extremum set by the first, so a reversal was never confirmed and
the entire Elliott-wave signal was a constant zero. There is a dedicated
regression test for this bug (`tests/test_coin_fundamentals.py`).

### Cycle 5: "100 → 300 in a year"

Not "better than the formula on Sharpe" but a measurable goal: triple the
capital over a fixed calendar year (August 2025 to August 2026, entirely in the
past). Differences from cycles 1-4:
- `market/split.py::make_anchored_split` makes val cover specific dates, not
  "the last N months" and not dates from the config;
- `train.py --select-by return` selects the checkpoint by return on val, not
  by Sharpe (a deliberately aggressive choice: drawdown is not penalized when
  selecting the epoch);
- `config/cycle5_growth.yaml` is its own override (initially `exposure_max: 6`,
  now irrelevant because of the story with the limits above).

Results (val return of the fly versus the formula, same year):

| Run | Per-coin ceiling | Fly return | Formula return | ×3 target |
|---|--:|--:|--:|--:|
| 1 | 0.40 | +20.1% | −0.9% | no |
| 2 | 0.75 | +6.6% | −3.1% | no |

Both times the fly outperforms the formula (the formula itself is negative in
this year), but both times it is far from the ×3 target (+200%), by an order of
magnitude. Honest conclusion: `--select-by return` and a raised position
ceiling are necessary but insufficient conditions; a static capital fraction
with 4h decisions physically cannot triple capital in a year without a
destructive drawdown on this data. An open question for the next iteration.

### Cycle 6: "seasons"

6 non-overlapping 2-month periods from the last 2 years: the 2 calmest, the 2
most active, and 2 medium by volatility (classification: mean realized
volatility of the block, `market/periods.py`). The goal is return in precisely
these periods, not generalization.

The key engineering detail: `agent.run()` marks equity and computes the
forward return of the reward using **adjacent indices of the slice** it is
given. Concatenating 6 disjoint periods into one array and calling
`agent.run()` once is a silent bug: at the junction of any two periods the
price of one period is dragged into the reward of the other as if one hour had
passed. The solution is not to patch `agent.py` but to call it **once per
block** (`train_seasonal.py`, a new entrypoint, not `flybrain.train`):
`FlyBrain.reset()`/`Plasticity.reset()` clear only the current activity and the
eligibility trace, while the trained weights and the dopamine baseline survive
between calls. The regression test for the original bug is
`tests/test_seasonal_no_leakage.py` (it compares the raw equity curve when
concatenated against a continuous series).

## Part 2. Methodology v2

Principle: the mushroom bodies are an associative memory "world pattern →
good/bad", trained by dopamine. Give it (1) a clean sparse state code, (2) an
informative reward with the right horizon and per-coin credit, (3) inputs that
contain information, including the teacher's forecast, and (4) an honest
walk-forward evaluation plus an ablation.

### 2.1 Feature store (new package `src/flybrain/features/`)

- `store.py`: a DuckDB file `data.nosync/features.duckdb`. Table
  `features(ts, available_at, coin, name, value)`. Every join is as-of by
  `available_at` (what was actually known at the moment of the decision). A
  leakage test through `available_at` is mandatory.
- `price.py`: port of the current 10 features plus new ones from 1h candles,
  sampled at 4h boundaries: returns over 4h/12h/24h/72h/168h, volatility
  24/168, vol-of-vol, ATR/range, drawdown from the 30-day high, distance to
  SMA 7d/30d, RSI14 on 4h, and cross-coin features: BTC return as a feature for
  ETH/SOL, relative coin/BTC strength. Calendar: day of week, 4h slot.
- `external.py`: fear and greed index (alternative.me, no key, daily,
  `available_at` = the next 00:00 UTC, history since 2018); BTC dominance.
  CoinGecko `/global` gives only the current value and history on the free
  tier is truncated → **fallback**: BTC's share of market cap within a
  BTC/ETH/SOL basket via `market_chart` (check the depth of history during
  implementation; if insufficient, compute BTC's relative strength against the
  basket from prices, which already exists in `price.py`).
- `sentiment.py`: history comes from a Kaggle news dataset with dates (for
  example "Crypto News+"), live data from RSS feeds of CoinDesk/CoinTelegraph/
  Decrypt. Groq `llama-3.3-70b` scores each headline from −1 to +1 per
  mentioned coin and for the market as a whole; the aggregate over a 4h window
  is mean, count, max|·|. Cached in SQLite by headline hash (Groq limit 30 rpm /
  14.4k per day, so the history takes a few nights). The prompt contains no
  date; we ask for "tone", not "forecast". **Honesty caveat:** the LLM was
  trained on history up to 2025 and "knows" what happened to BTC, so the
  historical scores may leak. Therefore all results are reported in two
  variants: with sentiment and without.
- Normalization: a causal rolling 30-day z-score, then tanh (as now).
- (Out of scope, but one line for the future: funding / open interest /
  long-short ratio from the public Binance fapi without a key is the only
  external signal with a documented edge; it can be added in a day.)

### 2.2 Teacher (`src/flybrain/teacher/`)

- `train.py`: `sklearn.ensemble.HistGradientBoostingRegressor` (no new
  dependencies) on the feature table, forecasting forward return at 12h and 24h
  per coin, plus a sign classifier with probability. Walk-forward: retrain
  every 30 days on a rolling 18 months; purged CV with an embargo equal to the
  horizon so that overlapping labels do not leak.
- `predict.py`: an as-of forecast for any `ts` in the store.
- Teacher metrics are reported separately: IC (rank correlation of forecast
  with outcome), hit rate, and its own simple backtest. **If the teacher has no
  edge, it is visible immediately, and the fly cannot be blamed.** Significance
  threshold: IC > 0.03 at 12h.
- Teacher outputs → 3 channels per coin ("teacher's smell"): μ̂_12h, μ̂_24h,
  confidence. The fly learns *how much and when* to trust the teacher given the
  rest of the state, which is exactly the biological role of the MB (valence of
  a compound stimulus).

### 2.3 Brain v2 (`brain/model.py`)

- **Sparse KC code (k-WTA):** after the propagation steps, keep the top-k
  active KCs (k = 5% ≈ 260 of 5,177), set the rest to 0. This is the "APL".
  The most important change for pattern separation. Config `kc_sparsity`.
- **Noise on MBON only** (the "approach/avoid" groups), not on the whole
  network. Config `noise_target: mbon`.
- **Input normalization:** the vector `B·x` is brought to unit norm on every
  candle (lateral inhibition of the antennal lobe), so a strong channel does
  not drown out weak ones. Config `input_normalize`.
- **Output in [−1, 1]:** `2·sigmoid − 1` with `allow_short: true`. Smoothing
  and z-score remain.
- **4h decision step:** `agent.py` walks over 1h data, but decisions and
  rewards happen at 4h boundaries (stride 4). Between them the network state
  drifts and no orders are placed.
- **Ensemble** (`brain/ensemble.py`): N=5 brains with different seeds (input
  layout, MBON split, noise). The decision is the mean. 5×10 ms per candle is
  cheap.

### 2.4 Reward and credit (`trading/reward.py`, `plasticity_hebb.py`)

- **Per coin:** `r_c(t) = pos_c(t) · fwd_ret_c(t, H) / vol_c − fee_c(t)`.
  The MBONs of coin `c` receive their own `δ_c = r_c − r̄_c`; only the KC→MBON
  edges entering the MBONs of that coin are updated.
- **Horizon H = 3 decisions (12h)**, configurable. The reward for moment `t` is
  known at `t+H` and is applied then; `eligibility_decay` is chosen so that the
  trace of `t` is still alive: `γ^H ≈ 0.5 → γ ≈ 0.8`. Test: the reward for `t`
  uses prices ≤ `t+H` and is applied no earlier than `t+H`.
- The turnover penalty remains; advantage clip ±3.

### 2.5 Learning rules (all switchable by flag)

- **A. Hebbian v2**: as above. The biological showcase.
- **B. Actor-critic** (`plasticity_rl.py`): the critic is a linear readout from
  MBON → V̂; advantage = r − V̂. Removes the main variance of REINFORCE. Policy:
  Beta on [0,1] or a squashed Gaussian on [−1,1].
- **C. "Instinct"** (`brain/pretrain.py`): supervised pretraining of KC→MBON
  with the delta rule: "approach" ← sign(teacher μ̂ or actual forward return on
  train) > 0, "avoid" ← < 0. Several passes over the train window. Then A or B
  fine-tune. The fly starts not from zero but from "trend up → buy" from the
  original plan, only learned.

### 2.6 Validation (`backtest/walk_forward.py`, `backtest/ablation.py`)

- **Walk-forward:** train 12 months → test 2 months, step 2 months, from
  2022-01 to 2026-09 → ~25 windows. The teacher and the fly are retrained in
  every window. Report: Sharpe/return per window, the share of windows beating
  buy-and-hold, the share of windows above the 75th percentile of random
  agents; the aggregate is the median and a bootstrap CI.
- **Fee sensitivity:** 0.05 / 0.1 / 0.2%.
- **Ablation (the answer to the project's main question):** on the same data
  and seeds: {connectome; random degree-preserving graph (edge rewiring);
  Erdős–Rényi of the same density; MLP with the same number of parameters} ×
  {A; B; C+B} × {with teacher; without}. One table with CIs. This is the result
  that is guaranteed to be interesting regardless of returns.

### 2.7 Hyperparameters and cloud (`search.py`, `scripts/pack_for_cloud.py`,
`notebooks/cloud_run.ipynb`)

- Optuna over: lr, noise, `eligibility_decay`, `kc_sparsity`, steps, leak,
  `output_smoothing`, `output_gain`, horizon H, turnover penalty. The
  objective is the mean Sharpe over the **val** windows of walk-forward (not
  test).
- Budget: a 4h train window ≈ 2,190 candles × 10 ms ≈ 22 s/epoch/seed. A full
  walk-forward × 5 seeds ≈ 2.3 h per config, which is too much for Optuna.
  Therefore: search on 5 representative windows × 1 seed × 2 epochs ≈ 4 min
  per trial → 100 trials ≈ 7 h = one Kaggle session (30 h/week of CPU for
  free). The top-3 configs are then run in full.
- `pack_for_cloud.py`: zip `src/ config/ data.nosync/features.duckdb`
  (~50 MB) → Kaggle Dataset; `cloud_run.ipynb`: pip install → `python -m
  flybrain.search --trials 100` → results into a folder for download.
  Colab is a fallback with the same notebook.
- Locally, only short runs and only with the `--gentle` flag in `train.py`:
  `nice -n 19`, `torch.set_num_threads(2)`, a 60 s pause between epochs, and a
  stop when `pmset -g therm` reports throttling. This keeps runs safe on
  fanless laptops.

### 2.8 LLM analyst (`analyst.py`)

Once a week: reads the week's decisions and equity, the teacher's feature
importances, and the correlations of MBON activity with outcomes; asks Groq (or
Claude) to write a structured report: what worked, where it was wrong, market
regime, ideas. Markdown goes to `runs.nosync/…/analyst_<date>.md` and to the
dashboard. It applies nothing itself; a human reads the advice.

### 2.9 Order of work

**Week 1: the learning core.** k-WTA, noise on MBON only, per-coin reward with
a horizon, shorts, 4h step, ensemble, walk-forward evaluator, `--gentle`.
Milestone: the median window at or above the 75th percentile of random agents;
a win over buy-and-hold at least in bearish windows.

**Week 2: knowledge.** Feature store, fear/greed, news history from Kaggle +
Groq sentiment, teacher, teacher channels, "instinct", actor-critic.
Milestone: teacher IC > 0.03; fly+teacher better than the teacher alone on
Sharpe or drawdown.

**Week 3: science.** Cloud package, Optuna on Kaggle, the ablation grid.
Milestone: a "connectome vs random graph vs MLP" table with CIs.

**Week 4: going live.** Live paper trading at 4h with the best config, the
dashboard (stage 7 of the original plan), the weekly analyst. After that,
monthly refit via walk-forward and iterations driven by the analyst's reports.

### 2.10 Setup requirements

- A Kaggle account. To upload the package to the cloud from code you also need
  an API token: on kaggle.com → Settings → API → "Create New Token", and put
  the `kaggle.json` file in `~/.kaggle/`. Without it the package can be
  uploaded by hand through the website ("New Dataset" button), which is also a
  working path.
- A Groq key goes into `.env` as `GROQ_API_KEY`. The `.env` file must never be
  committed.
- A Google account for Colab as a fallback.

### Files

Modified: `brain/model.py` (k-WTA, noise on MBON, input normalization, output
[−1,1]), `brain/plasticity_hebb.py` (per-coin δ, delayed reward),
`brain/plasticity_rl.py` (critic), `trading/reward.py` (per coin, horizon),
`trading/portfolio.py` (shorts: negative units, no leverage), `agent.py` (4h
stride, delayed-reward buffer, ensemble), `train.py` (`--gentle`,
walk-forward), `config/default.yaml` (new sections `features`, `teacher`,
`walk_forward`, `ablation`, `search`), `config.py`.

Created: `features/{store,price,external,sentiment}.py`,
`teacher/{train,predict}.py`, `brain/{ensemble,pretrain}.py`,
`backtest/{walk_forward,ablation}.py`, `search.py`, `analyst.py`,
`scripts/pack_for_cloud.py`, `notebooks/cloud_run.ipynb`,
`config/experiments/ablation_*.yaml`.

Reused: `market/fetch.py` (candles), `market/features.py` (base features, which
move to `features/price.py`), `trading/portfolio.py`,
`backtest/{metrics,benchmarks,report}.py`, `build.py`, `housekeeping.py`,
`storage.py`.

### Verification

- Tests: KC sparsity level is exactly k; noise does not touch KC/PN; per-coin
  reward is isolated (shift only BTC → only the edges into BTC's MBONs change);
  delayed reward has no leakage; the store's as-of join returns no rows with
  `available_at > ts`; purged CV for the teacher; shorts create no leverage;
  the ensemble is deterministic given the seed.
- Commands: `python -m flybrain.features.build`, `python -m
  flybrain.teacher.train --report`, `python -m flybrain.backtest.walk_forward
  --config config/experiments/v2.yaml`, `python -m flybrain.backtest.ablation`,
  `python -m flybrain.search --trials 5 --dry` (local, gentle).
- Acceptance criterion for week 1: a walk-forward report with ≥20 windows, the
  fly at or above the 75th percentile of random agents in the median window,
  and not a single window with turnover above 30 trades/month per coin.
- Honest recording: all numbers go into `DESIGN_LOG.md` whatever they are, including
  the "teacher without an edge" and "connectome no better than a random graph"
  outcomes.

---

## Part 3. How the fly thinks and how the project is organized right now (2026-09-16)

A snapshot of the current state. If anything in parts 1–1.9 above contradicts
this section, this one is authoritative.

### One tick of thought, from input to trade

1. **Market** (`market/fetch.py`): hourly candles for 5 coins (BTC, ETH, SOL,
   PAXG (gold), ZEC) from Binance (fallback exchanges: Bybit/OKX/Kraken), from
   2020-08 (PAXG a bit later, from 2020-08-28, as it has no shorter history).
   Cached in parquet; `align()` intersects the timestamps of all coins.
2. **Features** (`features/store.py::build_inputs`): **~140 channels**, causal
   rolling z-score (30-day window) → tanh, range [-0.9, 0.9]:
   - a common set per coin (`market.features`, ~31 features): multi-scale
     returns, volatility (plain + Parkinson + Garman-Klass), hour-of-week
     seasonality, RSI, EMA spread, drawdown, distances to SMA, calendar;
   - "fundamental" extra channels for a specific coin
     (`market.coin_features`, part 1.9 above): Elliott waves (ETH), trend
     (BTC), mean reversion (gold), liquidity (ZEC);
   - cross-coin (`cross_coin_features`): BTC return as a feature for all,
     relative strength of a coin against BTC;
   - the weight of the risk formula itself as an input (`base_weight`);
   - external: fear and greed index (alternative.me), Solana DeFi TVL
     (DefiLlama), both as-of by publication date, so the future does not leak;
   - optionally the teacher's forecast (gradient boosting, cycle 3/hard):
     three channels per coin, out-of-fold.
3. **Brain input** (`connectome/roles.py`): each of the ~140 channels → 24
   random projection neurons (PN), seed fixed (or distinct for each ensemble
   member). Sparse matrix `B` (6283 × ~140).
4. **Forward pass** (`brain/model.py::FlyBrain.forward`), 4 steps of
   `a ← 0.7·a + 0.3·tanh(W·a + B·x)` over the FlyWire mushroom-body subgraph
   (6283 neurons, 103647 connections, 21275 of them plastic KC→MBON; Dale's
   law: the sign is fixed by neurotransmitter and only the magnitude changes).
   Sparse code: k-WTA keeps the top 5% of Kenyon cells active (≈259 of 5177)
   and zeroes the rest. Input gain control keeps the total "loudness" of the
   odour constant. The state `a` is not reset between ticks, which gives
   recurrence across hours.
5. **Readout** (`read_out`, `position_mode: exposure`): the fly does NOT choose
   a side. Direction and base size are set by the risk-targeting formula
   (`trading/baseline.py`, `target_daily_vol`, inversely proportional to
   one-week volatility), and the brain output is a multiplier on top of it
   (0 = go to cash, 1 = exactly the formula, up to `exposure_max` = more
   aggressive than the formula). This is the decision from the measurements in
   part 1.5: price direction is unpredictable (IC ≈ 0), volatility is
   predictable (IC +0.61), so the fly learns exactly what the data contain.
6. **Ensemble**: 5 independent brains (different seeds: own input layout, own
   MBON split); the decision is the mean.
7. **Guards** (`trading/guards.py`) before execution: a cap of 50% of capital
   per coin, sum of fractions ≤ 1.0 (no leverage), a trade only if the target
   has moved by ≥ 8%, no more than 6 trades/hour, a daily stop-loss of -10%
   (full stop until a manual reset), and the first N trades wait for a button
   press on the dashboard.
8. **Reward** (`trading/reward.py::CoinRewardTracker`): its own for each coin,
   with a horizon of 3 decisions (12h): `pos · fwd_return / vol`, applied no
   earlier than the price is actually known (a delayed queue of tokens).
   Optionally (cycle 4/hard, not the default) `A − λA²` minus the same utility
   for the formula: a risk penalty instead of a linear reward, which does not
   care how much risk is taken when direction is unpredictable.
9. **Learning**: the Hebbian reward-modulated rule (default) or REINFORCE, on
   KC→MBON edges. The live brain also learns while trading, with a small step
   (`hebbian.lr_live`), a daily limit on weight drift, and an automatic stop
   when it is exceeded.

### How it all runs and where to look

One process: `python -m flybrain.serve --adapter dry_run
--online-learning` starts both the trading loop (an asyncio task that "thinks"
every 2 seconds, trading only at 4h boundaries) and the web dashboard
(`server/app.py`, FastAPI + WebSocket at `http://127.0.0.1:8000`): a 3D brain
made of points (mushroom bodies glow, spontaneous spikes provide visible
activity, dopamine neurons flash green/red), a chart of prices and positions, a
decision feed, a training panel (weight drift, Train/Pause/Stop buttons), and a
terminal command for the log.

### The six training cycles (dashboard buttons, `src/flybrain/cycles.py`)

| # | id | Question | Rule/selection |
|---|---|---|---|
| 1 | `cycle1_hebbian` | the biological recipe on the freshest 14 months | hebbian, by Sharpe |
| 2 | `cycle2_reinforce` | a different algorithm, same data | REINFORCE, by Sharpe |
| 3 | `cycle3_teacher` | does the teacher's forecast help as one of the inputs | hebbian + teacher, by Sharpe |
| 4 | `cycle4_hard` | a reward for risk timing instead of a linear one | hebbian, config/hard.yaml, by Sharpe |
| 5 | `cycle5_growth` | triple the capital in a specific year | hebbian, --select-by return |
| 6 | `cycle6_seasonal` | maximum in 6 specific seasonal periods | hebbian, own entrypoint |

Honest summary of cycles 1-4 on 9 walk-forward windows (part 1.8): median
advantage over the formula **−0.12**, a win in 44% of windows. Over the long
run the fly is at parity with the formula, not better. Cycle 5 does not reach
the ×3 target (part 1.9). No checkpoint is currently "production" by
walk-forward result; the current live checkpoint (`train_live_final`) is
historical, trained before the five-coin expansion, and is kept as a working
demonstration of the pipeline, not as a validated strategy.

### Repository structure

| Path | Contents |
|---|---|
| `config/` | `default.yaml` is the entire configuration; `hard.yaml`, `cycle5_growth.yaml` are overrides on top of default for individual cycles |
| `src/flybrain/connectome/` | Loading the FlyWire connectome, the mushroom-body subgraph, neuron roles |
| `src/flybrain/brain/` | `model.py` (network), `plasticity_hebb.py`/`plasticity_rl.py` (rules), `ensemble.py` |
| `src/flybrain/features/` | `price.py` (price + per-coin fundamentals), `external.py` (fear/greed, TVL), `store.py` (assembly + cache) |
| `src/flybrain/market/` | Candles (`fetch.py`), time split (`split.py`), seasonal blocks (`periods.py`) |
| `src/flybrain/trading/` | Portfolio, reward, guards, execution, baseline position formula |
| `src/flybrain/teacher/` | Gradient boosting alongside the fly, out-of-fold |
| `src/flybrain/backtest/` | Run, metrics, benchmarks, walk-forward, ablation |
| `src/flybrain/train.py`, `train_seasonal.py`, `cycles.py`, `live_engine.py`, `serve.py` | Training, live loop, dashboard cycle registry |
| `server/` `web/` | FastAPI server and HTML/three.js dashboard |
| `data.nosync/` `runs.nosync/` | Data and run results, not tracked in git (the `.nosync` suffix is a naming convention that keeps file-sync services from syncing them) |

### How to run it yourself

```bash
# live brain + dashboard (paper account, no real money)
python -m flybrain.serve --adapter dry_run --online-learning --auto-approve
# open http://127.0.0.1:8000, cycles are started with buttons there

# the same training from the terminal, without the dashboard
python -m flybrain.train --recent-months 14 --rule hebbian

# an honest check beyond a single window
python -m flybrain.backtest.walk_forward --config config/hard.yaml
```
