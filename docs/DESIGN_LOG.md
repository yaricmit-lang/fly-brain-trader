# FlyBrain Trader: design log

> This document describes the **actual** architecture and the decisions that
> were made, not the original draft. The initial plan diverged considerably
> from what was built (dashboard, exchanges, number of coins). The training
> methodology, the honest measurements, and "how the fly thinks right now" are
> in [METHODOLOGY.md](METHODOLOGY.md), especially Part 3 at the end. This file
> covers the engineering structure, the decisions taken, and the pitfalls hit
> along the way.

## Context and goal

A research project: the connectome of the *Drosophila* brain (FlyWire FAFB
v783) becomes a trainable network on the real synaptic topology, and trading
decisions are produced through reward modulation (dopamine). Goals, in order
of priority:

1. **Research.** Give an honest answer to whether the connectome's structure,
   or reward-based learning, helps on a task the brain did not evolve for.
2. **Live demonstration.** The fly "thinks" and trades on a paper account,
   with a 3D visualization of the brain.

No profit is promised in any form. The honest outcome of the measurements
(METHODOLOGY.md, Parts 1.5-1.9): price direction is unpredictable on this data,
and over long horizons the fly is on par with a simple risk-targeting formula;
an advantage was never confirmed. The value of the project is as a research
and demo testbed and for risk management, not as a source of alpha.

## Decisions (current, not draft)

| Topic | How it is actually done |
|---|---|
| Environment | Python 3.14, torch (CPU, plain `torch.sparse`, no torch-geometric). Heavy data lives in `data.nosync/` and `runs.nosync/`, which are git-ignored (the `.nosync` suffix also keeps iCloud from syncing them on macOS). |
| Subgraph | FlyWire v783 mushroom body (MB): 6,283 neurons (KC/MBON/DAN/PN/APL), 103,647 connections, 21,275 plastic KC->MBON connections. The visualization shows the whole brain (138,533 neurons) as points, with the MB drawn brighter. |
| Coins | 5: BTC, ETH, SOL, PAXG (gold), ZEC (extended from 3 on 2026-09-16). CRCL (a stock) was deliberately left out: it breaks the alignment of hourly data with NYSE trading hours. |
| What the fly decides | NOT the side of the trade. This was measured (METHODOLOGY, Part 1.5): price direction is unpredictable. A risk-targeting formula provides direction and base size, and the fly learns a **multiplier** on top of it (`position_mode: exposure`). |
| Learning | Two rules behind a common interface: reward-modulated Hebbian (biologically motivated) and REINFORCE, plus an optional ML teacher (gradient boosting) as an additional input. Selected by flag or config; see the six cycles below. |
| Network biology | Dale's law (sign fixed by neurotransmitter), only KC->MBON synapses are plastic, k-WTA sparse Kenyon-cell code (~5% active), exploration noise applied to MBONs only. |
| Live executor | **A built-in simulator**, not a third-party site. Decision of 2026-09-15: the Binance Futures Testnet required KYC on a main Binance account, and the Bybit demo API was not available for the account region. Both adapters (`binance_futures_testnet`, `bybit_demo`) remain in the code behind the common `Adapter` interface in case that changes; `paper` / `dry_run` are what is used. |
| Dashboard | **Not Next.js** as originally planned, but a single HTML page plus three.js from a CDN (`web/index.html`) on top of FastAPI + WebSocket (`server/app.py`): simpler and needs no build step. It shows a 3D brain as points (spontaneous spikes for visible activity; dopamine neurons flash green/red), a price chart, a decision feed, a training panel, and 6 cycle buttons. |
| Safeguards | Per-coin weight cap (currently 50%; it went 40% -> 75% -> rolled back), minimum position change of 8%, at most 6 trades per hour, daily stop-loss of -10% with manual reset, and manual approval of the first N trades. |
| MB skeletons (SWC meshes) | Not done and not planned: the whole brain as points turned out to be enough for the visual, and the extra dependency (`navis` / a CAVE token) is not worth it. |
| Sentiment/news, cloud Optuna | Not done. The methodology showed that the bottleneck is neither hyperparameters nor the tone of the news, but the fact that price direction is not predictable in the first place (see METHODOLOGY, Part 1.5), so these stages lost priority. |
| Disk | Budget respected. `housekeeping.py` rotates old runs. |
| Language | Code, comments, UI, and documentation are in English. |
| Git | One commit per substantive change. |

## Repository structure

```
flybrain-trader/
├── README.md
├── docs/                       # OVERVIEW.md, METHODOLOGY.md, DESIGN_LOG.md
├── config/
│   ├── default.yaml            # all configuration: coins, risk, cycles
│   ├── hard.yaml                # override: reward for risk timing (cycle 4)
│   ├── cycle5_growth.yaml       # override: "100 -> 300" goal
│   └── experiments/
├── .env.example                 # exchange keys (optional), GROQ_API_KEY
├── data.nosync/  runs.nosync/   # connectome, candles, feature cache, checkpoints, logs (not in git)
├── src/flybrain/
│   ├── config.py                 # pydantic schema for the whole config
│   ├── connectome/                # Codex loader, MB subgraph, neuron roles
│   ├── brain/                     # model.py, plasticity_hebb/rl.py, ensemble.py
│   ├── market/                    # fetch.py, features.py, split.py, periods.py
│   ├── features/                  # price.py (+ per-coin fundamentals), external.py, store.py
│   ├── teacher/                   # gradient boosting alongside the fly
│   ├── trading/                   # portfolio, reward, guards, executor, baseline, adapters/
│   ├── backtest/                  # run, metrics, benchmarks, walk_forward, ablation
│   ├── train.py, train_seasonal.py, cycles.py   # training + registry of the 6 cycles
│   ├── live_engine.py, serve.py, live.py         # live loop, CLI wrapper
│   ├── gentle.py, storage.py, housekeeping.py
├── server/app.py                  # FastAPI + WebSocket
├── web/index.html                 # the whole dashboard, one page
├── scripts/                       # check_access.py, export_geometry.py, tail_logs.py, synthetic_connectome.py
├── tests/                         # 89 tests
```

The core architecture (graph -> features -> brain -> reward -> live/cycles) is
described in detail in [METHODOLOGY.md](METHODOLOGY.md), Part 3, and is not
duplicated here.

## The six training cycles (dashboard buttons)

| # | Question | Outcome |
|---|---|---|
| 1 Hebbian | The biologically motivated recipe, most recent 14 months | Sharpe delta -0.08 vs. the formula |
| 2 REINFORCE | A different algorithm, same data | Delta +0.02 |
| 3 + teacher | Does an ML forecast help as an input | Delta -0.05 |
| 4 hard | Reward for risk timing, not for returns | Delta +0.51 on one window, but -0.12 over 9 walk-forward windows: not confirmed |
| 5 "100 -> 300" | Triple the capital in a year | +6...20% instead of +200%; never achieved |
| 6 "seasons" | Maximize in 6 specific periods | Trained block by block with no data concatenation (which would leak at the seams) |

Honest conclusion across all six (METHODOLOGY, Parts 1.8-1.9): the fly is on par
with the risk-targeting formula and never systematically beat it under an
honest evaluation (walk-forward, many windows). The connectome topology also
gives no measurable advantage over a random graph of the same size (ablation,
Part 1.7): a structure that is useful for olfactory memory need not transfer
to financial time series. This is an expected negative research result, not an
unfinished project.

## Pitfalls we actually hit (not hypothetical)

**Platforms and exchanges**
- The Binance Futures Testnet required KYC on the main account.
- The Bybit demo API was not available for the account's region. -> built-in
  `PaperAdapter`.
- The `ccxt` `binanceusdm` sandbox pointed at the wrong URLs, so an explicit
  override to `testnet.binancefuture.com` was needed. This no longer matters
  since the testnet is not used, but it is kept in the adapter for the future.

**Hardware**
- A fanless MacBook Air **rebooted from thermal stress** after two heavy runs
  in a row at 100% CPU. -> `gentle.py`: `nice 19`, `torch.set_num_threads(2)`,
  pauses between epochs, a throttling check via `pmset -g therm`, and
  `OMP_NUM_THREADS` / `MKL_NUM_THREADS` capped for the sklearn teacher as well
  (otherwise it consumed every core, bypassing the torch limit).
- Never run two heavy processes at once (training + walk-forward, or two
  cycles together). Background monitors enforce this rule.

**Data**
- `align()` intersects the hourly timestamps of all coins. Before adding
  PAXG/ZEC we checked that the history would not be cut significantly (PAXG
  starts 2020-08-28, four weeks after the common start: acceptable).
- Overlapping labels over long horizons produce a spurious IC (Part 1.5). The
  only defense is walk-forward with independent windows, not a single check
  over the whole history.
- The feature and teacher caches are keyed by config. For live mode we
  separately solved the case "the cache must not be retrained on every new
  hourly candle" (a key that excludes the row count, plus a "tail" model).

**Process**
- Output from training subprocesses redirected to a file is fully buffered
  (not line by line), so progress monitoring reads `windows.json` /
  `history.json` directly instead of grepping the log.
- Training cycles are separate SUBPROCESSES, not threads inside the dashboard
  process: a crash or hang in training must not take the trading loop down
  with it. After a restart the server reattaches to training that is already
  running (it finds the process via `ps aux` and does not lose progress).
- Concatenating several periods into one array for cycle 6 would have caused a
  silent data leak at the seams (the agent marks equity by neighboring slice
  indices). The fix was to call training once per period rather than patch the
  core.

## Disk

The `.nosync` suffix on `data.nosync/` and `runs.nosync/` keeps iCloud from
syncing the heavy folders. `python -m flybrain.housekeeping --report` shows
what takes how much space; `--clean` rotates old runs, except those marked
with a `KEEP` file inside the run folder.

## Setup checklist

- [x] FlyWire Codex v783 downloaded, connectome loaded
- [ ] Groq API key in `.env`: optional (only for the sentiment stage, which is not active)
- [ ] Binance/Bybit keys: optional, needed only if you want a real exchange
      testnet or demo instead of `paper`
- [ ] CAVE token / Kaggle: not needed; the corresponding stages (MB
      skeletons, news sentiment, cloud Optuna) are not being done, see the
      table above

## Verification

- `pytest tests/`: 89 tests covering absence of lookahead in all features
  (including per-coin fundamentals), Dale's law, guards, the portfolio
  simulator, causality of splits, no leakage between cycle 6 blocks, `ps`
  parsing, and the cycle registry.
- Commands: `python -m flybrain.serve --adapter dry_run --online-learning`
  (dashboard at `http://127.0.0.1:8000`), `python -m flybrain.train
  --recent-months 14`, `python -m flybrain.backtest.walk_forward --config
  config/hard.yaml` for an honest check beyond a single window.
- Final criterion: the fly against the risk-targeting formula and against 100
  random agents on walk-forward. The result is recorded in METHODOLOGY.md as it
  is, negative findings included.
