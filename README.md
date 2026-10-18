# FlyBrain Trader

A crypto trading agent built on the real wiring diagram of a fruit fly's brain.

The agent takes the mushroom-body subgraph of the *Drosophila* connectome
(FlyWire FAFB v783), puts trainable weights on the real synapse topology, feeds
market data into its "olfactory" inputs, and trains it with reward modulation:
profit strengthens the connections that fired, loss weakens them.

> **Research project, not a trading strategy.** Walk-forward evaluation shows
> no reliable edge over a simple volatility-targeting formula (see
> [Results](#results)). It trades through a built-in simulator by default.
> Do not run it with real money. Nothing here is financial advice.

## How it works

- **Inputs.** About 140 causal channels for five assets (BTC, ETH, SOL, PAXG
  gold and ZEC): price, volume and volatility features plus per-coin
  "fundamental" rules (Elliott-wave features for ETH, trend for BTC,
  mean reversion for gold, liquidity for ZEC, Solana DeFi TVL for SOL). Each
  channel drives its own group of projection neurons, the way one odour drives
  its own glomeruli in a real fly.
- **Brain.** The mushroom-body subgraph: Kenyon cells, MBON output neurons and
  the dopaminergic PAM and PPL1 neurons, which in the fly form the
  reward-learning centre. Activity propagates through a sparse synapse matrix
  for a few steps.
- **Output.** A risk-targeting formula sets the direction and base size of each
  position. The brain learns only a multiplier on top of it, how aggressive or
  cautious to be, read from the difference between the "approach" and "avoid"
  MBON groups. We measured that price direction is not predictable from these
  inputs, while volatility is (see [docs/METHODOLOGY.md](docs/METHODOLOGY.md)).
- **Learning.** A reward-modulated Hebbian rule (the biologically plausible
  one) and REINFORCE share one interface and are switched by a config flag.
  Reward arrives after a fixed horizon (12 h by default), not immediately.
  Several training cycles with different objectives can be launched from the
  dashboard.
- **Safety.** Per-coin exposure caps, a daily drawdown stop, a trade-rate limit
  and a drift fuse that halts online learning if the weights move too far.
- **Dashboard.** FastAPI plus a single-page 3D view of the brain: neurons flash
  green on reward and red on punishment, next to price charts and the agent's
  current decision log.

## Results

The fly was compared with buy-and-hold, an SMA crossover, volatility targeting
and 100 random agents over a 22-window walk-forward. It did not beat the
volatility-targeting formula, and shuffling the connectome (ablation) made no
noticeable difference. Negative results are reported as they came out. The
numbers and the reasoning behind each design change are in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Quick start

Developed and tested on Python 3.14. A CPU is enough; no GPU is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest                                  # builds a toy connectome on first run
cp .env.example .env                    # all keys are optional
python scripts/check_access.py          # which services are reachable from your network
```

To run on the real connectome you need to download the FlyWire v783 tables
from Codex (a Google login is required, so this step is manual):

```bash
python -m flybrain.connectome.download  # prints the exact steps and checks the files
python -m flybrain.connectome.loader    # builds the sparse graph
python -m flybrain.market.fetch         # public exchange candles, no API key needed
python -m flybrain.train                # train (add --synthetic for a smoke test)
```

Run the dashboard and the trading loop (simulated orders, nothing is sent to an
exchange):

```bash
python -m flybrain.serve --adapter dry_run --online-learning --auto-approve
```

Then open <http://127.0.0.1:8000>.

The server has **no authentication** and exposes trading-control endpoints. It
binds to `127.0.0.1` by default; only change `viz.server_host` on a network you
trust. Browser requests from other origins are rejected.

## Project layout

| Path | Contents |
|---|---|
| `config/` | All settings. There are no magic numbers in the code |
| `src/flybrain/connectome/` | Connectome loading, subgraph extraction, neuron roles |
| `src/flybrain/brain/` | Sparse-graph model, plasticity rules, ensemble |
| `src/flybrain/market/`, `features/` | Candles, features, time splits, leak-free as-of joins |
| `src/flybrain/trading/` | Portfolio, reward, guards, executors and exchange adapters |
| `src/flybrain/backtest/` | Runs, metrics, benchmarks, walk-forward, ablation |
| `server/`, `web/` | FastAPI server and the 3D dashboard |
| `docs/` | Overview, methodology with all measurements, design log |
| `data.nosync/`, `runs.nosync/` | Data and run output, git-ignored. The `.nosync` suffix keeps macOS iCloud from syncing them |

Housekeeping for run output:

```bash
python -m flybrain.housekeeping --report   # what takes how much space
python -m flybrain.housekeeping --clean    # delete old runs
```

Put an empty file named `KEEP` inside a run folder to protect it from cleanup.

## Documentation

- [docs/OVERVIEW.md](docs/OVERVIEW.md): the project in plain language
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): methodology, measurements and the
  current architecture step by step (Part 3 explains how the fly "thinks" right now)
- [docs/DESIGN_LOG.md](docs/DESIGN_LOG.md): engineering decisions and pitfalls

## License

The code is released under the [MIT License](LICENSE). This does not cover the
FlyWire connectome data, which has its own terms (see below).

## Data and attribution

The connectome comes from the FlyWire consortium (FAFB v783) via
[Codex](https://codex.flywire.ai). It is not redistributed here; follow the
FlyWire and Codex terms and citation requirements when you use it. Market data
is fetched from public exchange APIs.
