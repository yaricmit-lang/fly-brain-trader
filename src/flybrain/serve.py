"""
Run the dashboard together with the trading loop.

    python -m flybrain.serve                              # dry run
    python -m flybrain.serve --adapter paper --online-learning
    python -m flybrain.serve --checkpoint runs.nosync/.../best.pt

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import gentle
from .config import load_config

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Dashboard + live trading")
    ap.add_argument("--config", default=None)
    ap.add_argument("--adapter", default=None,
                    choices=["dry_run", "paper", "bybit_demo", "binance_futures_testnet"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--online-learning", action="store_true")
    ap.add_argument("--auto-approve", action="store_true",
                    help="do not wait for manual approval of the first trades")
    ap.add_argument("--think-every", type=float, default=2.0,
                    help="how often the brain thinks, in seconds (trades still happen once per bar)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="start from a clean slate: do not load the live experience "
                         "accumulated by a previous run (runs.nosync/live/live_state.pt), "
                         "only the cycle checkpoint. By default the accumulated "
                         "experience is continued")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.adapter:
        cfg.trading.executor.adapter = args.adapter
    if args.port:
        cfg.viz.server_port = args.port
    if args.online_learning:
        # Live mode uses a smaller step: reward arrives every 12h, there is
        # no rush, and mistakes cost more (applies to both learning rules).
        cfg.training.hebbian.lr = cfg.training.hebbian.lr_live
        cfg.training.reinforce.lr = cfg.training.reinforce.lr_live
    gentle.apply(cfg)

    geom = cfg.path("data") / "geometry.bin"
    if not geom.exists():
        raise SystemExit(
            "brain geometry not found. Run first:\n"
            "    python scripts/export_geometry.py"
        )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import uvicorn  # noqa: PLC0415

    from server.app import app, attach  # noqa: PLC0415

    from .live_engine import LiveEngine, read_checkpoint_pointer  # noqa: PLC0415

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is None:
        # The dashboard's "make live" button used to change the checkpoint
        # only in process memory, so any server restart silently reset the
        # live brain to None (initial connectome weights) and lost the
        # user's choice. The choice is now persisted as a pointer; an
        # explicit --checkpoint flag still takes precedence.
        run_dir = cfg.path("runs") / "live"
        checkpoint = read_checkpoint_pointer(run_dir)
        if checkpoint:
            print(f"  picked up the last selected live checkpoint: {checkpoint}")

    print(f"\n  execution mode: {cfg.trading.executor.adapter}")
    if cfg.trading.executor.adapter == "dry_run":
        print("  (dry run: nothing will be sent to the exchange)")
    print(f"  online learning: {'yes, step ' + str(cfg.training.hebbian.lr) if args.online_learning else 'no'}")

    engine = LiveEngine(
        cfg=cfg,
        checkpoint=checkpoint,
        online_learning=args.online_learning,
        auto_approve=args.auto_approve,
        think_every_sec=args.think_every,
        fresh=args.fresh,
    )
    attach(engine)
    print(f"  journal: {engine.run_dir / 'live.sqlite'}")
    if args.fresh:
        print("  --fresh: live experience from the previous run is ignored")

    uvicorn.run(app, host=cfg.viz.server_host, port=cfg.viz.server_port, log_level="warning")


if __name__ == "__main__":
    main()
