"""
Training loop.

    python -m flybrain.train --synthetic          # on the synthetic stand-in, for smoke-testing
    python -m flybrain.train                      # on the real connectome
    python -m flybrain.train --rule reinforce     # rule B instead of A

Early stopping uses the validation period, which the network never sees
during training. A checkpoint is saved only when val improves; otherwise we
would pick the best result on the very data we trained on and get a pretty
number instead of a working model.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import gentle
from .agent import Agent
from .backtest import benchmarks as B
from .backtest.report import make_run_dir
from .build import build_ensemble, checkpoint_manifest, save_checkpoint
from .config import load_config
from .features import build_inputs
from .market.fetch import get_all
from .market.split import make_anchored_split, make_recent_split, make_split


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the fly")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="use the toy connectome")
    ap.add_argument("--rule", choices=["hebbian", "reinforce"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-bars", type=int, default=0, help="truncate train for a quick check")
    ap.add_argument("--members", type=int, default=None, help="ensemble size (default: from config)")
    ap.add_argument("--no-gentle", action="store_true", help="disable gentle mode (not on a laptop!)")
    ap.add_argument("--teacher", dest="teacher", action="store_true", default=None,
                    help="add teacher forecast channels")
    ap.add_argument("--no-teacher", dest="teacher", action="store_false",
                    help="no teacher: pure fly")
    ap.add_argument("--recent-months", type=int, default=None,
                    help="train on the last N months of history instead of the "
                         "fixed config dates; train ends right at the boundary "
                         "of today minus --val-months")
    ap.add_argument("--val-months", type=int, default=2,
                    help="how much of recent-months to give to val (default 2)")
    ap.add_argument("--run-dir", default=None,
                    help="explicit output path instead of an auto-generated "
                         "timestamped one; training cycles need it so the "
                         "dashboard knows in advance where to find best.pt")
    ap.add_argument("--pause-sec", type=int, default=None,
                    help="pause between epochs (gentle mode); default: from config")
    ap.add_argument("--val-start", default=None,
                    help="val starts on a specific date (e.g. 2025-08-01), not 'the last N "
                         "months'; in this mode --recent-months means train_months BEFORE "
                         "val-start. For targeted cycles such as 'triple the capital August to August'")
    ap.add_argument("--val-end", default=None, help="val period end (see --val-start)")
    ap.add_argument("--select-by", choices=["sharpe", "return", "edge"], default="sharpe",
                    help="metric on val for picking the best checkpoint: default is "
                         "risk-adjusted Sharpe (safer); 'return' is aggressive, straight "
                         "return even at the cost of drawdown; 'edge' is the fly's Sharpe MINUS "
                         "the formula's Sharpe on the same val (what we actually want to maximize)")
    ap.add_argument("--goal-return", type=float, default=2.0,
                    help="return target on val for result.json[hit_goal], as a fraction "
                         "(2.0 = triple the capital, +200%%). Does not affect training, only the report")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.rule:
        cfg.training.rule = args.rule
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.no_gentle:
        cfg.training.gentle.enabled = False
    if args.pause_sec is not None:
        cfg.training.gentle.pause_between_epochs_sec = args.pause_sec
    if args.teacher:
        cfg.teacher.enabled = True      # so config_used.json tells the truth
    gentle.apply(cfg)
    t_start = time.time()

    # ---------------- data ----------------
    print("  loading market data...")
    data = get_all(cfg)
    X, dt, names, prices = build_inputs(cfg, data, with_teacher=args.teacher)
    n_t = sum(1 for n in names if ":teacher_" in n)
    print(f"  inputs: {X.shape[1]} channels ({n_t} from teacher), {len(X):,} hourly candles")
    if args.val_start or args.val_end:
        if not (args.val_start and args.val_end):
            raise SystemExit("--val-start and --val-end must be given together")
        train_months = args.recent_months or 12
        split = make_anchored_split(dt, train_months, args.val_start, args.val_end)
        print(f"  window: val {args.val_start} .. {args.val_end} (train: {train_months} months before it)")
    elif args.recent_months:
        split = make_recent_split(dt, args.recent_months, args.val_months)
        print(f"  window: last {args.recent_months} months (val: last {args.val_months})")
    else:
        split = make_split(dt, cfg)
    print(split.describe(dt))

    tr, va = split.train, split.val
    if args.limit_bars:
        tr = tr[-args.limit_bars:]
        print(f"  train truncated to {len(tr):,} candles (quick check)")

    # ---------------- brain ----------------
    print("  building brain...")
    brain, sg, rm, plast = build_ensemble(
        cfg, synthetic=args.synthetic, with_plasticity=True, n_members=args.members,
        channel_names=names,
    )
    print(brain.describe())
    first = getattr(brain, "members", [brain])[0]     # for weight statistics

    if len(names) != rm.n_channels:
        raise SystemExit(
            f"data has {len(names)} channels, but the brain expects {rm.n_channels}. "
            f"Check market.features and market.symbols in the config."
        )

    agent = Agent(brain, cfg, plast)

    # effective learning step: the config lr is calibrated for a reference
    # decision frequency (training.lr_reference_timeframe) and is rescaled
    # for any other; print it so the correction is not invisible
    scale = cfg.lr_stride_scale
    if scale != 1.0:
        base_lr = cfg.training.hebbian.lr if cfg.training.rule == "hebbian" else cfg.training.reinforce.lr
        print(f"  decision timeframe {cfg.market.decision_timeframe} vs reference "
              f"{cfg.training.lr_reference_timeframe}: lr {base_lr:g} x {scale:g} = {base_lr * scale:g}")

    # readout warm-up before val (see Agent.run(warm_X=...) and C2 in the
    # plan): excluded from the val metrics, it only lets _ro_mean/_ro_var
    # settle before the evaluated window starts
    def _warmup_slice(idx: np.ndarray) -> np.ndarray | None:
        hours = cfg.training.warmup_hours
        if hours <= 0 or len(idx) == 0:
            return None
        lo, hi = max(0, int(idx[0]) - hours), int(idx[0])
        return X[lo:hi] if hi > lo else None

    warm_va = _warmup_slice(va)

    # reference formula on val, computed once before the epoch loop: needed
    # both for --select-by edge (the fly's Sharpe relative to it each epoch)
    # and for the final result.json
    p_val = prices[va]
    vt, _ = B.vol_target(cfg, p_val)

    tag = args.tag or f"{cfg.training.rule}{'_synth' if args.synthetic else ''}"
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(cfg, f"train_{tag}")

    # config_used.json/channels.json are written BEFORE the epoch loop
    # (2026-09-16, F1): per-epoch lr/noise decay (h.lr *= h.lr_epoch_decay)
    # mutates cfg.training.hebbian IN PLACE, so a dump written after the loop
    # would reflect the final dose rather than the starting one, and the run
    # could not be reproduced from its own dump.
    # channels.json is the checkpoint manifest (see F1 in ensemble.py): input
    # channel list, seed, member count. switch_checkpoint must check against
    # them rather than silently load weights built for a different channel list.
    (run_dir / "config_used.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "channels.json").write_text(
        json.dumps({"names": names, "seed": cfg.project.seed,
                    "n_members": len(getattr(brain, "members", [brain])), "schema": 2},
                   indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------------- epochs ----------------
    best_val = -np.inf
    patience = 0
    history: list[dict] = []

    for epoch in range(1, cfg.training.epochs + 1):
        print(f"\n  epoch {epoch}/{cfg.training.epochs}  (rule: {cfg.training.rule})")

        train_res = agent.run(
            X[tr], prices[tr], learn=True, seed=cfg.project.seed + epoch,
            progress_every=200,
        )
        gentle.pause(cfg, "after train")
        val_res = agent.run(X[va], prices[va], learn=False, seed=cfg.project.seed, warm_X=warm_va)

        vm, tm = val_res.metrics, train_res.metrics
        if args.select_by == "sharpe":
            score = vm.sharpe
        elif args.select_by == "return":
            score = vm.total_return
        else:
            score = vm.sharpe - vt.sharpe
        mean_r = float(np.mean(train_res.rewards)) if train_res.rewards else 0.0
        print(
            f"    train: return {tm.total_return * 100:>7.1f}%  Sharpe {tm.sharpe:>6.2f}  "
            f"trades {tm.n_trades:>6,}  decisions {train_res.n_decisions:,}"
        )
        print(
            f"    val:   return {vm.total_return * 100:>7.1f}%  Sharpe {vm.sharpe:>6.2f}  "
            f"trades {vm.n_trades:>6,}   mean reward {mean_r:+.4f}"
        )

        history.append({
            "epoch": epoch,
            "train": tm.as_dict(),
            "val": vm.as_dict(),
            "mean_reward": mean_r,
            "n_decisions": train_res.n_decisions,
            "weight_mean": float(first.w_mag.detach().mean()),
            "weight_max": float(first.w_mag.detach().max()),
            "score": score,             # what the checkpoint was actually selected by
            "select_by": args.select_by,  # sharpe (default), return or edge
        })

        # dopamine protocol: tolerance, i.e. dose and noise decay per epoch.
        # HebbianPlasticity holds a reference to the same config object, so
        # the edit here takes effect next epoch without a rebuild.
        h = cfg.training.hebbian
        if h.lr_epoch_decay != 1.0 or h.noise_epoch_decay != 1.0:
            h.lr *= h.lr_epoch_decay
            h.exploration_noise *= h.noise_epoch_decay
            print(f"    dose for next epoch: lr {h.lr:.4f}, noise {h.exploration_noise:.3f}")

        if score > best_val:
            best_val = score
            patience = 0
            save_checkpoint(brain, run_dir / "best.pt", {"epoch": epoch, "score": score, "select_by": args.select_by},
                            manifest=checkpoint_manifest(cfg, brain, names))
            print("    val improved, checkpoint saved")
        else:
            patience += 1
            print(f"    no improvement ({patience}/{cfg.training.early_stopping_patience})")
            if patience >= cfg.training.early_stopping_patience:
                print("    early stopping")
                break

    save_checkpoint(brain, run_dir / "last.pt", {"epoch": len(history)},
                    manifest=checkpoint_manifest(cfg, brain, names))
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------------- summary vs the formula ----------------
    # The fly's val Sharpe means nothing on its own: what matters is whether it
    # beats the risk-targeting formula on the SAME window. That is the question
    # of "which cycle goes live", so it is computed right here.
    # "score" is the same metric the checkpoint was actually selected by (see
    # --select-by). This used to be hard-coded h["val"]["sharpe"], so with
    # select_by=return result.json declared a different epoch "best" than the
    # one actually saved as best.pt.
    best = max(history, key=lambda h: h["score"]) if history else None
    bh, _ = B.buy_and_hold(cfg, p_val)
    result = {
        "rule": cfg.training.rule,
        "teacher": bool(args.teacher),
        "select_by": args.select_by,
        "members": len(getattr(brain, "members", [brain])),
        "epochs_done": len(history),
        "best_epoch": best["epoch"] if best else None,
        "val_start": f"{dt[va[0]]:%Y-%m-%d}", "val_end": f"{dt[va[-1]]:%Y-%m-%d}",
        "train_start": f"{dt[tr[0]]:%Y-%m-%d}", "train_end": f"{dt[tr[-1]]:%Y-%m-%d}",
        "fly": best["val"] if best else None,
        "vol_target": vt.as_dict(),
        "buy_hold": bh.as_dict(),
        "edge_sharpe": (best["val"]["sharpe"] - vt.sharpe) if best else None,
        "edge_return": (best["val"]["total_return"] - vt.total_return) if best else None,
        # return target (default +200%, i.e. tripling): a measurable criterion
        # for the "100 -> 300" cycle, ignored by the other cycles
        "goal_return": args.goal_return,
        "hit_goal": bool(best and best["val"]["total_return"] >= args.goal_return),
        "duration_sec": round(time.time() - t_start),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if best:
        print(f"\n  val: fly Sharpe {best['val']['sharpe']:.2f} (return {best['val']['total_return']*100:+.1f}%)  "
              f"|  formula Sharpe {vt.sharpe:.2f} (return {vt.total_return*100:+.1f}%)  "
              f"|  buy & hold {bh.sharpe:.2f}\n"
              f"  edge: Sharpe {result['edge_sharpe']:+.2f}, return {result['edge_return']*100:+.1f} pp")
    print(f"  time: {result['duration_sec'] / 60:.1f} min")
    print(f"\n  done. Run: {run_dir}")
    print(f"  next: python -m flybrain.backtest.run --checkpoint '{run_dir / 'best.pt'}'\n")


if __name__ == "__main__":
    main()
