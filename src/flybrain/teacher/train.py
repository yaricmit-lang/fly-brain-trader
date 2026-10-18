"""
Teacher training with honest validation.

    python -m flybrain.teacher.train --report

Two traps make such models show an edge that does not exist:

1. Overlapping labels. A 12-hour-ahead forecast is almost identical for
   adjacent hours. With a naive consecutive train/test split, neighboring
   rows land on opposite sides of the boundary and the model "sees" the
   answer. The fix is an embargo: drop the H hours of the training part
   adjacent to the test part.

2. Fitting on the whole history. A model trained on 2020-2026 and evaluated
   on the same span always looks brilliant. Hence walk-forward only: train
   on the past, evaluate on the next slice, move forward.

The main metric is IC (rank correlation of forecast with outcome). For
hourly crypto an IC of 0.03-0.05 is already something; 0.1+ is suspicious.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..features import build_inputs
from ..market.fetch import get_all


def forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Log return of each coin over the next `horizon` hours. The tail is NaN."""
    out = np.full_like(prices, np.nan, dtype=np.float64)
    out[:-horizon] = np.log(prices[horizon:] / np.maximum(prices[:-horizon], 1e-12))
    return out


def coin_columns(names: list[str], coin: str) -> list[int]:
    """
    Columns the teacher sees for one coin: its own features plus
    market-wide ones (MKT:*). Other coins are withheld, otherwise the BTC
    model would peek at SOL features, which is a proxy leak.
    """
    return [i for i, n in enumerate(names)
            if n.startswith(f"{coin}:") or n.startswith("MKT:")]


def rank_ic(pred: np.ndarray, truth: np.ndarray) -> float:
    """Spearman rank correlation; robust to crypto outliers."""
    ok = np.isfinite(pred) & np.isfinite(truth)
    if ok.sum() < 30:
        return float("nan")
    p = pd.Series(pred[ok]).rank().to_numpy()
    t = pd.Series(truth[ok]).rank().to_numpy()
    return float(np.corrcoef(p, t)[0, 1])


def make_model(cfg: Config):
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

    t = cfg.teacher
    return HistGradientBoostingRegressor(
        max_iter=t.max_iter,
        learning_rate=t.learning_rate,
        max_depth=t.max_depth,
        min_samples_leaf=t.min_samples_leaf,
        l2_regularization=t.l2,
        early_stopping=False,
        random_state=cfg.project.seed,
    )


def walk_forward_ic(
    cfg: Config,
    X: np.ndarray,
    y: np.ndarray,
    cols: list[int],
    horizon: int,
    dt: pd.DatetimeIndex,
    train_months: int = 18,
    test_months: int = 2,
) -> list[dict]:
    """
    Rolling-window training with an embargo. Returns per window: IC,
    hit rate and row counts.
    """
    rows: list[dict] = []
    t0 = dt[0] + pd.DateOffset(months=train_months)
    while True:
        tr_end = t0
        te_end = t0 + pd.DateOffset(months=test_months)
        if te_end > dt[-1]:
            break
        tr_start = tr_end - pd.DateOffset(months=train_months)
        tr = np.where((dt >= tr_start) & (dt < tr_end))[0]
        te = np.where((dt >= tr_end) & (dt < te_end))[0]
        # embargo: labels of the last `horizon` train hours reach into test
        tr = tr[:-horizon] if len(tr) > horizon else tr
        ok = np.isfinite(y[tr])
        tr = tr[ok]
        if len(tr) < 500 or len(te) < 100:
            t0 = te_end
            continue

        model = make_model(cfg)
        model.fit(X[np.ix_(tr, cols)], y[tr])
        pred = model.predict(X[np.ix_(te, cols)])
        truth = y[te]
        ic = rank_ic(pred, truth)
        fin = np.isfinite(pred) & np.isfinite(truth)
        hit = float((np.sign(pred[fin]) == np.sign(truth[fin])).mean()) if fin.any() else float("nan")
        rows.append({
            "test_start": f"{dt[te[0]]:%Y-%m-%d}",
            "test_end": f"{dt[te[-1]]:%Y-%m-%d}",
            "ic": ic, "hit_rate": hit, "n_train": int(len(tr)), "n_test": int(len(te)),
        })
        t0 = te_end
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Teacher: training and honest validation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--report", action="store_true", help="walk-forward IC per window")
    ap.add_argument("--horizons", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    horizons = args.horizons or cfg.teacher.horizons_hours

    print("  loading market data and features...")
    data = get_all(cfg)
    X, dt, names, prices = build_inputs(cfg, data)
    coins = [s.split("/")[0] for s in cfg.market.symbols]
    print(f"  {X.shape[0]:,} hours x {X.shape[1]} channels, {len(coins)} coins\n")

    report: dict = {"horizons": {}}
    for h in horizons:
        fwd = forward_returns(prices, h)
        print(f"  === horizon {h}h ===")
        per_coin = {}
        for i, c in enumerate(coins):
            cols = coin_columns(names, c)
            rows = walk_forward_ic(cfg, X, fwd[:, i], cols, h, dt,
                                   cfg.teacher.train_months, cfg.teacher.test_months)
            ics = np.array([r["ic"] for r in rows], dtype=float)
            hits = np.array([r["hit_rate"] for r in rows], dtype=float)
            pos = float((ics > 0).mean() * 100) if len(ics) else float("nan")
            print(f"    {c}: windows {len(rows):>2}  mean IC {np.nanmean(ics):+.4f}  "
                  f"median {np.nanmedian(ics):+.4f}  positive windows {pos:.0f}%  "
                  f"hit-rate {np.nanmean(hits) * 100:.1f}%")
            per_coin[c] = {
                "windows": rows,
                "ic_mean": float(np.nanmean(ics)), "ic_median": float(np.nanmedian(ics)),
                "pct_positive_windows": pos, "hit_rate_mean": float(np.nanmean(hits)),
            }
        report["horizons"][str(h)] = per_coin
        all_ic = np.array([per_coin[c]["ic_mean"] for c in coins])
        verdict = ("edge present" if np.nanmean(all_ic) > 0.03 else
                   "weak but nonzero" if np.nanmean(all_ic) > 0.01 else "no edge")
        print(f"    mean IC across coins: {np.nanmean(all_ic):+.4f} - {verdict}\n")

    out = cfg.path("runs") / "teacher_ic.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  report: {out}\n")


if __name__ == "__main__":
    main()
