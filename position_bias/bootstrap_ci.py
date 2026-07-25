#!/usr/bin/env python3
"""B2 — bootstrap 95% CIs for the attack metrics, resampling over candidate SETS.

Ingests one or more adv_*.json files produced by adversarial_perm.py with a "raw"
block (--dump-raw, on by default). Each raw row is one (candidate-set, target) with
per-R indicators; the correct resampling unit is the candidate SET, so we resample
set_ids with replacement and recompute the metric mean over all targets in the
picked sets. Reports point estimate + percentile CI for each (file, stratum, R, metric).

Runs locally, no cluster / no model.

  # headline CI at each file's R_max, irrelevant stratum:
  python position_bias/bootstrap_ci.py adv_lft.json adv_invariant.json

  # full budget-curve CI bands (every R in the grid) -> feeds Fig 6:
  python position_bias/bootstrap_ci.py adv_lft.json --all-R \
         --out results/budget_curve_ci.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_METRICS = ["promotion_into_top5", "rank_gain_adv", "weak_into_top5"]
ROOT = Path(__file__).resolve().parent


def load_raw(path: Path):
    data = json.loads(path.read_text())
    raw = data.get("raw")
    if not raw:
        raise SystemExit(f"{path} has no 'raw' block. Re-run adversarial_perm.py with --dump-raw.")
    r_grid = [int(r) for r in data.get("r_grid", [data.get("budget_R", 50)])]
    return raw, r_grid, int(data.get("budget_R", max(r_grid)))


def bootstrap_metric(rows, R, metric, n_boot, rng):
    """rows: list of raw dicts (one stratum). Resample by set_id; return point, lo, hi."""
    by_set = defaultdict(list)
    for row in rows:
        by_set[row["set_id"]].append(float(row["per_R"][str(R)][metric]))
    set_ids = list(by_set.keys())
    if not set_ids:
        return float("nan"), float("nan"), float("nan"), 0, 0
    flat = [v for vals in by_set.values() for v in vals]
    point = float(np.mean(flat))
    idx = np.arange(len(set_ids))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(idx, len(idx), replace=True)
        vals = [v for j in pick for v in by_set[set_ids[j]]]
        boots[b] = np.mean(vals)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), len(set_ids), len(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="adv_*.json files (must have a 'raw' block)")
    ap.add_argument("--stratum", default="irrelevant", choices=["irrelevant", "weak_pos", "strong_pos"])
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                    help="comma-separated: promotion_into_top5, rank_gain_adv, weak_into_top5, weak_into_top1, ...")
    ap.add_argument("--all-R", action="store_true", help="report at every R in the grid (else only R_max)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "bootstrap_ci.csv"))
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rng = np.random.default_rng(args.seed)
    out_rows = []

    for path in args.inputs:
        path = Path(path)
        label = path.stem
        raw, r_grid, r_max = load_raw(path)
        rows = [r for r in raw if r["stratum"] == args.stratum]
        Rs = r_grid if args.all_R else [r_max]
        for R in Rs:
            for metric in metrics:
                point, lo, hi, n_sets, n_tgt = bootstrap_metric(rows, R, metric, args.n_boot, rng)
                out_rows.append({"file": label, "stratum": args.stratum, "R": R, "metric": metric,
                                 "point": point, "ci_lo": lo, "ci_hi": hi,
                                 "n_sets": n_sets, "n_targets": n_tgt})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "stratum", "R", "metric", "point",
                                          "ci_lo", "ci_hi", "n_sets", "n_targets"])
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"{'file':<22}{'R':>5}{'metric':>22}{'point':>9}{'  95% CI':>18}{'n_sets':>8}")
    for r in out_rows:
        print(f"{r['file']:<22}{r['R']:>5}{r['metric']:>22}{r['point']:>9.4f}"
              f"   [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]{r['n_sets']:>8}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
