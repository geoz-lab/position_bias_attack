#!/usr/bin/env python3
"""Collect prompt-check outputs into one comparison CSV (run locally after pulling
the scan_*.json / adv_*.json back from Sherlock's $PROJ/position_bias_check/).

  python position_bias_check/collect.py <dir_with_json> \
         --out position_bias_check/results/prompt_check.csv

One row per (variant, stratum). curve_range + best/worst rank come from scan_*.json;
promo@5 / rank_gain / into_top5 from adv_*.json. Compare each treatment row against the
p0_baseline row: a prompt that "works" drives curve_range and promo@5 toward ~0.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("indir", help="dir containing scan_<variant>.json and adv_<variant>.json")
    ap.add_argument("--out", default="position_bias_check/results/prompt_check.csv")
    args = ap.parse_args()

    indir = Path(args.indir)
    variants = sorted({p.name[len("scan_"):-len(".json")]
                       for p in indir.glob("scan_*.json")})
    if not variants:
        raise SystemExit(f"no scan_*.json found in {indir}")

    rows = []
    for v in variants:
        scan = json.loads((indir / f"scan_{v}.json").read_text())
        adv_path = indir / f"adv_{v}.json"
        adv = json.loads(adv_path.read_text()) if adv_path.exists() else {"summary": {}}
        strata = sorted(set(scan.get("summary", {})) | set(adv.get("summary", {})))
        for st in strata:
            sc = scan["summary"].get(st, {})
            a = adv["summary"].get(st, {})
            rows.append({
                "variant": v,
                "stratum": st,
                "curve_range": round(sc.get("position_effect_range", float("nan")), 3),
                "best_rank": round(sc.get("best_mean_rank", float("nan")), 3),
                "worst_rank": round(sc.get("worst_mean_rank", float("nan")), 3),
                "promo@5": round(a.get("promotion_into_top5", float("nan")), 4),
                "rank_gain": round(a.get("rank_gain_adv", float("nan")), 3),
                "into_top5": round(a.get("weak_into_top5", float("nan")), 4),
                "n": a.get("n", sc.get("n_cstar", "")),
            })

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {outp}")
    # quick echo of the irrelevant-stratum comparison (the headline)
    print("\nirrelevant stratum (headline):")
    print(f"  {'variant':26s} {'curve_range':>11s} {'promo@5':>8s}")
    for r in rows:
        if r["stratum"] == "irrelevant":
            print(f"  {r['variant']:26s} {r['curve_range']:11} {r['promo@5']:8}")


if __name__ == "__main__":
    main()
