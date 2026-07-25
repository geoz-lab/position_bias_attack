#!/usr/bin/env python3
"""RQ2 / §3b — targeted exploitability via a budget-R permutation-search attacker.

Attacker model (deliberately conservative -> a LOWER BOUND on a smarter attacker):
"the attacker may try R candidate orderings and keep the one that ranks their
target c* highest." We report, per target-stratum:
  typical_rank  = mean c* output rank over R_max random perms (attack-free baseline)
  adv_rank      = best (min) c* output rank over the first R perms (what the attacker gets)
  rank_gain_adv = typical_rank - adv_rank         (positions gained by ordering alone)
  weak_into_top5/1 = frac of targets whose adv_rank < 5 / == 0
  promotion@k   = frac of targets with typical_rank >= k but adv_rank < k
                  (i.e. genuinely OUTSIDE top-k on average, pushed IN by ordering)

InvariRank is position-invariant so adv_rank ~= typical_rank -> rank_gain ~= 0 and
weak_into_top5 collapses to the content-only base rate. LFT - InvariRank = the
position-attributable attack surface (RQ2 + RQ4).

B1 (budget curve): we run R_max = --budget perms ONCE, then report every metric at
each R in --r-grid via the running-min of adv_rank over the first R perms. The
attack-free baseline `typical_rank` is FIXED at R_max for all R, so the promo@5
gate ("outside top-5 on average") does not drift with the budget.

B2 (bootstrap): --dump-raw writes one row per (candidate-set, target) with the
per-R indicators, so bootstrap_ci.py can resample over candidate sets and put a
95% CI on promo@5 / rank_gain / into_top5. Grouping key is `set_id`.

  # headline single-budget run (backward compatible: summary reported at R=budget):
  python position_bias/adversarial_perm.py --config <rank_lft.yaml> -R 50 \
         --num-samples 300 --out $PROJ/position_bias/adv_lft.json

  # B1 full budget curve (one pass, running-min over R in {1,5,10,20,50,100}):
  python position_bias/adversarial_perm.py --config <rank_lft.yaml> -R 100 \
         --r-grid 1,5,10,20,50,100 --num-samples 300 --out $PROJ/position_bias/adv_lft.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from datasets.utils import load_jsonl, set_seed
from model import load_model_for_ranking, load_tokenizer, select_device
from prompts import build_prompt
from ranking import MeanLogProbListwiseScorer


def stratum(rel: int) -> str:
    if rel >= 3:
        return "strong_pos"
    if rel >= 1:
        return "weak_pos"
    return "irrelevant"


def parse_r_grid(spec: str, r_max: int) -> list[int]:
    """Parse "1,5,10,..." -> sorted unique ints in [1, r_max]; always include r_max."""
    grid = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            grid.add(int(tok))
    grid = {r for r in grid if 1 <= r <= r_max}
    grid.add(r_max)
    return sorted(grid)


def metrics_at_R(typical: float, adv: int) -> dict:
    """All per-target indicators given a fixed typical_rank and an adv_rank at some R."""
    return {
        "adv_rank": adv,
        "rank_gain_adv": typical - adv,
        "weak_into_top5": 1.0 if adv < 5 else 0.0,
        "weak_into_top1": 1.0 if adv == 0 else 0.0,
        "promotion_into_top5": 1.0 if (typical >= 5 and adv < 5) else 0.0,
        "promotion_into_top1": 1.0 if (typical >= 1 and adv == 0) else 0.0,
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-R", "--budget", type=int, default=50,
                    help="R_max: number of random perms actually run (also the headline budget for `summary`)")
    ap.add_argument("--r-grid", default="1,5,10,20,50,100",
                    help="budget points to report the curve at (clamped to <= R_max; R_max always added)")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--targets-per-set", type=int, default=2, help="1 irrelevant + 1 weak/strong")
    ap.add_argument("--dump-raw", dest="dump_raw", action="store_true", default=True,
                    help="write per-(set,target) rows for bootstrap CIs (default on)")
    ap.add_argument("--no-dump-raw", dest="dump_raw", action="store_false")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = select_device(cfg.device)
    cfg.device = str(device)

    r_grid = parse_r_grid(args.r_grid, args.budget)

    tokenizer = load_tokenizer(cfg)
    model = load_model_for_ranking(cfg, tokenizer, device)
    scorer = MeanLogProbListwiseScorer(model, tokenizer, cfg).to(device)

    samples = [s for s in load_jsonl(cfg.data_path) if s.get("candidates")][: args.num_samples]
    rng = random.Random(int(cfg.seed))

    # curve[R][stratum] -> accumulator lists;  raw -> per-(set,target) rows for bootstrap
    curve = {R: defaultdict(lambda: defaultdict(list)) for R in r_grid}
    typical_by_stratum = defaultdict(list)
    raw_rows = []
    n_used = 0
    set_id = 0

    with torch.no_grad():
        for sample in samples:
            cands = sample["candidates"]
            K = len(cands)
            rels = [int(c.get("relevance", 0)) for c in cands]

            by_stratum = defaultdict(list)
            for i, r in enumerate(rels):
                by_stratum[stratum(r)].append(i)
            targets = []
            if by_stratum["irrelevant"]:
                targets.append(rng.choice(by_stratum["irrelevant"]))
            if by_stratum["weak_pos"]:
                targets.append(rng.choice(by_stratum["weak_pos"]))
            elif by_stratum["strong_pos"]:
                targets.append(rng.choice(by_stratum["strong_pos"]))
            targets = targets[: args.targets_per_set]
            if not targets:
                continue

            # R_max random perms, shared across targets for efficiency
            ranks_by_cand = []  # list over perms: dict cand_index -> output rank (0=best)
            ok = True
            for _ in range(args.budget):
                perm = list(range(K))
                rng.shuffle(perm)
                prompt = build_prompt(sample, perm, cfg)
                enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=int(cfg.max_seq_length))
                scores = scorer(enc["input_ids"].to(device), enc["attention_mask"].to(device))
                if int(scores.numel()) != K:
                    ok = False
                    break
                s = [float(x) for x in scores.detach().float().cpu().tolist()]
                order = sorted(range(K), key=lambda j: s[j], reverse=True)  # perm-local positions
                rank_of_local = {loc: r for r, loc in enumerate(order)}
                rank_by_candidate = {perm[loc]: rank_of_local[loc] for loc in range(K)}
                ranks_by_cand.append(rank_by_candidate)
            if not ok or len(ranks_by_cand) < args.budget:
                continue

            for c in targets:
                st = stratum(rels[c])
                rr = [rb[c] for rb in ranks_by_cand]           # length R_max
                typical = mean(rr)                             # attack-free baseline, fixed at R_max
                typical_by_stratum[st].append(typical)

                # running-min adv over the first R perms, evaluated at each grid point
                per_R = {}
                run_min = None
                gi = 0
                for r_idx in range(1, args.budget + 1):
                    run_min = rr[r_idx - 1] if run_min is None else min(run_min, rr[r_idx - 1])
                    if gi < len(r_grid) and r_idx == r_grid[gi]:
                        m = metrics_at_R(typical, run_min)
                        per_R[r_grid[gi]] = m
                        for key, val in m.items():
                            curve[r_grid[gi]][st][key].append(val)
                        gi += 1

                if args.dump_raw:
                    raw_rows.append({
                        "set_id": set_id,
                        "stratum": st,
                        "typical_rank": typical,
                        "per_R": {str(R): per_R[R] for R in r_grid},
                    })
            n_used += 1
            set_id += 1

    # ---- summary (backward compatible): reported at R = R_max (== --budget) ----
    R_max = args.budget
    summary = {}
    for st in typical_by_stratum:
        acc = curve[R_max][st]
        summary[st] = {
            "n": len(acc["adv_rank"]),
            "typical_rank": mean(typical_by_stratum[st]),
            "adv_rank": mean(acc["adv_rank"]),
            "rank_gain_adv": mean(acc["rank_gain_adv"]),
            "weak_into_top5": mean(acc["weak_into_top5"]),
            "weak_into_top1": mean(acc["weak_into_top1"]),
            "promotion_into_top5": mean(acc["promotion_into_top5"]),
            "promotion_into_top1": mean(acc["promotion_into_top1"]),
        }

    # ---- budget curve: metric means per (R, stratum) ----
    curve_out = {}
    for R in r_grid:
        curve_out[str(R)] = {
            st: {k: mean(v) for k, v in acc.items()}
            for st, acc in curve[R].items()
        }

    out = {
        "config": args.config,
        "budget_R": R_max,
        "r_grid": r_grid,
        "num_sets_used": n_used,
        "summary": summary,          # at R_max (consumed by posbias_eval.sbatch SUMMARY)
        "budget_curve": curve_out,   # B1: metrics at every R in r_grid
    }
    if args.dump_raw:
        out["raw"] = raw_rows        # B2: per-(set,target) rows keyed by set_id

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== adversarial (R_max={R_max}, grid={r_grid}): {args.config} ({n_used} sets) ===")
    for st, s in summary.items():
        print(f"  {st:12s} typical_rank={s['typical_rank']:5.2f} adv_rank={s['adv_rank']:5.2f} "
              f"gain={s['rank_gain_adv']:4.2f} | into_top5={s['weak_into_top5']:.3f} "
              f"into_top1={s['weak_into_top1']:.3f} promo@5={s['promotion_into_top5']:.3f} (n={s['n']})")
    if len(r_grid) > 1:
        print("  budget curve promo@5 (irrelevant):", " ".join(
            f"R{R}={curve_out[str(R)].get('irrelevant', {}).get('promotion_into_top5', float('nan')):.3f}"
            for R in r_grid))


if __name__ == "__main__":
    main()
