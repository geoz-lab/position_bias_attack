#!/usr/bin/env python3
"""B1/B5 — pointwise reference on the attack-budget axis.

Pointwise scores each candidate in ISOLATION (history + that one candidate), so a
candidate's score does not depend on any permutation of the list. The budget-R
attacker therefore gets nothing: every one of its R orderings yields the identical
per-candidate scores, so adv_rank == typical_rank at all R -> rank_gain = 0 and
promo@5 = 0, exactly, by construction. The only non-trivial quantity is into_top5,
which equals the pure content base rate (constant in R).

We exploit that invariance to avoid wasted compute: score each candidate ONCE
(K forward passes per set), then fill every R in the grid with the fixed rank.
Output matches adversarial_perm.py's schema (summary / budget_curve / raw) so
bootstrap_ci.py and budget_curve.csv consume it identically (model=pointwise).

  python position_bias/pointwise_adv.py --config <rank_lft.yaml> \
         -R 100 --r-grid 1,5,10,20,50,100 --num-samples 300 \
         --out $PROJ/position_bias/adv_pointwise_curve.json
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

from adversarial_perm import mean, metrics_at_R, parse_r_grid, stratum


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a (causal) rank config; its adapter is scored pointwise")
    ap.add_argument("-R", "--budget", type=int, default=100)
    ap.add_argument("--r-grid", default="1,5,10,20,50,100")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--targets-per-set", type=int, default=2)
    ap.add_argument("--no-dump-raw", dest="dump_raw", action="store_false", default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = select_device(cfg.device)
    cfg.device = str(device)
    r_grid = parse_r_grid(args.r_grid, args.budget)

    tok = load_tokenizer(cfg)
    model = load_model_for_ranking(cfg, tok, device)
    scorer = MeanLogProbListwiseScorer(model, tok, cfg).to(device)

    samples = [s for s in load_jsonl(cfg.data_path) if s.get("candidates")][: args.num_samples]
    rng = random.Random(int(cfg.seed))

    curve = {R: defaultdict(lambda: defaultdict(list)) for R in r_grid}
    typical_by_stratum = defaultdict(list)
    raw_rows = []
    n_used, set_id = 0, 0

    with torch.no_grad():
        for sample in samples:
            cands = sample["candidates"]
            K = len(cands)
            rels = [int(c.get("relevance", 0)) for c in cands]

            # one isolated forward per candidate (order-independent)
            scores, ok = [], True
            for c in range(K):
                enc = tok(build_prompt(sample, [c], cfg), return_tensors="pt",
                          truncation=True, max_length=int(cfg.max_seq_length))
                sc = scorer(enc["input_ids"].to(device), enc["attention_mask"].to(device))
                if int(sc.numel()) != 1:
                    ok = False
                    break
                scores.append(float(sc[0]))
            if not ok:
                continue

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

            for c in targets:
                st = stratum(rels[c])
                fixed_rank = sum(1 for x in scores if x > scores[c])  # order-independent
                typical_by_stratum[st].append(float(fixed_rank))
                per_R = {}
                for R in r_grid:
                    m = metrics_at_R(float(fixed_rank), fixed_rank)  # adv == typical
                    per_R[R] = m
                    for k, v in m.items():
                        curve[R][st][k].append(v)
                if args.dump_raw:
                    raw_rows.append({"set_id": set_id, "stratum": st, "typical_rank": float(fixed_rank),
                                     "per_R": {str(R): per_R[R] for R in r_grid}})
            n_used += 1
            set_id += 1

    R_max = args.budget
    summary = {}
    for st in typical_by_stratum:
        acc = curve[R_max][st]
        summary[st] = {
            "n": len(acc["adv_rank"]), "typical_rank": mean(typical_by_stratum[st]),
            "adv_rank": mean(acc["adv_rank"]), "rank_gain_adv": mean(acc["rank_gain_adv"]),
            "weak_into_top5": mean(acc["weak_into_top5"]), "weak_into_top1": mean(acc["weak_into_top1"]),
            "promotion_into_top5": mean(acc["promotion_into_top5"]),
            "promotion_into_top1": mean(acc["promotion_into_top1"]),
        }
    curve_out = {str(R): {st: {k: mean(v) for k, v in acc.items()} for st, acc in curve[R].items()}
                 for R in r_grid}

    out = {"config": args.config, "budget_R": R_max, "r_grid": r_grid, "num_sets_used": n_used,
           "scorer": "pointwise (isolated single-candidate prompts; order-invariant by construction)",
           "summary": summary, "budget_curve": curve_out}
    if args.dump_raw:
        out["raw"] = raw_rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n=== pointwise adversarial (R_max={R_max}, grid={r_grid}): {args.config} ({n_used} sets) ===")
    for st, s in summary.items():
        print(f"  {st:12s} typical_rank={s['typical_rank']:5.2f} adv_rank={s['adv_rank']:5.2f} "
              f"gain={s['rank_gain_adv']:4.2f} | into_top5={s['weak_into_top5']:.3f} "
              f"promo@5={s['promotion_into_top5']:.3f} (n={s['n']})  [0 by construction]")


if __name__ == "__main__":
    main()
