#!/usr/bin/env python3
"""RQ1 / §3a — single-candidate position scan (causal curve of position bias).

For each test candidate set we pick target candidates c* (stratified by TRUE
relevance), then slide c* through every input position 0..K-1 while keeping the
other K-1 candidates in a fixed base order, score once per position, and record
c*'s OUTPUT rank (0 = ranked first). Aggregated over many sets this gives the
causal curve: mean output-rank of c* as a function of its input position.

Run it once with the LFT rank config and once with the InvariRank rank config
(same code path; only the config's attention_mask/position_ids differ). A sloped
curve for LFT + a flat curve for InvariRank is the RQ1 + RQ4 headline.

  python position_bias/scan_position.py --config <rank_lft.yaml> --num-samples 300 \
         --out $PROJ/position_bias/scan_lft.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo root (imports model/, ranking/, ...)
sys.path.insert(0, str(ROOT))

from config import load_config
from datasets.utils import load_jsonl, set_seed
from model import load_model_for_ranking, load_tokenizer, select_device
from prompts import build_prompt
from ranking import MeanLogProbListwiseScorer


def stratum(rel: int) -> str:
    if rel >= 3:
        return "strong_pos"      # forced-in future positives (rating>=3)
    if rel >= 1:
        return "weak_pos"        # graded-weak positives (rare)
    return "irrelevant"          # relevance-0 negatives / fillers


def output_rank(scores, idx: int) -> int:
    """0-indexed rank of the candidate at input position `idx` (0 = best).
    Ties broken optimistically for c* (counts strictly-greater scores)."""
    s = [float(x) for x in scores.detach().float().cpu().tolist()]
    target = s[idx]
    return sum(1 for x in s if x > target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="rank_lft.yaml or rank_invarirank.yaml")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--cstar-per-set", type=int, default=3,
                    help="how many c* to sample per set (stratified: 1 strong, 1 weak, 1 irrelevant)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = select_device(cfg.device)
    cfg.device = str(device)

    tokenizer = load_tokenizer(cfg)
    model = load_model_for_ranking(cfg, tokenizer, device)
    scorer = MeanLogProbListwiseScorer(model, tokenizer, cfg).to(device)

    samples = [s for s in load_jsonl(cfg.data_path) if s.get("candidates")][: args.num_samples]
    rng = random.Random(int(cfg.seed))

    # (stratum, position) -> list of output ranks;  also promotion into top-k
    ranks = defaultdict(list)
    into_top5 = defaultdict(list)   # stratum -> per-c* best (min over positions) reached top-5? (0/1)
    into_top1 = defaultdict(list)
    K_seen = defaultdict(int)
    n_used = 0

    with torch.no_grad():
        for sample in samples:
            cands = sample["candidates"]
            K = len(cands)
            rels = [int(c.get("relevance", 0)) for c in cands]
            base = list(range(K))

            by_stratum = defaultdict(list)
            for i, r in enumerate(rels):
                by_stratum[stratum(r)].append(i)
            cstars = [rng.choice(v) for st in ("strong_pos", "weak_pos", "irrelevant")
                      for v in ([by_stratum[st]] if by_stratum[st] else [])]
            if not cstars:
                continue

            used_this_sample = False
            for cstar in cstars:
                st = stratum(rels[cstar])
                others = [i for i in base if i != cstar]
                best_rank = None
                for pos in range(K):
                    perm = others[:pos] + [cstar] + others[pos:]
                    prompt = build_prompt(sample, perm, cfg)
                    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=int(cfg.max_seq_length))
                    input_ids = enc["input_ids"].to(device)
                    attn = enc["attention_mask"].to(device)
                    scores = scorer(input_ids, attn)
                    if int(scores.numel()) != K:
                        continue  # truncation dropped a candidate span; skip this measurement
                    orank = output_rank(scores, pos)
                    ranks[(st, pos)].append(orank)
                    best_rank = orank if best_rank is None else min(best_rank, orank)
                    used_this_sample = True
                if best_rank is not None:
                    into_top5[st].append(1.0 if best_rank < 5 else 0.0)
                    into_top1[st].append(1.0 if best_rank == 0 else 0.0)
                    K_seen[K] += 1
            if used_this_sample:
                n_used += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    strata = sorted({st for (st, _p) in ranks})
    positions = sorted({p for (_st, p) in ranks})
    curve = {st: {int(p): {"mean_output_rank": mean(ranks[(st, p)]), "n": len(ranks[(st, p)])}
                  for p in positions if (st, p) in ranks}
             for st in strata}

    summary = {}
    for st in strata:
        means = [curve[st][p]["mean_output_rank"] for p in sorted(curve[st])]
        summary[st] = {
            "position_effect_range": (max(means) - min(means)) if means else float("nan"),  # curve span
            "best_mean_rank": min(means) if means else float("nan"),   # most-favorable position
            "worst_mean_rank": max(means) if means else float("nan"),  # least-favorable position
            "best_case_into_top5_rate": mean(into_top5.get(st, [])),   # c* reachable into top-5 by position alone
            "best_case_into_top1_rate": mean(into_top1.get(st, [])),
            "n_cstar": len(into_top5.get(st, [])),
        }

    out = {
        "config": args.config,
        "num_samples_used": n_used,
        "K_distribution": dict(K_seen),
        "curve": curve,
        "summary": summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # readable print
    print(f"\n=== position scan: {args.config} ({n_used} sets used) ===")
    print("mean OUTPUT rank of c* by INPUT position (0=best). Flat => position-invariant.")
    header = "stratum".ljust(12) + " | " + " ".join(f"p{p:02d}" for p in positions)
    print(header)
    for st in strata:
        row = st.ljust(12) + " | " + " ".join(
            (f"{curve[st][p]['mean_output_rank']:4.1f}" if p in curve[st] else "   .") for p in positions)
        print(row)
    print("\nsummary (per stratum):")
    for st in strata:
        s = summary[st]
        print(f"  {st:12s} range={s['position_effect_range']:.2f}  "
              f"best_rank={s['best_mean_rank']:.2f} worst_rank={s['worst_mean_rank']:.2f}  "
              f"best->top5={s['best_case_into_top5_rate']:.3f} best->top1={s['best_case_into_top1_rate']:.3f}  "
              f"(n={s['n_cstar']})")


if __name__ == "__main__":
    main()
