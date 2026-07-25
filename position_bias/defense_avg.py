#!/usr/bin/env python3
"""Defense C — test-time order averaging. For the causal reranker, score each
candidate as the mean over P internal random permutations, then rank. This makes
the output (approximately) order-independent, so an ordering attacker gains nothing
— at O(P) forward-pass cost and only approximate invariance (finite P).

We report, per P: nDCG@10 (quality under averaged scoring), and the irrelevant
into_top5 rate (= the residual attack surface / content base rate, since ordering
no longer moves anything). Compare against the single-pass causal baseline (P=1).

  python position_bias/defense_avg.py --config <rank_lft.yaml> --avg-perms 1,5,10,20 \
         --num-samples 300 --out $PROJ/position_bias/defC/avg.json
"""
from __future__ import annotations

import argparse, json, math, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config
from datasets.utils import load_jsonl, set_seed
from model import load_model_for_ranking, load_tokenizer, select_device
from prompts import build_prompt
from ranking import MeanLogProbListwiseScorer


def ndcg_at_k(order, rel_by_cand, k=10):
    def dcg(rs): return sum((2.0**r - 1.0)/math.log2(i+2) for i, r in enumerate(rs))
    gains = [rel_by_cand.get(c, 0) for c in order[:k]]
    ideal = sorted(rel_by_cand.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains)/idcg if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a CAUSAL rank config")
    ap.add_argument("--avg-perms", default="1,5,10,20", help="comma list of P")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Ps = [int(x) for x in args.avg_perms.split(",")]
    Pmax = max(Ps)

    import torch
    cfg = load_config(args.config); set_seed(int(cfg.seed))
    device = select_device(cfg.device); cfg.device = str(device)
    tok = load_tokenizer(cfg); model = load_model_for_ranking(cfg, tok, device)
    scorer = MeanLogProbListwiseScorer(model, tok, cfg).to(device)
    samples = [s for s in load_jsonl(cfg.data_path) if s.get("candidates")][: args.num_samples]
    rng = random.Random(int(cfg.seed))

    # accumulate per P: nDCG@10 list, and irrelevant-into-top5 flags
    ndcg = defaultdict(list); into5 = defaultdict(list); n_used = 0
    with torch.no_grad():
        for sample in samples:
            cands = sample["candidates"]; K = len(cands)
            rels = [int(c.get("relevance", 0)) for c in cands]
            rel_by_cand = {i: rels[i] for i in range(K)}
            if max(rels) <= 0:
                continue
            # Pmax random perms; per-candidate cumulative score
            cum = [0.0]*K; got = 0
            for _ in range(Pmax):
                perm = list(range(K)); rng.shuffle(perm)
                enc = tok(build_prompt(sample, perm, cfg), return_tensors="pt",
                          truncation=True, max_length=int(cfg.max_seq_length))
                sc = scorer(enc["input_ids"].to(device), enc["attention_mask"].to(device))
                if int(sc.numel()) != K:
                    continue
                s = [float(x) for x in sc.detach().float().cpu().tolist()]
                for loc in range(K):
                    cum[perm[loc]] += s[loc]
                got += 1
                if got in Ps:
                    avg = [cum[c]/got for c in range(K)]
                    order = sorted(range(K), key=lambda c: avg[c], reverse=True)
                    ndcg[got].append(ndcg_at_k(order, rel_by_cand, 10))
                    top5 = set(order[:5])
                    for c in range(K):
                        if rels[c] == 0:
                            into5[got].append(1.0 if c in top5 else 0.0)
            n_used += 1

    def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
    summary = {P: {"ndcg@10": mean(ndcg[P]), "irrelevant_into_top5": mean(into5[P]),
                   "forward_passes": P} for P in Ps if ndcg[P]}
    out = {"config": args.config, "n_used": n_used, "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n=== Defense C (test-time averaging): {args.config} ({n_used} sets) ===")
    print("P=1 is the single-pass causal baseline. Averaging removes order-dependence at O(P).")
    for P in sorted(summary):
        s = summary[P]
        print(f"  P={P:2d}  nDCG@10={s['ndcg@10']:.4f}  irrelevant_into_top5={s['irrelevant_into_top5']:.3f}  cost={P}x forwards")


if __name__ == "__main__":
    main()
