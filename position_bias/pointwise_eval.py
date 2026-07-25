#!/usr/bin/env python3
"""Pointwise reference baseline. Score each candidate INDEPENDENTLY (history + that
one candidate), then rank by score. This is position-invariant by construction (no
other candidates in context → no order to exploit), but costs O(K) forward passes
per query and drops the listwise comparison signal.

We reuse the given (causal) rank config's trained adapter but score one candidate at
a time — i.e., the same reranker used pointwise (an inference-mode ablation). Reports
nDCG@10 / HR@10 (quality) and cost=K forwards. Attack surface is 0 by construction.

  python position_bias/pointwise_eval.py --config <rank_lft.yaml> --num-samples 500 \
         --out $PROJ/position_bias/pointwise/pw.json
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config
from datasets.utils import load_jsonl, set_seed
from model import load_model_for_ranking, load_tokenizer, select_device
from prompts import build_prompt
from ranking import MeanLogProbListwiseScorer


def ndcg_at_k(order, rel, k=10):
    def dcg(rs): return sum((2.0**r - 1.0)/math.log2(i+2) for i, r in enumerate(rs))
    gains = [rel[c] for c in order[:k]]
    ideal = sorted(rel, reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains)/idcg if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a (causal) rank config; its adapter is scored pointwise")
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    cfg = load_config(args.config); set_seed(int(cfg.seed))
    device = select_device(cfg.device); cfg.device = str(device)
    tok = load_tokenizer(cfg); model = load_model_for_ranking(cfg, tok, device)
    scorer = MeanLogProbListwiseScorer(model, tok, cfg).to(device)
    samples = [s for s in load_jsonl(cfg.data_path) if s.get("candidates")][: args.num_samples]

    ndcg5, ndcg10, hr10 = [], [], []
    total_forwards = 0; n = 0
    with torch.no_grad():
        for sample in samples:
            cands = sample["candidates"]; K = len(cands)
            rel = [int(c.get("relevance", 0)) for c in cands]
            if max(rel) <= 0:
                continue
            scores = []
            ok = True
            for c in range(K):
                enc = tok(build_prompt(sample, [c], cfg), return_tensors="pt",
                          truncation=True, max_length=int(cfg.max_seq_length))
                sc = scorer(enc["input_ids"].to(device), enc["attention_mask"].to(device))
                if int(sc.numel()) != 1:
                    ok = False; break
                scores.append(float(sc[0]))
            if not ok:
                continue
            total_forwards += K
            order = sorted(range(K), key=lambda c: scores[c], reverse=True)
            ndcg5.append(ndcg_at_k(order, rel, 5)); ndcg10.append(ndcg_at_k(order, rel, 10))
            hr10.append(1.0 if any(rel[c] > 0 for c in order[:10]) else 0.0)
            n += 1

    def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
    out = {"config": args.config, "n_used": n,
           "ndcg@5": mean(ndcg5), "ndcg@10": mean(ndcg10), "hr@10": mean(hr10),
           "avg_forwards_per_query": total_forwards / max(n, 1),
           "attack_surface": "0 by construction (each candidate scored in isolation)"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n=== pointwise: {args.config} ({n} queries) ===")
    print(f"  nDCG@10={out['ndcg@10']:.4f}  nDCG@5={out['ndcg@5']:.4f}  HR@10={out['hr@10']:.4f}  "
          f"cost={out['avg_forwards_per_query']:.0f} forwards/query  attack_surface=0 (invariant by construction)")


if __name__ == "__main__":
    main()
