#!/usr/bin/env python3
"""RQ2 stronger attacker — structured greedy vs. budget-R random search.

Structured greedy for a target c* (uses the RQ1 curve knowledge, robust across models):
place c* at a favorable input position (sweet spot) and scatter its strongest
competitors to the least-favorable positions (queue head + tail); fillers elsewhere.
One forward per set. Compare its result to the budget-R random-search attacker and to
the no-attack baseline. Structured >= random demonstrates our promo@5 is a LOWER bound.

  python position_bias/greedy_attack.py --config <rank_lft.yaml> -R 50 --fav-pos 5 \
         --num-samples 300 --out $PROJ/position_bias/greedy/greedy.json
"""
from __future__ import annotations

import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config
from datasets.utils import load_jsonl, set_seed
from model import load_model_for_ranking, load_tokenizer, select_device
from prompts import build_prompt
from ranking import MeanLogProbListwiseScorer


def stratum(rel):
    return "strong_pos" if rel >= 3 else ("weak_pos" if rel >= 1 else "irrelevant")


def score_perm(scorer, tok, cfg, sample, perm, device):
    import torch
    K = len(perm)
    enc = tok(build_prompt(sample, perm, cfg), return_tensors="pt",
              truncation=True, max_length=int(cfg.max_seq_length))
    sc = scorer(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    if int(sc.numel()) != K:
        return None
    return [float(x) for x in sc.detach().float().cpu().tolist()]


def rank_of(scores, pos):
    return sum(1 for x in scores if x > scores[pos])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-R", "--budget", type=int, default=50)
    ap.add_argument("--fav-pos", type=int, default=5, help="favorable position for c* (RQ1 sweet spot)")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    cfg = load_config(args.config); set_seed(int(cfg.seed))
    device = select_device(cfg.device); cfg.device = str(device)
    tok = load_tokenizer(cfg); model = load_model_for_ranking(cfg, tok, device)
    scorer = MeanLogProbListwiseScorer(model, tok, cfg).to(device)
    samples = [s for s in load_jsonl(cfg.data_path) if s.get("candidates")][: args.num_samples]
    rng = random.Random(int(cfg.seed))

    per = defaultdict(lambda: {"typical": [], "rand": [], "greedy": [],
                               "into5_rand": [], "into5_greedy": []})
    n = 0
    with torch.no_grad():
        for sample in samples:
            K = len(sample["candidates"])
            rels = [int(c.get("relevance", 0)) for c in sample["candidates"]]
            fav = min(max(args.fav_pos, 0), K - 1)
            by = defaultdict(list)
            for i, r in enumerate(rels):
                by[stratum(r)].append(i)
            targets = [rng.choice(by[s]) for s in ("irrelevant", "weak_pos") if by[s]]
            if not targets:
                continue

            # base (identity) scores -> competitor ranking by score
            base = score_perm(scorer, tok, cfg, sample, list(range(K)), device)
            if base is None:
                continue

            # budget-R random search (shared across targets)
            rand_ranks = defaultdict(list)
            for _ in range(args.budget):
                perm = list(range(K)); rng.shuffle(perm)
                sc = score_perm(scorer, tok, cfg, sample, perm, device)
                if sc is None:
                    continue
                loc = {perm[l]: l for l in range(K)}
                order = sorted(range(K), key=lambda j: sc[j], reverse=True)
                rk = {order[r]: r for r in range(K)}  # perm-local pos -> rank
                for c in range(K):
                    rand_ranks[c].append(rk[loc[c]])

            for c in targets:
                st = stratum(rels[c])
                # structured greedy arrangement
                others = [i for i in range(K) if i != c]
                others_by_score = sorted(others, key=lambda j: base[j], reverse=True)  # strongest first
                rem_pos = [p for p in range(K) if p != fav]
                # worst positions first = farthest from sweet spot (tail + head)
                worst_first = sorted(rem_pos, key=lambda p: abs(p - fav), reverse=True)
                arrangement = [None] * K
                arrangement[fav] = c
                for comp, pos in zip(others_by_score, worst_first):
                    arrangement[pos] = comp
                sc = score_perm(scorer, tok, cfg, sample, arrangement, device)
                greedy_rank = rank_of(sc, fav) if sc is not None else K

                typ = sum(rand_ranks[c]) / len(rand_ranks[c])
                radv = min(rand_ranks[c])
                d = per[st]
                d["typical"].append(typ); d["rand"].append(radv); d["greedy"].append(greedy_rank)
                d["into5_rand"].append(1.0 if radv < 5 else 0.0)
                d["into5_greedy"].append(1.0 if greedy_rank < 5 else 0.0)
            n += 1

    def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
    summary = {st: {"n": len(d["greedy"]), "typical_rank": mean(d["typical"]),
                    "random_adv_rank": mean(d["rand"]), "greedy_adv_rank": mean(d["greedy"]),
                    "into5_random": mean(d["into5_rand"]), "into5_greedy": mean(d["into5_greedy"])}
               for st, d in per.items()}
    out = {"config": args.config, "budget_R": args.budget, "fav_pos": args.fav_pos,
           "num_sets_used": n, "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n=== greedy vs random (R={args.budget}): {args.config} ({n} sets) ===")
    for st, s in summary.items():
        print(f"  {st:12s} typical={s['typical_rank']:5.2f} random_adv={s['random_adv_rank']:5.2f} "
              f"greedy_adv={s['greedy_adv_rank']:5.2f} | into5 random={s['into5_random']:.3f} "
              f"greedy={s['into5_greedy']:.3f} (n={s['n']})")
    print("  greedy_adv <= random_adv (lower rank = stronger) => random-search promo is a lower bound.")


if __name__ == "__main__":
    main()
