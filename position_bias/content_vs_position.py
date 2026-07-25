#!/usr/bin/env python3
"""RQ3 (light) — position vs. content, 2x2 separation. For a target c* (irrelevant),
cross two factors and measure c*'s output rank in each cell:

              content: original      content: boosted (inject user's top history genre)
  pos: orig     baseline                 pure content effect
  pos: fav      pure position effect     joint effect

Reports position_effect (fav vs orig at fixed content), content_effect (boosted vs
orig at fixed position), and their interaction — showing position is an independent,
comparable lever to content manipulation. Diagnostic only (no manipulation tooling).

  python position_bias/content_vs_position.py --config <rank_lft.yaml> --num-samples 300 \
         --fav-pos 4 --out $PROJ/position_bias/rq3/cvp.json
"""
from __future__ import annotations

import argparse, copy, json, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config
from datasets.utils import load_jsonl, set_seed
from model import load_model_for_ranking, load_tokenizer, select_device
from prompts import build_prompt
from ranking import MeanLogProbListwiseScorer


def output_rank(scores, idx):
    s = [float(x) for x in scores.detach().float().cpu().tolist()]
    return sum(1 for x in s if x > s[idx])


def top_history_genre(sample):
    c = Counter()
    for h in sample.get("history", []):
        for g in h.get("genres", []) or []:
            c[str(g)] += 1
    return c.most_common(1)[0][0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--fav-pos", type=int, default=4, help="favorable input position (RQ1 sweet spot)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    cfg = load_config(args.config); set_seed(int(cfg.seed))
    device = select_device(cfg.device); cfg.device = str(device)
    tok = load_tokenizer(cfg); model = load_model_for_ranking(cfg, tok, device)
    scorer = MeanLogProbListwiseScorer(model, tok, cfg).to(device)
    samples = [s for s in load_jsonl(cfg.data_path) if s.get("candidates")][: args.num_samples]
    rng = random.Random(int(cfg.seed))

    def rank_cell(sample, cstar, pos, boost_genre):
        smp = sample
        if boost_genre is not None:
            smp = copy.deepcopy(sample)
            g = list(smp["candidates"][cstar].get("genres", []) or [])
            if boost_genre not in g:
                g.append(boost_genre)
            smp["candidates"][cstar]["genres"] = g
        K = len(smp["candidates"])
        others = [i for i in range(K) if i != cstar]
        perm = others[:pos] + [cstar] + others[pos:]
        enc = tok(build_prompt(smp, perm, cfg), return_tensors="pt",
                  truncation=True, max_length=int(cfg.max_seq_length))
        sc = scorer(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        if int(sc.numel()) != K:
            return None
        return output_rank(sc, pos)

    pos_eff, cont_eff, inter, base_ranks = [], [], [], []
    n = 0
    with torch.no_grad():
        for sample in samples:
            cands = sample["candidates"]; K = len(cands)
            rels = [int(c.get("relevance", 0)) for c in cands]
            irr = [i for i in range(K) if rels[i] == 0]
            if not irr:
                continue
            cstar = rng.choice(irr)
            genre = top_history_genre(sample)
            if genre is None:
                continue
            orig_pos = cands[cstar].get("orig_index", 0)  # not used; base uses identity ordering
            r_oo = rank_cell(sample, cstar, 0, None)                 # orig pos, orig content
            r_fo = rank_cell(sample, cstar, args.fav_pos, None)      # fav pos,  orig content
            r_ob = rank_cell(sample, cstar, 0, genre)                # orig pos, boosted content
            r_jb = rank_cell(sample, cstar, args.fav_pos, genre)     # fav pos,  boosted content
            if None in (r_oo, r_fo, r_ob, r_jb):
                continue
            pos_eff.append(r_oo - r_fo)    # lower rank = better; positive = position helped
            cont_eff.append(r_oo - r_ob)   # positive = content boost helped
            inter.append((r_oo - r_jb) - (r_oo - r_fo) - (r_oo - r_ob))
            base_ranks.append(r_oo)
            n += 1

    def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
    out = {"config": args.config, "fav_pos": args.fav_pos, "n_used": n,
           "position_effect_ranks": mean(pos_eff),   # avg rank improvement from favorable position
           "content_effect_ranks": mean(cont_eff),   # avg rank improvement from genre-boost
           "interaction_ranks": mean(inter),
           "baseline_rank": mean(base_ranks)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n=== RQ3 position vs content: {args.config} ({n} irrelevant c*) ===")
    print(f"  position_effect={out['position_effect_ranks']:.2f} ranks  "
          f"content_effect={out['content_effect_ranks']:.2f} ranks  "
          f"interaction={out['interaction_ranks']:.2f}  (baseline_rank={out['baseline_rank']:.2f})")
    print("  => position is an independent lever; compare its magnitude to the content-boost lever.")


if __name__ == "__main__":
    main()
