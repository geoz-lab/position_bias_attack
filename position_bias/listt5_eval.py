#!/usr/bin/env python3
"""B4 driver — run the encoder-decoder (ListT5/Flan-T5) ranker through the SAME
three probes as the causal models, so its row is directly comparable:

  rank : write ranked_lists.json (build_rank_record schema) -> scripts/evaluate_ranking.py
         gives nDCG@10 / HR / Kendall tau. Also records the invalid-output rate.
  scan : RQ1 position curve (curve_range)                    -> scan_<tag>.json
  adv  : RQ2 budget-R attacker (promo@5, rank_gain, into5/1) -> adv_<tag>.json
         (same budget-curve + raw-dump format as adversarial_perm.py, so
          bootstrap_ci.py works on ListT5 too)

  python position_bias/listt5_eval.py --config <rank_listt5.yaml> --task all \
         --num-samples 300 -R 50 --out-dir $PROJ/position_bias/listt5
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
from datasets.utils import load_jsonl, set_seed, write_json
from model import select_device
from ranking.scoring import build_rank_record

# reuse the exact attack-metric helpers so ListT5 numbers are defined identically
from adversarial_perm import metrics_at_R, parse_r_grid, stratum
from listt5_ranker import ListT5Ranker, candidate_ids_for


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def output_rank(scores: list[float], pos: int) -> int:
    """0-indexed rank (0=best) of the candidate at local position `pos`."""
    target = scores[pos]
    return sum(1 for x in scores if x > target)


# --------------------------------------------------------------------------- rank
def run_rank(ranker, samples, cfg, rng, out_path):
    import torch

    n_perms = int(getattr(cfg, "eval_num_permutations", 10))
    records = []
    for si, sample in enumerate(samples):
        cands = sample["candidates"]
        K = len(cands)
        cand_ids = candidate_ids_for(sample)
        perms = []
        for _ in range(n_perms):
            perm = list(range(K))
            rng.shuffle(perm)
            perms.append(perm)
        sc_list = ranker.score_batch(sample, perms)
        scores_list = [torch.tensor(sc, dtype=torch.float32) for sc in sc_list]
        rel_seqs = [[int(cands[idx].get("relevance", 0)) for idx in perm] for perm in perms]
        batch = {
            "sample_index": si, "user_id": sample.get("user_id"), "split": sample.get("split"),
            "list_length": K, "num_items": K, "history": sample.get("history", []),
            "candidates": cands, "candidate_ids": cand_ids, "relevance": rel_seqs,
        }
        records.append(build_rank_record(batch, scores_list, perms))
    write_json(records, out_path)
    return {"n_records": len(records)}


# --------------------------------------------------------------------------- scan
def run_scan(ranker, samples, cfg, rng, out_path):
    ranks = defaultdict(list)
    into5, into1 = defaultdict(list), defaultdict(list)
    n_used = 0
    for sample in samples:
        cands = sample["candidates"]
        K = len(cands)
        rels = [int(c.get("relevance", 0)) for c in cands]
        base = list(range(K))
        by_stratum = defaultdict(list)
        for i, r in enumerate(rels):
            by_stratum[stratum(r)].append(i)
        cstars = [rng.choice(by_stratum[st]) for st in ("strong_pos", "weak_pos", "irrelevant")
                  if by_stratum[st]]
        if not cstars:
            continue
        used = False
        for cstar in cstars:
            st = stratum(rels[cstar])
            others = [i for i in base if i != cstar]
            perms = [others[:pos] + [cstar] + others[pos:] for pos in range(K)]
            sc_list = ranker.score_batch(sample, perms)
            best = None
            for pos in range(K):
                orank = output_rank(sc_list[pos], pos)
                ranks[(st, pos)].append(orank)
                best = orank if best is None else min(best, orank)
            used = True
            if best is not None:
                into5[st].append(1.0 if best < 5 else 0.0)
                into1[st].append(1.0 if best == 0 else 0.0)
        if used:
            n_used += 1

    strata = sorted({st for (st, _p) in ranks})
    positions = sorted({p for (_st, p) in ranks})
    curve = {st: {int(p): {"mean_output_rank": mean(ranks[(st, p)]), "n": len(ranks[(st, p)])}
                  for p in positions if (st, p) in ranks} for st in strata}
    summary = {}
    for st in strata:
        means = [curve[st][p]["mean_output_rank"] for p in sorted(curve[st])]
        summary[st] = {
            "position_effect_range": (max(means) - min(means)) if means else float("nan"),
            "best_mean_rank": min(means) if means else float("nan"),
            "worst_mean_rank": max(means) if means else float("nan"),
            "best_case_into_top5_rate": mean(into5.get(st, [])),
            "best_case_into_top1_rate": mean(into1.get(st, [])),
            "n_cstar": len(into5.get(st, [])),
        }
    write_json({"config": "listt5", "num_samples_used": n_used, "curve": curve, "summary": summary},
               out_path)
    return summary


# --------------------------------------------------------------------------- adv
def run_adv(ranker, samples, cfg, rng, out_path, r_max, r_grid, targets_per_set, dump_raw):
    curve = {R: defaultdict(lambda: defaultdict(list)) for R in r_grid}
    typical_by_stratum = defaultdict(list)
    raw_rows = []
    n_used, set_id = 0, 0
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
        targets = targets[:targets_per_set]
        if not targets:
            continue

        perms = []
        for _ in range(r_max):
            perm = list(range(K))
            rng.shuffle(perm)
            perms.append(perm)
        sc_list = ranker.score_batch(sample, perms)
        ranks_by_cand = []
        for perm, sc in zip(perms, sc_list):
            order = sorted(range(K), key=lambda j: sc[j], reverse=True)
            rank_of_local = {loc: r for r, loc in enumerate(order)}
            ranks_by_cand.append({perm[loc]: rank_of_local[loc] for loc in range(K)})

        for c in targets:
            st = stratum(rels[c])
            rr = [rb[c] for rb in ranks_by_cand]
            typical = mean(rr)
            typical_by_stratum[st].append(typical)
            per_R, run_min, gi = {}, None, 0
            for r_idx in range(1, r_max + 1):
                run_min = rr[r_idx - 1] if run_min is None else min(run_min, rr[r_idx - 1])
                if gi < len(r_grid) and r_idx == r_grid[gi]:
                    m = metrics_at_R(typical, run_min)
                    per_R[r_grid[gi]] = m
                    for k, v in m.items():
                        curve[r_grid[gi]][st][k].append(v)
                    gi += 1
            if dump_raw:
                raw_rows.append({"set_id": set_id, "stratum": st, "typical_rank": typical,
                                 "per_R": {str(R): per_R[R] for R in r_grid}})
        n_used += 1
        set_id += 1

    summary = {}
    for st in typical_by_stratum:
        acc = curve[r_max][st]
        summary[st] = {
            "n": len(acc["adv_rank"]), "typical_rank": mean(typical_by_stratum[st]),
            "adv_rank": mean(acc["adv_rank"]), "rank_gain_adv": mean(acc["rank_gain_adv"]),
            "weak_into_top5": mean(acc["weak_into_top5"]), "weak_into_top1": mean(acc["weak_into_top1"]),
            "promotion_into_top5": mean(acc["promotion_into_top5"]),
            "promotion_into_top1": mean(acc["promotion_into_top1"]),
        }
    curve_out = {str(R): {st: {k: mean(v) for k, v in acc.items()} for st, acc in curve[R].items()}
                 for R in r_grid}
    out = {"config": "listt5", "budget_R": r_max, "r_grid": r_grid, "num_sets_used": n_used,
           "summary": summary, "budget_curve": curve_out}
    if dump_raw:
        out["raw"] = raw_rows
    write_json(out, out_path)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="rank_listt5.yaml")
    ap.add_argument("--task", default="all", choices=["all", "rank", "scan", "adv"])
    ap.add_argument("--scorer", default="generate", choices=["generate", "encode"],
                    help="'generate' = ListT5-style generative ranker; 'encode' = T5 encoder LambdaRank scorer")
    ap.add_argument("--num-samples", type=int, default=300, help="samples for scan/adv probes")
    ap.add_argument("--rank-num-samples", type=int, default=None,
                    help="samples for the rank/Table-1 task (default: cfg.ranking_num_samples, else --num-samples) "
                         "-- use 2000 to match the causal Table-1 nDCG/tau")
    ap.add_argument("-R", "--budget", type=int, default=50)
    ap.add_argument("--r-grid", default="1,5,10,20,50,100")
    ap.add_argument("--targets-per-set", type=int, default=2)
    ap.add_argument("--no-dump-raw", dest="dump_raw", action="store_false", default=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(getattr(cfg, "seed", 42)))
    device = select_device(getattr(cfg, "device", "cuda"))
    cfg.device = str(device)
    if args.scorer == "encode":
        from listt5_scorer import T5EncoderScorer
        ranker = T5EncoderScorer(cfg, device, for_training=False)
    else:
        ranker = ListT5Ranker(cfg, device)

    all_samples = [s for s in load_jsonl(cfg.data_path) if s.get("candidates")]
    n_rank = args.rank_num_samples or int(getattr(cfg, "ranking_num_samples", 0)) or args.num_samples
    rank_samples = all_samples[:n_rank]           # Table-1 nDCG/tau (parity with causal: 2000)
    probe_samples = all_samples[:args.num_samples]  # scan/adv probes (300)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r_grid = parse_r_grid(args.r_grid, args.budget)

    results = {}
    if args.task in ("all", "rank"):
        rng = random.Random(int(getattr(cfg, "seed", 42)))
        ranked_path = Path(getattr(cfg, "ranked_lists_path", out_dir / "ranked_lists.json"))
        results["rank"] = run_rank(ranker, rank_samples, cfg, rng, ranked_path)
        print(f"[listt5] rank ({len(rank_samples)} samples) -> {ranked_path}  (run scripts/evaluate_ranking.py on it)")
    if args.task in ("all", "scan"):
        rng = random.Random(int(getattr(cfg, "seed", 42)) + 1)
        results["scan"] = run_scan(ranker, probe_samples, cfg, rng, out_dir / "scan_listt5.json")
    if args.task in ("all", "adv"):
        rng = random.Random(int(getattr(cfg, "seed", 42)) + 2)
        results["adv"] = run_adv(ranker, probe_samples, cfg, rng, out_dir / "adv_listt5.json",
                                 args.budget, r_grid, args.targets_per_set, args.dump_raw)

    inval = ranker.invalid_rate
    (out_dir / "invalid_rate.json").write_text(json.dumps(
        {"invalid_rate": inval, "n_calls": ranker.n_calls, "n_invalid": ranker.n_invalid}, indent=2))

    print(f"\n############ ListT5 SUMMARY (invalid_output_rate={inval:.4f}, calls={ranker.n_calls}) ############")
    if "scan" in results and "irrelevant" in results["scan"]:
        print(f"  curve_range(irrelevant) = {results['scan']['irrelevant']['position_effect_range']:.2f}")
    if "adv" in results and "irrelevant" in results["adv"]:
        a = results["adv"]["irrelevant"]
        print(f"  promo@5={a['promotion_into_top5']:.3f} rank_gain={a['rank_gain_adv']:.2f} "
              f"into5={a['weak_into_top5']:.3f} into1={a['weak_into_top1']:.3f}")


if __name__ == "__main__":
    main()
