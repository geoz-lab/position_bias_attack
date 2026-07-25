#!/usr/bin/env python3
"""B4 training — fine-tune the T5 encoder listwise scorer with LambdaRank.

SAME loss and hyperparameters as the causal LFT (LambdaRank, lr=5e-5, LoRA r16/a32
on [q,k,v,o], 500 steps, eff. batch 16, seed 42) — only the architecture differs
(bidirectional T5 encoder + linear head vs causal decoder + mean-logprob). A fresh
random input permutation is drawn each step; relevance is read in that same order.

  python position_bias/train_listt5_scorer.py --config <train_listt5_scorer.yaml>
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from datasets.utils import load_jsonl, set_seed
from model import select_device
from training.losses import lambda_rank_loss
from listt5_scorer import T5EncoderScorer
from prompts import build_prompt


def listwise_softmax_ce(scores, relevance):
    """RankT5-style listwise softmax cross-entropy: target mass proportional to gains
    (2^rel - 1), predicted = softmax over the list's scores. Larger, better-conditioned
    gradients than LambdaRank's dNDCG-weighted pairwise loss (which floored at lr=5e-5)."""
    import torch
    import torch.nn.functional as F

    gains = torch.pow(2.0, relevance.float()) - 1.0
    Z = gains.sum()
    if Z <= 0:
        return torch.tensor(0.0, device=scores.device)
    target = gains / Z
    logp = F.log_softmax(scores.float(), dim=-1)
    return -(target * logp).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="train_listt5_scorer.yaml")
    args = ap.parse_args()

    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = select_device(getattr(cfg, "device", "cuda"))

    scorer = T5EncoderScorer(cfg, device, for_training=True)
    scorer.train_mode()

    samples = [s for s in load_jsonl(cfg.train_path) if s.get("candidates")]
    if not samples:
        raise SystemExit("no training samples with candidates")
    rng = random.Random(int(cfg.seed))

    steps = int(getattr(cfg, "total_optimizer_steps", 500))
    accum = int(getattr(cfg, "gradient_accumulation_steps", 16))
    maxlen = int(getattr(cfg, "max_seq_length", 1536))
    mgn = float(getattr(cfg, "max_grad_norm", 1.0))
    sigma = float(getattr(cfg, "lambda_rank_sigma", 1.0))
    loss_name = str(getattr(cfg, "rank_loss", "softmax_ce")).lower()  # softmax_ce (RankT5) | lambdarank
    print(f"[train_listt5_scorer] loss={loss_name} lr={getattr(cfg, 'learning_rate', 5e-5)}", flush=True)
    params = scorer.trainable_parameters()
    opt = AdamW(params, lr=float(getattr(cfg, "learning_rate", 5e-5)),
                weight_decay=float(getattr(cfg, "weight_decay", 0.0)))
    sched = get_linear_schedule_with_warmup(opt, int(getattr(cfg, "warmup_steps", 10)), steps)

    def stream():
        while True:
            rng.shuffle(samples)
            for s in samples:
                yield s

    it = stream()
    opt.zero_grad()
    for step in range(steps):
        loss_acc = 0.0
        n_nonzero = 0
        for _ in range(accum):
            s = next(it)
            K = len(s["candidates"])
            perm = list(range(K))
            rng.shuffle(perm)
            prompt = build_prompt(s, perm, cfg)
            enc = scorer.tok(prompt, return_tensors="pt", truncation=True, max_length=maxlen)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            try:
                scores = scorer.score_ids(ids, am)
            except ValueError:
                continue
            if scores.numel() != K:
                continue  # truncation dropped a span
            rel = torch.tensor([int(s["candidates"][perm[i]].get("relevance", 0)) for i in range(K)],
                               device=device).float()
            loss = (lambda_rank_loss(scores, rel, sigma=sigma) if loss_name == "lambdarank"
                    else listwise_softmax_ce(scores, rel))
            if float(loss) == 0.0:
                continue  # all-equal relevance -> no signal
            (loss / accum).backward()
            loss_acc += float(loss) / accum
            n_nonzero += 1
        torch.nn.utils.clip_grad_norm_(params, mgn)
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 25 == 0 or step == steps - 1:
            print(f"[train_listt5_scorer] step {step + 1}/{steps} loss={loss_acc:.4f} (nz={n_nonzero}/{accum})",
                  flush=True)

    ckpt = getattr(cfg, "checkpoint_dir", None) or str(Path(cfg.run_dir) / "checkpoints")
    scorer.save(str(Path(ckpt) / "final"))


if __name__ == "__main__":
    main()
