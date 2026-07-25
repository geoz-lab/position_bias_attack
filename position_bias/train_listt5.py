#!/usr/bin/env python3
"""B4 training — fine-tune a T5 (encoder-decoder) listwise ranker with LoRA.

Learns to GENERATE the gold ranking (comma-separated LOCAL candidate indices, best
first) from the listwise prompt. A fresh random input permutation is drawn each step
so the model sees many orders (the target is the permutation-invariant relevance
ordering, mapped to local positions). Teacher-forced cross-entropy. After training,
listt5_eval.py loads the saved adapter and runs the same attack probes.

  python position_bias/train_listt5.py --config <train_listt5.yaml>
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
from model import resolve_dtype, select_device
from listt5_ranker import build_listwise_prompt, gold_local_order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="train_listt5.yaml")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.optim import AdamW
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = select_device(getattr(cfg, "device", "cuda"))

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    dtype = resolve_dtype(getattr(cfg, "dtype", "float32"))  # small model -> fp32 by default for stable training
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, dtype=dtype)
    lconf = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(getattr(cfg, "lora_r", 16)),
        lora_alpha=int(getattr(cfg, "lora_alpha", 32)),
        lora_dropout=float(getattr(cfg, "lora_dropout", 0.05)),
        target_modules=list(getattr(cfg, "lora_target_modules", ["q", "k", "v", "o"])),
    )
    model = get_peft_model(model, lconf)
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    samples = [s for s in load_jsonl(cfg.train_path) if s.get("candidates") and s.get("target_ranking")]
    if not samples:
        raise SystemExit("no training samples with candidates + target_ranking")
    rng = random.Random(int(cfg.seed))

    steps = int(getattr(cfg, "total_optimizer_steps", 500))
    accum = int(getattr(cfg, "gradient_accumulation_steps", 16))
    maxlen = int(getattr(cfg, "max_seq_length", 1536))
    tgtlen = int(getattr(cfg, "target_max_length", 128))
    mgn = float(getattr(cfg, "max_grad_norm", 1.0))
    opt = AdamW(model.parameters(), lr=float(getattr(cfg, "learning_rate", 1e-3)),
                weight_decay=float(getattr(cfg, "weight_decay", 0.0)))
    sched = get_linear_schedule_with_warmup(opt, int(getattr(cfg, "warmup_steps", 10)), steps)

    def sample_stream():
        while True:
            rng.shuffle(samples)
            for s in samples:
                yield s

    stream = sample_stream()
    opt.zero_grad()
    for step in range(steps):
        loss_acc = 0.0
        for _ in range(accum):
            s = next(stream)
            K = len(s["candidates"])
            perm = list(range(K))
            rng.shuffle(perm)
            prompt = build_listwise_prompt(s, perm)
            target = ", ".join(str(x) for x in gold_local_order(s, perm))
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=maxlen).to(device)
            labels = tok(target, return_tensors="pt", truncation=True, max_length=tgtlen).input_ids.to(device)
            labels[labels == tok.pad_token_id] = -100
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
            (out.loss / accum).backward()
            loss_acc += float(out.loss) / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), mgn)
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 25 == 0 or step == steps - 1:
            print(f"[train_listt5] step {step + 1}/{steps} loss={loss_acc:.4f}", flush=True)

    ckpt_dir = getattr(cfg, "checkpoint_dir", None) or str(Path(cfg.run_dir) / "checkpoints")
    outdir = Path(ckpt_dir) / "final"
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(outdir))
    tok.save_pretrained(str(outdir))
    print(f"[train_listt5] saved adapter -> {outdir}")


if __name__ == "__main__":
    main()
