#!/usr/bin/env python3
"""B4 — T5 ENCODER listwise scorer (bidirectional), trained with LambdaRank.

The generative ListT5 approach floored out (the ~22-item irrelevant tail of the gold
permutation is unlearnable noise -> invalid_rate=100%). This scorer avoids generation
entirely: the T5 ENCODER reads the whole candidate list at once (bidirectional
self-attention), we mean-pool each [ITEM]..[/ITEM] span from the encoder hidden states
and pass it through a linear head to a per-candidate score. Trained with the SAME
LambdaRank loss as the causal reranker (ties among irrelevants contribute ~0), so the
ONLY essential differences vs the causal model are (a) bidirectional vs causal
attention and (b) pooled-hidden+head vs mean-logprob scoring. No generation => no
invalid outputs.

Reuses the causal reranker's invarirank prompt, [SPAN]/[ITEM] special tokens, and
SpanExtractor, and exposes the same score_batch(sample, perms) interface as
ListT5Ranker so listt5_eval.py drives it unchanged (rank/scan/adv).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import resolve_dtype  # noqa: E402
from model.invarirank import SpanExtractor  # noqa: E402
from prompts import build_prompt  # noqa: E402

_SPECIAL_KEYS = ["span_start_token", "span_end_token", "item_start_token", "item_end_token"]


def build_scorer_tokenizer(cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)))
    specials = [getattr(cfg, k) for k in _SPECIAL_KEYS]
    tok.add_special_tokens({"additional_special_tokens": specials})
    return tok


class T5EncoderScorer:
    def __init__(self, cfg, device, for_training: bool = False):
        import torch.nn as nn
        from transformers import T5EncoderModel

        self.cfg = cfg
        self.device = device
        self.max_len = int(getattr(cfg, "max_seq_length", 1536))
        self.tok = build_scorer_tokenizer(cfg)

        dtype = resolve_dtype("float32") if for_training else resolve_dtype(getattr(cfg, "dtype", "bfloat16"))
        base = T5EncoderModel.from_pretrained(cfg.model_name, dtype=dtype,
                                              trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)))
        base.resize_token_embeddings(len(self.tok))
        hidden = base.config.d_model
        self.head = nn.Linear(hidden, 1)

        adapter = getattr(cfg, "adapter_path", None)
        if for_training:
            from peft import LoraConfig, TaskType, get_peft_model

            lconf = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=int(getattr(cfg, "lora_r", 16)),
                lora_alpha=int(getattr(cfg, "lora_alpha", 32)),
                lora_dropout=float(getattr(cfg, "lora_dropout", 0.05)),
                target_modules=list(getattr(cfg, "lora_target_modules", ["q", "k", "v", "o"])),
            )
            self.encoder = get_peft_model(base, lconf)
            self.encoder.print_trainable_parameters()
        elif adapter and Path(adapter).exists():
            import torch
            from peft import PeftModel

            self.encoder = PeftModel.from_pretrained(base, adapter)
            head_path = Path(adapter) / "head.pt"
            if head_path.exists():
                self.head.load_state_dict(torch.load(head_path, map_location="cpu"))
            self.head.to(dtype=self.head.weight.dtype)
            print(f"[listt5-scorer] loaded adapter + head <- {adapter}")
        else:
            print(f"[listt5-scorer] WARNING: no adapter at {adapter}; running an UNTRAINED scorer")
            self.encoder = base

        self.extractor = SpanExtractor(self.tok, cfg)
        self.encoder.to(device)
        self.head.to(device)
        self.n_calls = 0
        self.n_invalid = 0  # scorer never produces invalid outputs

    # ---- differentiable core: input_ids [1,L] -> per-candidate scores [K] (prompt order) ----
    def score_ids(self, input_ids, attention_mask):
        import torch

        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [1,L,H]
        span_info = self.extractor(input_ids)
        pooled = []
        for c0, c1 in span_info.candidate_spans:
            a, b = c0 + 1, c1 - 1  # drop the [ITEM]/[/ITEM] marker tokens
            if b <= a:
                a, b = c0, c1
            pooled.append(hs[0, a:b].mean(dim=0))
        pooled = torch.stack(pooled).to(self.head.weight.dtype)
        return self.head(pooled).squeeze(-1)  # [K]

    def train_mode(self):
        self.encoder.train()
        self.head.train()

    def eval_mode(self):
        self.encoder.eval()
        self.head.eval()

    def trainable_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad] + list(self.head.parameters())

    @property
    def invalid_rate(self) -> float:
        return 0.0

    def score_batch(self, sample: dict, perms: list[list[int]]) -> list[list[float]]:
        """Per-candidate scores aligned to LOCAL positions, one encoder forward per perm."""
        import torch

        out = []
        with torch.no_grad():
            for perm in perms:
                K = len(perm)
                prompt = build_prompt(sample, perm, self.cfg)
                enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_len)
                ids = enc["input_ids"].to(self.device)
                am = enc["attention_mask"].to(self.device)
                try:
                    sc = self.score_ids(ids, am)
                    vals = [float(x) for x in sc.detach().float().cpu().tolist()]
                except ValueError:
                    vals = []
                if len(vals) != K:  # truncation dropped a span (rare) -> neutral fallback
                    vals = [0.0] * K
                self.n_calls += 1
                out.append(vals)
        return out

    def save(self, ckpt_dir: str):
        import torch

        d = Path(ckpt_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(str(d))
        torch.save(self.head.state_dict(), str(d / "head.pt"))
        self.tok.save_pretrained(str(d))
        print(f"[listt5-scorer] saved adapter + head -> {d}")
