#!/usr/bin/env python3
"""B4 — encoder-decoder (ListT5 / Flan-T5-style) listwise ranker.

The paper's attack is characterized on decoder-only *causal* rerankers. This module
provides an encoder-decoder comparison point: does the order-only attack surface
persist when candidates are encoded bidirectionally and the ranking is *generated*
(a permutation of candidate indices) instead of read off per-candidate log-probs?

Design so the existing position-bias probes can consume it unchanged: the ranker's
`.score(sample, perm)` returns a length-K list of pseudo-scores aligned to the
prompt's LOCAL positions (position i = the candidate placed i-th under `perm`),
exactly like MeanLogProbListwiseScorer. Pseudo-score = K - output_rank, so
higher = ranked better; scan_position / adversarial_perm math works verbatim.

Invalid-output handling (must be reported): the model may emit indices that are
out of range, duplicated, or missing. We parse integers left-to-right, keep the
first occurrence of each valid in-range index, then append any missing indices in
ascending order. A generation is counted "invalid" (needed repair) if the parsed
valid-unique sequence is not already a full permutation of 0..K-1.

NOTE (scaffold): default backbone is Flan-T5 used ZERO-SHOT via an instruction to
emit the ranking. Swapping in real ListT5 weights (castorini/listt5-*) or fine-
tuning on our history-based format is a config change + a TODO documented in
render_listt5.sh; the tournament/FiD decoding of the original ListT5 is not
reproduced here.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import resolve_dtype  # noqa: E402
from prompts.base import candidate_id, format_candidate_item, format_user_history  # noqa: E402

_INT_RE = re.compile(r"\d+")


def load_seq2seq(cfg: Any, device: Any):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    trc = bool(getattr(cfg, "trust_remote_code", False))
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=trc)
    dtype = resolve_dtype(getattr(cfg, "dtype", "bfloat16"))
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, dtype=dtype, trust_remote_code=trc)
    adapter = getattr(cfg, "adapter_path", None)
    if adapter and Path(adapter).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"[listt5] loaded fine-tuned adapter <- {adapter}")
    else:
        if adapter:
            print(f"[listt5] WARNING: adapter_path set but not found ({adapter}); using base model zero-shot")
    model.to(device)
    model.eval()
    return tok, model


def build_listwise_prompt(sample: dict, perm: list[int]) -> str:
    """Shared by the trainer and the eval ranker so train/eval prompts match exactly."""
    cands = sample["candidates"]
    history_text = format_user_history(sample.get("history"), None)
    lines = ["User history:", history_text or "(none)", "", "Candidates:"]
    for local_pos, idx in enumerate(perm):
        lines.append(f"[{local_pos}] {format_candidate_item(cands[idx], None)}")
    lines += [
        "",
        "Rank the candidates from most to least relevant to the user's preferences.",
        "Output the candidate numbers separated by commas, best first.",
    ]
    return "\n".join(lines)


def gold_local_order(sample: dict, perm: list[int]) -> list[int]:
    """Training target: local positions (0..K-1 under `perm`) in canonical relevance
    order. Uses the sample's permutation-invariant target_ranking (rel desc, tie-broken
    by item identity) so the target does not depend on input order."""
    cands = sample["candidates"]
    local_of_global = {g: i for i, g in enumerate(perm)}
    tr = sample.get("target_ranking") or {}
    item_ids = tr.get("item_ids")
    if item_ids:
        id_to_global = {}
        for g, c in enumerate(cands):
            id_to_global.setdefault(str(c.get("item_id")), g)
        global_order = [id_to_global[str(i)] for i in item_ids if str(i) in id_to_global]
    else:
        rel = [int(c.get("relevance", 0)) for c in cands]
        global_order = sorted(range(len(cands)), key=lambda g: (-rel[g], g))
    return [local_of_global[g] for g in global_order if g in local_of_global]


def parse_order(text: str, K: int) -> tuple[list[int], bool]:
    """Return (full permutation of 0..K-1, invalid_flag). invalid = repair was needed."""
    raw = [int(x) for x in _INT_RE.findall(text)]
    seen, order = set(), []
    for v in raw:
        if 0 <= v < K and v not in seen:
            seen.add(v)
            order.append(v)
    invalid = (order != raw) or (len(order) != K)  # out-of-range / dup / missing / extra
    if len(order) != K:
        order += [i for i in range(K) if i not in seen]
    return order, invalid


def pseudo_scores_from_order(order: list[int], K: int) -> list[float]:
    """order = local positions best-first. Return scores[local_pos] = K - rank (higher=better)."""
    scores = [0.0] * K
    for rank, local_pos in enumerate(order):
        scores[local_pos] = float(K - rank)
    return scores


class ListT5Ranker:
    def __init__(self, cfg: Any, device: Any):
        self.cfg = cfg
        self.device = device
        self.tok, self.model = load_seq2seq(cfg, device)
        self.max_len = int(getattr(cfg, "max_seq_length", 2048))
        self.gen_max_new_tokens = int(getattr(cfg, "gen_max_new_tokens", 128))
        self.gen_batch_size = int(getattr(cfg, "gen_batch_size", 32))
        self.debug_n = int(getattr(cfg, "gen_debug_samples", 0))  # print this many raw decodes for diagnosis
        self._dbg = 0
        self.n_invalid = 0
        self.n_calls = 0

    def build_prompt(self, sample: dict, perm: list[int]) -> str:
        return build_listwise_prompt(sample, perm)

    def _generate_texts(self, prompts: list[str]) -> list[str]:
        """Batched greedy generation over a chunk of prompts."""
        import torch

        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.gen_max_new_tokens, num_beams=1, do_sample=False
            )
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def score_batch(self, sample: dict, perms: list[list[int]]) -> list[list[float]]:
        """Score many permutations of ONE candidate set in batched generate() calls.
        Returns a list (aligned to `perms`) of length-K pseudo-score arrays."""
        K = len(sample["candidates"])
        prompts = [self.build_prompt(sample, perm) for perm in perms]
        scores = []
        for i in range(0, len(prompts), self.gen_batch_size):
            texts = self._generate_texts(prompts[i:i + self.gen_batch_size])
            for text in texts:
                if self._dbg < self.debug_n:
                    print(f"[listt5 debug] raw gen #{self._dbg} (K={K}): {text[:240]!r}", flush=True)
                    self._dbg += 1
                order, invalid = parse_order(text, K)
                self.n_calls += 1
                self.n_invalid += int(invalid)
                scores.append(pseudo_scores_from_order(order, K))
        return scores

    def score(self, sample: dict, perm: list[int]):
        """Length-K pseudo-scores aligned to LOCAL positions (matches the causal scorer API)."""
        return self.score_batch(sample, [perm])[0]

    @property
    def invalid_rate(self) -> float:
        return self.n_invalid / self.n_calls if self.n_calls else float("nan")


def candidate_ids_for(sample: dict) -> list[str]:
    return [candidate_id(c, i) for i, c in enumerate(sample["candidates"])]
