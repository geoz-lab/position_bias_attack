# Position Bias as an Attack Surface in LLM Listwise Rerankers

Code and verified results accompanying our paper on **position bias in LLM
listwise rerankers, reframed as an exploitable attack surface**.

LLM listwise rerankers are strong, but as decoder-only models they score
candidates depending on **input order**. We show this position bias is not
benign noise: an adversary who can influence candidate order (e.g. a caller of a
multi-tenant / API reranker) can **manufacture top-k exposure for irrelevant
items without changing any content or relevance**. We characterize the surface
(structured, scale-independent, position ≫ content), evaluate defenses as a
*security property*, and surface a cheap audit signal that predicts
exploitability without running any attack.

> **We propose no new architecture.** Our contribution is the attack-surface
> analysis + defense evaluation. This is a **vulnerability audit + defense
> validation, not a manipulation tool**: we report aggregate magnitudes and a
> working defense, content edits are diagnostic only, and there is no optimized
> manipulation pipeline.

## Relationship to InvariRank (please cite)

The reranker implementation this study attacks and defends is **InvariRank**,
from a published paper. The top-level Python package here — `config.py`,
`configs/`, `datasets/`, `model/`, `prompts/`, `ranking/`, `retriever/`,
`scripts/`, `training/` — is a **redistributed copy of the InvariRank reference
implementation** (MIT, © 2026 Ethan Bito; see `LICENSE-InvariRank` and
`NOTICE`). We vendor it only so our analysis code runs out of the box.

If you use this repository, please also cite InvariRank:

```bibtex
@misc{bito2026onepassanyorder,
  title  = {One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation},
  author = {Ethan Bito and Yongli Ren and Estrid He},
  year   = {2026},
  eprint = {2604.27599},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  url    = {https://arxiv.org/abs/2604.27599},
}
```

**Terminology map** (ours ↔ InvariRank): the vulnerable default we attack is the
**causal** reranker (standard attention + RoPE) = InvariRank's **LFT** baseline
(`configs/*_lft.yaml`, internal tag `lft`); the **position-invariant** reranker
(structured attention mask + shared candidate position ids) = InvariRank's
defense mechanism (`prompt_style: invarirank`, internal tag `invarirank`), which
we evaluate as one defense among several.

## Contributions

- **C1 — structured & exploitable.** Position bias is a structured U-shaped
  position→rank curve (not noise). A plain random-order search promotes 10–57% of
  otherwise-below-top-5 irrelevant candidates into top-5 (`promo@5`). **No scaling
  law** — larger models are not immune.
- **C2 — position ≫ content.** The position lever is ≈5.6× a content edit, and
  leaves no content trace.
- **C3 — defense landscape.** Naive order-augmented training **fails**;
  consistency-KL training and architectural invariance close the surface in a
  single forward pass; test-time order averaging works at O(P) cost. Mechanism:
  the attention channel dominates over the position-id channel.
- **C4 — cheap audit signal.** Across 24 (model, domain) cells, a causal
  reranker's ordinary permutation stability (Kendall **τ**, attack-free) predicts
  its exploitability (`promo@5`) at Pearson **r ≈ −0.97** (partial r = −0.78
  controlling for domain). Practitioners can flag attack surface by measuring τ
  alone.
- **Domain-dependent severity.** `promo@5` on MovieLens (0.10–0.15) < Fashion
  (0.23–0.39) < Books (0.43–0.57): the weaker the content signal, the larger the
  hole. Defenses drive `promo@5 ≈ 0` in every domain, at a non-uniform quality
  cost (Books largest, ~17%).

## Repository layout

```text
position_bias_attack/
├── README.md            ← you are here
├── LICENSE              MIT (our contribution)
├── LICENSE-InvariRank   MIT (vendored InvariRank code)
├── NOTICE               provenance of every top-level component
├── requirements.txt
├── pyproject.toml
│
│   ── vendored InvariRank reference implementation (see NOTICE) ──
├── config.py            YAML/JSON config loading
├── configs/             example dataset / train / rank configs (incl. *_lft.yaml)
├── datasets/            MovieLens-32M + Amazon raw→JSONL builders
├── model/               tokenizer/model/LoRA loading; InvariRank mask+position logic
├── prompts/             prompt builders + JSON wording templates
├── ranking/             ranking pipeline + mean-log-prob listwise scorer
├── retriever/           LightGCN first-stage retrieval
├── scripts/             build_dataset / train_model / run_ranking / evaluate_ranking
├── training/            tokenized listwise dataset, losses, metrics, training loop
│
│   ── our contribution ──
├── position_bias/       attack-surface & defense analysis (RQ1–RQ5 + revisions)
│   ├── scan_position.py         RQ1  position→rank causal curve (curve_range)
│   ├── adversarial_perm.py      RQ2  budget-R permutation attacker (promo@5) — core
│   ├── greedy_attack.py         RQ2  structured greedy (lower bound; footnote)
│   ├── content_vs_position.py   RQ3  position vs content 2×2 attribution
│   ├── pointwise_eval.py             pointwise reference baseline
│   ├── defense_avg.py           C    test-time order averaging defense
│   ├── bootstrap_ci.py          B2   bootstrap CIs over candidate sets
│   ├── analyze_correlation.py   C4   domain-controlled τ↔promo@5 (runs locally)
│   ├── pointwise_adv.py         B1   pointwise on the attack-budget axis
│   ├── listt5_scorer.py         B4   T5 encoder listwise scorer
│   ├── listt5_ranker.py         B4   generative ListT5 ranker (deprecated route)
│   ├── listt5_eval.py           B4   run the T5 scorer through the same 3 probes
│   ├── train_listt5_scorer.py   B4   train the T5 encoder scorer
│   ├── train_listt5.py          B4   train the generative ListT5 (deprecated route)
│   └── results/                 verified numeric results (CSV) + data dictionary
│
└── position_bias_check/ ablation: can *prompting* remove the bias? (no)
    ├── collect.py               aggregate scan/adv JSON → prompt_check.csv
    ├── templates/               p0 baseline + p1–p5 anti-position prompt styles
    └── results/                 prompt_check.csv + raw per-variant JSON
```

The analysis scripts add the repository root to `sys.path`
(`Path(__file__).resolve().parents[1]`) and import the vendored InvariRank
modules, so **keep `position_bias/` and `position_bias_check/` as direct
subdirectories of the repo root** — that layout is what makes the imports resolve.

## Setup

Python ≥ 3.10 (3.12 recommended).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For gated Hugging Face models, authenticate with the HF CLI or set `HF_TOKEN`.
Fine-tuning and evaluation expect a CUDA GPU; the analysis scripts run against
the trained adapters produced by the training step below.

## Quick start

The pipeline is: **build data → train a reranker → run the analysis probes**.
All entry points take a `--config` (see `configs/` for the expected fields — edit
paths, model id, and sample counts before running anything large).

**1. Build a dataset** (MovieLens-32M or Amazon; writes
`data/processed/<dataset>/<K>/{train,val,test}.jsonl`):

```bash
python scripts/build_dataset.py --config configs/dataset.yaml
```

**2. Train a reranker.** Causal (the vulnerable default we attack) vs.
position-invariant (one defense) differ only by these config fields:

```yaml
# causal (LFT) — configs/train_lft.yaml
attention_mask: causal
position_ids: standard
prompt_style: invarirank      # markers only; standard attention/positions

# position-invariant defense — configs/train.yaml
attention_mask: block
position_ids: shared
prompt_style: invarirank
```

```bash
python scripts/train_model.py --config configs/train_lft.yaml   # causal
python scripts/train_model.py --config configs/train.yaml        # invariant
```

**3. Rank + evaluate quality/robustness** (nDCG, HR, Kendall τ, permutation
overlap):

```bash
python scripts/run_ranking.py      --config configs/rank_lft.yaml
python scripts/evaluate_ranking.py --config configs/rank_lft.yaml
```

## Running the attack-surface analysis

Each script takes a `rank`-style `--config` pointing at a trained adapter and
writes a JSON result. Representative invocations (see each script's `--help` and
module docstring for the full option set):

```bash
cd position_bias

# RQ1 — position→rank causal curve (curve_range). Slide a fixed target through
# every input position; large range + structure ⇒ exploitable.
python scan_position.py --config <rank_cfg> --num-samples 300 --out scan.json

# RQ2 — budget-R permutation attacker. "Try R orderings, keep the best for the
# target." promo@5 = fraction of below-top-5 irrelevant targets forced into top-5.
python adversarial_perm.py --config <rank_cfg> -R 50 --num-samples 300 --out adv.json
#   attack-budget curve (Fig. 5):  add  --r-grid 1,5,10,20,50,100

# RQ3 — position vs content 2×2 attribution.
python content_vs_position.py --config <rank_cfg> --num-samples 300 --out pvc.json

# Baselines / defenses
python pointwise_eval.py --config <rank_cfg> --num-samples 300 --out pw.json
python defense_avg.py    --config <rank_cfg> --avg-perms 20 --num-samples 300 --out defC.json
python greedy_attack.py  --config <rank_cfg> -R 50 --num-samples 300 --out greedy.json
```

Revision-round analyses:

```bash
# B2 — bootstrap 95% CIs (resamples over candidate SETS) from adv_*.json --dump-raw
python bootstrap_ci.py adv.json --all-R --out results/budget_curve_ci.csv

# C4 — domain-controlled τ↔promo@5 correlation. Runs locally off the shipped CSV:
python analyze_correlation.py     # reads results/correlation_data.csv

# B1 — pointwise on the budget axis (flat ≈0 by construction)
python pointwise_adv.py --config <rank_cfg> -R 100 --r-grid 1,5,10,20,50,100 --out pw_adv.json

# B4 — encoder-decoder (T5) comparison: train the scorer, then the same 3 probes
python train_listt5_scorer.py --config <listt5_cfg>
python listt5_eval.py --config <listt5_cfg> --scorer encode --task all --out-dir listt5_out
```

### Prompt ablation — can prompting remove the bias? (`position_bias_check/`)

Holds the **same trained causal adapter** fixed and changes **only the eval-time
prompt wording** (`templates/p0`–`p5`: baseline + five distinct anti-position
styles — declarative, imperative, persona, chain-of-thought, structured rules),
then re-runs `scan_position.py` (curve_range) and `adversarial_perm.py` (promo@5)
for each. Result: **prompting does not close the surface** (all variants stay at
`promo@5 ≈ 0.12`, vs ≈0 for architectural invariance) — the bias is
*mechanistic*, not *instructional*. Aggregate the per-variant JSON with:

```bash
python position_bias_check/collect.py <dir_with_scan_and_adv_json> \
       --out position_bias_check/results/prompt_check.csv
```

## Results

Verified numbers ship as CSVs in `position_bias/results/` — see
`position_bias/results/README.md` for the column dictionary and provenance, and
`position_bias_check/results/` for the prompt ablation. Headlines:

| Result | File |
|---|---|
| Main matrix — per (dataset, model, variant, seed) | `results_by_model_dataset.csv` |
| Defense landscape (anchor, MovieLens) | `defense_matrix.csv` |
| Cross-domain defense (ML / Books / Fashion) | `defense_cross_domain.csv` |
| C4 audit signal — 24-cell τ vs promo@5 | `correlation_data.csv`, `tau_promo_correlation_analysis.csv` |
| Attack-budget curve (Fig. 5) + CIs | `budget_curve.csv`, `budget_curve_ci.csv` |
| Encoder-decoder (T5) comparison | `mechanism_comparison.csv` |
| Anchor deep-dive (all metrics, both strata) | `anchor_deepdive.csv` |
| Dataset build stats | `dataset_stats.csv` |
| Prompt ablation (prompting fails) | `../position_bias_check/results/prompt_check.csv` |

Defense matrix at the anchor (Llama-3.2-3B, MovieLens):

| Defense | nDCG@10 | promo@5 | cost | exact |
|---|---|---|---|---|
| baseline causal | 0.829 | 0.120 | 1× | no |
| architectural invariance | 0.784 | **0.000** | 1× | yes |
| permutation-consistency KL | 0.777 | **0.007** | 1× | ≈ |
| order-augmented training | 0.849 | 0.117 *(fails)* | 1× | no |
| test-time averaging (P=20) | 0.848 | ≈0 | 20× | no |
| pointwise reference | 0.763 | 0 | 25× | yes |

## Ethics

This is a defensive security study. We report aggregate vulnerability magnitudes
and a validated defense; content edits are diagnostic (to separate position from
content) only; we ship no optimized manipulation pipeline. The intended use is
auditing and hardening deployed rerankers.

## Citation

Please cite both our paper (position-bias attack surface) and InvariRank (the
reranker we build on). The InvariRank BibTeX is above; our paper's citation will
be added on publication.

## License

Our contribution (`position_bias/`, `position_bias_check/`, this README) is MIT —
see `LICENSE`. The vendored InvariRank code is MIT © 2026 Ethan Bito — see
`LICENSE-InvariRank` and `NOTICE`.
