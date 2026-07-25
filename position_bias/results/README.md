# Results CSVs — provenance and column guide

All numbers were transcribed from exact terminal output pasted during the session (Sherlock SLURM job
logs), then cross-checked; see `../METHODOLOGY.md` for the full experimental setup these numbers came from.
One transcription error was caught and fixed during this pass: the MovieLens anchor (Llama-3.2-3B, seed 42)
has two distinct training runs — a 5000-user preliminary run (nDCG=0.8231, τ=0.8707) and the **official**
10000-user run (nDCG=0.8287, τ=0.8451) used everywhere else. All CSVs here use the official numbers.

## `dataset_stats.csv`
One row per dataset: raw source size, build parameters (history length, min interactions, whether future
positives require LightGCN retrieval), post-split sample counts, and first-stage retriever recall@k.
`NOT_CAPTURED` marks values that were never printed in a pasted terminal log during this session (noted
rather than guessed) — see column `notes` for the closest available reference number.

## `results_by_model_dataset.csv`
The main matrix: one row per (dataset, model, variant, seed). `variant` is `causal` (standard attention +
RoPE) or `invariant` (structured mask + shared positions). Metrics are all computed on the *irrelevant*
relevance stratum (true relevance = 0) unless noted:
- `ndcg_at_10`, `tau_kendall`: standard ranking quality / permutation-robustness (from `scripts/evaluate_ranking.py`).
- `curve_range_irrelevant`: from `scan_position.py` — worst-minus-best mean output rank of a target candidate as it's slid through every input position (RQ1, structure).
- `promo_at_5_irrelevant`, `rank_gain_irrelevant`, `into_top5_irrelevant`: from `adversarial_perm.py`, budget R=50 (RQ2, exploitability). `promo@5` = fraction of targets ranked outside top-5 on average that a random-order search can force into top-5 — the core "manufactured exposure" metric.
- `n_samples` = number of candidate sets (300 throughout, except the pointwise/ anchor-deepdive runs which used 500 — see `anchor_deepdive.csv`).

**Scope note:** the scaling axis (Qwen3 family) and cross-family points (Llama-3.1-8B, Mistral-7B) are
**causal-only** by design (attack characterization is the paper's contribution; re-running the invariant
defense at every scale/domain would be near-tautological — see `METHODOLOGY.md` §"scope decisions").
Invariant rows exist only for: the MovieLens/Books/Fashion anchor (Llama-3.2-3B), and Qwen3-1.7B on
MovieLens (a leftover from the harness-validation run, kept because it's real data).

## `defense_matrix.csv`
The defense landscape (anchor, Llama-3.2-3B, **MovieLens only** — not yet extended to Amazon domains).
Rows A/A1/A2 isolate the two invariance channels (attention mask vs. position ids) for the RQ5 mechanism
analysis. B1/B2 are training-time countermeasures; C is inference-time order-averaging (Permutation
Self-Consistency); pointwise is the trivially-invariant reference. Read the `notes` column carefully for
defense C — its `promo@5≈0` claim in the paper is an *inference* from the averaged-score behavior, not a
directly re-measured `adversarial_perm.py` run against the P-averaged scorer. That's flagged as a concrete
follow-up experiment.

## `anchor_deepdive.csv`
Every granular number (both relevance strata: `irrelevant` and `strong_pos`) from the very first, most
thoroughly-instrumented run: Llama-3.2-3B on MovieLens, both variants, all five position-bias scripts
(`scan_position.py`, `adversarial_perm.py`, `greedy_attack.py`, `content_vs_position.py`,
`pointwise_eval.py`). This is the source of truth for anything not captured in the terser per-model
SUMMARY lines used in `results_by_model_dataset.csv` (e.g. `into_top1`, the `strong_pos` stratum, the
greedy-vs-random comparison, the position-vs-content decomposition).

## `correlation_data.csv`
The exact 24 (model, domain) causal-only rows behind the paper's headline correlation claim: Pearson
`r = -0.9743` between a reranker's ordinary permutation stability (`tau_kendall`) and its exploitability
(`promo_at_5_irrelevant`). Verified by direct computation on this file:
```
python3 -c "import pandas as pd; df=pd.read_csv('correlation_data.csv'); print(df['tau_kendall'].corr(df['promo_at_5_irrelevant']))"
# -0.9742791464073596
```

---

# Revision results (B1–B6)

Added in the revision pass: attack-budget sensitivity (B1), bootstrap CIs (B2), domain-controlled
audit correlation (B3), encoder-decoder mechanism comparison (B4), cross-domain defense (B6). All
numbers are the Llama-3.2-3B / MovieLens anchor unless a `domain` column says otherwise. Producing
scripts (all in `position_bias/`): `adversarial_perm.py` (`--r-grid` budget curve + per-set `raw`
dump), `bootstrap_ci.py`, `analyze_correlation.py`, `pointwise_adv.py`, `listt5_scorer.py` +
`train_listt5_scorer.py` (T5 encoder scorer), `render_defense.sh`/`run_defenses.sh` (now `DATA_DIR`-aware).

## `budget_curve.csv` (B1) — attack-budget curve → Fig 5
One row per `(model, R)`. `model` ∈ {causal, ListT5, invariant, pointwise}; `R` ∈ {1,5,10,20,50,100};
`promo` = promo@5 on the irrelevant stratum; `ci_lo`/`ci_hi` = 95% bootstrap CI (blank where not filled).
Sources: `adversarial_perm.py -R 100 --r-grid …` (causal/invariant), `pointwise_adv.py` (pointwise, 0 by
construction), `listt5_eval.py --scorer encode` (ListT5).
**Headline:** exploitability *scales with attacker budget* for causal (0.027→0.180) and ListT5
(0.003→0.070), but is *budget-independent* for the defenses (invariant/pointwise flat ≈0). R=50 is not a
cherry-picked operating point.

## `budget_curve_ci.csv` (B2) — bootstrap CIs for attack metrics
Full `bootstrap_ci.py --all-R` output (10k resamples over candidate **sets**): one row per
`(file, stratum, R, metric)`, `metric` ∈ {promotion_into_top5, rank_gain_adv, weak_into_top5}, with
`point`, `ci_lo`, `ci_hi`, `n_sets`, `n_targets`. Also the source for CI-annotated main-table numbers
(e.g. causal R=50 promo@5 = 0.163 [0.123, 0.207]; rank_gain 3.44 [3.22, 3.67]). To regenerate for any
run: `python bootstrap_ci.py <adv_*_curve.json> --all-R`.

## `tau_promo_correlation_analysis.csv` (B3) — domain-controlled audit correlation → Fig 4
From `analyze_correlation.py` over the 24 cells in `correlation_data.csv`. Rows: `all_24_cells`,
`<domain>_only`, `leave_out_<domain>`, `partial_corr_control_domain`. Columns: `pearson_r`, `pearson_p`,
`spearman_rho`, `spearman_p`.
**Headline:** overall Pearson r=**-0.974** (95% bootstrap CI [-0.989, -0.956], permutation p<1e-4);
**partial r controlling for domain = -0.776 (p=8e-6)** → the τ↔promo@5 audit signal is a real
within-domain relationship, *not* just domain clustering. Within-domain: Books -0.86, Fashion -0.81,
MovieLens -0.54 (n.s. — promo@5 has almost no variance on MovieLens, a floor effect).

## `mechanism_comparison.csv` (B4) — architecture comparison → Table 2
MovieLens anchor, R=50, irrelevant stratum. Rows: `causal_llama3b`, `listt5_encoder`
(flan-t5-base encoder scorer), `architectural_invariance`, `pointwise_reference`. Columns include
`architecture`, `attention`, `ndcg_at_10`, `curve_range_irrelevant`, `tau_kendall`,
`promo_at_5_irrelevant`, `rank_gain_irrelevant`, `into_top5_irrelevant`.
**Headline:** the bidirectional T5 encoder scorer — at *higher* quality than causal (nDCG 0.876 vs
0.829) — has ~half the attack surface (promo@5 0.057 vs 0.120, curve_range 0.71 vs 3.09) but does not
zero it (residual from T5 relative-position bias). **Confounds** (loss softmax-CE vs LambdaRank; scoring
pooled-hidden+head vs mean-logprob; backbone flan-t5-base vs Llama-3B) are real — state the claim as
"encoder-decoder at equal-or-higher quality halves the surface," not single-variable attribution.
ListT5 training recipe + failed generative route: `../../summary.md §5`.

## `defense_cross_domain.csv` (B6) — defense landscape across domains → Table 3
One row per `(defense, domain)`. defenses: `baseline_causal`, `architectural_invariance`,
`permutation_consistency_kl` (the B2 training-time defense), `pointwise_reference`; domains:
movielens/amazon_books/amazon_fashion. Columns: `ndcg_at_10`, `promo_at_5_irrelevant`, `tau_kendall`,
`curve_range_irrelevant`, `exact_invariant`, `source`, `notes`.
**Headline:** the permutation-consistency-KL defense *generalizes* — promo@5 cut 84–94% in every domain
(ML 0.120→0.007, Books 0.480→0.067, Fashion 0.297→0.047) — but leaves a small residual on Amazon that
*exact* architectural invariance removes. Tradeoff is domain-dependent: on Books, permkl keeps more
quality (nDCG 0.552 vs invariance's 0.510) for a still-large 86% surface reduction. Neither defense
dominates → a landscape, not a single winner.
