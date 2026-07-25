#!/usr/bin/env python3
"""B3 — domain-controlled tau <-> promo@5 analysis (paper contribution C4).

The headline claim is that a causal reranker's ordinary permutation stability
(Kendall tau, measurable with NO attack) predicts its exploitability (promo@5).
Overall Pearson r across the 24 (model, domain) cells is ~ -0.97. A reviewer will
ask: is that a real within-domain relationship, or just domain clustering (three
tight clusters that happen to line up)? This script answers that with:

  - overall Pearson / Spearman (all 24 cells)
  - per-domain Pearson / Spearman (8 models each)
  - leave-one-domain-out Pearson (16 cells each)
  - partial correlation controlling for domain (residualize both on domain dummies)
  - bootstrap 95% CI + permutation-test p-value for the overall Pearson

Runs locally from results/correlation_data.csv -- no cluster needed.
  python position_bias/analyze_correlation.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
CSV_IN = ROOT / "results" / "correlation_data.csv"
CSV_OUT = ROOT / "results" / "tau_promo_correlation_analysis.csv"
SEED = 42
N_BOOT = 10000
N_PERM = 10000


def load():
    rows = list(csv.DictReader(CSV_IN.open()))
    tau = np.array([float(r["tau_kendall"]) for r in rows])
    promo = np.array([float(r["promo_at_5_irrelevant"]) for r in rows])
    dom = np.array([r["domain"] for r in rows])
    return tau, promo, dom


def partial_corr_controlling_domain(tau, promo, dom):
    """Pearson r between tau and promo after regressing each on domain dummies."""
    domains = sorted(set(dom))
    # design matrix: intercept + (k-1) domain dummies
    X = np.column_stack([np.ones(len(dom))] + [(dom == d).astype(float) for d in domains[1:]])
    beta_t, *_ = np.linalg.lstsq(X, tau, rcond=None)
    beta_p, *_ = np.linalg.lstsq(X, promo, rcond=None)
    res_t = tau - X @ beta_t
    res_p = promo - X @ beta_p
    r, p = stats.pearsonr(res_t, res_p)
    return r, p


def main():
    tau, promo, dom = load()
    domains = sorted(set(dom))
    out_rows = []

    def record(name, x, y):
        n = len(x)
        if n < 3 or np.std(x) == 0 or np.std(y) == 0:
            pr = ps = sr = sp = float("nan")
        else:
            pr, ps = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
        out_rows.append({"analysis": name, "n": n,
                         "pearson_r": pr, "pearson_p": ps,
                         "spearman_rho": sr, "spearman_p": sp})
        return pr

    r_all = record("all_24_cells", tau, promo)
    for d in domains:
        m = dom == d
        record(f"{d}_only", tau[m], promo[m])
    for d in domains:
        m = dom != d
        record(f"leave_out_{d}", tau[m], promo[m])

    # partial correlation controlling for domain
    pr_partial, pp_partial = partial_corr_controlling_domain(tau, promo, dom)
    out_rows.append({"analysis": "partial_corr_control_domain", "n": len(tau),
                     "pearson_r": pr_partial, "pearson_p": pp_partial,
                     "spearman_rho": float("nan"), "spearman_p": float("nan")})

    # bootstrap CI + permutation p-value for the overall Pearson
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(tau))
    boot = np.array([stats.pearsonr(tau[s], promo[s])[0]
                     for s in (rng.choice(idx, len(idx), replace=True) for _ in range(N_BOOT))])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    perm = np.array([stats.pearsonr(tau, rng.permutation(promo))[0] for _ in range(N_PERM)])
    p_perm = (np.sum(np.abs(perm) >= abs(r_all)) + 1) / (N_PERM + 1)

    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["analysis", "n", "pearson_r", "pearson_p",
                                          "spearman_rho", "spearman_p"])
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"{'analysis':<30}{'n':>4}{'pearson_r':>12}{'pearson_p':>12}{'spearman_rho':>14}")
    for r in out_rows:
        print(f"{r['analysis']:<30}{r['n']:>4}{r['pearson_r']:>12.4f}"
              f"{r['pearson_p']:>12.4g}{r['spearman_rho']:>14.4f}")
    print()
    print(f"overall Pearson r = {r_all:.4f}   "
          f"bootstrap 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]   "
          f"permutation p = {p_perm:.4g}")
    print(f"partial r (control domain) = {pr_partial:.4f} (p={pp_partial:.4g})")
    print(f"\nwrote {CSV_OUT.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
