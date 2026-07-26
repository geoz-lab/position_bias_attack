#!/bin/bash
#SBATCH --job-name=t5-budget
#SBATCH --partition=owners
#SBATCH --account=kovscek
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=t5-budget-%j.out
# B4 / Fig 6: ListT5 (T5 encoder-scorer) attack-budget curve under the R_max=50
# convention. Fills the ListT5 row of the budget-curve table and the R<=50 points
# of the budget figure, consistent with the causal p0 anchor (N=300, R_max=50).
# Inference only -- reuses the already-trained encoder-scorer adapter.
#
#   Prereq (once): render config + train the encoder scorer
#     STEPS=2000 SCORER_R=32 bash render_listt5.sh listt5 google/flan-t5-base 42
#     sbatch --export=ALL,MTAG=listt5 train_listt5_scorer.sbatch
#   Then run this budget curve:
#     sbatch --export=ALL,MTAG=listt5 b4_budget_t5.sbatch
#
# Env overrides: NSAMPLES (default 300), RGRID (default 1,5,10,20,50),
#                TARGETS (adv targets per set, default 2).
set -euo pipefail
source /home/groups/kovscek/gmzhang/reranker/sherlock/env.sh
cd "$CODE_DIR"

: "${MTAG:?set MTAG via --export (e.g. listt5)}"
CFGFILE="$CFG_DIR/$MTAG/rank_listt5_scorer.yaml"
[ -f "$CFGFILE" ] || { echo "[t5-budget] missing config: $CFGFILE (run render_listt5.sh first)"; exit 1; }

OUT="$PROJ/position_bias/${MTAG}_listt5_scorer_budget"; mkdir -p "$OUT"
NSAMPLES="${NSAMPLES:-300}"      # match the causal p0 anchor (N=300)
RGRID="${RGRID:-1,5,10,20,50}"   # R_max=50 convention -- NO R=100
TARGETS="${TARGETS:-2}"          # adv targets per candidate set

echo "[t5-budget] node=$(hostname) MTAG=$MTAG cfg=$CFGFILE  N=$NSAMPLES  Rmax=50  grid=$RGRID"
nvidia-smi -L || true

# -R is R_max (the pool size that defines each target's typical rank); it must equal
# the largest grid entry (50) so promo@k is measured against a fixed R_max=50 pool.
python position_bias/listt5_eval.py --config "$CFGFILE" --scorer encode --task adv \
  --num-samples "$NSAMPLES" -R 50 --r-grid "$RGRID" --targets-per-set "$TARGETS" \
  --out-dir "$OUT"

echo ""
echo "############ ListT5 BUDGET CURVE (R_max=50, irrelevant stratum) ############"
OUT="$OUT" python - <<'PY'
import json, os
adv = json.load(open(os.path.join(os.environ["OUT"], "adv_listt5.json")))
bc = adv.get("budget_curve", {})
print("R\tpromo@5")
for R in sorted(bc, key=lambda x: int(x)):
    v = bc[R].get("irrelevant", {}).get("promotion_into_top5", float("nan"))
    print(f"{R}\t{v:.4f}")
head = adv["summary"]["irrelevant"]["promotion_into_top5"]
print(f"\nR=50 headline promo@5 = {head:.4f}")
print("SANITY: this must match the ListT5 row of the mechanism table (0.057).")
print("Paste the R=1..50 values into the ListT5 row of the budget-curve table")
print("(paper Table 11) and, if you want the ListT5 line drawn in the figure,")
print("into paper/plot/plot_figure.py.")
PY
