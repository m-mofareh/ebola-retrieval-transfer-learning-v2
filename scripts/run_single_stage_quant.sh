#!/bin/bash
# ============================================================
# run_single_stage_quant.sh
# Single-stage DeepDTA + GraphDTA on quantitative Ebola dataset
# No transfer learning — trains directly on Ebola data only
#
# Usage:
#   chmod +x run_single_stage_quant.sh
#   nohup bash run_single_stage_quant.sh \
#       > single_stage_quant.log 2>&1 &
#   tail -f single_stage_quant.log
# ============================================================

SCRIPT="single_stage_deepdta_graphdta.py"
EBOLA_ZIP="ebola_quant.zip"
OUT_DIR="single_stage_quant_results"
SEEDS=(42 1 2 3 0)
MODELS=("deepdta" "graphdta")
EPOCHS=200
PATIENCE=40
BATCH=64
LR=1e-3

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  Single-stage DeepDTA + GraphDTA"
echo "  Dataset  : $EBOLA_ZIP (2842 rows, quantitative)"
echo "  Models   : ${MODELS[*]}"
echo "  Seeds    : ${SEEDS[*]}"
echo "  Epochs   : $EPOCHS  Patience: $PATIENCE"
echo "  Started  : $(date)"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "  $MODEL | seed=$SEED | $(date)"

        python -u "$SCRIPT" \
            --model    "$MODEL" \
            --seed     "$SEED" \
            --ebola_zip "$EBOLA_ZIP" \
            --output_dir   "$OUT_DIR" \
            --epochs    "$EPOCHS" \
            --patience  "$PATIENCE" \
            --batch_size     "$BATCH" \
            --lr        "$LR" \
            > "$OUT_DIR/${MODEL}_seed${SEED}.log" 2>&1

        # Print result immediately
        METRICS=$(find "$OUT_DIR" \
                  -name "metrics_${MODEL}_seed${SEED}.csv" \
                  2>/dev/null | head -1)
        if [ -f "$METRICS" ]; then
            python3 -c "
import pandas as pd
df = pd.read_csv('$METRICS')
print(f'  RMSE={df[\"RMSE\"].iloc[0]:.4f}  '
      f'Pearson={df[\"Pearson\"].iloc[0]:.4f}  '
      f'beats={df[\"beats_baseline\"].iloc[0]}')
"
        else
            echo "  WARNING: metrics not found"
        fi
    done
done

# Aggregate
python3 - << 'PYEOF'
import glob, numpy as np, pandas as pd

BASELINE = 0.5513
OUT_DIR  = "single_stage_quant_results"
metrics  = ["RMSE","MAE","R2","Pearson"]

rows = []
for model in ["deepdta","graphdta"]:
    files = glob.glob(f"{OUT_DIR}/metrics_{model}_seed*.csv")
    if not files:
        print(f"  WARNING: no results for {model}")
        continue
    df  = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    row = {"model": model, "n_seeds": len(df)}
    for m in metrics:
        v = df[m].dropna()
        row[f"{m}_mean"] = round(v.mean(), 4)
        row[f"{m}_std"]  = round(v.std(ddof=1), 4) if len(v)>1 else 0.0
    for _, r in df.iterrows():
        s = int(r.get("seed", -1))
        row[f"seed{s}_RMSE"] = round(float(r["RMSE"]), 4)
    row["beats_baseline"] = int((df["RMSE"] < BASELINE).sum())
    rows.append(row)
    df.to_csv(f"{OUT_DIR}/{model}_all_seeds.csv", index=False)

if not rows:
    print("No results found")
    exit(0)

summary = pd.DataFrame(rows)
summary.to_csv(f"{OUT_DIR}/summary_mean_std.csv", index=False)

print("\n" + "="*70)
print("  SINGLE-STAGE RESULTS — Quantitative Ebola (2842 rows)")
print("="*70)
print(f"  {'Model':<12} {'RMSE mean±std':<22} {'Pearson':<12} {'Beats'}")
print("  " + "-"*55)
for _, r in summary.sort_values("RMSE_mean").iterrows():
    beat = f"{r['beats_baseline']}/{r['n_seeds']}"
    print(f"  {r['model']:<12} "
          f"{r['RMSE_mean']:.4f} ± {r['RMSE_std']:.4f}       "
          f"{r['Pearson_mean']:.4f}       {beat}")

print(f"\n  Baseline RMSE : {BASELINE}")
print(f"\n  Per-seed RMSE:")
print(f"  {'Model':<12} {'s42':>7} {'s1':>7} {'s2':>7} "
      f"{'s3':>7} {'s0':>7} {'Mean':>7}")
print("  " + "-"*55)
for _, r in summary.iterrows():
    seeds = [r.get(f"seed{s}_RMSE", float("nan"))
             for s in [42,1,2,3,0]]
    seed_str = "  ".join([f"{v:.4f}" if not np.isnan(v) else "  —   "
                           for v in seeds])
    print(f"  {r['model']:<12} {seed_str}  {r['RMSE_mean']:.4f}")
print("="*70)
PYEOF

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Results: $OUT_DIR/summary_mean_std.csv"
echo "============================================================"
