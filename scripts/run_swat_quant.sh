#!/bin/bash
# ============================================================
# run_swat_quant.sh
# SWAT ablation: 4 retrieval methods x 2 models x 5 seeds
# Dataset  : ebola_quant.zip (2842 rows, quantitative)
# BDB      : 50k samples, no pool limit (full 584k)
# Methods  : random, protein, compound, hybrid
# Models   : deepdta, graphdta
#
# Usage:
#   chmod +x run_swat_quant.sh
#   nohup bash run_swat_quant.sh > swat_quant.log 2>&1 &
#   tail -f swat_quant.log
# ============================================================

SCRIPT="deepdta_graphdta_swat_fixed.py"
EBOLA_ZIP="ebola_quant.zip"
BINDINGDB_DIR="bindingdb_data/"
SEEDS=(42 1 2 3 0)
MODELS=("deepdta" "graphdta")
METHODS=("random" "protein" "compound" "hybrid")
BDB_SAMPLES=300000         # 50k samples
# No --bindingdb_pool_samples → full 584k pool

STAGE1_EPOCHS=100
STAGE1_PATIENCE=20
STAGE1_BATCH=256
STAGE1_LR=1e-3
PROBE_EPOCHS=200
PROBE_PATIENCE=40
PROBE_LR=5e-4
STAGE2_BATCH=64

BASE_OUT="swat_quant_results_300k"
mkdir -p "$BASE_OUT"

echo "================================================================"
echo "  SWAT Ablation — Quantitative Ebola Dataset"
echo "  Dataset  : $EBOLA_ZIP (2842 rows)"
echo "  BDB      : $BDB_SAMPLES samples (full pool, no limit)"
echo "  Models   : ${MODELS[*]}"
echo "  Methods  : ${METHODS[*]}"
echo "  Seeds    : ${SEEDS[*]}"
echo "  Started  : $(date)"
echo "================================================================"

for METHOD in "${METHODS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        METHOD_DIR="$BASE_OUT/${METHOD}_${MODEL}"
        mkdir -p "$METHOD_DIR"

        echo ""
        echo "================================================================"
        echo "  METHOD: $METHOD | MODEL: $MODEL"
        echo "================================================================"

        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo "  $MODEL | $METHOD | seed=$SEED | $(date)"

            python -u "$SCRIPT" \
                --model          "$MODEL" \
                --stage          12 \
                --seed           "$SEED" \
                --retrieval_type "$METHOD" \
                --hybrid_alpha   0.5 \
                --ebola_zip      "$EBOLA_ZIP" \
                --bindingdb_dir  "$BINDINGDB_DIR" \
                --bindingdb_samples "$BDB_SAMPLES" \
                --output_dir     "$METHOD_DIR/seed${SEED}" \
                --stage1_epochs  "$STAGE1_EPOCHS" \
                --stage1_patience "$STAGE1_PATIENCE" \
                --stage1_batch   "$STAGE1_BATCH" \
                --stage1_lr      "$STAGE1_LR" \
                --probe_epochs   "$PROBE_EPOCHS" \
                --probe_patience "$PROBE_PATIENCE" \
                --probe_lr       "$PROBE_LR" \
                --stage2_batch   "$STAGE2_BATCH" \
                > "$METHOD_DIR/seed${SEED}.log" 2>&1

            # Print result immediately
            METRICS=$(find "$METHOD_DIR/seed${SEED}" \
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
        echo "  $METHOD $MODEL done at $(date)"
    done
done

# ============================================================
# Aggregate all results
# ============================================================
echo ""
echo "================================================================"
echo "  AGGREGATING RESULTS ..."
echo "================================================================"

python3 - << 'PYEOF'
import os, glob
import numpy as np
import pandas as pd

BASELINE = 0.5513
metrics  = ["RMSE","MAE","R2","Pearson"]
METHODS  = ["random","protein","compound","hybrid"]
MODELS   = ["deepdta","graphdta"]
base     = "swat_quant_results"

rows = []
for method in METHODS:
    for model in MODELS:
        pattern = (f"{base}/{method}_{model}/seed*/"
                   f"stage2_{model}_seed*/"
                   f"metrics_{model}_seed*.csv")
        files = glob.glob(pattern)
        if not files:
            print(f"  WARNING: no results for {method} {model}")
            continue
        df  = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        row = {"method": method, "model": model, "n_seeds": len(df)}
        for m in metrics:
            v = df[m].dropna()
            row[f"{m}_mean"] = round(v.mean(), 4)
            row[f"{m}_std"]  = round(v.std(ddof=1), 4) if len(v)>1 else 0.0
        for _, r in df.iterrows():
            s = int(r.get("seed", -1))
            row[f"seed{s}_RMSE"] = round(float(r["RMSE"]), 4)
        row["beats_baseline"] = int((df["RMSE"] < BASELINE).sum())
        rows.append(row)

if not rows:
    print("No results found")
    exit(0)

summary = pd.DataFrame(rows)
summary.to_csv(f"{base}/ablation_summary.csv", index=False)

print("\n" + "="*80)
print("  SWAT ABLATION — Quantitative Ebola (2842 rows), 50k BDB, full pool")
print("="*80)
print(f"  {'Method':<10} {'Model':<10} {'RMSE mean±std':<22} "
      f"{'Pearson':<12} {'Beats'}")
print("  " + "-"*65)

for model in MODELS:
    sub = summary[summary["model"]==model].sort_values("RMSE_mean")
    print(f"\n  {model.upper()}:")
    for _, r in sub.iterrows():
        beat = f"{r['beats_baseline']}/{r['n_seeds']}"
        print(f"    {r['method']:<10} {r['model']:<10} "
              f"{r['RMSE_mean']:.4f} ± {r['RMSE_std']:.4f}       "
              f"{r['Pearson_mean']:.4f}       {beat}")

best = summary.loc[summary["RMSE_mean"].idxmin()]
print(f"\n  Baseline RMSE : {BASELINE}")
print(f"  Best overall  : {best['model']} + {best['method']} "
      f"(RMSE={best['RMSE_mean']:.4f} ± {best['RMSE_std']:.4f})")

# Per-seed table
print("\n  PER-SEED RMSE:")
print(f"  {'Method':<10} {'Model':<10} "
      f"{'s42':>7} {'s1':>7} {'s2':>7} {'s3':>7} {'s0':>7} {'Mean':>7}")
print("  " + "-"*68)
for _, r in summary.sort_values(["model","RMSE_mean"]).iterrows():
    seeds = [r.get(f"seed{s}_RMSE", float("nan")) for s in [42,1,2,3,0]]
    seed_str = "  ".join([f"{v:.4f}" if not np.isnan(v) else "  —   "
                           for v in seeds])
    print(f"  {r['method']:<10} {r['model']:<10} "
          f"{seed_str}  {r['RMSE_mean']:.4f}")

print("="*80)
print(f"\n  Saved: {base}/ablation_summary.csv")
PYEOF

echo ""
echo "================================================================"
echo "  ALL DONE: $(date)"
echo "  Results : $BASE_OUT/ablation_summary.csv"
echo "================================================================"
