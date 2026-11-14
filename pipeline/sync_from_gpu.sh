#!/bin/bash
# Pull console outputs and evaluation artifacts from the lab GPU machine into this repo.
# pipeline/results/            = console model run (E2: trained 2024+2025, tested on 2025 test regions)
# pipeline/results/experiments/ = all four year experiments (E1-E4) + the original 2025-only run
# pipeline/results/label_audit_<year>.*
set -e
HOST=${HTH_HOST:-mirtgpu@100.126.187.101}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R=hack-the-harvest/data
MAIN=results_e2_train2024-2025_test2025
mkdir -p "$REPO/public/nl" "$REPO/pipeline/results/experiments"
rsync -az --delete "$HOST:$R/console/" "$REPO/public/nl/"
rsync -az --exclude '*.parquet' --exclude models "$HOST:$R/$MAIN/" "$REPO/pipeline/results/"
rsync -az "$HOST:$R/$MAIN/predictions_all.parquet" "$HOST:$R/features/splits.csv" "$HOST:$R/features/feature_schema.json" "$REPO/pipeline/results/"
for e in results_e1_train2025_test2025 results_e2_train2024-2025_test2025 results_e3_train2024_test2025 results_e4_train2024-2025_test2024; do
  mkdir -p "$REPO/pipeline/results/experiments/$e"
  rsync -az --include 'metrics.json' --include 'model_comparison_test.csv' --include 'confusion_matrices.png' --include 'importance_*.csv' --exclude '*' "$HOST:$R/$e/" "$REPO/pipeline/results/experiments/$e/"
done
mkdir -p "$REPO/pipeline/results/experiments/indices_20m"
rsync -az --include 'index_experiment*' --exclude '*' "$HOST:$R/results_indices_e2/" "$REPO/pipeline/results/experiments/indices_20m/"
mkdir -p "$REPO/pipeline/results/experiments/original_2025_only"
rsync -az --include 'metrics.json' --include 'model_comparison_test.csv' --include 'deep_*seed*.json' --exclude '*' "$HOST:$R/results/" "$REPO/pipeline/results/experiments/original_2025_only/"
rsync -az "$HOST:$R/results/label_audit.md" "$REPO/pipeline/results/label_audit_2025.md"
rsync -az "$HOST:$R/results/label_audit.json" "$REPO/pipeline/results/label_audit_2025.json"
rsync -az "$HOST:hack-the-harvest/data2024/results/label_audit.md" "$REPO/pipeline/results/label_audit_2024.md"
rsync -az "$HOST:hack-the-harvest/data2024/results/label_audit.json" "$REPO/pipeline/results/label_audit_2024.json"
for f in pipeline.log s1.log models.log deep.log console.log; do rsync -az "$HOST:$R/$f" "$REPO/pipeline/results/logs_$f" 2>/dev/null || true; done
echo "synced"
