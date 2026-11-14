#!/bin/bash
# Spectral-index experiment: 20 m extraction (launched separately per block) -> features -> comparison (E2 setting).
set -e
H=~/hack-the-harvest; P=$H/.venv/bin/python; cd $H/pipeline
until [ "$(ls $H/data/s2_20m/obs20_*.parquet 2>/dev/null | wc -l)" -ge 20 ] && [ "$(ls $H/data2024/s2_20m/obs20_*.parquet 2>/dev/null | wc -l)" -ge 20 ] && ! pgrep -f 02b_extract >/dev/null; do sleep 15; done
echo "extraction done $(date)"
HTH_YEAR=2025 HTH_DATA=$H/data $P 09_index_features.py | grep -v "block .* done"
HTH_YEAR=2024 HTH_DATA=$H/data2024 $P 09_index_features.py | grep -v "block .* done"
HTH_ROOTS="2024:$H/data2024,2025:$H/data" HTH_TRAIN_YEARS=2024,2025 HTH_TEST_YEAR=2025 \
  HTH_RESULTS=$H/data/results_indices_e2 HTH_DATA=$H/data $P 10_index_experiment.py
