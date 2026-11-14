#!/bin/bash
set -e
export HTH_DATA=~/hack-the-harvest/data
P=../.venv/bin/python
until [ "$(ls $HTH_DATA/s1/*.parquet 2>/dev/null | wc -l)" -ge 20 ] && ! pgrep -f 03_extract_s1 >/dev/null; do sleep 15; done
echo "S1 complete $(date)"
T0=$(date +%s)
$P 04_build_features.py | grep -v "block .* done"
echo "features_seconds $(( $(date +%s) - T0 ))"
$P 00_label_audit.py > /dev/null && echo "audit ok"
$P 05_models.py
echo "MODELS_DONE $(date)"
for seed in 1 2 3; do
  HTH_SEED=$seed $P 06_deep.py s2 > ../deep_s2_$seed.log 2>&1 &
  HTH_SEED=$seed $P 06_deep.py s2s1 > ../deep_s2s1_$seed.log 2>&1 &
  HTH_SEED=$seed $P 06_deep.py s2s1_nocnn > ../deep_nocnn_$seed.log 2>&1 &
  wait
done
echo "ALL_DONE $(date)"
