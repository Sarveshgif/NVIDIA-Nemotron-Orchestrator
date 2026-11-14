#!/bin/bash
# Add crop year 2024 (same 20 blocks/regions) and run the multi-year experiments.
set -e
H=~/hack-the-harvest; P=$H/.venv/bin/python; cd $H/pipeline
D24=$H/data2024; D25=$H/data
until grep -q DONE $H/data/brp/dl24.log 2>/dev/null; do sleep 15; done
echo "gpkg ready $(date)"
export HTH_YEAR=2024 HTH_DATA=$D24
$P 01b_blocks_from_gpkg.py $H/data/brp/brp_2024.gpkg $D25
BLOCKS=$(ls $D25/s2/obs_*.parquet | sed 's/.*obs_//; s/.parquet//' | tr '\n' ' ')
set -- $BLOCKS
A="${@:1:7}"; B="${@:8:7}"; C="${@:15:6}"
HTH_THREADS=12 $P 02_extract_s2.py $A > $H/s2_24a.log 2>&1 &
HTH_THREADS=12 $P 02_extract_s2.py $B > $H/s2_24b.log 2>&1 &
HTH_THREADS=12 $P 02_extract_s2.py $C > $H/s2_24c.log 2>&1 &
# S1 queues start as soon as each block's S2 table exists
./run_s1_queue.sh $A > $H/s1_24a.log 2>&1 &
./run_s1_queue.sh $B > $H/s1_24b.log 2>&1 &
./run_s1_queue.sh $C > $H/s1_24c.log 2>&1 &
wait
echo "extraction done $(date)"
$P 04_build_features.py | grep -v "block .* done"
$P 00_label_audit.py > /dev/null && echo "audit 2024 ok"
unset HTH_YEAR HTH_DATA
export HTH_ROOTS="2024:$D24,2025:$D25"
run() { HTH_TRAIN_YEARS=$1 HTH_TEST_YEAR=$2 HTH_RESULTS=$D25/$3 HTH_DATA=$D25 $P 05_models.py; }
run 2025 2025 results_e1_train2025_test2025
run 2024,2025 2025 results_e2_train2024-2025_test2025
run 2024 2025 results_e3_train2024_test2025
run 2024,2025 2024 results_e4_train2024-2025_test2024
echo "MODELS_DONE $(date)"
for seed in 1 2 3; do
  for v in s2 s2s1 s2s1_nocnn; do
    HTH_SEED=$seed HTH_TRAIN_YEARS=2024,2025 HTH_TEST_YEAR=2025 HTH_RESULTS=$D25/results_e2_train2024-2025_test2025 HTH_DATA=$D25 $P 06_deep.py $v > $H/deep24_${v}_$seed.log 2>&1 &
  done
  wait
done
echo "ALL_DONE $(date)"
