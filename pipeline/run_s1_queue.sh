#!/bin/bash
# Run S1 for each block once its S2 field table exists.
export HTH_DATA=${HTH_DATA:-~/hack-the-harvest/data} HTH_THREADS=12
for b in "$@"; do
  while [ ! -f $HTH_DATA/s2/obs_$b.parquet ]; do sleep 20; done
  ../.venv/bin/python 03_extract_s1.py $b
done
echo S1_QUEUE_DONE
