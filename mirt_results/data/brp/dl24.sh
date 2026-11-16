#!/bin/bash
U=https://service.pdok.nl/rvo/gewaspercelen/atom/downloads/brpgewaspercelen_definitief_2024.gpkg
SIZE=3106852864; N=8; CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$(( s+CH-1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
  ( for t in 1 2 3 4 5; do curl -s --retry 5 -r $s-$e -o part_$i $U && [ $(stat -c%s part_$i) -eq $((e-s+1)) ] && break; sleep 5; done ) &
done
wait
cat $(for i in $(seq 0 $((N-1))); do echo part_$i; done) > brp_2024.gpkg && rm part_*
echo "DONE $(stat -c%s brp_2024.gpkg)"
