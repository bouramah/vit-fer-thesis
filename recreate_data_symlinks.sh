#!/bin/bash
# Recrée data_rafdb_fer/ (format FER) à partir de data_rafdb/DATASET/ (RAF-DB aligné).
set -e
cd "$(dirname "$0")"
BASE="$(pwd)/data_rafdb/DATASET"
FER="$(pwd)/data_rafdb_fer"
declare -A MAP=( [1]=surprise [2]=fear [3]=disgust [4]=happy [5]=sad [6]=angry [7]=neutral )
for sp in train test; do
  for k in "${!MAP[@]}"; do
    mkdir -p "$FER/$sp"
    ln -sfn "$BASE/$sp/$k" "$FER/$sp/${MAP[$k]}"
  done
done
echo "OK -> data_rafdb_fer/ recréé."
