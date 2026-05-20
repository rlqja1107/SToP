#!/usr/bin/env bash
# Download and prepare MVBench for evaluation.
#
# Layout expected by eval/benchmark_data.py:
#   <repo>/dataset/VQA/MVBench/json/{action_sequence,action_prediction,...}.json
#   <repo>/dataset/VQA/MVBench/video/{star,ssv2_video,Moments_in_Time_Raw,FunQA_test,
#                                     clevrer,perception,sta,scene_qa,nturgbd,tvqa,vlnqa}/...
#
# Idempotent: re-running skips work already done.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 1. Clone HuggingFace dataset repo (skip if already present) ---
if [ ! -d "MVBench" ]; then
    echo "[1/2] Cloning OpenGVLab/MVBench ..."
    git clone https://huggingface.co/datasets/OpenGVLab/MVBench
else
    echo "[1/2] MVBench/ already exists — skipping clone."
fi
cd MVBench/video

# Some snapshots ship a flat data0613.zip that mixes everything together;
# we don't want it (per-source zips below give the right layout).
[ -f data0613.zip ] && rm -f data0613.zip

# --- 2. Extract per-source zips (flatten alongside) ---
# Each zip is removed immediately after a successful extraction so disk
# usage stays bounded.
echo "[2/2] Extracting per-source video zips and removing zips ..."
shopt -s nullglob
for zip_file in *.zip; do
    echo "  - $zip_file"
    unzip -n "$zip_file" > /dev/null
    rm -f "$zip_file"
done

src_dirs=$(find . -maxdepth 1 -mindepth 1 -type d | wc -l)
json_count=$(ls ../json/*.json 2>/dev/null | wc -l)

echo
echo "Done. MVBench layout:"
echo "  $(pwd)/../json/  ($json_count json files)"
echo "  $(pwd)/         ($src_dirs source dirs)"
