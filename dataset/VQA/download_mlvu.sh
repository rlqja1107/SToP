#!/usr/bin/env bash
# Download and prepare MLVU (multiple-choice) for evaluation.
#
# Layout expected by eval/benchmark_data.py:
#   <repo>/dataset/VQA/MLVU/MLVU/json/{1_plotQA,2_needle,3_ego,4_count,5_order,
#                                      6_anomaly_reco,7_topic_reasoning}.json
#   <repo>/dataset/VQA/MLVU/MLVU/video/{1_plotQA,...}/{video_file.mp4}
#
# NOTE: the upstream HF repo is named "MVLU" (typo in the dataset author's repo).
# We rename the top-level clone to "MLVU" so paths match the loader.
#
# Idempotent: re-running skips work already done.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 1. Ensure MLVU/ is in place ---
if [ ! -d "MLVU" ]; then
    if [ -d "MVLU" ]; then
        echo "[1/1] Renaming pre-cloned MVLU/ -> MLVU/ to match loader path."
        mv MVLU MLVU
    else
        echo "[1/1] Cloning MLVU/MVLU and renaming to MLVU ..."
        git clone https://huggingface.co/datasets/MLVU/MVLU
        mv MVLU MLVU
    fi
else
    echo "[1/1] MLVU/ already exists — skipping."
fi

json_count=$(ls MLVU/MLVU/json/*.json 2>/dev/null | wc -l)
video_count=$(find MLVU/MLVU/video -type f -name "*.mp4" 2>/dev/null | wc -l)

echo
echo "Done. MLVU layout:"
echo "  $(pwd)/MLVU/MLVU/json/   ($json_count json files)"
echo "  $(pwd)/MLVU/MLVU/video/  ($video_count mp4 files across task subdirs)"
