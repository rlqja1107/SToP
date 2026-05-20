#!/usr/bin/env bash
# Download and prepare VCGBench (VideoChatGPT) for evaluation.
#
# Layout expected by eval/benchmark_data.py:
#   <repo>/dataset/fine_grained_task/VCGBench/
#       {Generic,Temporal,Consistency}/test-00000-of-00001.parquet
#       Test_Videos/<video_name>.<ext>     # ext ∈ {mp4, avi, mov, mkv}
#
# Idempotent: re-running skips work already done.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p VCGBench
cd VCGBench

# --- 1. Parquet annotations for 3 splits ---
echo "[1/2] Downloading parquet annotations ..."
for split in Generic Temporal Consistency; do
    mkdir -p "$split"
    if [ ! -f "$split/test-00000-of-00001.parquet" ]; then
        wget -nc "https://huggingface.co/datasets/lmms-lab/VideoChatGPT/resolve/main/$split/test-00000-of-00001.parquet"
        mv -f test-00000-of-00001.parquet "$split/"
    fi
done

# --- 2. Test_Videos ---
if [ -z "$(find Test_Videos -type f \( -name '*.mp4' -o -name '*.avi' -o -name '*.mov' -o -name '*.mkv' \) 2>/dev/null | head -1)" ]; then
    echo "[2/2] Downloading and extracting videos ..."
    if [ ! -f "videos.zip" ]; then
        wget -nc https://huggingface.co/datasets/lmms-lab/VideoChatGPT/resolve/main/videos.zip
    fi
    unzip -n videos.zip
    rm -f videos.zip
else
    echo "[2/2] Test_Videos/ already populated — skipping."
fi

video_count=$(find Test_Videos -type f \( -name '*.mp4' -o -name '*.avi' -o -name '*.mov' -o -name '*.mkv' \) 2>/dev/null | wc -l)
echo
echo "Done. VCGBench layout:"
for split in Generic Temporal Consistency; do
    echo "  $(pwd)/$split/test-00000-of-00001.parquet"
done
echo "  $(pwd)/Test_Videos/  ($video_count video files)"
