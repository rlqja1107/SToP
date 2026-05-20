#!/usr/bin/env bash
# Download and prepare VideoComp (ActivityNet + YouCook2) for evaluation.
#
# Layout expected by eval/benchmark_data.py:
#   <repo>/dataset/fine_grained_task/VideoComp/
#       activitynet_comp_val.json
#       youcook2_comp_val.json
#       activitynet_video/v_<id>.{mp4,mkv}
#       raw_videos/video/<id>.{mp4,mkv,webm}
#
# Idempotent: re-running skips work already done.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p VideoComp
cd VideoComp

# --- 1. Annotation JSONs (must live inside VideoComp/) ---
echo "[1/3] Downloading annotation JSONs ..."
wget -nc https://storage.googleapis.com/video_comp/activitynet_comp_val.json
wget -nc https://storage.googleapis.com/video_comp/youcook2_comp_val.json

# --- 2. ActivityNet videos (split tar in 8 parts) ---
if [ -z "$(find activitynet_video -type f \( -name '*.mp4' -o -name '*.mkv' \) 2>/dev/null | head -1)" ]; then
    echo "[2/3] Downloading ActivityNet videos (8 parts) ..."
    for i in $(seq -w 0 7); do
        wget -nc "https://huggingface.co/datasets/friedrichor/ActivityNet_Captions/resolve/main/ActivityNet_Videos.tar.part-00${i}"
    done
    echo "  Concatenating and extracting tar ..."
    cat ActivityNet_Videos.tar.part-* | tar -xf -
    [ -f val.tar.gz ] && tar -xzf val.tar.gz && rm -f val.tar.gz
    [ -d Activity_Videos ] && mv Activity_Videos activitynet_video
    rm -f ActivityNet_Videos.tar.part-*
else
    echo "[2/3] activitynet_video/ already populated — skipping."
fi

# --- 3. YouCook2 videos ---
if [ -z "$(find raw_videos/video -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.webm' \) 2>/dev/null | head -1)" ]; then
    echo "[3/3] Downloading YouCook2 videos ..."
    mkdir -p raw_videos/video
    if [ ! -f "YouCookIIVideos.zip" ]; then
        wget -nc https://huggingface.co/datasets/lmms-lab/YouCook2/resolve/main/YouCookIIVideos.zip
    fi
    unzip -n YouCookIIVideos.zip
    [ -d "YouCookIIVideos/test" ] && mv YouCookIIVideos/test/* raw_videos/video/ 2>/dev/null || true
    [ -d "YouCookIIVideos/val" ]  && mv YouCookIIVideos/val/*  raw_videos/video/ 2>/dev/null || true
    rm -rf YouCookIIVideos YouCookIIVideos.zip
else
    echo "[3/3] raw_videos/video/ already populated — skipping."
fi

an_count=$(find activitynet_video -type f \( -name '*.mp4' -o -name '*.mkv' \) 2>/dev/null | wc -l)
yc_count=$(find raw_videos/video -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.webm' \) 2>/dev/null | wc -l)
echo
echo "Done. VideoComp layout:"
echo "  $(pwd)/activitynet_comp_val.json"
echo "  $(pwd)/youcook2_comp_val.json"
echo "  $(pwd)/activitynet_video/   ($an_count video files)"
echo "  $(pwd)/raw_videos/video/    ($yc_count video files)"
