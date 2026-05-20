#!/usr/bin/env bash
# Download and prepare EventHallusion for evaluation.
#
# Layout expected by eval/benchmark_data.py:
#   <repo>/dataset/fine_grained_task/EventHallusion/questions/{entire,misleading,mix}_questions.json
#   <repo>/dataset/fine_grained_task/EventHallusion/videos/{entire,interleave,misleading}/*.mp4
#
# Idempotent: re-running skips work already done.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p EventHallusion
cd EventHallusion

# --- 1. Question JSONs (GitHub raw) ---
echo "[1/2] Downloading question JSON files ..."
mkdir -p questions
for f in entire_questions.json misleading_questions.json mix_questions.json; do
    if [ ! -f "questions/$f" ]; then
        wget -nc "https://raw.githubusercontent.com/Stevetich/EventHallusion/refs/heads/master/questions/$f"
        mv -f "$f" "questions/$f"
    fi
done

# --- 2. Videos (Google Drive zip) ---
if [ -z "$(find videos -mindepth 2 -name '*.mp4' 2>/dev/null | head -1)" ]; then
    echo "[2/2] Downloading and extracting videos ..."
    if [ ! -f "eventhallusion.zip" ]; then
        gdown --continue "https://drive.google.com/uc?id=1IPmx6Y80UrXwVPmZJh6zjCPHtlsw4p9n"
    fi
    unzip -n eventhallusion.zip
    rm -rf __MACOSX
    rm -f eventhallusion.zip
else
    echo "[2/2] videos/ already populated — skipping."
    rm -rf __MACOSX
fi

json_count=$(ls questions/*.json 2>/dev/null | wc -l)
video_count=$(find videos -type f -name '*.mp4' 2>/dev/null | wc -l)
echo
echo "Done. EventHallusion layout:"
echo "  $(pwd)/questions/  ($json_count json files)"
echo "  $(pwd)/videos/     ($video_count mp4 files)"
