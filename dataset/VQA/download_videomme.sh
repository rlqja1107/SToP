set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 1. Clone HuggingFace dataset repo (skip if already present) ---
if [ ! -d "Video_MME" ]; then
    echo "[1/3] Cloning lmms-lab/Video-MME ..."
    git clone https://huggingface.co/datasets/lmms-lab/Video-MME
    mv Video-MME Video_MME
else
    echo "[1/3] Video_MME/ already exists — skipping clone."
fi
cd Video_MME

# --- 2. Place parquet at the path the loader expects ---
if [ ! -f "test-00000-of-00001.parquet" ]; then
    if [ -f "videomme/test-00000-of-00001.parquet" ]; then
        echo "[2/3] Moving parquet up from videomme/ ..."
        mv videomme/test-00000-of-00001.parquet ./
        rmdir videomme 2>/dev/null || true
    else
        echo "ERROR: parquet not found. Expected at videomme/test-00000-of-00001.parquet" >&2
        exit 1
    fi
else
    echo "[2/3] parquet already in place — skipping."
fi

# --- 3. Extract video chunks into ./video/ (flattened, no overwrite) ---
# Each zip is removed immediately after a successful extraction so disk
# usage stays bounded (~100GB of zips total).
mkdir -p video
echo "[3/3] Extracting video chunks and removing zips (this can take a while) ..."
for zip_file in videos_chunked_*.zip; do
    [ -f "$zip_file" ] || continue
    echo "  - $zip_file"
    # -j: strip 'data/' prefix, -n: never overwrite existing files
    unzip -j -n "$zip_file" -d video/ > /dev/null
    rm -f "$zip_file"
done
# subtitle.zip isn't used by the loader; drop it too.
rm -f subtitle.zip

video_count=$(ls video/*.mp4 2>/dev/null | wc -l)
echo
echo "Done. Video_MME layout:"
echo "  $(pwd)/test-00000-of-00001.parquet"
echo "  $(pwd)/video/  ($video_count mp4 files)"
