
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../SToP/pruning
cd "$SCRIPT_DIR"

DEST="prunevid_upstream"
REPO="https://github.com/Visual-AI/PruneVid.git"

if [ ! -d "$DEST" ]; then
    echo "[1/1] Cloning upstream PruneVid (CC BY-NC-SA 4.0) into $DEST/ ..."
    git clone "$REPO" "$DEST"
else
    echo "[1/1] $DEST/ already exists — skipping clone."
fi

echo
echo "Done. Upstream PruneVid reference layout:"
echo "  $SCRIPT_DIR/$DEST/                (original PruneVid source, CC BY-NC-SA 4.0)"
echo "  $SCRIPT_DIR/$DEST/LICENSE         (PruneVid license — NonCommercial)"
echo
echo "NOTE: This directory is git-ignored. Do not commit it into SToP."
echo "      The code is NonCommercial (CC BY-NC-SA 4.0); derivatives inherit that license."
