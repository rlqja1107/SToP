#!/usr/bin/env bash
set -e

# Source conda so `conda activate` works inside this script.
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n stop python=3.10 -y
conda activate stop

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3 pandas decord datasets \
    opencv-python==4.10.0.84 einops accelerate openai tqdm
pip install numpy==1.26.1 gdown
