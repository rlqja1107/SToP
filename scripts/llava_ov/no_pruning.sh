
# Vanilla: no pruning at all.
# --pruning none, --retention_ratio 0.0
# --use_stop False forces μ_s = μ_t = 0

export CUDA_VISIBLE_DEVICES='0'
BACKBONE='lmms-lab/llava-onevision-qwen2-7b-ov' # lmms-lab/LLaVA-Video-7B-Qwen2, lmms-lab/llava-onevision-qwen2-7b-ov
DATASET='EventHallusion' # EventHallusion, VCGBench, VideoComp, Video_MME, MVBench, MLVU

OPENAI_KEY="YOUR_OPENAI_KEY" # For VCGBench, EventHallusion

python eval/evaluate.py --pruning none --backbone $BACKBONE --retention_ratio 0.0 --use_stop False --openai_key $OPENAI_KEY --dataset $DATASET
