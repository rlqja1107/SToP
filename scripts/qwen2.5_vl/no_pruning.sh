
# Qwen2.5-VL — Vanilla (no pruning).
# --pruning none, --retention_ratio 0.0
# --use_stop False forces μ_s = μ_t = 0

export CUDA_VISIBLE_DEVICES='0'
BACKBONE='Qwen/Qwen2.5-VL-7B-Instruct'
DATASET='EventHallusion' # EventHallusion, VCGBench, VideoComp, Video_MME, MVBench, MLVU

OPENAI_KEY="YOUR_OPENAI_KEY" # For VCGBench, EventHallusion

python eval/evaluate.py --pruning none --backbone $BACKBONE --retention_ratio 0.0 --use_stop False --openai_key $OPENAI_KEY --dataset $DATASET
