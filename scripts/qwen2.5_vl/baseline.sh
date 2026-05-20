
# Qwen2.5-VL — Baseline: VisionZip pruning WITHOUT SToP.
# --use_stop False forces μ_s = μ_t = 0
# Only VisionZip is implemented for Qwen2.5-VL — see scripts/qwen2.5_vl/README.md
# For the no-pruning (vanilla) baseline, see scripts/qwen2.5_vl/no_pruning.sh

export CUDA_VISIBLE_DEVICES='0'
PRUNING='visionzip' # Qwen2.5-VL supports visionzip only
BACKBONE='Qwen/Qwen2.5-VL-7B-Instruct'
RETENTION_RATIO=0.1
DATASET='EventHallusion' # EventHallusion, VCGBench, VideoComp, Video_MME, MVBench, MLVU

OPENAI_KEY="YOUR_OPENAI_KEY" # For VCGBench, EventHallusion

python eval/evaluate.py --pruning $PRUNING --backbone $BACKBONE --retention_ratio $RETENTION_RATIO --use_stop False --openai_key $OPENAI_KEY --dataset $DATASET
