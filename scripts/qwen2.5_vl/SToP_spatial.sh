
# Qwen2.5-VL — VisionZip + SToP
# Only VisionZip is implemented for Qwen2.5-VL — see scripts/qwen2.5_vl/README.md
# μ_s, μ_t are auto-set from --pruning via eval/utils/config.py (STOP_MU_DEFAULTS)
#   visionzip → μ_s=0.03, μ_t=0.0

export CUDA_VISIBLE_DEVICES='0'
PRUNING='visionzip' # Qwen2.5-VL supports visionzip only
BACKBONE='Qwen/Qwen2.5-VL-7B-Instruct'
RETENTION_RATIO=0.1
DATASET='EventHallusion' # EventHallusion, VCGBench, VideoComp, Video_MME, MVBench, MLVU

OPENAI_KEY="YOUR_OPENAI_KEY" # For VCGBench, EventHallusion

python eval/evaluate.py --pruning $PRUNING --backbone $BACKBONE --retention_ratio $RETENTION_RATIO --dataset $DATASET --openai_key $OPENAI_KEY
