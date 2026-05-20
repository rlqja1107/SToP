
# LLaVA-Video-7B-Qwen2 — Baseline: pruning method WITHOUT SToP.

export CUDA_VISIBLE_DEVICES='0'
PRUNING='visionzip' # visionzip, flashvid, fastvid, prunevid, holitom
BACKBONE='lmms-lab/LLaVA-Video-7B-Qwen2'
RETENTION_RATIO=0.1
DATASET='EventHallusion' # EventHallusion, VCGBench, VideoComp, Video_MME, MVBench, MLVU

OPENAI_KEY="YOUR_OPENAI_KEY" # For VCGBench, EventHallusion

python eval/evaluate.py --pruning $PRUNING --backbone $BACKBONE --retention_ratio $RETENTION_RATIO --use_stop False --openai_key $OPENAI_KEY --dataset $DATASET
