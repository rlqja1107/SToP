
# Baseline: pruning method WITHOUT SToP.
# --use_stop False forces μ_s = μ_t = 0
# For the no-pruning (vanilla) baseline, see scripts/no_pruning.sh

export CUDA_VISIBLE_DEVICES='7'
PRUNING='fastvid' # visionzip, flashvid, fastvid, prunevid, holitom
BACKBONE='lmms-lab/llava-onevision-qwen2-7b-ov' # lmms-lab/LLaVA-Video-7B-Qwen2, lmms-lab/llava-onevision-qwen2-7b-ov
RETENTION_RATIO=0.1
DATASET='EventHallusion' # EventHallusion, VCGBench, VideoComp, Video_MME, MVBench, MLVU

OPENAI_KEY="YOUR_OPENAI_KEY" # For VCGBench, EventHallusion

python eval/evaluate.py --pruning $PRUNING --backbone $BACKBONE --retention_ratio $RETENTION_RATIO --use_stop False --openai_key $OPENAI_KEY --dataset $DATASET
