
# HoliTom + SToP
# μ_s, μ_t are auto-set from --pruning via eval/utils/config.py (STOP_MU_DEFAULTS)
#   holitom → μ_s=0.04, μ_t=0.07

export CUDA_VISIBLE_DEVICES='0'
PRUNING='holitom' # visionzip, flashvid, fastvid, prunevid, holitom
BACKBONE='lmms-lab/llava-onevision-qwen2-7b-ov' # lmms-lab/LLaVA-Video-7B-Qwen2, lmms-lab/llava-onevision-qwen2-7b-ov
RETENTION_RATIO=0.1 # 0.1, 0.15, 0.2
DATASET='EventHallusion' # EventHallusion, VCGBench, VideoComp, Video_MME, MVBench, MLVU

OPENAI_KEY="YOUR_OPENAI_KEY" # For VCGBench, EventHallusion

python eval/evaluate.py --pruning $PRUNING --backbone $BACKBONE --retention_ratio $RETENTION_RATIO --dataset $DATASET --openai_key $OPENAI_KEY
