# Qwen2.5-VL Scripts

Evaluation scripts for **`Qwen/Qwen2.5-VL-7B-Instruct`**.

## Supported pruning methods

For Qwen2.5-VL, **only VisionZip-style pruning is implemented**
([`pruning/qwen2vl/wrapper.py`](../../pruning/qwen2vl/wrapper.py)).
HoliTom, FastVid, FlashVid, and PruneVid are **not available** as separate
methods on this backbone — the wrapper runs the same VisionZip-style
algorithm regardless of the `--pruning` value when pruning is enabled.


## Scripts

| Script | Configuration |
|---|---|
| `no_pruning.sh` | `--pruning none --use_stop False` |
| `baseline.sh` | `--pruning visionzip --use_stop False` |
| `SToP_spatial.sh` | `--pruning visionzip` with default SToP μ values (μ_s = 3.7) |

Edit `BACKBONE`, `DATASET`, `RETENTION_RATIO`, or `CUDA_VISIBLE_DEVICES`
inside each script to vary the experiment.
