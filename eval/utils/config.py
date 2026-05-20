import argparse
import os

# Default dataset root: SToP/dataset/ (resolved relative to this file).
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))           # SToP/eval/utils
_DEFAULT_DATASET_ROOT = os.path.normpath(
    os.path.join(_CONFIG_DIR, "..", "..", "dataset")               # SToP/dataset
)


# Default (μ_s, μ_t) per pruning method when SToP is enabled.
# NOTE: HoliTom's hyperparameter values are sensitive to the retention ratio; the
# defaults below are tuned for a 10% retention ratio.
STOP_MU_DEFAULTS = {
    "visionzip": (0.03, 0.0),
    "fastvid":   (0.02, 0.0),
    "holitom":   (0.04, 0.07),
}


STOP_MU_BACKBONE_OVERRIDES = {
    "qwen2.5-vl": {
        "visionzip": (3.7, 0.0),
    },
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Video-LLM on standard video understanding benchmarks."
    )
    # Model selection
    parser.add_argument(
        "--pruning", type=str, default="holitom", choices=['none', 'prunevid', 'holitom', 'visionzip', 'fastvid', 'flashvid']
    )
    parser.add_argument("--backbone", default="lmms-lab/llava-onevision-qwen2-7b-ov", type=str,
                        help="Model backbone. Supported: "
                             "lmms-lab/llava-onevision-qwen2-7b-ov, "
                             "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct, "
                             "lmms-lab/LLaVA-Video-7B-Qwen2, "
                             "Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", default="EventHallusion", type=str)
    parser.add_argument("--root_dataset", type=str, default=_DEFAULT_DATASET_ROOT,
                        help="Root dir containing VQA/, fine_grained_task/, etc. "
                             "Defaults to SToP/dataset/")
    parser.add_argument("--n_frame", default=32, type=int, help="Number of video frames to sample.")
    parser.add_argument("--seed", default=4, type=int)
    parser.add_argument("--openai_key", default="", type=str)
    parser.add_argument("--save_pred_result", default=False, type=str2bool)

    # Token pruning / retention settings
    parser.add_argument("--retention_ratio", type=float, default=0.2,
                        help="Fraction of visual tokens to retain.")

    # SToP switch: when False, μ_s = μ_t = 0 (baseline / vanilla pruning).
    # When True, μ_s and μ_t are set from STOP_MU_DEFAULTS based on --pruning.
    parser.add_argument("--use_stop", type=str2bool, default=True,
                        help="Apply SToP sink-aware adjustment with per-method μ defaults. "
                             "Set to False to run the corresponding baseline (μ_s = μ_t = 0).")

    parser.add_argument("--gamma", type=float, default=1.1,
                        help="γ: context module scaling factor.")
    return parser


def apply_stop_defaults(args):
    """Set args.mu_s / args.mu_t based on --backbone, --pruning and --use_stop."""
    if not args.use_stop:
        args.mu_s, args.mu_t = 0.0, 0.0
        return args

    # Backbone-specific override takes precedence
    backbone_lower = args.backbone.lower()
    for bk_key, method_map in STOP_MU_BACKBONE_OVERRIDES.items():
        if bk_key in backbone_lower:
            args.mu_s, args.mu_t = method_map.get(args.pruning, (0.0, 0.0))
            return args

    args.mu_s, args.mu_t = STOP_MU_DEFAULTS.get(args.pruning, (0.0, 0.0))
    return args
