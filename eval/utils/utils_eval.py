import os
import re
import sys
import json
import torch
import random
from tqdm import tqdm
import numpy as np
from eval.generators import (
    generate_ov,
    generate_qwen2_5_vl_prune,
)

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../SToP/eval/utils
EVAL_DIR   = os.path.dirname(_UTILS_DIR)                  # .../SToP/eval
ROOT_DIR   = os.path.dirname(EVAL_DIR)                    # .../SToP
PRUNING_DIR = os.path.join(ROOT_DIR, "pruning")           # .../SToP/pruning

# Make pruning sub-packages (visionzip, holitom, fastvid, …) importable
if PRUNING_DIR not in sys.path:
    sys.path.insert(0, PRUNING_DIR)




def _backbone_short_name(backbone: str) -> str:
    """Map a HuggingFace model ID to a short directory-friendly name."""
    _map = {
        "lmms-lab/llava-onevision-qwen2-7b-ov": "LLaVA_OV_7B",
        "lmms-lab/LLaVA-Video-7B-Qwen2": "LLaVA_Video_7B",
        "liuhaotian/llava-v1.5-7b": "LLaVA_v1.5_7B",
        "liuhaotian/llava-v1.5-13b": "LLaVA_v1.5_13B",
        "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct": "LLaVA_OV1.5_8B",
        "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct": "LLaVA_OV1.5_4B",
        "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5_VL_7B",
        "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5_VL_3B",
    }
    return _map.get(backbone, backbone.split("/")[-1])


def save_results(args, answer_list) -> bool:
    """
    Determine the output file path based on active experimental settings and
    write *answer_list* to disk as JSON.

    Returns ``True`` when a specialised path was used (caller should exit
    afterwards); ``False`` when the default naming scheme should be used.
    """
    backbone_name = _backbone_short_name(args.backbone)
    base_dir = os.path.join("result", backbone_name, args.dataset)
    os.makedirs(base_dir, exist_ok=True)
    if args.retention_ratio != 0.0:
    # --- Explicit pruning method ---
        base_dir = os.path.join(base_dir, args.pruning)
        os.makedirs(base_dir, exist_ok=True)
        save_path = os.path.join(base_dir, f"pred_ret_{args.retention_ratio}") if args.mu_s == 0.0 else os.path.join(base_dir, f"SToP_pred_ret_{args.retention_ratio}")
    else:
        save_path = os.path.join(base_dir, f"pred_vanilla")
    if args.dataset in ['EventHallusion', 'VCGBench'] and args.save_pred_result:
        with open(save_path+".json", 'w') as f:
            json.dump(answer_list, f)
    return base_dir


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _set_pruning_env(args):
    """Set environment variables for the pruning method."""
    if "visionzip" in args.pruning:
        os.environ["WRAPPER"] = "visionzip"
        # CLIP-based models (LLaVA-1.5) have 576 tokens (24x24), SigLIP has 729 (27x27)
        is_clip = "llava-v1.5" in args.backbone.lower()
        if is_clip:
            ratio_map = {0.10: 14, 0.15: 22, 0.20: 29, 0.25: 36}
            base_tokens = 144
        else:
            ratio_map = {0.10: 20, 0.15: 30, 0.20: 40, 0.25: 50}
            base_tokens = 196
        spatial_tokens = ratio_map.get(
            args.retention_ratio, int(base_tokens * args.retention_ratio)
        )
        os.environ["SPATIAL_TOKENS"] = str(spatial_tokens)

    elif "prunevid" in args.pruning:
        os.environ["WRAPPER"] = "prunevid"
        os.environ["RETAIN_RATIO"] = str(args.retention_ratio)

    elif "fastvid" in args.pruning:
        os.environ["WRAPPER"] = "fastvid"
        os.environ["fastvid_retention_ratio"] = str(args.retention_ratio)
        os.environ["fastvid_DySeg_c"] = "8"
        os.environ["fastvid_DySeg_tau"] = "0.9"
        os.environ["fastvid_STPrune_d"] = "0.4"
        os.environ["fastvid_DTM_p"] = "4"
        os.environ["fastvid_DTM_alpha"] = "0.6"
        os.environ["_load_vision_abstract"] = "1"

    elif "flashvid" in args.pruning:
        os.environ["WRAPPER"] = "flashvid"
        os.environ["retention_ratio"] = str(args.retention_ratio)
        os.environ["do_segment"] = "True"
        os.environ["segment_threshold"] = "0.9"
        os.environ["min_segment_num"] = "8"
        os.environ["complementary_segment"] = "True"
        os.environ["token_selection_method"] = "attn_div_v2"
        os.environ["temporal_threshold"] = "0.8"
        os.environ["expansion"] = "1.25"
        os.environ["pruning_layer"] = "20"
        os.environ["llm_retention_ratio"] = "0.3"

    elif 'holitom' in args.pruning:
        os.environ["WRAPPER"] = "holitom"
        os.environ["RETAIN_RATIO"] = str(args.retention_ratio)
        os.environ["T"] = "0.8"


def _load_llava(args, model_name):
    """Load a LLaVA-family model (OneVision or v1.5)."""
    sys.path.append(os.path.join(PRUNING_DIR, "LLaVA_NeXT_"))
    from pruning.LLaVA_NeXT.llava.model.builder import load_pretrained_model

    # LLaVA-1.5 needs eager attention (no SDPA support for output_attentions)
    attn_impl = "eager" if "llava-v1.5" in args.backbone.lower() else "sdpa"

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.backbone, None, model_name, load_8bit=False,
        attn_implementation=attn_impl,
    )

    # LLaVA-1.5 doesn't have spatial pool config; add defaults for pruning
    if not hasattr(model.config, "mm_spatial_pool_mode"):
        model.config.mm_spatial_pool_mode = "bilinear"
    if not hasattr(model.config, "mm_spatial_pool_stride"):
        model.config.mm_spatial_pool_stride = 2
    if not hasattr(model.config, "mm_newline_position"):
        model.config.mm_newline_position = "one_token"
    if not hasattr(model.config, "mm_patch_merge_type"):
        model.config.mm_patch_merge_type = "spatial_unpad"

    model = wrapper(model, args)
    model.eval()
    return model, tokenizer, image_processor


def _load_qwen2vl(args):
    """Load a Qwen2.5-VL model with pruning support."""
    from pruning.qwen2vl import Qwen2VLPruneWrapper
    model_wrapper = Qwen2VLPruneWrapper(args.backbone, args)
    return model_wrapper, model_wrapper.tokenizer, model_wrapper.processor


def _load_llava_ov15(args):
    """Load a LLaVA-OneVision-1.5 model with pruning support."""
    from pruning.llava_ov15 import LLaVAOV15PruneWrapper
    model_wrapper = LLaVAOV15PruneWrapper(args.backbone, args)
    return model_wrapper, model_wrapper.tokenizer, model_wrapper.processor


def load_model(args):
    if args.pruning != "none":
        _set_pruning_env(args)
    os.chdir(ROOT_DIR)

    backbone = args.backbone.lower()
    if "qwen2.5-vl" in backbone or "qwen2-vl" in backbone:
        return _load_qwen2vl(args)
    elif "llava-onevision-1.5" in backbone:
        return _load_llava_ov15(args)
    elif "llava-v1.5" in backbone:
        return _load_llava(args, "llava_llama")
    else:
        # Default: LLaVA-OneVision / LLaVA-Video (Qwen2 backbone)
        return _load_llava(args, "llava_qwen")




def wrapper(model, args):
    """Apply the configured wrapper module to *model* and return the wrapped model."""
    wrapper_name = os.environ.get("WRAPPER")
    if wrapper_name in ("visionzip", "holitom", "prunevid", "fastvid", "flashvid"):
        wrapper_module = __import__(wrapper_name)
        wrapper_cls = getattr(wrapper_module, wrapper_name)
        return wrapper_cls(model)
    if wrapper_name is not None:
        raise ValueError(f"Unknown wrapper: '{wrapper_name}'")
    return model


def run_inference(args, model, tokenizer, video, input_ids, attention_masks) -> str:
    """Route inference to the appropriate model-specific generation function."""
    backbone = args.backbone.lower()

    if "qwen2.5-vl" in backbone or "qwen2-vl" in backbone:
        return generate_qwen2_5_vl_prune(model, 1024, args, input_ids)
    elif "llava-onevision-1.5" in backbone:
        return generate_qwen2_5_vl_prune(model, 1024, args, input_ids)
    elif "llava" in backbone:
        image_sizes = attention_masks
        return generate_ov(model, video, image_sizes, tokenizer, input_ids, args)

    raise ValueError(f"No generation function matched backbone='{args.backbone}'")






def record_prediction(args, dataset, answer_list, video_path, prompt, gt_answer, response):
    """Append one prediction record to *answer_list*."""
    if hasattr(dataset, "eval_type"):
        answer_list.append({
            "video_id": "/".join(video_path.split("/")[-2:]),
            "prompt": prompt,
            "gt": gt_answer,
            "pred": response,
            "eval_type": dataset.eval_type,
        })
        return

    if isinstance(video_path, (list, tuple)) and len(video_path) == 2:
        answer_list.append({
            "video_id": "/".join(video_path[0].split("/")[-2:]),
            "needle_id": "/".join(video_path[1].split("/")[-1:]),
            "prompt": prompt,
            "gt": gt_answer,
            "pred": response,
        })
    else:
        vid = (
            "/".join(video_path.split("/")[-2:])
            if isinstance(video_path, str)
            else "/".join(video_path[-2:])
        )
        answer_list.append({"video_id": vid, "prompt": prompt, "gt": gt_answer, "pred": response})


def set_seed(seed: int = 8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Single standalone letter (no neighboring letters) for MCQA letter extraction.
_MCQA_LETTER_RE = re.compile(r'(?<![A-Za-z])([A-Za-z])(?![A-Za-z])')


def _mcqa_match(gt, pred) -> int:
    """Return 1 if `pred` matches `gt` under MCQA-style tolerant matching, else 0.

    - int gt: parse `pred` as int; non-numeric pred → wrong (no crash).
    - single-letter gt: extract the first standalone letter from `pred`
      (handles "A", "A.", "The answer is A", etc.); compare case-insensitively.
    - other: strip + case-insensitive exact match.
    """
    if pred is None:
        return 0
    try:
        if isinstance(gt, int):
            return int(int(str(pred).strip()) == gt)
        gt_s = str(gt).strip()
        pred_s = str(pred).strip()
        if len(gt_s) == 1 and gt_s.isalpha():
            m = _MCQA_LETTER_RE.search(pred_s)
            chosen = (m.group(1) if m else pred_s[:1])
            return int(chosen.upper() == gt_s.upper())
        return int(gt_s.lower() == pred_s.lower())
    except (ValueError, TypeError):
        return 0


def compute_save_mcqa(answer_list, base_dir, args):
    total = len(answer_list)
    if total == 0:
        print("No samples evaluated — skipping accuracy computation.")
        return

    correct = sum(_mcqa_match(i['gt'], i['pred']) for i in answer_list)
    acc = correct / total
    acc_dict = {"Overall accuracy": np.round(acc, 4)}
    print(f"Accuracy: {acc:.4f}")

    # Video_MME: also report per-duration accuracy (short / medium / long).
    if "Video_MME" in args.dataset and "eval_type" in answer_list[0]:
        from collections import defaultdict
        bucket = defaultdict(list)
        for i in answer_list:
            bucket[str(i['eval_type']).lower()].append(_mcqa_match(i['gt'], i['pred']))
        for duration in ("short", "medium", "long"):
            if duration in bucket:
                dur_acc = float(np.round(np.mean(bucket[duration]), 4))
                acc_dict[duration] = dur_acc
                print(f"  {duration:6s}: {dur_acc:.4f}  (n={len(bucket[duration])})")

    save_path = (
        os.path.join(base_dir, f"ret_{args.retention_ratio}_acc")
        if args.mu_s == 0.0
        else os.path.join(base_dir, f"SToP_ret_{args.retention_ratio}_acc")
    )
    with open(save_path + ".json", "w") as f:
        json.dump(acc_dict, f)

def compute_save_eventhallusion(answer_list, base_dir, args):
    from eval.utils.eval_eventhallusion import interleave, entire, misleading, get_chat_gpt_response, compute_eventhallusion_qa_result
    import os
    from openai import OpenAI
    
    os.environ['OPENAI_API_KEY'] = args.openai_key
    client = OpenAI()

    # ---- Description (GPT-judged) ----
    print("Evaluating EventHallusion")
    desc_result = []
    for i in tqdm(answer_list):
        eval_type = i['eval_type']
        if '_desc' not in eval_type:
            continue
        split = eval_type.split('_')[0]
        if split == 'interleave':
            prompt = interleave.format(i['pred'], i['gt'])
        elif split == 'entire':
            prompt = entire.format(i['pred'], i['gt'])
        else:
            prompt = misleading.format(i['pred'], i['gt'])
        response = get_chat_gpt_response(prompt, client)
        desc_result.append(1 if response.lower().startswith("yes") else 0)

    # ---- Binary ----
    binary_result = []
    for h in answer_list:
        compute_eventhallusion_qa_result(h, binary_result)

    desc_acc = float(np.mean(desc_result))
    binary_acc = float(np.mean(binary_result))
    overall_acc = (sum(binary_result) + sum(desc_result)) / (len(desc_result) + len(binary_result))

    print(f"Desc Acc: {desc_acc:.4f} | Binary Acc: {binary_acc:.4f} | Overall Acc: {overall_acc:.4f}")
    result_dict = {
        "desc_acc":    round(desc_acc, 4),
        "binary_acc":  round(binary_acc, 4),
        "overall_acc": round(overall_acc, 4),
    }
    prefix = "SToP_" if args.mu_s != 0.0 else ""
    result_save_path = os.path.join(base_dir, f"{prefix}ret_{args.retention_ratio}_acc_result.json")
    with open(result_save_path, "w") as f:
        json.dump(result_dict, f, indent=4)
    
    
def compute_save_vcgbench(answer_list, base_dir, args):
    import os
    from openai import OpenAI
    
    os.environ['OPENAI_API_KEY'] = args.openai_key
    client = OpenAI()
    from eval.utils.eval_vcgbench.consistency import consistency_chat_gpt_response
    from eval.utils.eval_vcgbench.temporal import temporal_chat_gpt_response
    from eval.utils.eval_vcgbench.context import context_chat_gpt_response
    from eval.utils.eval_vcgbench.orientation import orientation_chat_gpt_response
    from eval.utils.eval_vcgbench.correctness import correct_chat_gpt_response
    
    generic_type_data = []; consistency_type_data = []; temporal_type_data = []
    for j in answer_list:
        type_ = j['eval_type'].split("/")[-1]
        if type_ == 'generic':
            generic_type_data.append(j)
        elif type_ == 'consistency':
            consistency_type_data.append(j)
        elif type_ == 'temporal':
            temporal_type_data.append(j)
    consistency_score_list = []; temporal_score_list = []; context_score_list = []; orientation_score_list = []; correct_score_list = []; overall_score = []
    print("Evalating Consistency")
    question_list = []
    answer_list = []
    pred_list = []
    for k in tqdm(consistency_type_data):
        if k['prompt'] == 'None': continue
        question_list.append(k['prompt'])
        answer_list.append(k['gt'])
        pred_list.append(k['pred'])
        if len(pred_list) == 2:
            try:
                score = eval(consistency_chat_gpt_response(question_list[0], question_list[1], answer_list[0], pred_list[0], pred_list[1], client))['score'] # question1, question2,  answer, pred1, pred2
                consistency_score_list.append(float(score))
                overall_score.append(float(score))
            except:
                pass
            question_list.clear(); answer_list.clear(); pred_list.clear()
            
    print("Evaluating Temporal")
    for result in tqdm(temporal_type_data):
        try:
            score = eval(temporal_chat_gpt_response(result['prompt'], result['gt'], result['pred'], client))['score'] # question, answer, pred
            temporal_score_list.append(float(score))
            overall_score.append(float(score))
        except:
            pass
    
    print("Evaluating Generic")
    for result in tqdm(generic_type_data):
        try:
            context_score = eval(context_chat_gpt_response(result['prompt'], result['gt'], result['pred'], client))['score']
            orientation_score = eval(orientation_chat_gpt_response(result['prompt'], result['gt'], result['pred'], client))['score']
            correct_score = eval(correct_chat_gpt_response(result['prompt'], result['gt'], result['pred'], client))['score']
            
            context_score_list.append(float(context_score))
            overall_score.append(float(context_score))
            orientation_score_list.append(float(orientation_score))
            overall_score.append(float(orientation_score))
            correct_score_list.append(float(correct_score))
            overall_score.append(float(correct_score))
        except:
            pass
    overall = np.mean(overall_score)
    consistency = np.mean(consistency_score_list)
    temporal = np.mean(temporal_score_list)
    context = np.mean(context_score_list)
    orientation = np.mean(orientation_score_list)
    correct = np.mean(correct_score_list)
    print("================================")
    print(f"Overall Score: {overall:.4f}")
    print(f"Consistency: {consistency:.4f}, Temporal: {temporal:.4f}, Context: {context:.4f}, Orientation: {orientation:.4f}, Correct: {correct:.4f}")
    print("================================")
    result_save_path = os.path.join(base_dir, f"ret_{args.retention_ratio}_acc_result") if args.mu_s == 0.0 else os.path.join(base_dir, f"SToP_ret_{args.retention_ratio}_acc_result")
    result_dict = {"overall": overall, 'consistency': consistency, 'temporal':temporal, 'context': context, 'orientation': orientation, "correct": correct}
    with open(result_save_path+".json", "w") as f:
        json.dump(result_dict, f, indent=4)
    print("Save Complete")