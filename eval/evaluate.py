
import os
import sys
import json
import torch
from tqdm import tqdm
sys.path.append(".")
torch.set_num_threads(4)
sys.path.append("pruning/LLaVA_NeXT")
from benchmark_data import EvalDataset
from eval.utils.config import build_arg_parser, apply_stop_defaults
from eval.utils.utils_eval import set_seed, load_model, run_inference, record_prediction, save_results, compute_save_mcqa, compute_save_eventhallusion, compute_save_vcgbench

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args = apply_stop_defaults(args)
    set_seed(args.seed)

    print(
        f"[Config] model={args.pruning} | dataset={args.dataset} | "
        f"n_frame={args.n_frame} | retention={args.retention_ratio} | "
        f"backbone={args.backbone} | use_stop={args.use_stop} | "
        f"mu_s={args.mu_s} | mu_t={args.mu_t}"
    )

    model, tokenizer, image_processor = load_model(args)

    dataset = EvalDataset(
        tokenizer, args,
        model=model,
    )
    dataset.image_processor = image_processor

    answer_list = []
    for i, (input_ids, video, attention_masks, video_path, prompt, answer) in tqdm(
        enumerate(dataset), total=len(dataset)
    ):
        response = run_inference(
            args=args,
            model=model,
            tokenizer=tokenizer,
            video=video,
            input_ids=input_ids,
            attention_masks=attention_masks,
        )
        record_prediction(args, dataset, answer_list, video_path, prompt, answer, response)


    # --- Save predictions ---
    base_dir = save_results(args, answer_list)
    if args.dataset not in ['EventHallusion', 'VCGBench']:
        # MCQA
        compute_save_mcqa(answer_list, base_dir, args)
    elif args.dataset == 'EventHallusion':
        compute_save_eventhallusion(answer_list, base_dir, args)
    elif args.dataset == 'VCGBench':
        compute_save_vcgbench(answer_list, base_dir, args)
        
        