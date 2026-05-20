import torch

try:
    from llava.mm_utils import KeywordsStoppingCriteria
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Qwen2.5-VL (standard + pruned)
# ---------------------------------------------------------------------------


def generate_qwen2_5_vl_prune(model_wrapper, max_new_tokens: int, args, inputs) -> str:
    """Greedy decoding through the pruning-enabled Qwen2.5-VL / LLaVA-OV-1.5 wrapper."""
    pad_token_id = model_wrapper.tokenizer.pad_token_id

    # Update wrapper args so the patched visual encoder uses current settings
    model_wrapper.args = args

    # Filter out keys that model.generate() doesn't accept
    gen_inputs = {k: v for k, v in inputs.items()
                  if k not in ("second_per_grid_ts",)}

    outputs = model_wrapper.model.generate(
        **gen_inputs,
        eos_token_id=model_wrapper.tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=model_wrapper.use_cache,
    )
    trimmed = outputs[0, inputs.input_ids.shape[1]:]
    return model_wrapper.processor.decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


# ---------------------------------------------------------------------------
# LLaVA-OV wrapper
# ---------------------------------------------------------------------------

def generate_ov(model, image_tensors, image_sizes, tokenizer, input_ids, args) -> str:
    with torch.no_grad():
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            top_k=None,
            top_p=None,
            temperature=None,
            max_new_tokens=4096,
            modalities=["video"],
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
            args=args,
        )
    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

# ---------------------------------------------------------------------------
# DyCoke
# ---------------------------------------------------------------------------

def generate_dycoke(model, input_ids, attention_masks, pad_token_ids, image_tensors, tokenizer) -> str:
    """Run DyCoke generation with keyword stopping."""
    stopping_criteria = KeywordsStoppingCriteria(["<|im_end|>"], tokenizer, input_ids)
    gen_kwargs = dict(
        max_new_tokens=1024,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        do_sample=False,
        modalities=["video"],
        stopping_criteria=[stopping_criteria],
    )
    with torch.inference_mode():
        cont = model.generate(
            input_ids,
            attention_mask=attention_masks,
            pad_token_id=pad_token_ids,
            images=image_tensors,
            use_cache=True,
            **gen_kwargs,
        )
    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

