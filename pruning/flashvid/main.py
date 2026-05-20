import os
from .llava_arch import LlavaMetaForCausalLM_flashvid

def flashvid(model):
    
    print("################################")
    print("########## FlashVid ###########")
    print("################################")
    
    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_flashvid.prepare_inputs_labels_for_multimodal
    LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_flashvid.encode_images
    LlavaMetaForCausalLM.encode_images_multi = LlavaMetaForCausalLM_flashvid.encode_images_multi
    LlavaMetaForCausalLM.attention_sink_cnt = []
    return model
