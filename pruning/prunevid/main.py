import os
from .llava_arch import LlavaMetaForCausalLM_prunevid
from .modeling_qwen2 import Qwen2Model_prunevid

def prunevid(model):
    
    print("################################")
    print("############ PruneVid ###########")
    print("################################")

    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_prunevid.prepare_inputs_labels_for_multimodal
    LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_prunevid.encode_images
    LlavaMetaForCausalLM.encode_images_multi = LlavaMetaForCausalLM_prunevid.encode_images_multi
    
    LlavaMetaForCausalLM.compute_cluster_vectors = LlavaMetaForCausalLM_prunevid.compute_cluster_vectors
    LlavaMetaForCausalLM.spatial_merge_tokens = LlavaMetaForCausalLM_prunevid.spatial_merge_tokens
    LlavaMetaForCausalLM.index_points = LlavaMetaForCausalLM_prunevid.index_points
    LlavaMetaForCausalLM.segment_lengths = LlavaMetaForCausalLM_prunevid.segment_lengths
    LlavaMetaForCausalLM.refine_clusters = LlavaMetaForCausalLM_prunevid.refine_clusters
    LlavaMetaForCausalLM.cluster_dpc_knn = LlavaMetaForCausalLM_prunevid.cluster_dpc_knn
    LlavaMetaForCausalLM.merge_frames_dynamic = LlavaMetaForCausalLM_prunevid.merge_frames_dynamic
    # LlavaMetaForCausalLM.merge_tokens_by_attention_density = LlavaMetaForCausalLM_prunevid.merge_tokens_by_attention_density
    # LlavaMetaForCausalLM.merge_tokens_by_density = LlavaMetaForCausalLM_prunevid.merge_tokens_by_density
    # LlavaMetaForCausalLM.merge_tokens_by_clustering = LlavaMetaForCausalLM_prunevid.merge_tokens_by_clustering
    LlavaMetaForCausalLM.add_newline_token = LlavaMetaForCausalLM_prunevid.add_newline_token
    

    # from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    # Qwen2Model.forward = Qwen2Model_prunevid.forward

    return model
