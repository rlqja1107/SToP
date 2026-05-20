import os
from .llava_arch import LlavaMetaForCausalLM_fastvid
from .modeling_qwen2 import Qwen2Model_fastvid

def fastvid(model):
    
    print("################################")
    print("############ Fastvid ###########")
    print("################################")

    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_fastvid.prepare_inputs_labels_for_multimodal
    LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_fastvid.encode_images
    LlavaMetaForCausalLM.encode_images_multi = LlavaMetaForCausalLM_fastvid.encode_images_multi
    
    LlavaMetaForCausalLM.compute_cluster_vectors = LlavaMetaForCausalLM_fastvid.compute_cluster_vectors
    LlavaMetaForCausalLM.spatial_merge_tokens = LlavaMetaForCausalLM_fastvid.spatial_merge_tokens
    LlavaMetaForCausalLM.index_points = LlavaMetaForCausalLM_fastvid.index_points
    LlavaMetaForCausalLM.segment_lengths = LlavaMetaForCausalLM_fastvid.segment_lengths
    LlavaMetaForCausalLM.refine_clusters = LlavaMetaForCausalLM_fastvid.refine_clusters
    LlavaMetaForCausalLM.cluster_dpc_knn = LlavaMetaForCausalLM_fastvid.cluster_dpc_knn
    LlavaMetaForCausalLM.merge_frames_dynamic = LlavaMetaForCausalLM_fastvid.merge_frames_dynamic
    # LlavaMetaForCausalLM.merge_tokens_by_attention_density = LlavaMetaForCausalLM_prunevid.merge_tokens_by_attention_density
    # LlavaMetaForCausalLM.merge_tokens_by_density = LlavaMetaForCausalLM_prunevid.merge_tokens_by_density
    # LlavaMetaForCausalLM.merge_tokens_by_clustering = LlavaMetaForCausalLM_prunevid.merge_tokens_by_clustering
    LlavaMetaForCausalLM.add_newline_token = LlavaMetaForCausalLM_fastvid.add_newline_token
    
    LlavaMetaForCausalLM.fastvid_retention_ratio = float(os.environ['fastvid_retention_ratio'])
    LlavaMetaForCausalLM.fastvid_DySeg_c = int(os.environ['fastvid_DySeg_c'])
    LlavaMetaForCausalLM.fastvid_DySeg_tau = float(os.environ['fastvid_DySeg_tau'])
    LlavaMetaForCausalLM.fastvid_STPrune_d = float(os.environ['fastvid_STPrune_d'])
    LlavaMetaForCausalLM.fastvid_DTM_p = int(os.environ['fastvid_DTM_p'])
    LlavaMetaForCausalLM.fastvid_DTM_alpha = float(os.environ['fastvid_DTM_alpha'])
    
    return model
