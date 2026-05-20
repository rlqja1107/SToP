from abc import ABC, abstractmethod

import math
import re
import os
import time
import torch
import torch.nn as nn
from torch import einsum
import numpy as np
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
import torch.nn.functional as F


def minmax_norm(x: torch.Tensor, eps: float = 1e-12):
    x_min = x.amin(dim=1)[..., None]
    x_max = x.amax(dim=1)[..., None]
    return (x - x_min) / (x_max - x_min + eps)

class LlavaMetaForCausalLM_fastvid(ABC):

    def get_vision_abstract(self):
        vision_abstract = getattr(self, "vision_abstract", None)
        if type(vision_abstract) is list:
            vision_abstract = vision_abstract[0]
        return vision_abstract


    def encode_images(self, images):
        image_features, _ = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_images_multi(self, images):
        image_features, attn_weights, metric, images_dtype= self.get_model().get_vision_tower()(images)
        frame_global_features, _ = self.vision_abstract(image_features)
        image_features = self.get_model().mm_projector(image_features)

        return image_features, attn_weights, metric, images_dtype, frame_global_features
    
    def cluster_dpc_knn(self, x, cluster_num, k=7):
        with torch.no_grad():
            batch_size, seq_len, embed_dim = x.shape
            
            dist_matrix = torch.cdist(x.float(), x.float()) / (embed_dim ** 0.5)    # (batch_size, seq_len, seq_len)
            
            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k, dim=-1, largest=False) # (batch_size, seq_len, k)
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp() # (batch_size, seq_len)
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6
            
            # get distance indicator
            mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1).values[:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            
            # select the cluster center according to the score
            score = dist * density
            _, index_center = score.topk(cluster_num, dim=-1)
            
            return index_center, dist_matrix
    
    def select_static_windows(self, feature_sim, batch_size, tau, max_window_size):
        # pruned_static_count[s,e]
        # tau = 0.
        def get_pruned_static_count_vectorized(feature_sim, batch_size, tau):
            similarity_matrix = torch.ones((batch_size, batch_size, feature_sim.shape[1]), device=feature_sim.device)
            
            for start in range(batch_size-1):
                cum_similarity = torch.cumprod(feature_sim[start:] > tau, dim=0)
                similarity_matrix[start, start+1:start+1+len(cum_similarity)] = cum_similarity
            
            window_lengths = torch.arange(batch_size, device=feature_sim.device).unsqueeze(0) - \
                           torch.arange(batch_size, device=feature_sim.device).unsqueeze(1)
            window_lengths = window_lengths.clamp(min=0)
            
            pruned_static_count = (similarity_matrix.sum(dim=-1) * window_lengths).float()
            return pruned_static_count
        
        pruned_static_count = get_pruned_static_count_vectorized(feature_sim, batch_size, tau)
                
        dp = torch.zeros(batch_size, device=pruned_static_count.device)
        prev = torch.zeros(batch_size, dtype=torch.long, device=pruned_static_count.device)
        # [prev[i], i]
        
        for i in range(batch_size):
            max_val = dp[i-1] if i > 0 else 0
            best_j = i
            
            for window_size in range(2, min(i+1, max_window_size) + 1):
                j = i - window_size
                current_val = (dp[j] if j>=0 else 0) + pruned_static_count[j+1, i]  # [-, j] + [j+1, i]
                if current_val > max_val:
                    max_val = current_val
                    best_j = j+1
            
            dp[i] = max_val
            prev[i] = best_j    # [best_j, i]
        
        selected_frames = []
        i = batch_size - 1
        while i >= 0:
            selected_frames.append((prev[i].item(), i))
            i = prev[i].item() - 1
        
        selected_frames = selected_frames[::-1]
        total_reduced = dp[-1].item()
        
        return selected_frames, total_reduced

    def merge_tokens_by_clustering(self, feat, target_indices, dist_matrix, cluster_num, Beta):
        batch_size, seq_len, embed_dim = feat.shape
        all_indices = torch.arange(seq_len, device=feat.device)
        all_indices = all_indices.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        non_target_indices = torch.zeros((batch_size, seq_len-cluster_num), dtype=torch.long, device=feat.device)
        for b in range(batch_size):
            non_target_mask = ~torch.isin(all_indices[b], target_indices[b])
            non_target_indices[b] = all_indices[b][non_target_mask]
        # non_target_indices (batch_size, seq_len-cluster_num)
        
        non_target_feat = torch.gather(
            feat,
            dim=1,
            index=non_target_indices.unsqueeze(-1).expand(-1, -1, feat.size(-1))
        )   # (batch_size, seq_len-cluster_num, embed_dim)
        
        dist_matrix = torch.gather(
            dist_matrix, 
            dim=1, 
            index=non_target_indices.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1))
        )   # (batch_size, seq_len-cluster_num, seq_len)
        dist_matrix = torch.gather(
            dist_matrix, 
            dim=2, 
            index=target_indices.unsqueeze(1).expand(-1, dist_matrix.size(1), -1)
        )   # (batch_size, seq_len-cluster_num, cluster_num)
        
        idx_cluster = torch.argmin(dist_matrix, dim=-1) # (batch_size, seq_len-cluster_num)
        
        cluster_tokens = []
        for b in range(batch_size):
            batch_tokens = []
            for i in range(cluster_num):
                mask = (idx_cluster[b] == i)
                if mask.any():
                    cluster_features = non_target_feat[b][mask]
                    import os
                    if os.environ.get("NO_BETA", "0") == "0":
                        # rank0_print("USE_BETA")
                        cluster_means = cluster_features.mean(dim=0)
                        batch_tokens.append(Beta * feat[b][target_indices[b][i]] + (1 - Beta) * cluster_means)
                    else:
                        # rank0_print("NO_BETA")
                        all_features = torch.cat([feat[b][target_indices[b][i]].unsqueeze(0), cluster_features], dim=0)
                        batch_tokens.append(all_features.mean(dim=0))
                else:
                    batch_tokens.append(feat[b][target_indices[b][i]])
            cluster_tokens.append(torch.stack(batch_tokens))
        cluster_tokens = torch.stack(cluster_tokens)  # shape: (batch_size, cluster_num, embed_dim)
        
        return cluster_tokens

    def merge_tokens_by_attention_density(self, feat, attn, pos, retain_ratio, D, Beta, K):
        batch_size, seq_len, embed_dim = feat.shape
        dominant_num = round(math.ceil(seq_len * retain_ratio) * (1-D))
        contextual_num = math.ceil(seq_len * retain_ratio) - dominant_num
        
        ## Dominant Visual Tokens
        if dominant_num > 0:
            all_indices = attn.topk(dominant_num, dim=1).indices
            mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device).scatter_(1, all_indices, False)  # (batch_size, seq_len) False means retained tokens
            # finally, (batch_size, dominant_num, embed_dim) compare with feat
            dominant_tokens = feat.masked_select(~mask.unsqueeze(-1)).view(batch_size, dominant_num, embed_dim)
            dominant_pos = pos.masked_select(~mask).view(batch_size, dominant_num)
        else:
            mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device)
            dominant_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
            dominant_pos = torch.empty((batch_size, 0), device=feat.device)
        
        ## Contextual Visual Tokens
        if contextual_num > 0:
            ### Filter 
            # feat_filtered: (batch_size, seq_len-dominant_num, embed_dim)
            feat_filtered = feat.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num, embed_dim) 
            contextual_pos = pos.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num)
            target_indices, dist_matrix = self.cluster_dpc_knn(feat_filtered, contextual_num, k=min(K,contextual_num))
            target_indices = torch.sort(target_indices, dim=-1)[0]
            contextual_pos = torch.stack([contextual_pos[b][target_indices[b]] for b in range(batch_size)]) # (batch_size, contextual_num)
            # target_indices (batch_size, contextual_num)
            # dist_matrix (batch_size, seq_len-dominant_num, seq_len-dominant_num)
            # assign tokens to the nearest center
            
            contextual_tokens = self.merge_tokens_by_clustering(feat_filtered, target_indices, dist_matrix, contextual_num, Beta)
        else:
            contextual_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
            contextual_pos = torch.empty((batch_size, 0), device=feat.device)
        
        image_feat = []
        image_pos = []
        for b in range(batch_size):
            batch_tokens = torch.cat([dominant_tokens[b], contextual_tokens[b]], dim=0)
            batch_pos = torch.cat([dominant_pos[b], contextual_pos[b]], dim=0)
            image_feat.append(batch_tokens)
            image_pos.append(batch_pos)
        image_feat = torch.stack(image_feat)  # shape: (batch_size, dominant_num + contextual_num, embed_dim)
        image_pos = torch.stack(image_pos)
        
        return image_feat, image_pos
    
    def merge_tokens_by_density(self, feat, pos, retain_ratio, Beta, K):
        batch_size, seq_len, embed_dim = feat.shape
        cluster_num = round(seq_len * retain_ratio)
        if cluster_num > 0:
            target_indices, dist_matrix = self.cluster_dpc_knn(feat, cluster_num, k=min(K,cluster_num))
            target_indices = torch.sort(target_indices, dim=-1)[0]
            image_pos = torch.stack([pos[b][target_indices[b]] for b in range(batch_size)])
            
            cluster_tokens = self.merge_tokens_by_clustering(feat, target_indices, dist_matrix, cluster_num, Beta)
            image_feat = cluster_tokens
        else:
            image_feat = torch.empty((batch_size, 0, embed_dim), device=feat.device)
            image_pos = torch.empty((batch_size, 0), device=feat.device)
        
        return image_feat, image_pos


    def add_newline_token(self, feat, pos, grid_size, newline_token):
        row_pos = pos // grid_size
        expanded_feat_list = []
        for cur_feat, cur_row_pos in zip(feat, row_pos):
            expanded_feat = []
            for row in range(grid_size):
                find_row_feat = cur_feat[cur_row_pos == row]
                if len(find_row_feat) > 0:
                    expanded_feat.append(torch.cat((find_row_feat, newline_token), dim=0))
                else:
                    expanded_feat.append(find_row_feat)
            batch_feat = torch.cat(expanded_feat, dim=0)
            expanded_feat_list.append(batch_feat)
            
        image_feat = torch.cat(expanded_feat_list, dim=0)
        return image_feat


    def segment_lengths(self, tensor):
        # 获取设备信息（CPU 或 GPU）
        device = tensor.device
        B, N = tensor.shape

        # 列表用于存储每个视频的段长度
        segment_lengths_list = []
        max_segments = 0  # 记录最大段数

        for i in range(B):
            seq = tensor[i]
            # 计算值发生变化的位置
            change_points = torch.where(seq[1:] != seq[:-1])[0] + 1
            # 包含起始和结束位置
            boundaries = torch.cat([torch.tensor([0], device=device), change_points, torch.tensor([N], device=device)])
            # 计算每个段的长度
            lengths = boundaries[1:] - boundaries[:-1]
            segment_lengths_list.append(lengths)
            max_segments = max(max_segments, lengths.numel())

        # 初始化结果张量，填充为0
        result = torch.zeros((B, max_segments), dtype=torch.long, device=device)
        # 将每个视频的段长度填入结果张量
        for i in range(B):
            lengths = segment_lengths_list[i]
            result[i, :lengths.numel()] = lengths

        return result


    def refine_clusters(self, cluster_idx):
        import torch
        B, N = cluster_idx.shape
        refined_cluster_idx = cluster_idx.clone()
        for b in range(B):
            clusters = torch.unique(cluster_idx[b])
            segment_info = {}
            # 步骤1：对于每个 cluster，找到其所有的连续片段
            for cluster_label in clusters:
                indices = (cluster_idx[b] == cluster_label).nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue
                # 找到连续片段
                segments = []
                start = indices[0].item()
                prev = indices[0].item()
                for idx in indices[1:]:
                    idx = idx.item()
                    if idx == prev + 1:
                        prev = idx
                    else:
                        # 新的片段
                        segments.append((start, prev))
                        start = idx
                        prev = idx
                # 添加最后一个片段
                segments.append((start, prev))
                segment_info[cluster_label.item()] = segments

            # 步骤2：保留每个 cluster 中最长的片段，其余片段需要重新归类
            for cluster_label, segments in segment_info.items():
                # 找到最长的片段长度
                max_length = 0
                for (start, end) in segments:
                    length = end - start + 1
                    if length > max_length:
                        max_length = length
                # 如果最长的片段长度为1，且只有长度为1的片段，该 cluster 需要移除
                if max_length == 1:
                    for (start, end) in segments:
                        refined_cluster_idx[b, start:end+1] = -1  # -1表示需要重新归类
                    continue
                # 保留最长的片段，重新归类其他片段
                for (start, end) in segments:
                    length = end - start + 1
                    if length == max_length:
                        continue  # 保留最长的片段
                    else:
                        refined_cluster_idx[b, start:end+1] = -1  # 需要重新归类

            # 步骤3：对于需要重新归类的片段，按照左右邻居最长的片段的 cluster 进行归类
            idx = 0
            while idx < N:
                if refined_cluster_idx[b, idx] == -1:
                    # 找到需要重新归类的片段
                    start = idx
                    while idx < N and refined_cluster_idx[b, idx] == -1:
                        idx += 1
                    end = idx - 1
                    # 找到左侧和右侧的邻居 cluster 及其片段长度
                    left_cluster_label = None
                    left_length = 0
                    if start > 0:
                        left_label = refined_cluster_idx[b, start - 1].item()
                        # 左侧片段长度
                        l_idx = start - 1
                        while l_idx >= 0 and refined_cluster_idx[b, l_idx] == left_label:
                            l_idx -= 1
                        left_length = start - l_idx - 1
                        left_cluster_label = left_label
                    right_cluster_label = None
                    right_length = 0
                    if end < N - 1:
                        right_label = refined_cluster_idx[b, end + 1].item()
                        # 右侧片段长度
                        r_idx = end + 1
                        while r_idx < N and refined_cluster_idx[b, r_idx] == right_label:
                            r_idx += 1
                        right_length = r_idx - end - 1
                        right_cluster_label = right_label
                    # 选择片段长度较长的邻居 cluster 进行归类，若长度相同，选择左侧
                    if left_length > right_length:
                        new_label = left_cluster_label
                    elif right_length > left_length:
                        new_label = right_cluster_label
                    else:
                        new_label = left_cluster_label if left_cluster_label is not None else right_cluster_label
                    # 如果左右邻居都不存在，默认归类为 cluster 0
                    if new_label is None:
                        new_label = 0
                    # 重新归类
                    refined_cluster_idx[b, start:end+1] = new_label
                else:
                    idx += 1
        return refined_cluster_idx


    def index_points(self, points, idx):
        """Sample features following the index.
        Returns:
            new_points:, indexed points data, [B, S, C]

        Args:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


    def cluster_dpc_knn(self, x, cluster_num, k=5, token_mask=None):
        with torch.no_grad():
            B, N, C = x.shape

            dist_matrix = torch.cdist(x.float(), x.float()) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density

            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # # assign tokens to the nearest center
            dist_matrix = self.index_points(dist_matrix, index_down)

            idx_cluster = dist_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        return idx_cluster, cluster_num


    
    def compute_cluster_vectors(self, image_key_vectors, cluster_key_idx, num_cluster):
        """
        Args:
            image_key_vectors: Tensor of shape (B, L, D), the feature vectors
            cluster_key_idx: Tensor of shape (B, L), cluster indices for each vector
            num_cluster: int, the total number of clusters

        Returns:
            cluster_vectors: Tensor of shape (B, num_cluster, D), the averaged features for each cluster
        """
        # image_key_vectors: (B, L, D)
        # cluster_key_idx: (B, L)
        # num_cluster: integer, number of clusters

        B, L, D = image_key_vectors.shape

        # Step 1: 将cluster_key_idx进行one-hot编码
        # 得到的cluster_key_idx_onehot形状为 (B, L, num_cluster)
        cluster_key_idx_onehot = F.one_hot(cluster_key_idx, num_classes=num_cluster).to(dtype=image_key_vectors.dtype)

        # Step 2: 计算每个cluster的特征和
        # 首先调整cluster_key_idx_onehot的维度，使其变为 (B, num_cluster, L)
        cluster_key_idx_onehot_t = cluster_key_idx_onehot.permute(0, 2, 1)

        # 然后通过矩阵乘法计算每个cluster的特征和，得到的cluster_sums形状为 (B, num_cluster, D)
        cluster_sums = torch.bmm(cluster_key_idx_onehot_t, image_key_vectors)

        # Step 3: 计算每个cluster的元素数量
        # cluster_counts形状为 (B, num_cluster)
        cluster_counts = cluster_key_idx_onehot.sum(dim=1)

        # Step 4: 计算每个cluster的平均特征
        # 先避免除以0，将cluster_counts中为0的值替换为1
        cluster_counts_nonzero = cluster_counts.clone()
        cluster_counts_nonzero[cluster_counts_nonzero == 0] = 1

        # 计算平均值，结果cluster_features形状为 (B, num_cluster, D)
        cluster_features = cluster_sums / cluster_counts_nonzero.unsqueeze(-1)

        # Step 5: 对于没有元素的cluster，将其特征设置为0
        zero_mask = (cluster_counts == 0).unsqueeze(-1)  # (B, num_cluster, 1)
        cluster_features = cluster_features.masked_fill(zero_mask, 0)

        return cluster_features  # (B, num_cluster, D)
    
    def spatial_merge_tokens(self, feature, num_cluster, k):
        cluster_idx, _ = self.cluster_dpc_knn(feature, cluster_num=num_cluster, k=k)
        feature = self.compute_cluster_vectors(feature, cluster_idx, num_cluster=num_cluster)
        return feature


    def merge_frames_dynamic(self, frame_global_features, frame_attn_weights, frames, threshold=0.8, k=7, args=None, min_frame=None):
        self.frame_num = frame_global_features.shape[0]
        self.video_token_len = frames.shape[1]
        device_type = self.device
        hidden_states_dim = frames.shape[-1]
        #device_type = hidden_states.device
        #hidden_states_dim = hidden_states.shape[-1]
        frame_token_len = self.video_token_len // self.frame_num
        batchframe_indices = torch.arange(self.frame_num, device=device_type).unsqueeze(1)
        alltoken_indices = torch.arange(self.video_token_len, device=device_type).view(self.frame_num, frame_token_len)
        
        video_hidden_states = frames.reshape(self.frame_num, frame_token_len, -1)

        ############ DySeg ############
        frame_global_features = frame_global_features / frame_global_features.norm(dim=1, keepdim=True) 
        similarity_matrix = (frame_global_features[:-1] * frame_global_features[1:]).sum(dim=1)

        cut_indices_topk = torch.topk(similarity_matrix, self.fastvid_DySeg_c - 1, largest=False).indices
        cut_indices_cos = torch.nonzero(similarity_matrix < self.fastvid_DySeg_tau, as_tuple=False).squeeze(1)
        cut_indices = torch.unique(torch.cat([cut_indices_topk, cut_indices_cos])).sort().values
        padded = F.pad(cut_indices, (1, 1), value=-1)
        padded[-1] = self.frame_num - 1
        segment_sizes = padded.diff().tolist()
        
        ############ STPrune ############
        keep_indexs = ()

        final_tokens = []
        
        frame_retain_num = int(frame_token_len * self.fastvid_retention_ratio)

        frame_salient_num = frame_retain_num - int(frame_retain_num * self.fastvid_STPrune_d)
        
        frm_context_num_list = torch.zeros(self.frame_num, dtype=torch.int, device=device_type)
        frame_context_num = frame_retain_num - frame_salient_num

        ############ Compute Anchor Token Distribution ############
        offset = 0
        for seg_i_len in segment_sizes:
            seg_context_num = frame_context_num * seg_i_len
            temp_num = (seg_i_len + self.fastvid_DTM_p - 1) //  self.fastvid_DTM_p
            cur_frm_context_num = seg_context_num // temp_num

            end = offset + seg_i_len
            seg_indices = torch.arange(seg_i_len - 1, -1, -1, device=device_type) 
            mask = (seg_indices % self.fastvid_DTM_p == 0)
        
            frm_context_num_list[offset:end][mask] = cur_frm_context_num
            offset = end

        ############ ATS ############
        pooled_image_feat = frames[0].reshape(frame_attn_weights.shape[0], frame_attn_weights.shape[1], -1)
        sink_score = minmax_norm(frame_attn_weights.mean(0).repeat(pooled_image_feat.shape[0], 1)**(args.gamma))
        attn_ = frame_attn_weights - args.mu_s * sink_score
        salient_indexes = torch.topk(attn_, frame_salient_num, dim=1).indices                                            

        batch_indices = batchframe_indices.expand(-1, frame_salient_num)
        salient_tokens = video_hidden_states[batch_indices, salient_indexes]
        salient_global_indexes = alltoken_indices[batch_indices, salient_indexes]
        
        sampled_token_idx_list = [salient_indexes.cpu().numpy()] # Save Token Index
        final_tokens.append(salient_tokens.view(-1, hidden_states_dim)) # 32개 frame에 대해서 salient 
        keep_indexs += (salient_global_indexes.view(-1),)

        ############ Parallel Density Score Computation ############
        all_indices = torch.arange(frame_token_len, device=device_type).unsqueeze(0).expand(self.frame_num, -1)
        all_indices_mask = torch.ones_like(all_indices, dtype=torch.bool)
        all_indices_mask.scatter_(1, salient_indexes, False)
        filtered_indices = all_indices[all_indices_mask].view(self.frame_num, frame_token_len - frame_salient_num)
        
        batch_indices = batchframe_indices.expand(-1, frame_token_len - frame_salient_num)
        token_filtered = video_hidden_states[batch_indices, filtered_indices]
        alltoken_filtered_indices = alltoken_indices[batch_indices, filtered_indices]
        
        tmp_frm_hidden_states = token_filtered # 178 => rest one
        dist_matrix = torch.cdist(tmp_frm_hidden_states.float(), tmp_frm_hidden_states.float()) / (hidden_states_dim ** 0.5) # 178 x 178

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=4, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(
            density.shape, device=device_type, dtype=density.dtype) * 1e-6

        density_mask = density[:, None, :] > density[:, :, None]
        density_mask = density_mask.type(tmp_frm_hidden_states.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist_0, index_parent = (dist_matrix * density_mask + dist_max * (1 - density_mask)).min(dim=-1)

        density_score = dist_0 * density
        context_hyper = 0.3 if args.mu_s != 0.0 else 0.0
        density_score = density_score - context_hyper * torch.gather(sink_score, dim=1, index = filtered_indices)
        sampled_indexs = torch.topk(density_score, k=frame_context_num, dim=-1).indices
        absolute_indexs = []
        for i in range(len(sampled_indexs)):
            absolute_indexs.append(filtered_indices[i][sampled_indexs[i]])
        absolute_indexs = torch.stack(absolute_indexs, dim=0)
            
        batch_indices = batchframe_indices.expand(-1, frame_context_num)
        frm_context_tokens = token_filtered[batch_indices, sampled_indexs]
        frm_context_global_indexes = alltoken_filtered_indices[batch_indices, sampled_indexs]
        
        to_be_merge_tokens = token_filtered / token_filtered.norm(dim=-1, keepdim=True)
        merge_target_tokens = to_be_merge_tokens[batch_indices, sampled_indexs]

        similarity = torch.bmm(to_be_merge_tokens, merge_target_tokens.transpose(1,2))
        assign_one_hot = torch.zeros(self.frame_num, frame_token_len - frame_salient_num, frame_context_num, dtype=token_filtered.dtype, device=device_type)
        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)

        avg_weights = (1 / (assign_one_hot.sum(dim=1).unsqueeze(-1) + 1)).clamp(min=self.fastvid_DTM_alpha)

        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), token_filtered) / counts
    
        frm_context_tokens = avg_weights * frm_context_tokens + (1 - avg_weights) * aggregated_hidden

        context_for_frame_mask = (frm_context_num_list == frame_context_num)
        
        context_for_frame_tokens = frm_context_tokens[context_for_frame_mask]
        context_for_frame_global_indexes = frm_context_global_indexes[context_for_frame_mask]
        
        final_tokens.append(context_for_frame_tokens.view(-1, hidden_states_dim))
        keep_indexs += (context_for_frame_global_indexes.view(-1),)
        
        ############ DTM for Multi-Frame Segment ############
        idx_seg_start = 0
        
        for seg_i_len in segment_sizes:
            if seg_i_len > 1: 
                cur_seg_context_num_list = frm_context_num_list[idx_seg_start:idx_seg_start+seg_i_len]
                cur_seg_context_num = cur_seg_context_num_list[-1]
                
                cur_seg_target_mask = (cur_seg_context_num_list > frame_context_num)
                cur_seg_target_num = cur_seg_target_mask.sum()

                cur_seg_density_score = density_score[idx_seg_start:idx_seg_start+seg_i_len]
                cur_seg_density_score = cur_seg_density_score[cur_seg_target_mask]
                
                cur_seg_token_filtered = token_filtered[idx_seg_start:idx_seg_start+seg_i_len]
                cur_seg_token_target = cur_seg_token_filtered[cur_seg_target_mask]
                cur_seg_token_filtered = cur_seg_token_filtered.view(1, -1, hidden_states_dim).expand(cur_seg_target_num,-1,-1)
                
                cur_seg_alltoken_indices = alltoken_filtered_indices[idx_seg_start:idx_seg_start+seg_i_len]
                cur_seg_alltoken_indices = cur_seg_alltoken_indices[cur_seg_target_mask]
                
                cur_seg_density_score_copy = cur_seg_density_score - context_hyper * torch.gather(sink_score[idx_seg_start:idx_seg_start+seg_i_len][cur_seg_target_mask], dim=1, index = filtered_indices[idx_seg_start:idx_seg_start+seg_i_len][cur_seg_target_mask])
                sampled_indexs = torch.topk(cur_seg_density_score_copy, k=cur_seg_context_num, dim=-1).indices
                    
                batch_indices = batchframe_indices[:cur_seg_target_num].expand(-1, cur_seg_context_num)
                cur_context_tokens = cur_seg_token_target[batch_indices, sampled_indexs]
                cur_context_global_indexes = cur_seg_alltoken_indices[batch_indices, sampled_indexs]

                to_be_merge_tokens = cur_seg_token_filtered / cur_seg_token_filtered.norm(dim=-1, keepdim=True)
                merge_target_tokens = cur_context_tokens / cur_context_tokens.norm(dim=-1, keepdim=True)
        
                similarity = torch.bmm(to_be_merge_tokens, merge_target_tokens.transpose(1,2))
                assign_one_hot = torch.zeros(cur_seg_target_num, to_be_merge_tokens.shape[1], cur_seg_context_num, dtype=token_filtered.dtype, device=device_type)
                assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
    
                avg_weights = (1 / (assign_one_hot.sum(dim=1).unsqueeze(-1) + 1)).clamp(min=self.fastvid_DTM_alpha)
    
                counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), cur_seg_token_filtered) / counts
                
                cur_context_tokens = avg_weights * cur_context_tokens + (1 - avg_weights) * aggregated_hidden

                final_tokens.append(cur_context_tokens.view(-1, hidden_states_dim))
                keep_indexs += (cur_context_global_indexes.view(-1),)
            
            idx_seg_start += seg_i_len

        hidden_states = torch.cat(final_tokens, dim=0)
        keep_indexs = torch.cat(keep_indexs, dim=0)

        sorted_indexs = torch.argsort(keep_indexs)
        hidden_states = hidden_states[sorted_indexs].unsqueeze(0)
        keep_indexs = keep_indexs[sorted_indexs]
        return hidden_states
        

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, args=None):
        import os
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
            
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features, attn_weights, _, images_dtype, frame_global_features = self.encode_images_multi(concat_images)            
            max_window_size = int(os.environ.get("MAX_WINDOW_SIZE", 1024))
            NO_BETA = os.environ.get("NO_BETA", "1")
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    # [modify]

                    pooled_image_feat = self.get_2dPool(image_feat) # (batch_size, seq_len', embed_dim)
                    attn_weights = attn_weights.unsqueeze(-1)
                    attn_weights = self.get_2dPool(attn_weights)
                    attn_weights = attn_weights.squeeze(-1) # (batch_size, seq_len')
                    batch_size, seq_len, embed_dim = pooled_image_feat.shape
                    image_feat = self.merge_frames_dynamic(frame_global_features, attn_weights, pooled_image_feat.reshape(1, -1, pooled_image_feat.shape[2]), threshold=0.8, k=7, args=args, min_frame = pooled_image_feat.shape[0])

                    segment_features = []
                    segment_features.append(image_feat[0])
                        
                    image_features.append(torch.cat(segment_features, dim=0))

                else:
                    image_features.append(image_feat)

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):

                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # # Grid-wise

                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            # image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        if os.getenv("HOLITOM_k") is not None and os.getenv("HOLITOM_r") is not None:
            # [modified]
            image_token_posi = []
            prompt_len = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if os.getenv("HOLITOM_k") is not None and os.getenv("HOLITOM_r") is not None:
                # [modified]
                # record image position for further dropping
                image_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                if image_index == []:
                    image_token_posi.append(-1)
                else:
                    image_token_posi.append(image_index[0])

                # record input instruction length in inference mode
                if not self.training:  
                    if image_index == []:
                        prompt_len.append(cur_input_ids.shape[0])
                    else:
                        prompt_len.append(cur_input_ids.shape[0] - 1)   # consider image place holder
            
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            # [modify]
            text_token_count = sum([x.shape[0] for x in cur_labels_noim])
            vision_token_count = len(image_features[cur_image_idx])
            # rank0_print(f"Batch {batch_idx}: Text tokens: {text_token_count} Original Vision tokens: {vision_token_count}")
    
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)


        if os.getenv("HOLITOM_k") is not None and os.getenv("HOLITOM_r") is not None:
            # [modified]
            self.model.image_token_posi = image_token_posi
            self.model.prompt_len = prompt_len
            self.model.image_tokens = [image_feature.shape[0] for image_feature in image_features]
            self.model.all_attention_sink_pos = all_attention_sink_pos
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

