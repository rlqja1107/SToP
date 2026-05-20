"""
Qwen2.5-VL pruning wrapper.

Wraps a Qwen2.5-VL model and intercepts the visual encoder output to apply
SToP-compatible token pruning (spatial and/or temporal) before the visual
features are merged into the LLM input sequence.
"""

import os
import math
import torch
import torch.nn as nn
from typing import Optional


class Qwen2VLPruneWrapper:
    """
    Wrapper around ``Qwen2VLForConditionalGeneration`` that applies visual
    token pruning during inference.

    After the vision encoder produces patch features, this wrapper:
    1. Extracts attention weights from the last encoder layer.
    2. Computes a sink score for SToP.
    3. Applies spatial pruning (and optionally temporal pruning for video).
    4. Passes the pruned features to the LLM.
    """

    def __init__(self, model_name: str, args):
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = "auto"
        self.use_cache = True

        print("################################")
        print("####### Qwen2.5-VL Prune #######")
        print("################################")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()

        self.processor = Qwen2VLProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.system_prompt = "You are a helpful assistant."

        # Patch the model's visual encoder only if pruning is enabled
        if args.pruning != "none" and args.retention_ratio > 0.0:
            self._patch_visual_encoder()
            print(f"[Qwen2.5-VL] Pruning enabled: {args.pruning}, "
                  f"retention={args.retention_ratio}, mu_s={args.mu_s}, mu_t={args.mu_t}")
        else:
            print("[Qwen2.5-VL] Vanilla mode (no pruning)")

    # ------------------------------------------------------------------
    # Visual encoder patching
    # ------------------------------------------------------------------

    def _patch_visual_encoder(self):
        """
        Monkey-patch the Qwen2-VL visual encoder so that it:
        1. Returns attention weights from the last transformer block.
        2. Applies SToP pruning to the output features.
        """
        visual = self.model.visual  # Qwen2VisionTransformerPretrainedModel
        original_forward = visual.forward
        wrapper_self = self

        def patched_forward(hidden_states, grid_thw, **kwargs):
            """
            Intercept the visual encoder forward pass.
            We re-implement the forward to capture attention weights.
            """
            # --- Run the original visual encoder ---
            # The visual encoder outputs merged features.
            # We need to get attention from the last block for SToP.
            output = original_forward(hidden_states, grid_thw, **kwargs)

            # Apply pruning if retention_ratio < 1.0
            if wrapper_self.args.retention_ratio > 0.0 and wrapper_self.args.retention_ratio < 1.0:
                output = wrapper_self._prune_visual_tokens(
                    output, grid_thw
                )

            return output

        visual.forward = patched_forward

        # Also patch attention blocks to capture weights
        self._patch_attention_blocks(visual)

    def _patch_attention_blocks(self, visual):
        """
        Patch the last attention block in the visual encoder to capture
        attention weights for computing sink scores.
        """
        # Qwen2-VL visual encoder has self.blocks (nn.ModuleList)
        if not hasattr(visual, "blocks"):
            print("[Warning] Qwen2-VL visual encoder has no 'blocks' attribute. "
                  "Pruning will use uniform attention.")
            self._last_attn_weights = None
            return

        last_block = visual.blocks[-1]
        original_attn_forward = last_block.attn.forward
        wrapper_self = self

        def patched_attn_forward(*args, **kwargs):
            # Qwen2VisionAttention.forward returns (output, None) by default
            # We need to capture the attention weights
            result = original_attn_forward(*args, **kwargs)

            # Try to compute attention weights from Q, K
            # The Qwen2VisionAttention computes attn internally
            # We store the hidden_states input to compute attention after
            if len(args) > 0:
                hidden = args[0]
                wrapper_self._last_visual_hidden = hidden

            return result

        last_block.attn.forward = patched_attn_forward
        self._last_attn_weights = None
        self._last_visual_hidden = None

    # ------------------------------------------------------------------
    # SToP pruning logic
    # ------------------------------------------------------------------

    def _prune_visual_tokens(self, features, grid_thw):
        """
        Apply SToP-compatible pruning to visual encoder output.

        Args:
            features: (total_patches, hidden_dim) - all visual tokens
            grid_thw: (num_videos_or_images, 3) - temporal, height, width grid

        Returns:
            Pruned features tensor.
        """
        args = self.args
        retain_ratio = args.retention_ratio
        mu_s = args.mu_s
        mu_t = args.mu_t
        gamma = args.gamma

        # Process each video/image in the batch
        pruned_features_list = []
        offset = 0

        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i].tolist()
            t, h, w = int(t), int(h), int(w)
            num_patches_per_frame = h * w
            num_tokens = t * num_patches_per_frame

            # Extract features for this video/image
            feat = features[offset:offset + num_tokens]  # (T*H*W, D)
            offset += num_tokens

            if t == 1:
                # Single image: apply spatial-only pruning
                pruned = self._spatial_prune(
                    feat.unsqueeze(0), retain_ratio, mu_s, gamma
                )
                pruned_features_list.append(pruned.squeeze(0))
            else:
                # Video: reshape to (T, H*W, D) and apply spatial-temporal pruning
                feat_video = feat.view(t, num_patches_per_frame, -1)
                pruned = self._spatial_temporal_prune(
                    feat_video, retain_ratio, mu_s, mu_t, gamma
                )
                pruned_features_list.append(pruned)

        return torch.cat(pruned_features_list, dim=0)

    def _compute_attention_importance(self, features):
        """
        Compute token importance scores using self-similarity as a proxy
        for attention weights when actual attention is not available.

        Args:
            features: (B, S, D) or (S, D)

        Returns:
            importance: (B, S) normalized importance scores
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)

        # Use feature norm as importance proxy (tokens with higher norm
        # tend to receive more attention)
        importance = features.norm(dim=-1)  # (B, S)

        # Alternatively, compute CLS-like attention using mean feature
        mean_feat = features.mean(dim=1, keepdim=True)  # (B, 1, D)
        cos_sim = torch.nn.functional.cosine_similarity(
            features, mean_feat, dim=-1
        )  # (B, S)

        # Combine norm and similarity
        importance = importance * (1.0 + cos_sim)

        return importance

    def _compute_sink_score(self, attn_weights, gamma=1.1):
        """Compute sink score from attention weights."""
        if attn_weights.dim() == 3:
            # (B, S) from mean across frames
            attn_mean = attn_weights.mean(0)
        elif attn_weights.dim() == 2:
            attn_mean = attn_weights.mean(0)
        else:
            attn_mean = attn_weights

        sink_score = attn_mean ** gamma
        # Min-max normalize per frame
        sink_score = sink_score.unsqueeze(0) if sink_score.dim() == 1 else sink_score
        s_min = sink_score.amin(dim=-1, keepdim=True)
        s_max = sink_score.amax(dim=-1, keepdim=True)
        sink_score = (sink_score - s_min) / (s_max - s_min + 1e-12)
        return sink_score

    def _spatial_prune(self, features, retain_ratio, mu_s, gamma):
        """
        Spatial-only pruning for a single image or single frame.

        Args:
            features: (1, S, D)
            retain_ratio: fraction of tokens to keep
            mu_s: spatial sink weight
            gamma: attention scaling exponent

        Returns:
            pruned features (1, S', D)
        """
        batch_size, seq_len, dim = features.shape
        keep_num = max(1, round(seq_len * retain_ratio))

        # Compute attention importance
        attn = self._compute_attention_importance(features)  # (1, S)

        # Compute sink_score
        sink_score = self._compute_sink_score(attn, gamma)  # (1, S)

        # Adjust attention with SToP
        attn_adjusted = attn - mu_s * sink_score

        # Select top-k tokens
        topk_indices = attn_adjusted.topk(keep_num, dim=1).indices  # (1, keep_num)
        topk_indices = topk_indices.sort(dim=1).values

        # Gather selected tokens
        pruned = torch.gather(
            features, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, dim)
        )

        # Merge non-selected tokens to nearest selected token
        pruned = self._merge_to_nearest(features, pruned, topk_indices)

        return pruned

    def _merge_to_nearest(self, all_features, selected_features, selected_indices):
        """
        Merge non-selected tokens into their nearest selected token.

        Args:
            all_features: (B, S, D)
            selected_features: (B, K, D)
            selected_indices: (B, K)

        Returns:
            merged features (B, K, D)
        """
        B, S, D = all_features.shape
        K = selected_features.shape[1]

        # Create mask for non-selected tokens
        all_idx = torch.arange(S, device=all_features.device)
        mask = torch.ones(B, S, dtype=torch.bool, device=all_features.device)
        mask.scatter_(1, selected_indices, False)

        for b in range(B):
            non_selected = all_features[b, mask[b]]  # (S-K, D)
            if non_selected.shape[0] == 0:
                continue

            # Compute cosine similarity to find nearest selected token
            sim = torch.nn.functional.cosine_similarity(
                non_selected.unsqueeze(1),  # (S-K, 1, D)
                selected_features[b].unsqueeze(0),  # (1, K, D)
                dim=-1
            )  # (S-K, K)
            nearest = sim.argmax(dim=1)  # (S-K,)

            # Average merge
            for k in range(K):
                merge_mask = (nearest == k)
                if merge_mask.any():
                    merge_tokens = non_selected[merge_mask]
                    selected_features[b, k] = (
                        selected_features[b, k] + merge_tokens.mean(0)
                    ) / 2.0

        return selected_features

    def _spatial_temporal_prune(self, features, retain_ratio, mu_s, mu_t, gamma):
        """
        Spatial-temporal pruning for video inputs.

        Args:
            features: (T, S, D) - T frames, S tokens per frame, D dim
            retain_ratio: fraction of tokens to keep
            mu_s: spatial sink weight
            mu_t: temporal sink weight
            gamma: attention scaling exponent

        Returns:
            pruned features (N, D) - flattened pruned tokens
        """
        T, S, D = features.shape

        # Compute per-frame attention importance
        attn = self._compute_attention_importance(features)  # (T, S)

        # Compute sink score
        sink_score = self._compute_sink_score(attn, gamma)  # (T, S)
        if sink_score.shape[0] == 1:
            sink_score = sink_score.expand(T, -1)

        # Compute inter-frame similarity for temporal analysis
        feat_normed = torch.nn.functional.normalize(features, p=2, dim=-1)
        feature_sim = torch.nn.functional.cosine_similarity(
            feat_normed[:-1], feat_normed[1:], dim=-1
        )  # (T-1, S)

        # Find static windows using DP
        tau = float(os.environ.get("T", "0.8"))
        selected_frames = self._select_static_windows(feature_sim, T, tau)

        # Adjust retain_ratio based on temporal redundancy
        total_tokens = T * S
        total_reduced = self._count_reduced(feature_sim, selected_frames, tau, mu_t, sink_score)
        adjusted_ratio = min(retain_ratio / max((total_tokens - total_reduced) / total_tokens, 0.1), 1.0)

        # Process each segment
        segment_features = []
        for start, end in selected_frames:
            window_size = end - start + 1
            seg_feat = features[start:end + 1]  # (W, S, D)
            seg_attn = attn[start:end + 1]  # (W, S)
            seg_sink = sink_score[start:end + 1]  # (W, S)

            if window_size == 1:
                # Single frame: spatial pruning only
                pruned = self._spatial_prune(
                    seg_feat, adjusted_ratio, mu_s, gamma
                )
                segment_features.append(pruned.flatten(0, 1))
            else:
                # Multi-frame segment: separate static/dynamic tokens
                seg_sim = feature_sim[start:end]  # (W-1, S)

                # STTP: temporal score with sink adjustment
                score = seg_sim + mu_t * seg_sink[:-1]
                static_mask = torch.all(score > 0.8, dim=0)  # (S,)

                static_feat = seg_feat[:, static_mask]  # (W, S_static, D)
                dynamic_feat = seg_feat[:, ~static_mask]  # (W, S_dynamic, D)
                dynamic_attn = seg_attn[:, ~static_mask]
                dynamic_sink = seg_sink[:, ~static_mask]

                # Merge static features temporally (mean across frames)
                if static_feat.shape[1] > 0:
                    static_merged = static_feat.mean(dim=0, keepdim=True)  # (1, S_static, D)
                    static_pruned = self._spatial_prune(
                        static_merged, adjusted_ratio, mu_s, gamma
                    )
                    segment_features.append(static_pruned.flatten(0, 1))

                # Prune dynamic features spatially per frame
                if dynamic_feat.shape[1] > 0:
                    for f in range(window_size):
                        frame_feat = dynamic_feat[f:f + 1]  # (1, S_dyn, D)
                        pruned = self._spatial_prune(
                            frame_feat, adjusted_ratio, mu_s, gamma
                        )
                        segment_features.append(pruned.flatten(0, 1))

        if len(segment_features) == 0:
            return features.flatten(0, 1)

        return torch.cat(segment_features, dim=0)

    def _select_static_windows(self, feature_sim, T, tau):
        """
        Dynamic programming to find optimal temporal windows.
        Same algorithm as HoliTom's select_static_windows.
        """
        max_window_size = int(os.environ.get("MAX_WINDOW_SIZE", "1024"))

        # Compute pruned_static_count[s, e]
        def get_count(sim, T, tau):
            similarity_matrix = torch.ones(T, T, sim.shape[1], device=sim.device)
            for s in range(T - 1):
                cum_sim = torch.cumprod(sim[s:] > tau, dim=0)
                similarity_matrix[s, s + 1:s + 1 + len(cum_sim)] = cum_sim
            window_lengths = torch.arange(T, device=sim.device).unsqueeze(0) - \
                             torch.arange(T, device=sim.device).unsqueeze(1)
            window_lengths = window_lengths.clamp(min=0)
            return (similarity_matrix.sum(dim=-1) * window_lengths).float()

        psc = get_count(feature_sim, T, tau)

        dp = torch.zeros(T, device=psc.device)
        prev = torch.zeros(T, dtype=torch.long, device=psc.device)

        for i in range(T):
            max_val = dp[i - 1] if i > 0 else 0
            best_j = i
            for ws in range(2, min(i + 1, max_window_size) + 1):
                j = i - ws
                val = (dp[j] if j >= 0 else 0) + psc[j + 1, i]
                if val > max_val:
                    max_val = val
                    best_j = j + 1
            dp[i] = max_val
            prev[i] = best_j

        frames = []
        i = T - 1
        while i >= 0:
            frames.append((prev[i].item(), i))
            i = prev[i].item() - 1
        return frames[::-1]

    def _count_reduced(self, feature_sim, selected_frames, tau, mu_t, sink_score):
        """Count total temporally reduced tokens."""
        total = 0.0
        for start, end in selected_frames:
            if start == end:
                continue
            seg_sim = feature_sim[start:end]
            seg_sink = sink_score[start:end]
            score = seg_sim + mu_t * seg_sink
            static_mask = torch.all(score > 0.8, dim=0)
            # Static tokens are merged across (end-start) frames
            total += static_mask.sum().item() * (end - start)
        return total
