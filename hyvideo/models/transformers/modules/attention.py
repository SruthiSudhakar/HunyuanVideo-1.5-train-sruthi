# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import einops
import torch
from typing import Optional
from loguru import logger
import numpy as np
import torch.nn.functional as F
from typing import Optional, Literal

from hyvideo.commons.parallel_states import get_parallel_state
from hyvideo.utils.communications import (
    all_gather,
    all_to_all_4D,
)
from hyvideo.utils.flash_attn_no_pad import (
    flash_attn_no_pad,
    flash_attn_no_pad_v3,
)
from hyvideo.commons import maybe_fallback_attn_mode

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention = torch.compile(flex_attention, dynamic=False)
    torch._dynamo.config.cache_size_limit = 192
    torch._dynamo.config.accumulated_cache_size_limit = 192
    flex_mask_cache = {}
except Exception:
    logger.warning("Could not load Sliding Tile Attention of FlexAttn.")

from hyvideo.models.transformers.modules.ssta_attention import ssta_3d_attention



@torch.compiler.disable
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    drop_rate: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    attn_mode: str = "flash",
) -> torch.Tensor:
    """
    Compute attention using flash_attn_no_pad or torch scaled_dot_product_attention.

    Args:
        q: Query tensor of shape [B, L, H, D]
        k: Key tensor of shape [B, L, H, D]
        v: Value tensor of shape [B, L, H, D]
        drop_rate: Dropout rate for attention weights.
        attn_mask: Optional attention mask of shape [B, L].
        causal: Whether to apply causal masking.
        attn_mode: Attention mode, either "flash" or "torch". Defaults to "flash".

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """
    attn_mode = maybe_fallback_attn_mode(attn_mode)

    if attn_mode == "torch":
        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = q.transpose(1, 2)  # B * H * L * D
        key = k.transpose(1, 2)    # B * H * L * D
        value = v.transpose(1, 2)  # B * H * L * D
        
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [torch.int64, torch.int32]:
                assert attn_mask.max() <= 1 and attn_mask.min() >= 0, f'Integer attention mask must be between 0 and 1 for torch attention.'
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(f'Float attention mask is not implemented for torch attention.')
            attn_mask1 = einops.rearrange(attn_mask, 'b l -> b 1 l 1')
            attn_mask2 = einops.rearrange(attn_mask1, 'b 1 l 1 -> b 1 1 l')
            attn_mask = attn_mask1 & attn_mask2
        
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
        
        # transpose back
        x = x.transpose(1, 2)  # B * L * H * D
        b, s, h, d = x.shape
        out = x.reshape(b, s, -1)
        return out
    else:
        # flash mode (default)
        qkv = torch.stack([q, k, v], dim=2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()
        x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None)
        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)
        return out


@torch.compiler.disable
def parallel_attention(q, k, v, img_q_len, img_kv_len, 
                       attn_mode=None, text_mask=None, 
                       attn_param=None,
                       block_idx=None,
                       ):
    return sequence_parallel_attention(q, k, v, img_q_len, img_kv_len, attn_mode, text_mask, attn_param=attn_param, block_idx=block_idx)


def sequence_parallel_attention(q, k, v, 
                                img_q_len, img_kv_len, 
                                attn_mode=None, text_mask=None,
                                attn_param=None,
                                block_idx=None,
                                ):
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    if enable_sp:
        sp_group = parallel_dims.sp_group
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank
    
    if enable_sp:
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, sp_group, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, sp_group, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, sp_group, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(
                dim, sp_rank * local_heads, local_heads
            )

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    attn_mode = maybe_fallback_attn_mode(attn_mode)
    
    if attn_mode == "sageattn":
        from sageattention import sageattn
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        hidden_states = sageattn(query, key, value, tensor_layout="NHD", is_causal=False)
    elif attn_mode == "torch":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            attn_mask = None

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [torch.int64, torch.int32]:
                assert attn_mask.max() <= 1 and attn_mask.min() >= 0, f'Integer attention mask must be between 0 and 1 for torch attention.'
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(f'Float attention mask is not implemented for torch attention.')
            
        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = query.transpose(1, 2)  # B * Head_num * length * dim
        key = key.transpose(1, 2)      # B * Head_num * length * dim
        value = value.transpose(1, 2)  # B * Head_num * length * dim

        def score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(attn_mask[b, q_idx] & attn_mask[b, kv_idx], score, float('-inf'))

        hidden_states = flex_attention(query, key, value, score_mod=score_mod)
        
        # transpose back
        hidden_states = hidden_states.transpose(1, 2)
        
    elif attn_mode == "flash2":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)

        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)
        
    elif attn_mode == "flash3":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad_v3(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

    elif attn_mode == "flex-block-attn":
        sparse_type = attn_param["attn_sparse_type"]  # sta/block_attn/ssta
        ssta_threshold = attn_param["ssta_threshold"]
        ssta_lambda = attn_param["ssta_lambda"]
        ssta_sampling_type = attn_param["ssta_sampling_type"]
        ssta_adaptive_pool = attn_param["ssta_adaptive_pool"]

        attn_pad_type = attn_param["attn_pad_type"]  # repeat/zero
        attn_use_text_mask = attn_param["attn_use_text_mask"]
        attn_mask_share_within_head = attn_param["attn_mask_share_within_head"]

        ssta_topk = attn_param["ssta_topk"]
        thw = attn_param["thw"]
        tile_size = attn_param["tile_size"]
        win_size = attn_param["win_size"][0].copy()

        def get_image_tile(tile_size):
            block_size = np.prod(tile_size)
            if block_size == 384:
                tile_size = (1, 16, 24)
            elif block_size == 128:
                tile_size = (1, 16, 8)
            elif block_size == 64:
                tile_size = (1, 8, 8)
            elif block_size == 16:
                tile_size = (1, 4, 4)
            else:
                raise ValueError(f"Error tile_size {tile_size}, only support in [16, 64, 128, 384]")
            return tile_size

        if thw[0] == 1:
            tile_size = get_image_tile(tile_size)
            win_size = [1, 1, 1]
        elif thw[0] <= 31: # 16fps: 5 * 16 / 4 + 1 = 21; 24fps: 5 * 24 / 4 + 1 = 31
            ssta_topk = ssta_topk // 2

        # Concatenate and permute query, key, value to (B, H, S, D)
        query = torch.cat([query, encoder_query], dim=1).permute(0, 2, 1, 3)
        key = torch.cat([key, encoder_key], dim=1).permute(0, 2, 1, 3)
        value = torch.cat([value, encoder_value], dim=1).permute(0, 2, 1, 3)

        assert (
            query.shape[-1] == 128
        ), "The last dimension of query, key and value must be 128 for flex-block-attn."

        hidden_states = ssta_3d_attention(
            query,
            key,
            value,
            thw,
            topk=ssta_topk,
            tile_thw=tile_size,
            kernel_thw=win_size,
            text_len=encoder_sequence_length,
            sparse_type=sparse_type,
            threshold=ssta_threshold,
            lambda_=ssta_lambda,
            pad_type=attn_pad_type,
            text_mask=text_mask if attn_use_text_mask else None,
            sampling_type=ssta_sampling_type,
            adaptive_pool=ssta_adaptive_pool,
            mask_share_within_head=attn_mask_share_within_head,
        )
        hidden_states, sparse_ratio = hidden_states
        hidden_states = hidden_states.permute(0, 2, 1, 3)

    else:
        raise NotImplementedError(
            f'Unsupported attention mode: {attn_mode}. Only torch, flash, flash3, sageattn and flex-block-attn are supported.'
        )


    if enable_sp:
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes((sequence_length, encoder_sequence_length), dim=1)
        hidden_states = all_to_all_4D(hidden_states, sp_group, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states, dim=2, group=sp_group).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states


@torch.compiler.disable
def single_stream_sequence_parallel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mode: Literal[
        "self_full",          # full bidirectional self-attention over all tokens (q==k==v typically)
        "cross_frame_aligned",# each query frame attends only to tokens of its corresponding compressed frame
        "cross_full",         # each query attends to all kv tokens
    ] = "self_full",
    tokens_per_q_frame: int = 0,
    tokens_per_kv_frame: int = 0,
    q_is_fully_gathered: bool = False,
    k_is_fully_gathered: bool = False,
    v_is_fully_gathered: bool = False,
) -> torch.Tensor:
    """
    cross_frame_aligned:
        q: [B, T * Qf, H, D]
        k,v: [B, T * Kvf, H, D]
    Each query frame attends only to the KV tokens from the same frame.
    """
    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    query = q
    key = k
    value = v

    # Ulysses: input is seq-sharded, convert to (global_seq, local_heads)
    if enable_sp:
        sp_group = parallel_dims.sp_group
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank

        def shrink_head(state: torch.Tensor, dim: int) -> torch.Tensor:
            local_heads = state.shape[dim] // sp_size
            return state.narrow(dim, sp_rank * local_heads, local_heads)

        if q_is_fully_gathered:
            query = shrink_head(query, dim=2)
        else:
            query = all_to_all_4D(query, sp_group, scatter_dim=2, gather_dim=1)  # (B, Lq_global, H_local, D)

        if k_is_fully_gathered:
            key = shrink_head(key, dim=2)
        else:
            key = all_to_all_4D(key, sp_group, scatter_dim=2, gather_dim=1)  # (B, Lkv_global, H_local, D)

        if v_is_fully_gathered:
            value = shrink_head(value, dim=2)
        else:
            value = all_to_all_4D(value, sp_group, scatter_dim=2, gather_dim=1)

    B, Lq, Hh, Dd = query.shape
    _, Lkv, _, _ = key.shape

    # SDPA expects (B, H, L, D) for 4D or (N, L, D) for 3D.
    query_t = query.transpose(1, 2)  # (B, H_local, Lq, D)
    key_t   = key.transpose(1, 2)    # (B, H_local, Lkv, D)
    value_t = value.transpose(1, 2)  # (B, H_local, Lkv, D)

    if mode == "self_full":
        # Keep your flash fast-path if present; replace flex fallback with SDPA.
        attn_mode = maybe_fallback_attn_mode("flash")
        if attn_mode == "flash3":
            qkv = torch.stack([query, key, value], dim=2)  # (B, Lq, 3, H_local, D)
            hidden = flash_attn_no_pad_v3(qkv, None, causal=False, dropout_p=0.0, softmax_scale=None)
        elif attn_mode == "flash2":
            qkv = torch.stack([query, key, value], dim=2)  # (B, Lq, 3, H_local, D)
            hidden = flash_attn_no_pad(qkv, None, causal=False, dropout_p=0.0, softmax_scale=None)
        else:
            # Torch SDPA (no mask, non-causal). Likely dispatches to flash/mem-effic kernels when available.
            hidden = F.scaled_dot_product_attention(
                query_t, key_t, value_t,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)  # (B, Lq, H_local, D)

    elif mode == "cross_full":
        # Full cross-attn without mask (Lq is usually small here, so memory is fine).
        hidden = F.scaled_dot_product_attention(
            query_t, key_t, value_t,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)  # (B, Lq, H_local, D)
    elif mode == "cross_frame_aligned":
        if tokens_per_kv_frame <= 0:
            raise ValueError("tokens_per_kv_frame must be > 0 for cross_frame_aligned")

        if Lkv % tokens_per_kv_frame != 0:
            raise ValueError(
                f"Lkv={Lkv} must be divisible by tokens_per_kv_frame={tokens_per_kv_frame}"
            )

        T = Lkv // tokens_per_kv_frame

        if tokens_per_q_frame <= 0:
            if Lq % T != 0:
                raise ValueError(
                    f"Lq={Lq} must be divisible by number of frames T={T}. "
                    "Pass tokens_per_q_frame explicitly."
                )
            tokens_per_q_frame = Lq // T

        if Lq != T * tokens_per_q_frame:
            raise ValueError(
                f"Lq mismatch: expected {T} * {tokens_per_q_frame} = {T * tokens_per_q_frame}, got {Lq}"
            )

        # reshape into per-frame groups
        # query_t: [B, H, T*Qf, D] -> [B, H, T, Qf, D]
        # key_t:   [B, H, T*Kvf, D] -> [B, H, T, Kvf, D]
        query_tf = query_t.view(B, Hh, T, tokens_per_q_frame, Dd)
        key_tf   = key_t.view(B, Hh, T, tokens_per_kv_frame, Dd)
        value_tf = value_t.view(B, Hh, T, tokens_per_kv_frame, Dd)

        # merge (B,H,T) into batch so each frame attends only within itself
        query_flat = query_tf.permute(0, 2, 1, 3, 4).reshape(B * T, Hh, tokens_per_q_frame, Dd)
        key_flat   = key_tf.permute(0, 2, 1, 3, 4).reshape(B * T, Hh, tokens_per_kv_frame, Dd)
        value_flat = value_tf.permute(0, 2, 1, 3, 4).reshape(B * T, Hh, tokens_per_kv_frame, Dd)

        hidden_flat = F.scaled_dot_product_attention(
            query_flat,
            key_flat,
            value_flat,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )  # [B*T, H, Qf, D]

        hidden = (
            hidden_flat
            .view(B, T, Hh, tokens_per_q_frame, Dd)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, Hh, Lq, Dd)
            .transpose(1, 2)
            .contiguous()
        )  # [B, Lq, H, D]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Inverse Ulysses transform back to seq-sharded layout with global heads
    if enable_sp:
        sp_group = parallel_dims.sp_group
        if q_is_fully_gathered:
            hidden = all_gather(hidden, dim=2, group=sp_group)  # (B, Lq_global, H, D)
        else:
            hidden = all_to_all_4D(hidden, sp_group, scatter_dim=1, gather_dim=2)  # (B, Lq_shard, H, D)
        hidden = hidden.to(q.dtype)

    b, s, h, d = hidden.shape
    hidden = hidden.reshape(b, s, -1)

    return hidden


@torch.compiler.disable
def double_stream_sequence_parallel_attention(
    q: tuple[torch.Tensor, torch.Tensor],
    k: tuple[torch.Tensor, torch.Tensor],
    v: tuple[torch.Tensor, torch.Tensor],
    *,
    drop_rate: float = 0.0,
    attn_mode: str = "flash",   # "flash" / "flash2" / "flash3" / "torch"
) -> torch.Tensor:
    """
    Full bidirectional self-attention over the *concatenation* of two streams, with Ulysses-style
    sequence parallel (SP) assumed for BOTH streams.

    Inputs (each stream):
      q_i, k_i, v_i: [B, L_i_shard, H, D]   (sequence-sharded, full heads)

    Behavior:
      1) If SP enabled: all_to_all_4D(..., scatter_dim=2, gather_dim=1) per-stream to get
         [B, L_i_global, H_local, D]
      2) Concatenate streams along sequence: L_total = L0_global + L1_global
      3) Run full (non-causal) attention with NO mask over the concatenated tokens
      4) Split back to two streams
      5) If SP enabled: inverse all_to_all_4D(..., scatter_dim=1, gather_dim=2) per-stream
      6) Return per-stream outputs as [B, L_i_shard, H*D]

    Returns:
      out0, out1: each [B, L_i_shard, H*D]
    """
    q0, q1 = q
    k0, k1 = k
    v0, v1 = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled

    # Ulysses: (seq-sharded, full heads) -> (global seq, local heads)
    if enable_sp:
        sp_group = parallel_dims.sp_group
        q0 = all_to_all_4D(q0, sp_group, scatter_dim=2, gather_dim=1)  # [B, L0_global, H_local, D]
        k0 = all_to_all_4D(k0, sp_group, scatter_dim=2, gather_dim=1)
        v0 = all_to_all_4D(v0, sp_group, scatter_dim=2, gather_dim=1)

        q1 = all_to_all_4D(q1, sp_group, scatter_dim=2, gather_dim=1)  # [B, L1_global, H_local, D]
        k1 = all_to_all_4D(k1, sp_group, scatter_dim=2, gather_dim=1)
        v1 = all_to_all_4D(v1, sp_group, scatter_dim=2, gather_dim=1)

    L0 = q0.size(1)
    L1 = q1.size(1)

    # Concatenate along sequence (joint self-attn across both streams)
    q_cat = torch.cat([q0, q1], dim=1)  # [B, Ltot, H{local/full}, D]
    k_cat = torch.cat([k0, k1], dim=1)
    v_cat = torch.cat([v0, v1], dim=1)

    # Run full attention, no mask, non-causal
    attn_mode = maybe_fallback_attn_mode(attn_mode)

    if attn_mode in ("flash3", "flash2"):
        # flash expects [B, S, 3, H, D]
        qkv = torch.stack([q_cat, k_cat, v_cat], dim=2)
        if attn_mode == "flash3":
            hidden = flash_attn_no_pad_v3(
                qkv, None, causal=False, dropout_p=drop_rate, softmax_scale=None
            )  # [B, S, H, D]
        else:
            hidden = flash_attn_no_pad(
                qkv, None, causal=False, dropout_p=drop_rate, softmax_scale=None
            )
    elif attn_mode == "torch":
        # torch SDPA path: (B,H,S,D)
        query = q_cat.transpose(1, 2)
        key = k_cat.transpose(1, 2)
        value = v_cat.transpose(1, 2)
        hidden = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=drop_rate, is_causal=False
        ).transpose(1, 2)  # [B, S, H, D]
    else:
        # fallback: flex_attention expects (B,H,S,D)
        query = q_cat.transpose(1, 2)
        key = k_cat.transpose(1, 2)
        value = v_cat.transpose(1, 2)
        hidden = flex_attention(query, key, value).transpose(1, 2)  # [B, S, H, D]

    # Split back into streams in the current layout (global seq / local heads if SP enabled)
    h0, h1 = hidden.split_with_sizes((L0, L1), dim=1)

    # Inverse Ulysses: (global seq, local heads) -> (seq-sharded, full heads)
    if enable_sp:
        sp_group = parallel_dims.sp_group
        h0 = all_to_all_4D(h0, sp_group, scatter_dim=1, gather_dim=2).contiguous()  # [B, L0_shard, H, D]
        h1 = all_to_all_4D(h1, sp_group, scatter_dim=1, gather_dim=2).contiguous()  # [B, L1_shard, H, D]

    # Concatenate in (sharded) sequence dimension like original function output
    hidden_states = torch.cat([h0, h1], dim=1).to(q0.dtype)  # [B, L0_shard+L1_shard, H, D]

    b, s, h, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)  # [B, S_shard_total, H*D]

    return hidden_states
