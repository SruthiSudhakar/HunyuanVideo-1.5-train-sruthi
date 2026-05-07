from typing import Any, List, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer, RMSNorm
from .attention import single_stream_sequence_parallel_attention, double_stream_sequence_parallel_attention
from .posemb_layers import apply_rotary_emb, apply_rotary_emb_single

class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act_fun, factory_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.factory_kwargs = factory_kwargs
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, **self.factory_kwargs)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, **self.factory_kwargs)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, **self.factory_kwargs)
        self.act_fn = hidden_act_fun

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class CustomMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act_fun, factory_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.factory_kwargs = factory_kwargs
        self.mlp_up = nn.Linear(self.hidden_size, self.intermediate_size, **self.factory_kwargs)
        self.mlp_down = nn.Linear(self.intermediate_size, self.hidden_size, **self.factory_kwargs)
        self.act_fn = hidden_act_fun

    def forward(self, x):
        x = self.mlp_down(self.act_fn(self.mlp_up(x)))
        return x

class ActionCrossAttnBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        video_state_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        cross_attn_mode: str = "cross_frame_aligned",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.cross_attn_mode = cross_attn_mode

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim

        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_o = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_g = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear2_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear2_k = nn.Linear(video_state_size, hidden_size, **factory_kwargs)
        self.linear2_v = nn.Linear(video_state_size, hidden_size, **factory_kwargs)
        self.linear2_o = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear2_g = nn.Linear(hidden_size, hidden_size, **factory_kwargs)

        self.mlp = CustomMLP(hidden_size, mlp_hidden_dim, get_activation_layer(mlp_act_type)(), factory_kwargs)
        # self.mlp = Qwen3MLP(hidden_size, mlp_hidden_dim, get_activation_layer(mlp_act_type)(), factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm_1 = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm_1 = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.q_norm_2 = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm_2 = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.norm_1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.norm_2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.norm_3 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        video_states: torch.Tensor,
        freqs_cis_self: Tuple[torch.Tensor, torch.Tensor] = None,
        freqs_cis_cross: Tuple[torch.Tensor, torch.Tensor] = None,
        tokens_per_q_frame: int = 0,
        tokens_per_kv_frame: int = 0,
    ) -> torch.Tensor:
        # Do self attention
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)

        q = self.linear1_q(hidden_states)
        k = self.linear1_k(hidden_states)
        v = self.linear1_v(hidden_states)
        g = self.linear1_g(hidden_states)   # attention gating

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)
        
        q = self.q_norm_1(q).to(v)
        k = self.k_norm_1(k).to(v)

        qq, kk = apply_rotary_emb(q, k, freqs_cis_self, head_first=False)
        assert (
            qq.shape == q.shape and kk.shape == k.shape
        ), f"qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}"
        q, k = qq, kk

        hidden_states = single_stream_sequence_parallel_attention(
            q,
            k,
            v,
            mode="self_full",
        )

        hidden_states = hidden_states * torch.sigmoid(g).to(hidden_states)
        hidden_states = residual + self.linear1_o(hidden_states)

        # Do cross attention
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)

        q = self.linear2_q(hidden_states)
        k = self.linear2_k(video_states)
        v = self.linear2_v(video_states)
        g = self.linear2_g(hidden_states)   # attention gating

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)

        q = self.q_norm_2(q).to(v)
        k = self.k_norm_2(k).to(v)

        qq = apply_rotary_emb_single(q, freqs_cis_self, head_first=False)
        kk = apply_rotary_emb_single(k, freqs_cis_cross, head_first=False)
        assert (
            qq.shape == q.shape and kk.shape == k.shape
        ), f"qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}"
        q, k = qq, kk

        hidden_states = single_stream_sequence_parallel_attention(
            q,
            k,
            v,
            mode=self.cross_attn_mode,
            tokens_per_q_frame=tokens_per_q_frame,
            tokens_per_kv_frame=tokens_per_kv_frame,
        )

        hidden_states = hidden_states * torch.sigmoid(g).to(hidden_states)
        hidden_states = residual + self.linear2_o(hidden_states)

        # Do mlp
        residual = hidden_states
        hidden_states = self.norm_3(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return residual + hidden_states


class ActionSelfAttnBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim

        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_o = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_g = nn.Linear(hidden_size, hidden_size, **factory_kwargs)

        self.mlp = CustomMLP(hidden_size, mlp_hidden_dim, get_activation_layer(mlp_act_type)(), factory_kwargs)
        # self.mlp = Qwen3MLP(hidden_size, mlp_hidden_dim, get_activation_layer(mlp_act_type)(), factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm_1 = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm_1 = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.norm_1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.norm_2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis_self: Tuple[torch.Tensor, torch.Tensor] = None,
        hidden_states_are_fully_gathered: bool = False,
    ) -> torch.Tensor:
        # Do self attention
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)

        q = self.linear1_q(hidden_states)
        k = self.linear1_k(hidden_states)
        v = self.linear1_v(hidden_states)
        g = self.linear1_g(hidden_states)   # attention gating

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)
        
        q = self.q_norm_1(q).to(v)
        k = self.k_norm_1(k).to(v)

        qq, kk = apply_rotary_emb(q, k, freqs_cis_self, head_first=False)
        assert (
            qq.shape == q.shape and kk.shape == k.shape
        ), f"qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}"
        q, k = qq, kk

        hidden_states = single_stream_sequence_parallel_attention(
            q,
            k,
            v,
            mode="self_full",
            q_is_fully_gathered=hidden_states_are_fully_gathered,
            k_is_fully_gathered=hidden_states_are_fully_gathered,
            v_is_fully_gathered=hidden_states_are_fully_gathered,
        )

        ############### Exclusive Self Attention (XSA) START ###############
        hidden_states = rearrange(hidden_states, "B L (H D) -> B L H D", H=self.heads_num)
        v_hat = F.normalize(v, dim=-1)
        proj_coeff = (hidden_states * v_hat).sum(dim=-1, keepdim=True)
        hidden_states = hidden_states - proj_coeff * v_hat
        hidden_states = rearrange(hidden_states, "B L H D -> B L (H D)")
        ############### Exclusive Self Attention (XSA) END ###############

        hidden_states = hidden_states * torch.sigmoid(g).to(hidden_states)
        hidden_states = residual + self.linear1_o(hidden_states)

        # Do mlp
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return residual + hidden_states

class ActionDecodingInit(nn.Module):
    """
    Build per-latent-frame K action/pose latents initialized from:
      - learned action slot embeddings (K, Da)
      - learned temporal slot embeddings for *compressed* latent frames (R, Da)
      - learned temporal slot embedding for the *first* latent frame (1, Da)
      - plus a per-latent-frame pooled context derived from motion latents

    Special case:
      - latent frame 0 corresponds to a single (uncompressed) video frame
      - latent frames 1..T-1 each correspond to R video frames

    Input:
      motion_states: [B, L, Dm] where L = T_latent * K_motion (tokens are time-major)
      num_latent_frames: latent frame count (T_latent)

    Output:
      action_latents: [B, total_tokens, Da]
        where total_tokens = (1 + (T_latent-1)*R) * K
    """
    def __init__(
        self,
        motion_state_size: int,
        hidden_size: int,
        action_tokens_per_video_frame: int,
        temporal_compression_ratio: int = 4,
    ):
        super().__init__()
        self.K = int(action_tokens_per_video_frame)
        self.R = int(temporal_compression_ratio)
        self.hidden_size = int(hidden_size)

        self.action_slot = nn.Parameter(torch.randn(self.K, self.hidden_size) * 0.02)
        self.temporal_slot = nn.Parameter(torch.randn(self.R, self.hidden_size) * 0.02)
        self.first_frame_slot = nn.Parameter(torch.randn(1, self.hidden_size) * 0.02)

        self.proj = nn.Sequential(
            nn.LayerNorm(motion_state_size, elementwise_affine=True, eps=1e-6),
            nn.Linear(motion_state_size, self.hidden_size),
        )


    def forward(self, motion_states: torch.Tensor, num_latent_frames: int) -> torch.Tensor:
        """
        motion_states: [B, L, Dm]
        num_latent_frames: int
        returns: [B, ((1 + (T-1)*R) * K), Da]
        """
        if motion_states.dim() != 3:
            raise ValueError(f"Expected motion_states [B, L, D], got {tuple(motion_states.shape)}")

        B, L, Dm = motion_states.shape
        T = int(num_latent_frames)

        if L % T != 0:
            raise ValueError(f"Sequence length L={L} must be divisible by num_latent_frames T={T}")

        # [B, T_latent, K_motion, Dm] -> pooled -> [B, T_latent, Dm] -> proj -> [B, T_latent, Da]
        x = motion_states.reshape(B, T, L // T, Dm)
        motion_ctx = self.proj(x.mean(dim=2))

        action_slot = self.action_slot.to(motion_states)          # [K, Da]
        temporal_slots = self.temporal_slot.to(motion_states)     # [R, Da]
        first_slot = self.first_frame_slot.to(motion_states)      # [1, Da]

        # [B, 1, K, Da] = action_slot + first_slot + ctx0
        ctx0 = motion_ctx[:, 0:1, :]  # [B, 1, Da]
        action0 = (
            action_slot[None, None, :, :] +        # [1, 1, K, Da]
            first_slot[None, :, None, :] +         # [1, 1, 1, Da]
            ctx0[:, :, None, :]                    # [B, 1, 1, Da]
        )                                          # -> [B, 1, K, Da]

        if T > 1:
            ctx_rest = motion_ctx[:, 1:, :]                 # [B, T-1, Da]
            action_rest = (
                action_slot[None, None, None, :, :] +       # [1, 1, 1, K, Da]
                temporal_slots[None, None, :, None, :] +    # [1, 1, R, 1, Da]
                ctx_rest[:, :, None, None, :]               # [B, T-1, 1, 1, Da]
            )                                               # -> [B, T-1, R, K, Da]

            # concat along the "video-frame-expanded" axis: 1 + (T-1)*R
            action0_flat = action0.reshape(B, 1, self.K, self.hidden_size)                          # [B, 1, K, Da]
            action_rest_flat = action_rest.reshape(B, (T - 1) * self.R, self.K, self.hidden_size)   # [B, (T-1)*R, K, Da]
            action_states = torch.cat([action0_flat, action_rest_flat], dim=1)                      # [B, 1+(T-1)*R, K, Da]
        else:
            action_states = action0.reshape(B, 1, self.K, self.hidden_size)                         # [B, 1, K, Da]

        action_states = action_states.reshape(B, -1, self.hidden_size)                              # [B, (1+(T-1)*R)*K, Da]
        return action_states

class ActionOutProj(nn.Module):
    """
    Project action states [B, L, D] -> action outputs [B, L, Dj].
    """
    def __init__(
        self,
        hidden_size: int,
        pos_dim: int,
        num_joints: int,
    ):
        super().__init__()
        self.num_joints = int(num_joints)

        self.pos_proj = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6),
            nn.Linear(hidden_size, pos_dim),
        )

    def forward(self, action_states: torch.Tensor) -> torch.Tensor:
        """
        action_states: [B, L, D]

        Returns:
            action outputs [B, L, Dj]
        """
        if action_states.dim() != 3:
            raise ValueError(f"Expected action_states [B, L, D], got {tuple(action_states.shape)}")

        return self.pos_proj(action_states)

# class JointPosEmbedMlp(nn.Module):
#     def __init__(self, pos_dim: int, d_model: int, film_hidden_dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps

#         self.position_to_embed = nn.Linear(pos_dim, d_model)
#         self.embed_norm = nn.LayerNorm(d_model)

#         self.magnitude_to_film = nn.Sequential(
#             nn.Linear(1, film_hidden_dim),
#             nn.SiLU(),
#             nn.Linear(film_hidden_dim, 2 * d_model),
#         )

#         nn.init.zeros_(self.magnitude_to_film[-1].weight)
#         nn.init.zeros_(self.magnitude_to_film[-1].bias)

#     def forward(self, joint_pos_xyz: torch.Tensor) -> torch.Tensor:
#         # joint_pos_xyz: [..., 3]
#         joint_radius = joint_pos_xyz.norm(dim=-1, keepdim=True).clamp_min(self.eps)

#         embed = self.position_to_embed(joint_pos_xyz)
#         embed = self.embed_norm(embed)

#         film_params = self.magnitude_to_film(joint_radius)
#         film_scale, film_shift = film_params.chunk(2, dim=-1)

#         return embed * (1.0 + film_scale) + film_shift

class JointPosEmbedMlp(nn.Module):
    def __init__(self, pos_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pos_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, x):
        return self.net(x)

class ActionEncodingInit(nn.Module):

    def __init__(
            self, 
            hidden_size: int, 
            num_joints: int, 
            pos_dim: int, 
            temporal_compression_ratio: int = 4
        ):
        super().__init__()
        self.R = int(temporal_compression_ratio)
        self.hidden_size = int(hidden_size)
        self.K = int(num_joints)
        self.pos_dim = int(pos_dim)

        self.joint_pos_mlp = JointPosEmbedMlp(pos_dim=pos_dim, d_model=self.hidden_size)

        self.joint_pos = nn.Parameter(torch.randn(self.K, hidden_size) * 0.02)              # [J, Da]
        self.temporal_slot = nn.Parameter(torch.randn(self.R, hidden_size) * 0.02)          # [R, Da]
        self.first_frame_slot = nn.Parameter(torch.randn(1, hidden_size) * 0.02)            # [1, Da]

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        if joints.dim() != 4:
            raise ValueError(f"Expected joints [B,T,J,D], got {tuple(joints.shape)}")

        B, T, J, Dj = joints.shape
        if J != self.K:
            raise ValueError(f"Got J={J} joints but expected num_joints={self.K}")
        if Dj != self.pos_dim:
            raise ValueError(f"Expected joint_dim == {self.pos_dim}, got Dj={Dj}")
        if (T - 1) % self.R != 0:
            raise ValueError(f"Expected temporal dimension T in 4n+1, got joints in shape {tuple(joints.shape)}")

        pos = joints.clone()
        x = self.joint_pos_mlp(pos)  # [B, T, J, Da]

        joint_pos = self.joint_pos.to(x)        # [J, Da]
        x = x + joint_pos[None, None, :, :]     # [B, T, J, Da]

        first = self.first_frame_slot.to(x)     # [1, Da]
        temporal = self.temporal_slot.to(x)     # [R, Da]

        # Build per-video-frame embedding: [T, Da]
        if T == 1:
            frame_emb = first  # [1, Da]
        else:
            rest = temporal.repeat((T - 1) // self.R, 1)        # [T-1, Da] = 0..R-1 repeating
            frame_emb = torch.cat([first, rest], dim=0)         # [T, Da]

        # Flatten first, then add frame embedding with repeat_interleave over joints
        tokens = x.reshape(B, T * J, self.hidden_size)          # [B, L, Da], L=T*J
        frame_tok = frame_emb.repeat_interleave(J, dim=0)       # [T*J, Da]
        tokens = tokens + frame_tok[None, :, :]                 # [B, L, Da]

        return tokens

class ActionDecodingDoubleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.action_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.action_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.action_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.action_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.action_o = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.action_g = nn.Linear(hidden_size, hidden_size, **factory_kwargs)  # attention gate

        self.action_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)

        self.action_mlp = CustomMLP(hidden_size, mlp_hidden_dim, get_activation_layer(mlp_act_type)(), factory_kwargs)
        # self.action_mlp = Qwen3MLP(hidden_size, mlp_hidden_dim, get_activation_layer(mlp_act_type)(), factory_kwargs)

        self.motion_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.motion_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.motion_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.motion_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.motion_o = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.motion_g = nn.Linear(hidden_size, hidden_size, **factory_kwargs)  # attention gate

        self.motion_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.motion_mlp = CustomMLP(hidden_size, mlp_hidden_dim, get_activation_layer(mlp_act_type)(), factory_kwargs)
        # self.motion_mlp = Qwen3MLP(hidden_size, mlp_hidden_dim, get_activation_layer(mlp_act_type)(), factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.action_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.action_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.motion_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.motion_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        action: torch.Tensor,
        motion: torch.Tensor,
        freqs_cis_action: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        freqs_cis_motion: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        action_resid = action
        motion_resid = motion

        action_ln = self.action_norm1(action)
        motion_ln = self.motion_norm1(motion)

        action_q = self.action_q(action_ln)
        action_k = self.action_k(action_ln)
        action_v = self.action_v(action_ln)
        action_g = self.action_g(action_ln)  # [B, L_action, D]

        motion_q = self.motion_q(motion_ln)
        motion_k = self.motion_k(motion_ln)
        motion_v = self.motion_v(motion_ln)
        motion_g = self.motion_g(motion_ln)  # [B, L_motion, D]

        # reshape to [B, L, H, Dh]
        action_q = rearrange(action_q, "B L (H D) -> B L H D", H=self.heads_num)
        action_k = rearrange(action_k, "B L (H D) -> B L H D", H=self.heads_num)
        action_v = rearrange(action_v, "B L (H D) -> B L H D", H=self.heads_num)
        motion_q = rearrange(motion_q, "B L (H D) -> B L H D", H=self.heads_num)
        motion_k = rearrange(motion_k, "B L (H D) -> B L H D", H=self.heads_num)
        motion_v = rearrange(motion_v, "B L (H D) -> B L H D", H=self.heads_num)

        action_q = self.action_q_norm(action_q).to(action_v)
        action_k = self.action_k_norm(action_k).to(action_v)
        motion_q = self.motion_q_norm(motion_q).to(motion_v)
        motion_k = self.motion_k_norm(motion_k).to(motion_v)

        if freqs_cis_action is not None:
            action_qq, action_kk = apply_rotary_emb(action_q, action_k, freqs_cis_action, head_first=False)
            action_q, action_k = action_qq, action_kk
        if freqs_cis_motion is not None:
            motion_qq, motion_kk = apply_rotary_emb(motion_q, motion_k, freqs_cis_motion, head_first=False)
            motion_q, motion_k = motion_qq, motion_kk

        attn_out = double_stream_sequence_parallel_attention(
            (action_q, motion_q),
            (action_k, motion_k),
            (action_v, motion_v),
        )
        action_attn = attn_out[:, : action_resid.shape[1]].contiguous()
        motion_attn = attn_out[:, action_resid.shape[1] :].contiguous()

        action_attn = action_attn * torch.sigmoid(action_g).to(action_attn)
        motion_attn = motion_attn * torch.sigmoid(motion_g).to(motion_attn)

        action = action_resid + self.action_o(action_attn)
        motion = motion_resid + self.motion_o(motion_attn)

        action_resid2 = action
        action = self.action_norm2(action)
        action = action_resid2 + self.action_mlp(action)

        motion_resid2 = motion
        motion = self.motion_norm2(motion)
        motion = motion_resid2 + self.motion_mlp(motion)

        return action, motion
