# SPDX-License-Identifier: Apache-2.0
# Adapted from diffusers:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_cosmos.py

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm as DistributedRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Standalone modules (no TP needed, ported directly from diffusers)
# ---------------------------------------------------------------------------


def apply_rotary_emb_cosmos(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    # hidden_states: [B, seq, num_heads, head_dim]
    # freqs_cos/sin: [seq, head_dim]
    # Match diffusers Cosmos RoPE: split into first-half/second-half (unbind_dim=-2)
    cos = freqs_cos
    sin = freqs_sin
    # Unsqueeze to [1, seq, 1, head_dim] for broadcasting over B and num_heads
    if cos.ndim == 2 and hidden_states.ndim == 4:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

    # Split into real/imag halves (first D//2 and last D//2)
    x_real, x_imag = hidden_states.reshape(
        *hidden_states.shape[:-1], 2, -1
    ).unbind(-2)
    x_rotated = torch.cat([-x_imag, x_real], dim=-1)

    out = (hidden_states.float() * cos + x_rotated.float() * sin).to(hidden_states.dtype)
    return out


class CosmosPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: tuple[int, int, int],
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            out_channels,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size, num_channels, num_frames // p_t, p_t, height // p_h, p_h, width // p_w, p_w
        )
        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class CosmosRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        base_fps: int = 24,
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
    ):
        super().__init__()
        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.base_fps = base_fps

        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w

        self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def forward(
        self,
        hidden_states: torch.Tensor,
        fps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [
            num_frames // self.patch_size[0],
            height // self.patch_size[1],
            width // self.patch_size[2],
        ]
        device = hidden_states.device

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)

        dim_h_range = torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        dim_w_range = torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        dim_t_range = torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t

        h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
        temporal_freqs = 1.0 / (t_theta**dim_t_range)

        emb_h = torch.outer(seq[: pe_size[1]], h_spatial_freqs)[None, :, None, :].repeat(pe_size[0], 1, pe_size[2], 1)
        emb_w = torch.outer(seq[: pe_size[2]], w_spatial_freqs)[None, None, :, :].repeat(pe_size[0], pe_size[1], 1, 1)

        if fps is None:
            emb_t = torch.outer(seq[: pe_size[0]], temporal_freqs)
        else:
            emb_t = torch.outer(seq[: pe_size[0]] / fps * self.base_fps, temporal_freqs)

        emb_t = emb_t[:, None, None, :].repeat(1, pe_size[1], pe_size[2], 1)
        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin


class CosmosLearnablePositionalEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()
        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.eps = eps

        self.pos_emb_t = nn.Parameter(torch.zeros(self.max_size[0], hidden_size))
        self.pos_emb_h = nn.Parameter(torch.zeros(self.max_size[1], hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(self.max_size[2], hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [
            num_frames // self.patch_size[0],
            height // self.patch_size[1],
            width // self.patch_size[2],
        ]

        emb_t = self.pos_emb_t[: pe_size[0]][None, :, None, None, :].repeat(batch_size, 1, pe_size[1], pe_size[2], 1)
        emb_h = self.pos_emb_h[: pe_size[1]][None, None, :, None, :].repeat(batch_size, pe_size[0], 1, pe_size[2], 1)
        emb_w = self.pos_emb_w[: pe_size[2]][None, None, None, :, :].repeat(batch_size, pe_size[0], pe_size[1], 1, 1)
        emb = emb_t + emb_h + emb_w
        emb = emb.flatten(1, 3)

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(self.eps, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return (emb / norm).type_as(hidden_states)


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(timesteps.to(self.linear_1.weight.dtype))
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return emb


class CosmosEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int):
        super().__init__()
        self.time_proj = Timesteps(embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = nn.RMSNorm(embedding_dim, eps=1e-6, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.LongTensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep).type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(timesteps_proj)
        return temb, embedded_timestep


class CosmosAdaLayerNorm(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.embedding_dim = in_features
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 2 * in_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb[..., : 2 * self.embedding_dim]

        shift, scale = embedded_timestep.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale = (x.unsqueeze(1) for x in (shift, scale))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class CosmosAdaLayerNormZero(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None):
        super().__init__()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()

        if hidden_features is None:
            self.linear_1 = nn.Identity()
        else:
            self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)

        self.linear_2 = nn.Linear(hidden_features, 3 * in_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb

        shift, scale, gate = embedded_timestep.chunk(3, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale, gate = (x.unsqueeze(1) for x in (shift, scale, gate))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states, gate


# ---------------------------------------------------------------------------
# TP-adapted attention & FFN (adapted from WAN2.2 pattern)
# ---------------------------------------------------------------------------


class CosmosSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            bias=False,
        )

        self.num_heads = self.to_qkv.num_heads
        self.num_kv_heads = self.to_qkv.num_kv_heads
        self.tp_inner_dim = self.num_heads * head_dim

        # Per-head RMSNorm (matches NVIDIA/diffusers checkpoint format)
        self.norm_q = DistributedRMSNorm(head_dim, eps=eps)
        self.norm_k = DistributedRMSNorm(head_dim, eps=eps)

        self.to_out = RowParallelLinear(
            self.inner_dim, dim, bias=False, input_is_parallel=True, return_bias=False,
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        qkv, _ = self.to_qkv(hidden_states)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Unflatten into heads first, then apply per-head norm
        query = query.unflatten(2, (self.num_heads, self.head_dim))
        key = key.unflatten(2, (self.num_kv_heads, self.head_dim))
        value = value.unflatten(2, (self.num_kv_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb
            query = apply_rotary_emb_cosmos(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb_cosmos(key, freqs_cos, freqs_sin)

        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = self.to_out(hidden_states)
        return hidden_states


class CosmosCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        cross_attention_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.to_q = ColumnParallelLinear(
            dim, self.inner_dim, bias=False, gather_output=False, return_bias=False,
        )
        self.to_k = ColumnParallelLinear(
            cross_attention_dim, self.inner_dim, bias=False, gather_output=False, return_bias=False,
        )
        self.to_v = ColumnParallelLinear(
            cross_attention_dim, self.inner_dim, bias=False, gather_output=False, return_bias=False,
        )

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = num_heads // tp_size
        self.tp_inner_dim = self.num_heads * head_dim

        # Per-head RMSNorm (matches NVIDIA/diffusers checkpoint format)
        self.norm_q = DistributedRMSNorm(head_dim, eps=eps)
        self.norm_k = DistributedRMSNorm(head_dim, eps=eps)

        self.to_out = RowParallelLinear(
            self.inner_dim, dim, bias=False, input_is_parallel=True, return_bias=False,
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Unflatten into heads first, then apply per-head norm
        query = query.unflatten(2, (self.num_heads, self.head_dim))
        key = key.unflatten(2, (self.num_heads, self.head_dim))
        value = value.unflatten(2, (self.num_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = self.to_out(hidden_states)
        return hidden_states


class ColumnParallelGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str = "tanh", bias: bool = False):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in, dim_out, bias=bias, gather_output=False, return_bias=False,
        )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


class CosmosFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.net_0 = ColumnParallelGELU(dim, inner_dim, approximate="tanh", bias=False)
        self.net_1 = nn.Identity()
        self.net_2 = RowParallelLinear(
            inner_dim, dim, bias=False, input_is_parallel=True, return_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_1(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Transformer block & top-level model
# ---------------------------------------------------------------------------


class CosmosTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.attn1 = CosmosSelfAttention(
            dim=hidden_size, num_heads=num_attention_heads, head_dim=attention_head_dim,
        )

        self.norm2 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.attn2 = CosmosCrossAttention(
            dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )

        self.norm3 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        inner_dim = int(hidden_size * mlp_ratio)
        self.ff = CosmosFeedForward(dim=hidden_size, inner_dim=inner_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hs, gate = self.norm1(hidden_states, embedded_timestep, temb)
        attn_out = self.attn1(norm_hs, rotary_emb=(freqs_cos, freqs_sin))
        hidden_states = hidden_states + gate * attn_out

        norm_hs, gate = self.norm2(hidden_states, embedded_timestep, temb)
        attn_out = self.attn2(norm_hs, encoder_hidden_states)
        hidden_states = hidden_states + gate * attn_out

        norm_hs, gate = self.norm3(hidden_states, embedded_timestep, temb)
        ff_out = self.ff(norm_hs)
        hidden_states = hidden_states + gate * ff_out

        return hidden_states


class CosmosPredict25Transformer3DModel(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
        concat_padding_mask: bool = True,
        extra_pos_embed_type: str | None = "learnable",
        **kwargs,
    ):
        super().__init__()
        self.od_config = od_config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.concat_padding_mask = concat_padding_mask
        self.extra_pos_embed_type = extra_pos_embed_type

        hidden_size = num_attention_heads * attention_head_dim

        patch_embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels, hidden_size, patch_size, bias=False)

        self.rope = CosmosRotaryPosEmbed(
            hidden_size=attention_head_dim, max_size=max_size, patch_size=patch_size, rope_scale=rope_scale,
        )

        self.learnable_pos_embed = None
        if extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=hidden_size, max_size=max_size, patch_size=patch_size,
            )

        self.time_embed = CosmosEmbedding(hidden_size, hidden_size)

        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=text_embed_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = CosmosAdaLayerNorm(hidden_size, adaln_lora_dim)
        self.proj_out = nn.Linear(
            hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        fps: int | None = None,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.concat_padding_mask and padding_mask is not None:
            from torchvision import transforms

            padding_mask_resized = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST,
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask_resized.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1,
            )

        latent_shape = hidden_states.shape

        freqs_cos, freqs_sin = self.rope(hidden_states, fps=fps)

        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.extra_pos_embed_type else None

        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)

        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (batch_size, 1, num_frames, 1, 1), (
                f"Expected timestep shape [B, 1, T, 1, 1], got {timestep.shape}"
            )
            timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )
        else:
            raise ValueError(f"Expected timestep shape [B] or [B, 1, T, 1, 1], got {timestep.shape}")

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, embedded_timestep, freqs_cos, freqs_sin, temb,
            )

        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(1, (post_patch_num_frames, post_patch_height, post_patch_width))
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return hidden_states

    @staticmethod
    def _remap_nvidia_name(name: str) -> str | None:
        """Remap NVIDIA-native checkpoint names to model parameter names.

        Returns the remapped name, or None if the weight should be skipped.
        """
        import re

        # Skip internal state and non-parameter tensors
        if name.endswith("._extra_state"):
            return None
        # RoPE embedder stores buffers, not parameters
        if name.startswith("net.pos_embedder."):
            return None
        # Cross-attention text projection is handled by the pipeline
        if name.startswith("net.crossattn_proj."):
            return None

        # --- Top-level modules ---
        # Patch embedding: net.x_embedder.proj.1 → patch_embed.proj
        m = re.match(r"net\.x_embedder\.proj\.\d+\.(.+)", name)
        if m:
            return f"patch_embed.proj.{m.group(1)}"

        # Timestep embedding: net.t_embedder.1.* → time_embed.t_embedder.*
        m = re.match(r"net\.t_embedder\.\d+\.(.+)", name)
        if m:
            return f"time_embed.t_embedder.{m.group(1)}"

        # Timestep embedding norm: net.t_embedding_norm.* → time_embed.norm.*
        m = re.match(r"net\.t_embedding_norm\.(.+)", name)
        if m:
            return f"time_embed.norm.{m.group(1)}"

        # Final layer norm: net.final_layer.adaln_modulation.{1,2} → norm_out.linear_{1,2}
        m = re.match(r"net\.final_layer\.adaln_modulation\.(\d+)\.(.+)", name)
        if m:
            return f"norm_out.linear_{m.group(1)}.{m.group(2)}"

        # Final layer projection: net.final_layer.linear.* → proj_out.*
        m = re.match(r"net\.final_layer\.linear\.(.+)", name)
        if m:
            return f"proj_out.{m.group(1)}"

        # --- Transformer blocks ---
        m = re.match(r"net\.blocks\.(\d+)\.(.+)", name)
        if not m:
            return name  # Not an NVIDIA-format name, pass through
        block_idx, rest = m.group(1), m.group(2)
        prefix = f"transformer_blocks.{block_idx}"

        # AdaLN modulation: adaln_modulation_{type}.{1,2} → norm{n}.linear_{1,2}
        adaln_map = {
            "adaln_modulation_self_attn": "norm1",
            "adaln_modulation_cross_attn": "norm2",
            "adaln_modulation_mlp": "norm3",
        }
        for src, dst in adaln_map.items():
            m2 = re.match(rf"{src}\.(\d+)\.(.+)", rest)
            if m2:
                return f"{prefix}.{dst}.linear_{m2.group(1)}.{m2.group(2)}"

        # Self-attention
        sa_map = {
            "self_attn.q_proj": "attn1.to_q",
            "self_attn.k_proj": "attn1.to_k",
            "self_attn.v_proj": "attn1.to_v",
            "self_attn.output_proj": "attn1.to_out",
            "self_attn.q_norm": "attn1.norm_q",
            "self_attn.k_norm": "attn1.norm_k",
        }
        for src, dst in sa_map.items():
            m2 = re.match(rf"{re.escape(src)}\.(.+)", rest)
            if m2:
                return f"{prefix}.{dst}.{m2.group(1)}"

        # Cross-attention
        ca_map = {
            "cross_attn.q_proj": "attn2.to_q",
            "cross_attn.k_proj": "attn2.to_k",
            "cross_attn.v_proj": "attn2.to_v",
            "cross_attn.output_proj": "attn2.to_out",
            "cross_attn.q_norm": "attn2.norm_q",
            "cross_attn.k_norm": "attn2.norm_k",
        }
        for src, dst in ca_map.items():
            m2 = re.match(rf"{re.escape(src)}\.(.+)", rest)
            if m2:
                return f"{prefix}.{dst}.{m2.group(1)}"

        # FFN: mlp.layer1 → ff.net_0.proj, mlp.layer2 → ff.net_2
        m2 = re.match(r"mlp\.layer1\.(.+)", rest)
        if m2:
            return f"{prefix}.ff.net_0.proj.{m2.group(1)}"
        m2 = re.match(r"mlp\.layer2\.(.+)", rest)
        if m2:
            return f"{prefix}.ff.net_2.{m2.group(1)}"

        return name  # Unrecognized, pass through

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Self-attention QKV fusion: separate to_q/k/v → fused to_qkv
        stacked_params_mapping = []
        for i in range(len(self.transformer_blocks)):
            stacked_params_mapping.extend([
                (f"transformer_blocks.{i}.attn1.to_qkv", f"transformer_blocks.{i}.attn1.to_q", "q"),
                (f"transformer_blocks.{i}.attn1.to_qkv", f"transformer_blocks.{i}.attn1.to_k", "k"),
                (f"transformer_blocks.{i}.attn1.to_qkv", f"transformer_blocks.{i}.attn1.to_v", "v"),
            ])

        # Diffusers-format name remapping (kept for compatibility)
        weight_name_remapping = {}
        for i in range(len(self.transformer_blocks)):
            weight_name_remapping[f"transformer_blocks.{i}.attn1.to_out.0.weight"] = (
                f"transformer_blocks.{i}.attn1.to_out.weight"
            )
            weight_name_remapping[f"transformer_blocks.{i}.attn2.to_out.0.weight"] = (
                f"transformer_blocks.{i}.attn2.to_out.weight"
            )
            weight_name_remapping[f"transformer_blocks.{i}.ff.net.0.proj.weight"] = (
                f"transformer_blocks.{i}.ff.net_0.proj.weight"
            )
            weight_name_remapping[f"transformer_blocks.{i}.ff.net.2.weight"] = (
                f"transformer_blocks.{i}.ff.net_2.weight"
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Try NVIDIA-native remapping first
            remapped = self._remap_nvidia_name(name)
            if remapped is None:
                # Explicitly skipped (e.g., _extra_state, pos_embedder)
                loaded_params.add(name)
                continue
            name = remapped

            # Then apply diffusers-format remapping
            name = weight_name_remapping.get(name, name)
            original_name = name

            # Handle x_embedder shape mismatch: checkpoint may have extra
            # condition channels that the model doesn't expect
            if name == "patch_embed.proj.weight" and name in params_dict:
                param = params_dict[name]
                if loaded_weight.shape[1] > param.data.shape[1]:
                    logger.info(
                        f"Trimming patch_embed weight from {loaded_weight.shape[1]} "
                        f"to {param.data.shape[1]} input features"
                    )
                    loaded_weight = loaded_weight[:, : param.data.shape[1]]

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                name = original_name.replace(weight_name, param_name)
                if name not in params_dict:
                    logger.warning(f"Skipping stacked weight {original_name} -> {name}")
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    logger.warning(f"Skipping weight {original_name} -> {name}")
                    continue

                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)

        return loaded_params
