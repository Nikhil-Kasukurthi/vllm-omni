# SPDX-License-Identifier: Apache-2.0
# Adapted from diffusers:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from diffusers import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.cosmos_predict2_5.cosmos_predict2_5_transformer import (
    CosmosPredict25Transformer3DModel,
)
from vllm_omni.diffusion.models.cosmos_predict2_5.utils import DEFAULT_NEGATIVE_PROMPT
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)


def load_transformer_config(
    model_path: str,
    subfolder: str = "transformer",
    local_files_only: bool = True,
    revision: str | None = None,
) -> dict:
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
                revision=revision,
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_cosmos_predict25_post_process_func(od_config: OmniDiffusionConfig):
    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


class CosmosPredict25Pipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)
        revision = od_config.revision

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
            revision=revision,
            use_fast=False,
        )

        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=dtype,
            revision=revision,
            local_files_only=local_files_only,
        ).to(self.device)

        self.vae = AutoencoderKLWan.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=dtype,
            revision=revision,
            local_files_only=local_files_only,
        ).to(self.device)

        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temperal_downsample) if hasattr(self.vae, "temperal_downsample") else 4
        )
        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.temperal_downsample) if hasattr(self.vae, "temperal_downsample") else 8
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).float()
            if getattr(self.vae.config, "latents_mean", None) is not None
            else None
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).float()
            if getattr(self.vae.config, "latents_std", None) is not None
            else None
        )
        self.register_buffer("latents_mean", latents_mean)
        self.register_buffer("latents_std", latents_std)

        transformer_config = load_transformer_config(
            model,
            subfolder="transformer",
            local_files_only=local_files_only,
            revision=revision,
        )

        self.transformer = CosmosPredict25Transformer3DModel(
            od_config=od_config,
            **transformer_config,
        )

        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
            revision=revision,
        )
        if hasattr(self.scheduler, "alphas_cumprod") and isinstance(self.scheduler.alphas_cumprod, torch.Tensor):
            if self.scheduler.alphas_cumprod.is_cuda:
                self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.cpu()
        if hasattr(self.scheduler, "betas") and isinstance(self.scheduler.betas, torch.Tensor):
            if self.scheduler.betas.is_cuda:
                self.scheduler.betas = self.scheduler.betas.cpu()

    # -----------------------------------------------------------------------
    # Text encoding
    # -----------------------------------------------------------------------

    def _get_prompt_embeds(
        self,
        prompt: str | list[str],
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        input_ids_batch = []
        for text in prompt:
            conversations = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant who will provide prompts to an image generator."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                },
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=max_sequence_length,
                truncation=True,
                padding="max_length",
            )
            if isinstance(input_ids, dict):
                input_ids = input_ids["input_ids"]
            input_ids_batch.append(torch.LongTensor(input_ids))

        input_ids_batch = torch.stack(input_ids_batch, dim=0)

        with torch.no_grad():
            outputs = self.text_encoder(input_ids_batch.to(device), output_hidden_states=True)

        hidden_states = outputs.hidden_states
        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            h = hidden_states[layer_idx]
            normalized = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8)
            normalized_hidden_states.append(normalized)

        prompt_embeds = torch.cat(normalized_hidden_states, dim=-1)
        return prompt_embeds.to(dtype=dtype, device=device)

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prompt_embeds = self._get_prompt_embeds(prompt, max_sequence_length, device, dtype)

        negative_prompt_embeds = None
        if negative_prompt is not None:
            negative_prompt_embeds = self._get_prompt_embeds(negative_prompt, max_sequence_length, device, dtype)

        return prompt_embeds, negative_prompt_embeds

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def __call__(
        self,
        request: OmniDiffusionRequest | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        num_inference_steps: int = 36,
        guidance_scale: float = 7.0,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> DiffusionOutput:
        if request is not None:
            prompt = request.prompt
            num_inference_steps = getattr(request, "num_inference_steps", num_inference_steps)
            guidance_scale = self.od_config.guidance_scale or guidance_scale

        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT

        device = self.device
        transformer_dtype = next(self.transformer.parameters()).dtype

        # 1. Encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt if guidance_scale > 1.0 else None,
            device=device,
        )

        # 2. Prepare latents (Text2World: pure noise, no conditioning)
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        num_channels_latents = self.vae.config.z_dim

        latent_shape = (1, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=torch.float32)

        # Text2World: no conditioning frames
        # cond_latent matches latent shape, cond_mask is single-channel binary mask
        cond_latent = torch.zeros_like(latents)
        cond_mask = torch.zeros(1, 1, num_latent_frames, latent_height, latent_width,
                                device=device, dtype=latents.dtype)
        cond_indicator = torch.zeros(1, 1, num_latent_frames, 1, 1, device=device, dtype=transformer_dtype)

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        # 3. Denoising loop
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        gt_velocity = (latents - cond_latent) * cond_mask

        for i, t in enumerate(timesteps):
            sigma_t = (
                torch.tensor(self.scheduler.sigmas[i].item())
                .unsqueeze(0)
                .to(device=device, dtype=transformer_dtype)
            )

            in_latents = cond_mask * cond_latent + (1 - cond_mask) * latents
            in_latents = in_latents.to(transformer_dtype)
            in_timestep = cond_indicator * 0.1 + (1 - cond_indicator) * sigma_t

            noise_pred = self.transformer(
                hidden_states=in_latents,
                condition_mask=cond_mask.to(transformer_dtype),
                timestep=in_timestep,
                encoder_hidden_states=prompt_embeds,
                padding_mask=padding_mask,
            )
            noise_pred = gt_velocity + noise_pred * (1 - cond_mask)

            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                noise_pred_neg = self.transformer(
                    hidden_states=in_latents,
                    condition_mask=cond_mask.to(transformer_dtype),
                    timestep=in_timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    padding_mask=padding_mask,
                )
                noise_pred_neg = gt_velocity + noise_pred_neg * (1 - cond_mask)
                noise_pred = noise_pred + guidance_scale * (noise_pred - noise_pred_neg)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 4. VAE decode
        if self.latents_mean is not None and self.latents_std is not None:
            latents = latents * self.latents_std.to(latents.device, latents.dtype) + self.latents_mean.to(
                latents.device, latents.dtype
            )

        video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

        return DiffusionOutput(output=video)
