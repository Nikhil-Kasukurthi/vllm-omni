"""Test: download Cosmos-Predict2.5-2B, load weights, run forward pass."""

import json
import os

os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

MODEL_ID = "KyleShao/Cosmos-Predict2.5-2B-Diffusers"

# 1. Download transformer weights (cached after first run)
print("Downloading transformer weights...")
model_path = snapshot_download(MODEL_ID, allow_patterns=["transformer/*"])
print(f"Downloaded to: {model_path}")

# 2. Load config
config_path = os.path.join(model_path, "transformer", "config.json")
with open(config_path) as f:
    config = json.load(f)

# 3. Init distributed env
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29501")
init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")

device_config = DeviceConfig(device="cpu")

with set_current_vllm_config(VllmConfig(device_config=device_config)):
    initialize_model_parallel()

    import vllm_omni.platforms as platforms
    platforms.current_omni_platform.get_diffusion_attn_backend_cls = (
        lambda **kw: "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"
    )

    # Patch SDPA attention to work on CPU (no platform-specific dispatch)
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
    SDPAImpl.forward = SDPAImpl.forward_cuda

    from vllm_omni.diffusion.models.cosmos_predict2_5.cosmos_predict2_5_transformer import (
        CosmosPredict25Transformer3DModel,
    )

    known_keys = {
        "in_channels", "out_channels", "num_attention_heads", "attention_head_dim",
        "num_layers", "mlp_ratio", "text_embed_dim", "adaln_lora_dim",
        "max_size", "patch_size", "rope_scale", "concat_padding_mask",
        "extra_pos_embed_type",
    }
    filtered_config = {k: v for k, v in config.items() if k in known_keys}

    model = CosmosPredict25Transformer3DModel(**filtered_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created — {total_params / 1e9:.2f}B parameters")

    # 4. Load weights
    transformer_dir = os.path.join(model_path, "transformer")
    weight_files = sorted(f for f in os.listdir(transformer_dir) if f.endswith(".safetensors"))

    def weight_iterator():
        for wf in weight_files:
            state_dict = load_file(os.path.join(transformer_dir, wf), device="cpu")
            yield from state_dict.items()

    print("Loading weights...")
    loaded = model.load_weights(weight_iterator())
    print(f"Loaded {len(loaded)} weight tensors")

    # 5. Post-process weights for CPU dispatch, then convert to bfloat16
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)

    model = model.to(dtype=torch.bfloat16)
    model.eval()

    # 6. Forward pass with small dummy input
    # Config: in_channels=16, patch_size=[1,2,2], text_embed_dim=1024
    # H,W must be divisible by patch spatial size (2), T by temporal (1)
    B, C, T, H, W = 1, 16, 2, 16, 16
    text_seq_len = 16
    text_embed_dim = config["text_embed_dim"]  # 1024

    hidden_states = torch.randn(B, C, T, H, W, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], dtype=torch.bfloat16)  # scalar sigma
    encoder_hidden_states = torch.randn(B, text_seq_len, text_embed_dim, dtype=torch.bfloat16)
    padding_mask = torch.zeros(B, 1, H, W, dtype=torch.bfloat16)

    print(f"\nRunning forward pass...")
    print(f"  hidden_states:         {list(hidden_states.shape)}")
    print(f"  timestep:              {list(timestep.shape)}")
    print(f"  encoder_hidden_states: {list(encoder_hidden_states.shape)}")
    print(f"  padding_mask:          {list(padding_mask.shape)}")

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            condition_mask=None,
            padding_mask=padding_mask,
        )

    print(f"\nForward pass succeeded!")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Output mean:  {output.mean().item():.4f}")

cleanup_dist_env_and_memory()
