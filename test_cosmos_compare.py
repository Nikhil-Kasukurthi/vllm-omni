"""Compare vLLM-omni Cosmos transformer output against diffusers baseline."""

import json
import os

os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

MODEL_ID = "KyleShao/Cosmos-Predict2.5-2B-Diffusers"
DTYPE = torch.bfloat16

# ── Download ─────────────────────────────────────────────────────────────
print("Downloading transformer weights...")
model_path = snapshot_download(MODEL_ID, allow_patterns=["transformer/*"])
transformer_dir = os.path.join(model_path, "transformer")

with open(os.path.join(transformer_dir, "config.json")) as f:
    config = json.load(f)
print(f"Config: num_heads={config['num_attention_heads']}, "
      f"head_dim={config['attention_head_dim']}, "
      f"layers={config['num_layers']}")

# ── 1. Load diffusers baseline ───────────────────────────────────────────
print("\n=== Loading diffusers baseline ===")
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel

# Instantiate from config (non-standard weight filename in this repo)
diffusers_init_keys = {
    "in_channels", "out_channels", "num_attention_heads", "attention_head_dim",
    "num_layers", "mlp_ratio", "text_embed_dim", "adaln_lora_dim",
    "max_size", "patch_size", "rope_scale", "concat_padding_mask",
    "extra_pos_embed_type", "use_crossattn_projection",
}
ref_config = {k: v for k, v in config.items() if k in diffusers_init_keys}
ref_model = CosmosTransformer3DModel(**ref_config).to(dtype=DTYPE)

# Load weights from safetensors — remap NVIDIA-native names to diffusers names
import re

def remap_nvidia_to_diffusers(name):
    """Remap NVIDIA checkpoint names to diffusers CosmosTransformer3DModel names."""
    if name.endswith("._extra_state"):
        return None
    if name.startswith("net.pos_embedder."):
        return None

    # crossattn_proj: skip (shape mismatch between NVIDIA and diffusers)
    if name.startswith("net.crossattn_proj."):
        return None

    # x_embedder: net.x_embedder.proj.1 -> patch_embed.proj
    m = re.match(r"net\.x_embedder\.proj\.\d+\.(.*)", name)
    if m:
        return f"patch_embed.proj.{m.group(1)}"

    # t_embedder: net.t_embedder.1.* -> time_embed.t_embedder.*
    m = re.match(r"net\.t_embedder\.\d+\.(.*)", name)
    if m:
        return f"time_embed.t_embedder.{m.group(1)}"

    # t_embedding_norm: net.t_embedding_norm.* -> time_embed.norm.*
    m = re.match(r"net\.t_embedding_norm\.(.*)", name)
    if m:
        return f"time_embed.norm.{m.group(1)}"

    # final_layer: net.final_layer.adaln_modulation.{1,2} -> norm_out.linear_{1,2}
    m = re.match(r"net\.final_layer\.adaln_modulation\.(\d+)\.(.*)", name)
    if m:
        return f"norm_out.linear_{m.group(1)}.{m.group(2)}"

    # final_layer linear: net.final_layer.linear.* -> proj_out.*
    m = re.match(r"net\.final_layer\.linear\.(.*)", name)
    if m:
        return f"proj_out.{m.group(1)}"

    # transformer blocks
    m = re.match(r"net\.blocks\.(\d+)\.(.*)", name)
    if not m:
        return name
    idx, rest = m.group(1), m.group(2)
    prefix = f"transformer_blocks.{idx}"

    # AdaLN modulation
    adaln_map = {
        "adaln_modulation_self_attn": "norm1",
        "adaln_modulation_cross_attn": "norm2",
        "adaln_modulation_mlp": "norm3",
    }
    for src, dst in adaln_map.items():
        m2 = re.match(rf"{src}\.(\d+)\.(.*)", rest)
        if m2:
            return f"{prefix}.{dst}.linear_{m2.group(1)}.{m2.group(2)}"

    # Self-attention (diffusers uses separate q/k/v, to_out.0)
    sa_map = {
        "self_attn.q_proj": "attn1.to_q",
        "self_attn.k_proj": "attn1.to_k",
        "self_attn.v_proj": "attn1.to_v",
        "self_attn.output_proj": "attn1.to_out.0",
        "self_attn.q_norm": "attn1.norm_q",
        "self_attn.k_norm": "attn1.norm_k",
    }
    for src, dst in sa_map.items():
        m2 = re.match(rf"{re.escape(src)}\.(.*)", rest)
        if m2:
            return f"{prefix}.{dst}.{m2.group(1)}"

    # Cross-attention
    ca_map = {
        "cross_attn.q_proj": "attn2.to_q",
        "cross_attn.k_proj": "attn2.to_k",
        "cross_attn.v_proj": "attn2.to_v",
        "cross_attn.output_proj": "attn2.to_out.0",
        "cross_attn.q_norm": "attn2.norm_q",
        "cross_attn.k_norm": "attn2.norm_k",
    }
    for src, dst in ca_map.items():
        m2 = re.match(rf"{re.escape(src)}\.(.*)", rest)
        if m2:
            return f"{prefix}.{dst}.{m2.group(1)}"

    # FFN
    m2 = re.match(r"mlp\.layer1\.(.*)", rest)
    if m2:
        return f"{prefix}.ff.net.0.proj.{m2.group(1)}"
    m2 = re.match(r"mlp\.layer2\.(.*)", rest)
    if m2:
        return f"{prefix}.ff.net.2.{m2.group(1)}"

    return name

weight_files = [f for f in os.listdir(transformer_dir) if f.endswith(".safetensors")]
remapped_state_dict = {}
for wf in weight_files:
    sd = load_file(os.path.join(transformer_dir, wf), device="cpu")
    for orig_name, tensor in sd.items():
        new_name = remap_nvidia_to_diffusers(orig_name)
        if new_name is not None:
            # Trim patch_embed if needed
            if new_name == "patch_embed.proj.weight":
                expected = ref_model.patch_embed.proj.weight.shape[1]
                if tensor.shape[1] > expected:
                    tensor = tensor[:, :expected]
            remapped_state_dict[new_name] = tensor

missing, unexpected = ref_model.load_state_dict(remapped_state_dict, strict=False)
print(f"Diffusers model loaded: {sum(p.numel() for p in ref_model.parameters()) / 1e9:.2f}B params")
print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
if missing:
    print(f"  Missing samples: {missing[:5]}")
ref_model.eval()

# ── 2. Load vLLM-omni model ─────────────────────────────────────────────
print("\n=== Loading vLLM-omni model ===")

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
    test_model = CosmosPredict25Transformer3DModel(
        **{k: v for k, v in config.items() if k in known_keys}
    )

    def weight_iterator():
        for wf in sorted(os.listdir(transformer_dir)):
            if wf.endswith(".safetensors"):
                sd = load_file(os.path.join(transformer_dir, wf), device="cpu")
                yield from sd.items()

    loaded = test_model.load_weights(weight_iterator())
    print(f"vLLM-omni model loaded: {len(loaded)} weight tensors")

    # Post-process for CPU GEMM dispatch
    for module in test_model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)

    test_model = test_model.to(dtype=DTYPE)
    test_model.eval()

    # ── 3. Prepare identical inputs ──────────────────────────────────────
    print("\n=== Preparing inputs ===")
    B, C, T, H, W = 1, 16, 2, 16, 16
    text_seq_len = 16
    text_dim = config["text_embed_dim"]

    gen = torch.manual_seed(42)
    hidden_states = torch.randn(B, C, T, H, W, dtype=DTYPE, generator=gen)
    timestep = torch.tensor([0.5], dtype=DTYPE)
    encoder_hidden_states = torch.randn(B, text_seq_len, text_dim, dtype=DTYPE, generator=gen)
    padding_mask = torch.zeros(B, 1, H, W, dtype=DTYPE)

    print(f"  hidden_states:         {list(hidden_states.shape)}")
    print(f"  timestep:              {list(timestep.shape)}")
    print(f"  encoder_hidden_states: {list(encoder_hidden_states.shape)}")

    # ── 4. Run both models ───────────────────────────────────────────────
    print("\n=== Running diffusers baseline ===")
    with torch.no_grad():
        ref_out = ref_model(
            hidden_states=hidden_states.clone(),
            timestep=timestep.clone(),
            encoder_hidden_states=encoder_hidden_states.clone(),
            condition_mask=None,
            padding_mask=padding_mask.clone(),
            return_dict=False,
        )[0]
    print(f"  Output shape: {list(ref_out.shape)}, mean={ref_out.mean().item():.4f}")

    print("\n=== Running vLLM-omni model ===")
    with torch.no_grad():
        test_out = test_model(
            hidden_states=hidden_states.clone(),
            timestep=timestep.clone(),
            encoder_hidden_states=encoder_hidden_states.clone(),
            condition_mask=None,
            padding_mask=padding_mask.clone(),
        )
    print(f"  Output shape: {list(test_out.shape)}, mean={test_out.mean().item():.4f}")

    # ── 5. Compare ───────────────────────────────────────────────────────
    print("\n=== Comparison ===")
    abs_diff = (ref_out - test_out).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    ref_norm = ref_out.abs().mean().item()
    rel_diff = mean_diff / (ref_norm + 1e-8)

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_out.flatten().unsqueeze(0).float(),
        test_out.flatten().unsqueeze(0).float(),
    ).item()

    print(f"  Max absolute diff:    {max_diff:.6f}")
    print(f"  Mean absolute diff:   {mean_diff:.6f}")
    print(f"  Relative diff:        {rel_diff:.6f}")
    print(f"  Cosine similarity:    {cos_sim:.6f}")
    print(f"  Shapes match:         {ref_out.shape == test_out.shape}")

    # Tolerance: DistributedRMSNorm vs nn.RMSNorm causes small numerical diffs
    # that accumulate across 28 blocks. cos_sim > 0.995 is expected.
    if cos_sim > 0.995:
        print("\n  PASS -- outputs match (small diffs from RMSNorm implementation variance)")
    elif cos_sim > 0.95:
        print("\n  CLOSE -- outputs are similar but not identical (check weight mapping)")
    else:
        print("\n  FAIL -- outputs diverge significantly")

cleanup_dist_env_and_memory()
