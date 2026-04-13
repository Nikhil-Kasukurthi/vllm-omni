"""Compare vLLM-omni Cosmos transformer output against diffusers baseline."""

import json
import os

os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

MODEL_ID = "nvidia/Cosmos-Predict2.5-2B"
REVISION = "diffusers/base/post-trained"
DTYPE = torch.bfloat16

# ── Download ─────────────────────────────────────────────────────────────
print(f"Downloading transformer weights from {MODEL_ID} (revision: {REVISION})...")
model_path = snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=["transformer/*"])
transformer_dir = os.path.join(model_path, "transformer")

with open(os.path.join(transformer_dir, "config.json")) as f:
    config = json.load(f)
print(f"Config: num_heads={config['num_attention_heads']}, "
      f"head_dim={config['attention_head_dim']}, "
      f"layers={config['num_layers']}")

# ── 1. Load diffusers baseline ───────────────────────────────────────────
print("\n=== Loading diffusers baseline ===")
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel

# Official NVIDIA diffusers branch has standard naming — load directly
diffusers_init_keys = {
    "in_channels", "out_channels", "num_attention_heads", "attention_head_dim",
    "num_layers", "mlp_ratio", "text_embed_dim", "adaln_lora_dim",
    "max_size", "patch_size", "rope_scale", "concat_padding_mask",
    "extra_pos_embed_type", "use_crossattn_projection",
    "crossattn_proj_in_channels", "encoder_hidden_states_channels",
}
ref_config = {k: v for k, v in config.items() if k in diffusers_init_keys}
# Disable crossattn_proj for testing — we pass pre-projected encoder_hidden_states
ref_config["use_crossattn_projection"] = False
ref_model = CosmosTransformer3DModel(**ref_config).to(dtype=DTYPE)

weight_files = [f for f in os.listdir(transformer_dir) if f.endswith(".safetensors")]
state_dict = {}
for wf in weight_files:
    state_dict.update(load_file(os.path.join(transformer_dir, wf), device="cpu"))

missing, unexpected = ref_model.load_state_dict(state_dict, strict=False)
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
    B, C_latent, T, H, W = 1, 16, 2, 16, 16
    text_seq_len = 16
    text_dim = config["text_embed_dim"]

    gen = torch.manual_seed(42)
    hidden_states = torch.randn(B, C_latent, T, H, W, dtype=DTYPE, generator=gen)
    timestep = torch.tensor([0.5], dtype=DTYPE)
    encoder_hidden_states = torch.randn(B, text_seq_len, text_dim, dtype=DTYPE, generator=gen)
    padding_mask = torch.zeros(B, 1, H, W, dtype=DTYPE)
    condition_mask = torch.zeros(B, 1, T, H, W, dtype=DTYPE)

    print(f"  hidden_states:         {list(hidden_states.shape)}")
    print(f"  condition_mask:        {list(condition_mask.shape)}")
    print(f"  timestep:              {list(timestep.shape)}")
    print(f"  encoder_hidden_states: {list(encoder_hidden_states.shape)}")

    # ── 4. Run both models ───────────────────────────────────────────────
    print("\n=== Running diffusers baseline ===")
    with torch.no_grad():
        ref_out = ref_model(
            hidden_states=hidden_states.clone(),
            timestep=timestep.clone(),
            encoder_hidden_states=encoder_hidden_states.clone(),
            condition_mask=condition_mask.clone(),
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
            condition_mask=condition_mask.clone(),
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
