"""Profile vLLM-Omni Cosmos transformer with torch profiler."""
import os

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("MODEL", "nvidia/Cosmos-Predict2.5-2B")
REVISION = os.environ.get("REVISION", "diffusers/base/post-trained")
PROFILE_DIR = os.environ.get("PROFILE_DIR", "./profile_traces")

print(f"Profiling model: {MODEL} (revision: {REVISION})")
print(f"Traces will be saved to: {PROFILE_DIR}")

omni = Omni(model=MODEL, revision=REVISION)
params = OmniDiffusionSamplingParams(
    height=704,
    width=1280,
    num_frames=93,
    num_inference_steps=5,  # fewer steps for profiling
    guidance_scale=7.0,
    seed=42,
)

# Warmup
omni.generate(prompts=["warmup"], sampling_params=params)
torch.cuda.synchronize()

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=0, warmup=1, active=2, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(PROFILE_DIR),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(3):
        omni.generate(
            prompts=["A cat walking through a garden."],
            sampling_params=params,
        )
        torch.cuda.synchronize()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print(f"\nProfile traces saved to {PROFILE_DIR}/")
print(f"View with: tensorboard --logdir {PROFILE_DIR}")
