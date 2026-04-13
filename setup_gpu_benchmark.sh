#!/bin/bash
# setup_gpu_benchmark.sh
# Sets up a cloud GPU instance (RunPod, Vast.ai, Lambda, etc.) for
# benchmarking vLLM-Omni vs diffusers baseline on Cosmos Predict 2.5.
#
# Tested on: Ubuntu 22.04 + CUDA 12.x + Python 3.12
# Recommended GPU: A100 80GB or H100 (2B model fits on a single GPU)
#
# Usage:
#   chmod +x setup_gpu_benchmark.sh
#   ./setup_gpu_benchmark.sh

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-https://github.com/vllm-project/vllm-omni.git}"
VLLM_OMNI_BRANCH="${VLLM_OMNI_BRANCH:-main}"
COSMOS_MODEL="nvidia/Cosmos-Predict2.5-2B"
COSMOS_REVISION="diffusers/base/post-trained"

echo "============================================"
echo " vLLM-Omni GPU Benchmark Setup"
echo "============================================"
echo "Work directory:  $WORKDIR"
echo "Branch:          $VLLM_OMNI_BRANCH"
echo ""

# ── 0. Sanity checks ────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. This script requires a CUDA GPU."
    exit 1
fi

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── 1. System dependencies ──────────────────────────────────────────────
echo ">>> Installing system packages..."
apt-get update -qq
apt-get install -y -qq git sox libsox-fmt-all jq curl wget htop nvtop 2>/dev/null || true

# ── 2. Install uv (fast Python package manager) ─────────────────────────
if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# ── 3. Create Python environment ────────────────────────────────────────
echo ">>> Creating Python 3.12 venv..."
cd "$WORKDIR"
uv venv --python 3.12 .venv
source .venv/bin/activate
echo "Python: $(python --version)"

# ── 4. Install vLLM (GPU build) ─────────────────────────────────────────
echo ">>> Installing vLLM..."
uv pip install vllm

# ── 5. Clone and install vLLM-Omni ──────────────────────────────────────
echo ">>> Cloning vllm-omni ($VLLM_OMNI_BRANCH)..."
if [ ! -d "$WORKDIR/vllm-omni" ]; then
    git clone --branch "$VLLM_OMNI_BRANCH" "$VLLM_OMNI_REPO" "$WORKDIR/vllm-omni"
fi
cd "$WORKDIR/vllm-omni"

echo ">>> Installing vllm-omni..."
uv pip install -e .

# ── 6. Install profiling and benchmark tools ─────────────────────────────
echo ">>> Installing profiling tools..."
uv pip install \
    py-spy \
    torch-tb-profiler \
    tensorboard \
    memory_profiler \
    gdown \
    imageio[ffmpeg]

# ── 7. Download Cosmos Predict 2.5 2B model ─────────────────────────────
echo ">>> Pre-downloading model: $COSMOS_MODEL (revision: $COSMOS_REVISION)"
python -c "
from huggingface_hub import snapshot_download
print('Downloading full model (transformer + vae + text_encoder + tokenizer + scheduler)...')
path = snapshot_download('$COSMOS_MODEL', revision='$COSMOS_REVISION')
print(f'Model cached at: {path}')
"

# ── 8. Write benchmark scripts ──────────────────────────────────────────
echo ">>> Writing benchmark scripts..."

mkdir -p "$WORKDIR/benchmarks"

# 8a. Diffusers baseline benchmark
cat > "$WORKDIR/benchmarks/bench_diffusers_baseline.py" << 'PYEOF'
"""Benchmark: diffusers baseline for Cosmos Predict 2.5 2B (single-GPU)."""
import argparse
import time
import torch
from diffusers import CosmosPredict25Pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Cosmos-Predict2.5-2B")
    parser.add_argument("--revision", default="diffusers/base/post-trained")
    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-inference-steps", type=int, default=36)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Loading diffusers pipeline: {args.model} (revision: {args.revision})")
    pipe = CosmosPredict25Pipeline.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype)
    pipe = pipe.to("cuda")

    gen = torch.Generator(device="cuda").manual_seed(42)

    # Warmup
    print(f"Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        with torch.no_grad():
            pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                generator=gen,
            )
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({args.repeats} runs)...")
    times = []
    for i in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                generator=gen,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n=== Diffusers Baseline Results ===")
    print(f"  Resolution:   {args.height}x{args.width}, {args.num_frames} frames")
    print(f"  Steps:        {args.num_inference_steps}")
    print(f"  Mean latency: {sum(times)/len(times):.2f}s")
    print(f"  Min latency:  {min(times):.2f}s")
    print(f"  Max latency:  {max(times):.2f}s")
    print(f"  Peak GPU mem: {peak_mem:.1f} GB")

if __name__ == "__main__":
    main()
PYEOF

# 8b. vLLM-Omni offline benchmark
cat > "$WORKDIR/benchmarks/bench_vllm_omni.py" << 'PYEOF'
"""Benchmark: vLLM-Omni for Cosmos Predict 2.5 2B (single-GPU, offline)."""
import argparse
import time
import torch
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Cosmos-Predict2.5-2B")
    parser.add_argument("--revision", default="diffusers/base/post-trained")
    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-inference-steps", type=int, default=36)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cfg-parallel-size", type=int, default=1)
    parser.add_argument("--cache-backend", default=None, help="e.g. cache_dit")
    args = parser.parse_args()

    print(f"Loading vLLM-Omni: {args.model} (revision: {args.revision})")
    omni_kwargs = dict(
        model=args.model,
        revision=args.revision,
        cfg_parallel_size=args.cfg_parallel_size,
    )
    if args.cache_backend:
        omni_kwargs["cache_backend"] = args.cache_backend

    omni = Omni(**omni_kwargs)

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=7.0,
        seed=42,
    )

    # Warmup
    print(f"Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        omni.generate(prompts=[args.prompt], sampling_params=sampling_params)
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({args.repeats} runs)...")
    times = []
    for i in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = omni.generate(prompts=[args.prompt], sampling_params=sampling_params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n=== vLLM-Omni Results ===")
    print(f"  Resolution:   {args.height}x{args.width}, {args.num_frames} frames")
    print(f"  Steps:        {args.num_inference_steps}")
    print(f"  CFG parallel: {args.cfg_parallel_size}")
    print(f"  Cache:        {args.cache_backend or 'none'}")
    print(f"  Mean latency: {sum(times)/len(times):.2f}s")
    print(f"  Min latency:  {min(times):.2f}s")
    print(f"  Max latency:  {max(times):.2f}s")
    print(f"  Peak GPU mem: {peak_mem:.1f} GB")

if __name__ == "__main__":
    main()
PYEOF

# 8c. Torch profiler script
cat > "$WORKDIR/benchmarks/profile_vllm_omni.py" << 'PYEOF'
"""Profile vLLM-Omni Cosmos transformer with torch profiler."""
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("MODEL", "nvidia/Cosmos-Predict2.5-2B")
PROFILE_DIR = os.environ.get("PROFILE_DIR", "./profile_traces")

print(f"Profiling model: {MODEL}")
print(f"Traces will be saved to: {PROFILE_DIR}")

omni = Omni(model=MODEL)
params = OmniDiffusionSamplingParams(
    height=704, width=1280, num_frames=93,
    num_inference_steps=5,  # fewer steps for profiling
    guidance_scale=7.0, seed=42,
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
        omni.generate(prompts=["A cat walking through a garden."], sampling_params=params)
        torch.cuda.synchronize()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print(f"\nProfile traces saved to {PROFILE_DIR}/")
print(f"View with: tensorboard --logdir {PROFILE_DIR}")
PYEOF

# 8d. Runner script
cat > "$WORKDIR/benchmarks/run_all.sh" << 'RUNEOF'
#!/bin/bash
# Run the full benchmark suite: diffusers baseline vs vLLM-Omni
set -euo pipefail

cd /workspace
source .venv/bin/activate

SMALL_ARGS="--height 480 --width 832 --num-frames 33 --num-inference-steps 10 --warmup 1 --repeats 3"
FULL_ARGS="--height 704 --width 1280 --num-frames 93 --num-inference-steps 36 --warmup 1 --repeats 3"

echo ""
echo "============================================"
echo " Quick sanity test (small resolution)"
echo "============================================"

echo ""
echo "--- Diffusers baseline (small) ---"
python benchmarks/bench_diffusers_baseline.py $SMALL_ARGS 2>&1 | tee results_diffusers_small.txt

echo ""
echo "--- vLLM-Omni (small) ---"
python benchmarks/bench_vllm_omni.py $SMALL_ARGS 2>&1 | tee results_vllm_small.txt

echo ""
echo "============================================"
echo " Full benchmark"
echo "============================================"

echo ""
echo "--- Diffusers baseline (full) ---"
python benchmarks/bench_diffusers_baseline.py $FULL_ARGS 2>&1 | tee results_diffusers_full.txt

echo ""
echo "--- vLLM-Omni (full) ---"
python benchmarks/bench_vllm_omni.py $FULL_ARGS 2>&1 | tee results_vllm_full.txt

echo ""
echo "--- vLLM-Omni + Cache-DiT (full) ---"
python benchmarks/bench_vllm_omni.py $FULL_ARGS --cache-backend cache_dit 2>&1 | tee results_vllm_cachedit.txt

echo ""
echo "============================================"
echo " Results Summary"
echo "============================================"
echo ""
for f in results_*.txt; do
    echo "=== $f ==="
    grep -E "Mean latency|Peak GPU mem|Resolution" "$f" || true
    echo ""
done
RUNEOF
chmod +x "$WORKDIR/benchmarks/run_all.sh"

# ── 9. Print summary ────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Activate the environment:"
echo "  cd $WORKDIR && source .venv/bin/activate"
echo ""
echo "Quick test (small resolution, fast):"
echo "  python benchmarks/bench_diffusers_baseline.py --height 480 --width 832 --num-frames 33 --num-inference-steps 5 --repeats 1"
echo "  python benchmarks/bench_vllm_omni.py --height 480 --width 832 --num-frames 33 --num-inference-steps 5 --repeats 1"
echo ""
echo "Full benchmark suite:"
echo "  bash benchmarks/run_all.sh"
echo ""
echo "Profile with torch profiler:"
echo "  python benchmarks/profile_vllm_omni.py"
echo "  tensorboard --logdir ./profile_traces --bind_all"
echo ""
echo "Profile with py-spy (flame graph):"
echo "  py-spy record -o profile.svg -- python benchmarks/bench_vllm_omni.py --repeats 1"
echo ""
echo "Use the repo's built-in serving benchmark:"
echo "  # Start server:"
echo "  vllm serve $COSMOS_MODEL --omni --port 8091 &"
echo "  # Run benchmark:"
echo "  python vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py \\"
echo "      --backend v1/videos --dataset random --task t2v --num-prompts 5 \\"
echo "      --max-concurrency 1 --enable-negative-prompt"
echo ""
