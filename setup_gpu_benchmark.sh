#!/bin/bash
# setup_gpu_benchmark.sh
# Sets up a cloud GPU instance (RunPod, Vast.ai, Lambda, etc.) for
# benchmarking vLLM-Omni vs diffusers baseline on Cosmos Predict 2.5.
#
# Tested on: Ubuntu 22.04 + CUDA 12.x + Python 3.12
# Recommended GPU: A100 80GB or H100 (2B model fits on a single GPU)
#
# Usage:
#   cd /workspace/vllm-omni
#   chmod +x setup_gpu_benchmark.sh
#   ./setup_gpu_benchmark.sh

set -euo pipefail

COSMOS_MODEL="nvidia/Cosmos-Predict2.5-2B"
COSMOS_REVISION="diffusers/base/post-trained"

echo "============================================"
echo " vLLM-Omni GPU Benchmark Setup"
echo "============================================"
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

# ── 3. Create venv and install all dependencies ─────────────────────────
echo ">>> Running uv sync..."
uv sync
source .venv/bin/activate
echo "Python: $(python --version)"

# ── 4. Install PyTorch + vLLM (GPU build) ────────────────────────────────
echo ">>> Installing PyTorch and vLLM..."
CUDA_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ -n "$CUDA_MAJOR" ]; then
    echo "  Detected CUDA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi
# Install torch + torchvision + torchaudio together to keep versions in sync
uv pip install torch torchvision torchaudio --force-reinstall
# Upgrade NCCL to match the version PyTorch was built against
uv pip install --upgrade nvidia-nccl-cu12
# Pin transformers to v4 — diffusers is incompatible with transformers v5
uv pip install 'transformers>=4.51.0,<5.0.0'
uv pip install vllm
uv pip install peft==0.17.0

# ── 5. Install vLLM-Omni (editable) ─────────────────────────────────────
echo ">>> Installing vllm-omni..."
uv pip install -e .


# ── 6. Install profiling tools ───────────────────────────────────────────
echo ">>> Installing profiling tools..."
uv pip install \
    py-spy \
    torch-tb-profiler \
    tensorboard \
    memory_profiler \
    gdown \
    imageio[ffmpeg] \
    cosmos_guardrail

# ── 7. Download Cosmos Predict 2.5 2B model ─────────────────────────────
echo ">>> Pre-downloading model: $COSMOS_MODEL (revision: $COSMOS_REVISION)"
python -c "
from huggingface_hub import snapshot_download
print('Downloading full model (transformer + vae + text_encoder + tokenizer + scheduler)...')
path = snapshot_download('$COSMOS_MODEL', revision='$COSMOS_REVISION')
print(f'Model cached at: {path}')
"

# ── 8. Print summary ────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Quick test (small resolution, fast):"
echo "  python benchmarks/cosmos/bench_diffusers_baseline.py --height 480 --width 832 --num-frames 33 --num-inference-steps 5 --repeats 1"
echo "  python benchmarks/cosmos/bench_vllm_omni.py --height 480 --width 832 --num-frames 33 --num-inference-steps 5 --repeats 1"
echo ""
echo "Full benchmark suite:"
echo "  bash benchmarks/cosmos/run_all.sh"
echo ""
echo "Profile with torch profiler:"
echo "  python benchmarks/cosmos/profile_vllm_omni.py"
echo "  tensorboard --logdir ./profile_traces --bind_all"
echo ""
echo "Profile with py-spy (flame graph):"
echo "  py-spy record -o profile.svg -- python benchmarks/cosmos/bench_vllm_omni.py --repeats 1"
echo ""
