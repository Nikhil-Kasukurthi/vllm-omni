#!/bin/bash
# Run the full Cosmos Predict 2.5 benchmark suite: diffusers baseline vs vLLM-Omni vs NVIDIA native
set -uo pipefail  # don't -e: keep going past per-bench failures so other benches still produce numbers

export SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export SMALL_ARGS="--height 480 --width 832 --num-frames 33 --num-inference-steps 10 --warmup 3 --repeats 5"
export FULL_ARGS="--height 704 --width 1280 --num-frames 93 --num-inference-steps 36 --warmup 3 --repeats 5"

echo ""
echo "============================================"
echo " Quick sanity test (small resolution)"
echo "============================================"

echo ""
echo "--- Diffusers baseline (small) ---"
python "$SCRIPT_DIR/bench_diffusers_baseline.py" $SMALL_ARGS 2>&1 | tee results_diffusers_small.txt

echo ""
echo "--- vLLM-Omni (small) ---"
python "$SCRIPT_DIR/bench_vllm_omni.py" $SMALL_ARGS 2>&1 | tee results_vllm_small.txt

echo ""
echo "--- NVIDIA native (small) ---"
python "$SCRIPT_DIR/bench_nvidia_native.py" $SMALL_ARGS 2>&1 | tee results_nvidia_small.txt

echo ""
echo "============================================"
echo " Full benchmark"
echo "============================================"

echo ""
echo "--- Diffusers baseline (full) ---"
python "$SCRIPT_DIR/bench_diffusers_baseline.py" $FULL_ARGS 2>&1 | tee results_diffusers_full.txt

echo ""
echo "--- vLLM-Omni (full) ---"
python "$SCRIPT_DIR/bench_vllm_omni.py" $FULL_ARGS 2>&1 | tee results_vllm_full.txt

echo ""
echo "--- vLLM-Omni + Cache-DiT (full) ---"
python "$SCRIPT_DIR/bench_vllm_omni.py" $FULL_ARGS --cache-backend cache_dit 2>&1 | tee results_vllm_cachedit.txt

echo ""
echo "--- NVIDIA native (full) ---"
python "$SCRIPT_DIR/bench_nvidia_native.py" $FULL_ARGS 2>&1 | tee results_nvidia_full.txt

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
