#!/bin/bash
# Save one sample mp4 per backend at small + full configs for visual diffing.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SMALL="--height 480 --width 832 --num-frames 33 --num-inference-steps 10"
FULL="--height 704 --width 1280 --num-frames 93 --num-inference-steps 36"

mkdir -p samples

for cfg_name in small full; do
    if [ "$cfg_name" = "small" ]; then args="$SMALL"; else args="$FULL"; fi
    for backend in diffusers vllm_omni nvidia; do
        echo "=== ${backend} ${cfg_name} ==="
        python "$SCRIPT_DIR/save_sample.py" --backend $backend \
            --out samples/${backend}_${cfg_name}.mp4 $args
    done
    echo "=== vllm_omni cache_dit ${cfg_name} ==="
    python "$SCRIPT_DIR/save_sample.py" --backend vllm_omni --cache-backend cache_dit \
        --out samples/vllm_cachedit_${cfg_name}.mp4 $args
done
