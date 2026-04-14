"""Benchmark: diffusers baseline for Cosmos Predict 2.5 2B (single-GPU)."""
import argparse
import time

import torch
from diffusers import Cosmos2_5_PredictBasePipeline


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
    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype)
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
            pipe(
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
