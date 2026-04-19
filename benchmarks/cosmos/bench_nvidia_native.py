"""Benchmark: NVIDIA native Cosmos Predict 2.5 2B pipeline (single-GPU)."""
import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="2B/post-trained")
    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-inference-steps", type=int, default=36)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--disable-guardrails", action="store_true", default=True)
    args = parser.parse_args()

    resolution = f"{args.height},{args.width}"

    from pathlib import Path
    from cosmos_predict2.config import SetupArguments, InferenceArguments

    # Resolve HF-cached checkpoint path to avoid S3 fallback
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(
        repo_id="nvidia/Cosmos-Predict2.5-2B",
        filename="base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
        revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
    )
    setup = SetupArguments(
        model=args.model,
        checkpoint_path=ckpt_path,
        output_dir=Path("/tmp/cosmos_bench_output"),
        disable_guardrails=args.disable_guardrails,
    )

    # Patch checkpoint_db to handle missing HF revisions by falling back to main
    import cosmos_predict2._src.imaginaire.utils.checkpoint_db as _cdb
    _orig_hf_download = _cdb._hf_download
    def _patched_hf_download(cmd_args):
        import subprocess
        try:
            return _orig_hf_download(cmd_args)
        except subprocess.CalledProcessError:
            # Retry with the known-good revision
            from huggingface_hub import hf_hub_download
            repo = cmd_args[cmd_args.index("download") + 1]
            filename = cmd_args[-1]
            rev_idx = cmd_args.index("--revision") + 1 if "--revision" in cmd_args else None
            return hf_hub_download(repo, filename, revision="15a82a2ec231bc318692aa0456a36537c806e7d4")
    _cdb._hf_download = _patched_hf_download

    from cosmos_predict2.inference import Inference
    print(f"Loading NVIDIA native pipeline: {args.model}")
    t_load_start = time.perf_counter()
    inference = Inference(setup)
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {t_load:.1f}s")

    sample = InferenceArguments(
        name="benchmark",
        prompt=args.prompt,
        inference_type="text2world",
        num_output_frames=args.num_frames,
        num_steps=args.num_inference_steps,
        seed=args.seed,
        guidance=args.guidance,
        resolution=resolution,
    )

    def _run_inference():
        inference.pipe.generate_vid2world(
            prompt=sample.prompt,
            input_path=None,
            guidance=sample.guidance,
            num_video_frames=sample.num_output_frames,
            num_latent_conditional_frames=0,
            resolution=sample.resolution,
            seed=sample.seed,
            negative_prompt=sample.negative_prompt,
            num_steps=sample.num_steps,
        )

    # Warmup
    print(f"Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        _run_inference()
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({args.repeats} runs)...")
    times = []
    for i in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_inference()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n=== NVIDIA Native Pipeline Results ===")
    print(f"  Resolution:   {resolution}, {args.num_frames} frames")
    print(f"  Steps:        {args.num_inference_steps}")
    print(f"  Guidance:     {args.guidance}")
    print(f"  Mean latency: {sum(times)/len(times):.2f}s")
    print(f"  Min latency:  {min(times):.2f}s")
    print(f"  Max latency:  {max(times):.2f}s")
    print(f"  Peak GPU mem: {peak_mem:.1f} GB")


if __name__ == "__main__":
    main()
