"""Generate one Cosmos Predict 2.5 video per backend at a fixed seed for visual diff.

Usage:
    python save_sample.py --backend diffusers --out samples/diffusers_small.mp4 \
        --height 480 --width 832 --num-frames 33 --num-inference-steps 10
    python save_sample.py --backend vllm_omni  --out samples/vllm_small.mp4 ...
    python save_sample.py --backend nvidia     --out samples/nvidia_small.mp4 ...
"""
import argparse
from pathlib import Path

import numpy as np
import torch


PROMPT_DEFAULT = "A serene lakeside sunrise with mist over the water."


def _save_pil_frames(frames, out_path: Path, fps: int = 24) -> None:
    from diffusers.utils import export_to_video
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=fps)


def _save_any(obj, out_path: Path, fps: int = 24) -> None:
    """Save a video given either a list-of-PIL, a single tensor/ndarray, or a list of one tensor."""
    from PIL import Image as PILImage

    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], PILImage.Image):
        _save_pil_frames(obj, out_path, fps)
        return
    if isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]
    if isinstance(obj, torch.Tensor):
        _save_tensor_video(obj, out_path, fps)
        return
    if isinstance(obj, np.ndarray):
        _save_tensor_video(torch.from_numpy(obj), out_path, fps)
        return
    raise TypeError(f"Don't know how to save object of type {type(obj)}")


def _save_tensor_video(video: torch.Tensor, out_path: Path, fps: int = 24) -> None:
    """video: any of (B,C,T,H,W) / (C,T,H,W) / (B,T,H,W,C) / (T,H,W,C); values in [-1,1] or [0,1]."""
    import imageio.v3 as iio

    if video.dim() == 5:
        video = video[0]
    # Now (C, T, H, W) or (T, H, W, C). Detect by last-dim size.
    arr = video.detach().to(torch.float32).cpu().numpy().copy()
    if arr.shape[-1] in (1, 3, 4):
        pass  # already (T, H, W, C)
    else:
        arr = np.transpose(arr, (1, 2, 3, 0))  # (C, T, H, W) -> (T, H, W, C)
    if arr.min() < -0.01:
        arr = (arr + 1.0) * 0.5
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out_path), arr, fps=fps, codec="libx264")


def run_diffusers(args):
    from diffusers import Cosmos2_5_PredictBasePipeline

    # Stub safety checker to avoid Aegis download (matches bench).
    import diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict as mod

    class _NoOp:
        def check_text_safety(self, *_a, **_kw): return True
        def check_video_safety(self, video, *_a, **_kw): return video
        def to(self, *_a, **_kw): return self
    mod.CosmosSafetyChecker = _NoOp

    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16,
    ).to("cuda")
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    out = pipe(
        prompt=args.prompt,
        height=args.height, width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance,
        generator=gen,
    )
    frames = out.frames[0]
    _save_any(frames, Path(args.out))


def run_vllm_omni(args):
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni_kwargs = dict(model=args.model, revision=args.revision, cfg_parallel_size=1)
    if args.cache_backend:
        omni_kwargs["cache_backend"] = args.cache_backend
    omni = Omni(**omni_kwargs)
    sp = OmniDiffusionSamplingParams(
        height=args.height, width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )
    results = omni.generate(prompts=[args.prompt], sampling_params_list=sp)
    result = results[0]
    img0 = result.images[0] if result.images else None
    img0_info = type(img0).__name__
    if hasattr(img0, "shape"):
        img0_info += f" shape={tuple(img0.shape)}"
    if hasattr(img0, "dtype"):
        img0_info += f" dtype={img0.dtype}"
    if hasattr(img0, "size") and not hasattr(img0, "shape"):
        img0_info += f" size={img0.size}"
    if hasattr(img0, "mode"):
        img0_info += f" mode={img0.mode}"
    print(f"[vllm_omni] len(images)={len(result.images)} images[0]={img0_info} "
          f"peak_mem_mb={result.peak_memory_mb:.0f}")
    _save_any(result.images, Path(args.out))


def run_nvidia(args):
    from huggingface_hub import hf_hub_download
    from cosmos_predict2.config import SetupArguments, InferenceArguments
    import cosmos_predict2._src.imaginaire.utils.checkpoint_db as _cdb

    ckpt_path = hf_hub_download(
        repo_id="nvidia/Cosmos-Predict2.5-2B",
        filename="base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
        revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
    )
    setup = SetupArguments(
        model="2B/post-trained",
        checkpoint_path=ckpt_path,
        output_dir=Path("/tmp/cosmos_sample_output"),
        disable_guardrails=True,
    )

    _orig = _cdb._hf_download
    def _patched(cmd_args):
        import subprocess
        try:
            return _orig(cmd_args)
        except subprocess.CalledProcessError:
            from huggingface_hub import hf_hub_download as _h
            repo = cmd_args[cmd_args.index("download") + 1]
            return _h(repo, cmd_args[-1], revision="15a82a2ec231bc318692aa0456a36537c806e7d4")
    _cdb._hf_download = _patched

    from cosmos_predict2.inference import Inference
    inference = Inference(setup)

    sample = InferenceArguments(
        name="sample",
        prompt=args.prompt,
        inference_type="text2world",
        num_output_frames=args.num_frames,
        num_steps=args.num_inference_steps,
        seed=args.seed,
        guidance=args.guidance,
        resolution=f"{args.height},{args.width}",
    )
    video = inference.pipe.generate_vid2world(
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
    print(f"[nvidia] video shape={tuple(video.shape)} dtype={video.dtype} "
          f"min={float(video.min()):.3f} max={float(video.max()):.3f}")
    _save_any(video, Path(args.out))


BACKENDS = {"diffusers": run_diffusers, "vllm_omni": run_vllm_omni, "nvidia": run_nvidia}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=list(BACKENDS))
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="nvidia/Cosmos-Predict2.5-2B")
    p.add_argument("--revision", default="diffusers/base/post-trained")
    p.add_argument("--prompt", default=PROMPT_DEFAULT)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-backend", default=None, help="vllm_omni only, e.g. cache_dit")
    args = p.parse_args()
    BACKENDS[args.backend](args)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
