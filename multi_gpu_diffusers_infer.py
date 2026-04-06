#!/usr/bin/env python3
"""Multi-GPU sharded inference for Z-Image via diffusers, with optional quantization.

Example:
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 uv run python multi_gpu_diffusers_infer.py \
    --model-path ckpts/Z-Image-Turbo --gpu-memory-gb 6 --height 512 --width 512

8-bit quantization (requires bitsandbytes):
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 uv run python multi_gpu_diffusers_infer.py \
    --model-path ckpts/Z-Image-Turbo --quantization 8bit --dtype fp16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sharded multi-GPU inference for Z-Image (diffusers)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ckpts/Z-Image-Turbo",
        help="Local model path or HF repo id",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic portrait of a traveler in a neon-lit ancient alley, ultra detailed",
    )
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=9)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu-memory-gb", type=int, default=6, help="Per-visible-GPU memory cap in GiB"
    )
    parser.add_argument(
        "--cpu-memory-gb",
        type=int,
        default=0,
        help="CPU memory cap in GiB for offload. Set 0 to disable CPU pool in max_memory.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--device-map", choices=["balanced", "cuda", "cpu"], default="balanced"
    )
    parser.add_argument(
        "--attention-backend",
        choices=["default", "flash", "_flash_3"],
        default="default",
    )
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Quantize transformer weights with bitsandbytes",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
        help="4-bit quantization type when --quantization 4bit",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Compute dtype for 4-bit quantized linear ops",
    )
    parser.add_argument(
        "--sanitize-output",
        action="store_true",
        help="Replace NaN/Inf in output image tensor before saving",
    )
    parser.add_argument("--output", type=str, default="outputs/multi_gpu_example.png")
    return parser.parse_args()


def get_dtype(dtype_str: str):
    import torch

    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def get_4bit_compute_dtype(args, torch_module):
    if args.bnb_4bit_compute_dtype == "bf16":
        return torch_module.bfloat16
    if args.bnb_4bit_compute_dtype == "fp16":
        return torch_module.float16
    if args.bnb_4bit_compute_dtype == "fp32":
        return torch_module.float32
    # auto: prefer bf16 on modern GPUs for better numeric stability
    if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    return torch_module.float16


def build_max_memory(
    num_visible_gpus: int, gpu_memory_gb: int, cpu_memory_gb: int
) -> dict:
    max_memory = {i: f"{gpu_memory_gb}GiB" for i in range(num_visible_gpus)}
    if cpu_memory_gb > 0:
        max_memory["cpu"] = f"{cpu_memory_gb}GiB"
    return max_memory


def load_pipeline(args, dtype, max_memory):
    import torch
    from diffusers import ZImagePipeline

    model_id_or_path = args.model_path
    if Path(model_id_or_path).exists():
        model_id_or_path = str(Path(model_id_or_path).resolve())

    common_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "max_memory": max_memory,
        "low_cpu_mem_usage": True,
    }

    if args.quantization == "none":
        return ZImagePipeline.from_pretrained(model_id_or_path, **common_kwargs)

    try:
        import bitsandbytes as _  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "bitsandbytes is required for quantization. Install with: uv pip install bitsandbytes"
        ) from e

    try:
        from diffusers import BitsAndBytesConfig, ZImageTransformer2DModel
    except Exception as e:
        raise RuntimeError(
            "Your diffusers version does not expose BitsAndBytesConfig/ZImageTransformer2DModel. "
            "Update with: uv pip install git+https://github.com/huggingface/diffusers"
        ) from e

    if args.quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        compute_dtype = get_4bit_compute_dtype(args, torch)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # Quantize only the DiT transformer, keep VAE/text encoder in regular dtype.
    transformer = ZImageTransformer2DModel.from_pretrained(
        model_id_or_path,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=dtype,
    )

    return ZImagePipeline.from_pretrained(
        model_id_or_path,
        transformer=transformer,
        **common_kwargs,
    )


def main() -> int:
    args = parse_args()

    try:
        import torch
    except Exception as e:
        print(f"Missing torch: {e}")
        return 1

    try:
        from diffusers import ZImagePipeline  # noqa: F401
    except Exception as e:
        print("Missing diffusers. Install with:")
        print("  uv pip install git+https://github.com/huggingface/diffusers")
        print(f"Details: {e}")
        return 1

    if not torch.cuda.is_available():
        print("CUDA is not available. This script needs GPUs.")
        return 1

    num_visible_gpus = torch.cuda.device_count()
    dtype = get_dtype(args.dtype)
    max_memory = build_max_memory(
        num_visible_gpus, args.gpu_memory_gb, args.cpu_memory_gb
    )

    print(f"Visible GPUs: {num_visible_gpus}")
    for i in range(num_visible_gpus):
        print(f"  cuda:{i}: {torch.cuda.get_device_name(i)}")
    print(f"device_map: {args.device_map}")
    print(f"max_memory: {max_memory}")
    print(f"quantization: {args.quantization}")
    if args.quantization == "4bit" and args.num_inference_steps < 8:
        print(
            "Warning: Z-Image-Turbo usually needs 8 DiT forwards (~9 steps in diffusers)."
        )

    start_load = time.time()
    pipe = load_pipeline(args, dtype, max_memory)
    load_secs = time.time() - start_load
    print(f"Pipeline loaded in {load_secs:.2f}s")

    if args.attention_backend != "default":
        try:
            pipe.transformer.set_attention_backend(args.attention_backend)
            print(f"Set attention backend: {args.attention_backend}")
        except Exception as e:
            print(f"Failed to set attention backend {args.attention_backend}: {e}")

    if args.enable_model_cpu_offload:
        if getattr(pipe, "hf_device_map", None) is not None and hasattr(
            pipe, "reset_device_map"
        ):
            pipe.reset_device_map()
        pipe.enable_model_cpu_offload()
        print("Enabled model CPU offload")

    generator = torch.Generator(device="cuda:0").manual_seed(args.seed)

    start_gen = time.time()
    try:
        print(args.prompt)
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="pt",
        )
        image_tensor = result.images[0]

        if args.sanitize_output:
            image_tensor = torch.nan_to_num(
                image_tensor, nan=0.0, posinf=1.0, neginf=0.0
            )
        image_tensor = image_tensor.clamp(0, 1)

        nan_ratio = torch.isnan(result.images[0]).float().mean().item()
        print(
            f"Output stats: min={image_tensor.min().item():.4f}, max={image_tensor.max().item():.4f}, "
            f"mean={image_tensor.mean().item():.4f}, nan_ratio={nan_ratio:.6f}"
        )
        if args.quantization == "4bit" and nan_ratio > 0.99:
            print("4-bit quantization became numerically unstable (almost all NaN).")
            print("Please switch to: --quantization 8bit")
            return 2

        image = (
            (image_tensor.permute(1, 2, 0).detach().to(torch.float32).cpu().numpy() * 255.0)
            .round()
            .astype("uint8")
        )
        from PIL import Image

        image = Image.fromarray(image)
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            print("Device placement mismatch detected during sharded forward.")
            print(
                "This is commonly triggered when balanced sharding spills some layers to CPU under tight VRAM."
            )
            print("Try one of these:")
            print("  1) Free more VRAM per visible GPU and keep --device-map balanced")
            print("  2) Reduce visible GPUs to only freer cards")
            print("  3) Use --quantization 8bit or --quantization 4bit")
        raise
    gen_secs = time.time() - start_gen

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"Saved: {output_path}")
    print(f"Generation time: {gen_secs:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
