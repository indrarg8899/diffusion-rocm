#!/usr/bin/env python3
"""
CLI Image Generation - Generate images with Stable Diffusion on ROCm.

Usage:
    python scripts/generate.py --prompt "A castle in the clouds" --model sd-xl
    python scripts/generate.py --prompt "Portrait of a cat" --model sd-1.5 --steps 20
    python scripts/generate.py --prompt "Abstract art" --model sd-xl --output ./output/
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import StableDiffusionPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "sd-xl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd-2.1": "stabilityai/stable-diffusion-2-1",
    "sd-1.5": "runwayml/stable-diffusion-v1-5",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with Stable Diffusion (ROCm optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate.py --prompt "A sunset over mountains"
    python scripts/generate.py --prompt "Digital art" --model sd-xl --steps 25
    python scripts/generate.py --prompt "Portrait" --width 768 --height 1024
    python scripts/generate.py --prompt "Landscape" --num-images 4 --seed 123
        """,
    )

    parser.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative", "-n", type=str, default="", help="Negative prompt")
    parser.add_argument("--model", "-m", choices=MODELS.keys(), default="sd-xl", help="Model")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=None, help="Image width")
    parser.add_argument("--height", type=int, default=None, help="Image height")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--scheduler", choices=["ddim", "dpm++", "euler_a", "pndm"], default="euler_a")
    parser.add_argument("--output", type=str, default="./output/", help="Output directory")
    parser.add_argument("--tiled-vae", action="store_true", help="Use tiled VAE decoding")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    parser.add_argument("--flash-attn", action="store_true", default=True, help="Use flash attention")
    parser.add_argument("--channels-last", action="store_true", default=True, help="Use channels-last format")

    return parser.parse_args()


DEFAULT_RESOLUTIONS = {
    "sd-xl": (1024, 1024),
    "sd-2.1": (768, 768),
    "sd-1.5": (512, 512),
}

DEFAULT_STEPS = {
    "sd-xl": 30,
    "sd-2.1": 25,
    "sd-1.5": 20,
}


def main():
    args = parse_args()

    # Resolve defaults
    default_w, default_h = DEFAULT_RESOLUTIONS[args.model]
    width = args.width or default_w
    height = args.height or default_h
    steps = args.steps or DEFAULT_STEPS[args.model]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup pipeline
    config = PipelineConfig(
        model_name=MODELS[args.model],
        dtype=__import__("torch").float16,
        use_flash_attention=args.flash_attn,
        channels_last=args.channels_last,
        vae_tiled=args.tiled_vae,
        compile_model=args.compile,
    )

    logger.info(f"Model:     {args.model} ({MODELS[args.model]})")
    logger.info(f"Size:      {width}×{height}")
    logger.info(f"Steps:     {steps}")
    logger.info(f"Scheduler: {args.scheduler}")
    logger.info(f"Guidance:  {args.guidance}")

    pipeline = StableDiffusionPipeline(config)

    # Generate
    seed = args.seed or __import__("random").randint(0, 2**31)
    logger.info(f"Seed:      {seed}")

    start_time = time.perf_counter()
    result = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=args.guidance,
        num_images=args.num_images,
        seed=seed,
        scheduler_name=args.scheduler,
    )
    elapsed = time.perf_counter() - start_time

    # Save images
    timestamp = int(time.time())
    saved_paths = []
    for i, img in enumerate(result.images):
        filename = f"gen_{timestamp}_s{seed}_n{i}.png"
        path = output_dir / filename
        img.save(str(path))
        saved_paths.append(path)
        logger.info(f"Saved: {path}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Generation complete!")
    logger.info(f"Time:        {elapsed:.2f}s ({result.inference_time_ms:.0f} ms)")
    logger.info(f"Peak memory: {result.memory_used_mb:.0f} MB")
    logger.info(f"Images:      {len(saved_paths)}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
