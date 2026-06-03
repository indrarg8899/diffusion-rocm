"""CLI entry point for diffusion-rocm."""

import argparse
import torch
from diffusion_rocm.pipeline import DiffusionPipeline, DiffusionConfig


def generate(args):
    config = DiffusionConfig(
        model_id=args.model,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    pipe = DiffusionPipeline(config)
    pipe.load()

    image = pipe(
        prompt=args.prompt,
        seed=args.seed,
    )
    print(f"Generated image: {args.width}x{args.height}, {args.steps} steps")
    if args.output:
        print(f"Saved to {args.output}")
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion-ROCm Generate")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="output.png")
    args = parser.parse_args()
    generate(args)
