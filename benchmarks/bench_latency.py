#!/usr/bin/env python3
"""
Latency Benchmark - End-to-end inference latency for Stable Diffusion.

Usage:
    python benchmarks/bench_latency.py --model sd-xl --steps 30
    python benchmarks/bench_latency.py --model sd-2.1 --steps 25 --batch-size 4
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import StableDiffusionPipeline, PipelineConfig
from src.benchmark import benchmark_fn, BenchmarkSuite, get_hardware_info, generate_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "sd-xl": {
        "name": "stabilityai/stable-diffusion-xl-base-1.0",
        "default_width": 1024,
        "default_height": 1024,
        "default_steps": 30,
    },
    "sd-2.1": {
        "name": "stabilityai/stable-diffusion-2-1",
        "default_width": 768,
        "default_height": 768,
        "default_steps": 25,
    },
    "sd-1.5": {
        "name": "runwayml/stable-diffusion-v1-5",
        "default_width": 512,
        "default_height": 512,
        "default_steps": 20,
    },
}


def benchmark_pipeline(
    model_name: str,
    steps: int,
    width: int,
    height: int,
    batch_size: int,
    warmup: int = 3,
    iterations: int = 10,
    scheduler: str = "euler_a",
):
    """Benchmark full pipeline inference."""
    config = PipelineConfig(model_name=MODELS[model_name]["name"])
    pipeline = StableDiffusionPipeline(config)

    prompt = "A beautiful landscape with mountains and a lake" if batch_size == 1 else \
        ["A beautiful landscape"] * batch_size

    def run_pipeline():
        return pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            scheduler_name=scheduler,
            seed=42,
        )

    # Warmup
    logger.info(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        run_pipeline()

    # Benchmark
    logger.info(f"Running latency benchmark ({iterations} iterations)...")
    suite = BenchmarkSuite()
    suite.hardware_info = get_hardware_info()

    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start_event.record()
        run_pipeline()
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
        logger.info(f"  Iteration {i+1}/{iterations}: {latencies[-1]:.1f} ms")

    import numpy as np
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Resolution: {width}x{height}")
    logger.info(f"Steps: {steps}, Scheduler: {scheduler}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"{'─'*60}")
    logger.info(f"Mean latency:   {np.mean(latencies):.1f} ms")
    logger.info(f"Median latency: {np.median(latencies):.1f} ms")
    logger.info(f"P95 latency:    {np.percentile(latencies, 95):.1f} ms")
    logger.info(f"Std deviation:  {np.std(latencies):.1f} ms")
    logger.info(f"Peak memory:    {peak_mem:.0f} MB")
    logger.info(f"Throughput:     {1000.0 / np.mean(latencies):.1f} img/s")
    logger.info(f"{'='*60}")

    return latencies


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for Stable Diffusion")
    parser.add_argument("--model", choices=MODELS.keys(), default="sd-xl")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--scheduler", default="euler_a", choices=["ddim", "dpm++", "euler_a", "pndm"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    model_info = MODELS[args.model]
    steps = args.steps or model_info["default_steps"]
    width = args.width or model_info["default_width"]
    height = args.height or model_info["default_height"]

    latencies = benchmark_pipeline(
        model_name=args.model,
        steps=steps,
        width=width,
        height=height,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        scheduler=args.scheduler,
    )

    if args.output:
        import json
        Path(args.output).write_text(json.dumps({
            "model": args.model,
            "resolution": f"{width}x{height}",
            "steps": steps,
            "scheduler": args.scheduler,
            "batch_size": args.batch_size,
            "latencies_ms": latencies,
        }, indent=2))


if __name__ == "__main__":
    main()
