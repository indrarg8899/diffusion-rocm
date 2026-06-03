#!/usr/bin/env python3
"""
Throughput Benchmark - Maximize images/second with batching.

Usage:
    python benchmarks/bench_throughput.py --model sd-xl --max-batch 8
    python benchmarks/bench_throughput.py --model sd-2.1 --batch-size 4
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import StableDiffusionPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "sd-xl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd-2.1": "stabilityai/stable-diffusion-2-1",
    "sd-1.5": "runwayml/stable-diffusion-v1-5",
}

RESOLUTIONS = {
    "sd-xl": (1024, 1024),
    "sd-2.1": (768, 768),
    "sd-1.5": (512, 512),
}


def benchmark_throughput(
    model_key: str,
    max_batch: int,
    steps: int = 25,
    iterations: int = 5,
    scheduler: str = "euler_a",
):
    """Find optimal batch size for maximum throughput."""
    model_name = MODELS[model_key]
    height, width = RESOLUTIONS[model_key]

    config = PipelineConfig(model_name=model_name)
    pipeline = StableDiffusionPipeline(config)

    results = []

    for batch_size in range(1, max_batch + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing batch_size={batch_size}...")

        prompt = ["A beautiful landscape painting"] * batch_size

        try:
            # Warmup
            for _ in range(2):
                pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    scheduler_name=scheduler,
                    seed=42,
                )

            # Measure
            torch.cuda.reset_peak_memory_stats()
            latencies = []

            for i in range(iterations):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    scheduler_name=scheduler,
                    seed=42,
                )

                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))

            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            mean_latency = np.mean(latencies)
            throughput = (batch_size * 1000.0) / mean_latency

            result = {
                "batch_size": batch_size,
                "mean_latency_ms": mean_latency,
                "std_latency_ms": np.std(latencies),
                "throughput_img_s": throughput,
                "memory_peak_mb": peak_mem_mb,
                "success": True,
            }
            results.append(result)

            logger.info(f"  Latency:   {mean_latency:.1f} ± {np.std(latencies):.1f} ms")
            logger.info(f"  Throughput: {throughput:.1f} img/s")
            logger.info(f"  Memory:    {peak_mem_mb:.0f} MB")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"  OOM at batch_size={batch_size}")
                results.append({
                    "batch_size": batch_size,
                    "success": False,
                    "error": "OOM",
                })
                break
            raise

    # Find optimal
    successful = [r for r in results if r.get("success")]
    if successful:
        best = max(successful, key=lambda r: r["throughput_img_s"])
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimal batch size: {best['batch_size']}")
        logger.info(f"Peak throughput:    {best['throughput_img_s']:.1f} img/s")
        logger.info(f"Latency at optimal: {best['mean_latency_ms']:.1f} ms")
        logger.info(f"Memory at optimal:  {best['memory_peak_mb']:.0f} MB")
        logger.info(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark")
    parser.add_argument("--model", choices=MODELS.keys(), default="sd-xl")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--scheduler", default="euler_a")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    results = benchmark_throughput(
        model_key=args.model,
        max_batch=args.batch_size or args.max_batch,
        steps=args.steps,
        iterations=args.iterations,
        scheduler=args.scheduler,
    )

    if args.output:
        import json
        Path(args.output).write_text(json.dumps({
            "model": args.model,
            "results": results,
        }, indent=2))


if __name__ == "__main__":
    main()
