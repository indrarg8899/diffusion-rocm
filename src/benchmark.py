"""
Inference Benchmark - Comprehensive benchmarking for ROCm Stable Diffusion.

Measures latency, throughput, memory usage, and per-component timing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    component: str
    latency_ms: float
    memory_peak_mb: float
    memory_allocated_mb: float
    iterations: int
    batch_size: int = 1
    dtype: str = "float16"
    device: str = "cuda"

    @property
    def latency_per_iter_ms(self) -> float:
        return self.latency_ms / self.iterations


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)
    hardware_info: dict = field(default_factory=dict)
    software_info: dict = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def summary(self) -> dict:
        """Generate summary statistics."""
        return {
            "total_latency_ms": sum(r.latency_ms for r in self.results),
            "peak_memory_mb": max(r.memory_peak_mb for r in self.results) if self.results else 0,
            "components": {r.component: r.latency_per_iter_ms for r in self.results},
        }

    def to_json(self, path: str):
        """Save results to JSON."""
        data = {
            "hardware": self.hardware_info,
            "software": self.software_info,
            "results": [
                {
                    "component": r.component,
                    "latency_ms": r.latency_ms,
                    "latency_per_iter_ms": r.latency_per_iter_ms,
                    "memory_peak_mb": r.memory_peak_mb,
                    "batch_size": r.batch_size,
                    "dtype": r.dtype,
                }
                for r in self.results
            ],
            "summary": self.summary(),
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info(f"Results saved to: {path}")


def get_hardware_info() -> dict:
    """Collect hardware information."""
    info = {
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "name": props.name,
            "total_memory_gb": props.total_mem / (1024**3),
            "major": props.major,
            "minor": props.minor,
            "multiprocessor_count": props.multi_processor_count,
        })

    if hasattr(torch.version, "hip") and torch.version.hip:
        info["rocm_version"] = torch.version.hip

    return info


def benchmark_fn(
    fn,
    args: tuple = (),
    kwargs: dict = None,
    warmup: int = 3,
    iterations: int = 10,
    component: str = "unknown",
    batch_size: int = 1,
) -> BenchmarkResult:
    """
    Benchmark a function.

    Args:
        fn: Function to benchmark.
        args: Positional arguments.
        kwargs: Keyword arguments.
        warmup: Number of warmup iterations.
        iterations: Number of measurement iterations.
        component: Name of component being benchmarked.
        batch_size: Batch size used.

    Returns:
        BenchmarkResult with timing and memory stats.
    """
    kwargs = kwargs or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            fn(*args, **kwargs)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        with torch.inference_mode():
            fn(*args, **kwargs)
    end.record()

    torch.cuda.synchronize()
    latency_ms = start.elapsed_time(end)
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)

    return BenchmarkResult(
        component=component,
        latency_ms=latency_ms,
        memory_peak_mb=peak_memory_mb,
        memory_allocated_mb=allocated_mb,
        iterations=iterations,
        batch_size=batch_size,
        dtype="float16",
        device=torch.cuda.get_device_name(0),
    )


def benchmark_model(
    model: torch.nn.Module,
    input_fn,
    warmup: int = 3,
    iterations: int = 10,
    component: str = "model",
    batch_size: int = 1,
) -> BenchmarkResult:
    """
    Benchmark a model with input generation function.

    Args:
        model: PyTorch model to benchmark.
        input_fn: Function that returns input tensors.
        warmup: Warmup iterations.
        iterations: Measurement iterations.
        component: Component name.
        batch_size: Batch size.

    Returns:
        BenchmarkResult.
    """
    model.eval()
    return benchmark_fn(
        model,
        args=(input_fn(),),
        warmup=warmup,
        iterations=iterations,
        component=component,
        batch_size=batch_size,
    )


class ModelProfiler:
    """Profile model execution with per-layer timing."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.hooks = []
        self.timings: dict[str, list[float]] = {}

    def profile(self, input_tensor: torch.Tensor, iterations: int = 5) -> dict:
        """Profile model execution."""
        self._register_hooks()

        with torch.inference_mode():
            for _ in range(iterations):
                self.model(input_tensor)

        self._remove_hooks()
        return self.timings

    def _register_hooks(self):
        """Register forward hooks for timing."""
        for name, module in self.model.named_modules():
            hook = module.register_forward_hook(
                lambda mod, inp, out, name=name: self._record_time(name)
            )
            self.hooks.append(hook)

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _record_time(self, name: str):
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(time.perf_counter() * 1000)


def generate_report(suite: BenchmarkSuite, output_path: str = "benchmark_report.md"):
    """Generate markdown benchmark report."""
    lines = ["# Benchmark Report\n"]
    lines.append(f"## Hardware\n")

    if suite.hardware_info:
        for dev in suite.hardware_info.get("devices", []):
            lines.append(f"- **{dev['name']}**: {dev['total_memory_gb']:.1f} GB VRAM\n")

    lines.append("\n## Results\n")
    lines.append("| Component | Latency (ms) | Per-iteration (ms) | Peak Memory (MB) | Batch |")
    lines.append("|-----------|--------------|-------------------|------------------|-------|")

    for r in suite.results:
        lines.append(
            f"| {r.component} | {r.latency_ms:.1f} | {r.latency_per_iter_ms:.1f} | "
            f"{r.memory_peak_mb:.0f} | {r.batch_size} |"
        )

    summary = suite.summary()
    lines.append(f"\n**Total Latency**: {summary['total_latency_ms']:.1f} ms\n")
    lines.append(f"**Peak Memory**: {summary['peak_memory_mb']:.0f} MB\n")

    report = "\n".join(lines)
    Path(output_path).write_text(report)
    logger.info(f"Report saved to: {output_path}")
    return report
