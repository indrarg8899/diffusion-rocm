"""
diffusion-rocm - Stable Diffusion optimized for AMD ROCm GPUs.
"""

__version__ = "1.0.0"
__author__ = "diffusion-rocm contributors"

from src.pipeline import StableDiffusionPipeline, PipelineConfig, GenerationResult

__all__ = [
    "StableDiffusionPipeline",
    "PipelineConfig",
    "GenerationResult",
]
