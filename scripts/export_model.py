#!/usr/bin/env python3
"""
Model Export Script - Export pipeline to ONNX/TensorRT for ROCm.

Usage:
    python scripts/export_model.py --model sd-xl --output-dir ./exported
    python scripts/export_model.py --model sd-xl --build-engines
    python scripts/export_model.py --model sd-2.1 --output-dir ./exported
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import StableDiffusionPipeline, PipelineConfig
from src.tensorrt_export import TensorRTExporter, ExportConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "sd-xl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd-2.1": "stabilityai/stable-diffusion-2-1",
    "sd-1.5": "runwayml/stable-diffusion-v1-5",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Stable Diffusion pipeline to ONNX/TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/export_model.py --model sd-xl
    python scripts/export_model.py --model sd-2.1 --output-dir ./my_export
    python scripts/export_model.py --model sd-xl --build-engines --workspace 8
        """,
    )

    parser.add_argument("--model", choices=MODELS.keys(), default="sd-xl")
    parser.add_argument("--output-dir", type=str, default="./exported_models")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--build-engines", action="store_true", help="Build TensorRT engines")
    parser.add_argument("--workspace", type=int, default=4, help="TRT workspace size in GB")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Precision: {args.precision}")

    # Load pipeline
    config = PipelineConfig(model_name=MODELS[args.model])
    pipeline = StableDiffusionPipeline(config)

    # Export
    export_config = ExportConfig(
        model_name=MODELS[args.model],
        output_dir=args.output_dir,
        precision=args.precision,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        opset_version=args.opset,
        workspace_size_gb=args.workspace,
    )

    exporter = TensorRTExporter(export_config)

    results = exporter.export_pipeline(
        unet=pipeline.unet,
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
        device="cuda",
        build_engines=args.build_engines,
    )

    logger.info("\nExport complete!")
    for component, path in results.items():
        if path:
            size_mb = Path(path).stat().st_size / (1024**2) if Path(path).exists() else 0
            logger.info(f"  {component}: {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
