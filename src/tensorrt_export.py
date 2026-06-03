"""
TensorRT Export - Export pipeline components to TensorRT for ROCm.

Supports ONNX export and TRT engine compilation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for TensorRT export."""

    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    output_dir: str = "./exported_models"
    precision: str = "fp16"
    batch_size: int = 1
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 30
    opset_version: int = 18
    dynamic_axes: bool = True
    simplify: bool = True
    workspace_size_gb: int = 4


class TensorRTExporter:
    """
    Export Stable Diffusion pipeline to TensorRT-ROCm.

    Steps:
    1. Export PyTorch models to ONNX
    2. Optimize ONNX graphs
    3. Compile to TensorRT engines (if TRT available)

    Note: TensorRT-ROCm support requires ROCm 5.7+ and
    compatible TensorRT-ROCm package.
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._check_dependencies()

    def _check_dependencies(self):
        """Check for required dependencies."""
        try:
            import onnx

            logger.info(f"ONNX version: {onnx.__version__}")
        except ImportError:
            raise ImportError("ONNX not installed. Install with: pip install onnx")

        self.has_trt = False
        try:
            import tensorrt as trt

            self.has_trt = True
            logger.info(f"TensorRT version: {trt.__version__}")
        except ImportError:
            logger.warning("TensorRT not found. ONNX export only.")

    def export_unet(self, model: torch.nn.Module, device: str = "cuda"):
        """Export UNet to ONNX."""
        logger.info("Exporting UNet to ONNX...")

        h = self.config.height // 8
        w = self.config.width // 8

        dummy_latent = torch.randn(
            (self.config.batch_size, 4, h, w),
            device=device,
            dtype=torch.float16,
        )
        dummy_timestep = torch.tensor([500], device=device, dtype=torch.float32)
        dummy_context = torch.randn(
            (self.config.batch_size, 77, 1024),
            device=device,
            dtype=torch.float16,
        )

        onnx_path = self.output_dir / "unet.onnx"

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_latent, dummy_timestep, dummy_context),
                str(onnx_path),
                opset_version=self.config.opset_version,
                input_names=["latent", "timestep", "context"],
                output_names=["noise_pred"],
                dynamic_axes={
                    "latent": {0: "batch"},
                    "context": {0: "batch"},
                    "noise_pred": {0: "batch"},
                } if self.config.dynamic_axes else None,
            )

        logger.info(f"UNet exported to: {onnx_path}")
        return onnx_path

    def export_vae(self, model: torch.nn.Module, device: str = "cuda"):
        """Export VAE decoder to ONNX."""
        logger.info("Exporting VAE to ONNX...")

        h = self.config.height // 8
        w = self.config.width // 8

        dummy_latent = torch.randn(
            (self.config.batch_size, 4, h, w),
            device=device,
            dtype=torch.float16,
        )

        onnx_path = self.output_dir / "vae.onnx"

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_latent,),
                str(onnx_path),
                opset_version=self.config.opset_version,
                input_names=["latent"],
                output_names=["image"],
                dynamic_axes={
                    "latent": {0: "batch", 2: "height", 3: "width"},
                    "image": {0: "batch", 2: "height", 3: "width"},
                } if self.config.dynamic_axes else None,
            )

        logger.info(f"VAE exported to: {onnx_path}")
        return onnx_path

    def export_text_encoder(self, model: torch.nn.Module, device: str = "cuda"):
        """Export text encoder to ONNX."""
        logger.info("Exporting text encoder to ONNX...")

        dummy_input_ids = torch.zeros(
            (self.config.batch_size, 77),
            device=device,
            dtype=torch.long,
        )
        dummy_mask = torch.ones(
            (self.config.batch_size, 77),
            device=device,
            dtype=torch.long,
        )

        onnx_path = self.output_dir / "text_encoder.onnx"

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_mask),
                str(onnx_path),
                opset_version=self.config.opset_version,
                input_names=["input_ids", "attention_mask"],
                output_names=["hidden_states"],
                dynamic_axes={
                    "input_ids": {0: "batch"},
                    "attention_mask": {0: "batch"},
                    "hidden_states": {0: "batch"},
                } if self.config.dynamic_axes else None,
            )

        logger.info(f"Text encoder exported to: {onnx_path}")
        return onnx_path

    def build_trt_engine(self, onnx_path: str, engine_path: Optional[str] = None):
        """
        Build TensorRT engine from ONNX model.

        Args:
            onnx_path: Path to ONNX model.
            engine_path: Output path for TRT engine.
        """
        if not self.has_trt:
            logger.warning("TensorRT not available. Skipping engine build.")
            return None

        import tensorrt as trt

        engine_path = engine_path or str(Path(onnx_path).with_suffix(".engine"))

        logger.info(f"Building TensorRT engine: {onnx_path} → {engine_path}")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(error)}")
                raise RuntimeError("ONNX parsing failed")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config.workspace_size_gb * (1024**3),
        )

        if self.config.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = inp.shape
            if -1 in shape:
                min_shape = [1 if s == -1 else s for s in shape]
                opt_shape = [self.config.batch_size if s == -1 else s for s in shape]
                max_shape = [self.config.batch_size * 2 if s == -1 else s for s in shape]
                profile.set_shape(inp.name, min_shape, opt_shape, max_shape)

        config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed")

        with open(engine_path, "wb") as f:
            f.write(engine)

        logger.info(f"TensorRT engine saved: {engine_path}")
        return engine_path

    def export_pipeline(
        self,
        unet: torch.nn.Module,
        vae: torch.nn.Module,
        text_encoder: torch.nn.Module,
        device: str = "cuda",
        build_engines: bool = True,
    ):
        """Export all pipeline components."""
        logger.info(f"Exporting pipeline to: {self.output_dir}")

        results = {}

        results["unet"] = self.export_unet(unet, device)
        results["vae"] = self.export_vae(vae, device)
        results["text_encoder"] = self.export_text_encoder(text_encoder, device)

        if build_engines and self.has_trt:
            for name, onnx_path in results.items():
                results[name] = self.build_trt_engine(str(onnx_path))

        logger.info("Pipeline export complete!")
        return results
