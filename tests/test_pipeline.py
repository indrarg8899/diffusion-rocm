"""
Tests for Stable Diffusion ROCm pipeline.

Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_config(self):
        from src.pipeline import PipelineConfig

        config = PipelineConfig()
        assert config.model_name == "stabilityai/stable-diffusion-xl-base-1.0"
        assert config.dtype == torch.float16
        assert config.device == "cuda"
        assert config.vae_tiled is True
        assert config.use_flash_attention is True
        assert config.channels_last is True

    def test_custom_config(self):
        from src.pipeline import PipelineConfig

        config = PipelineConfig(
            model_name="stabilityai/stable-diffusion-2-1",
            dtype=torch.float32,
            device="cpu",
            vae_tiled=False,
        )
        assert config.model_name == "stabilityai/stable-diffusion-2-1"
        assert config.dtype == torch.float32
        assert config.device == "cpu"
        assert config.vae_tiled is False


class TestGenerationResult:
    """Test GenerationResult dataclass."""

    def test_default_result(self):
        from src.pipeline import GenerationResult

        result = GenerationResult()
        assert result.images == []
        assert result.nsfw_detected == []
        assert result.seed == 0
        assert result.inference_time_ms == 0.0

    def test_result_with_data(self):
        from PIL import Image
        from src.pipeline import GenerationResult

        img = Image.new("RGB", (64, 64), "red")
        result = GenerationResult(
            images=[img],
            seed=42,
            inference_time_ms=150.0,
            memory_used_mb=2048.0,
        )
        assert len(result.images) == 1
        assert result.seed == 42


class TestSchedulers:
    """Test noise schedulers."""

    def test_ddim_scheduler(self):
        from src.scheduler import DDIMScheduler

        scheduler = DDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(20)
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 20

    def test_euler_ancestral_scheduler(self):
        from src.scheduler import EulerAncestralScheduler

        scheduler = EulerAncestralScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(30)
        assert len(scheduler.timesteps) == 30

    def test_dpm_solver_scheduler(self):
        from src.scheduler import DPMSolverScheduler

        scheduler = DPMSolverScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(25)
        assert len(scheduler.timesteps) == 25

    def test_pndm_scheduler(self):
        from src.scheduler import PNDMScheduler

        scheduler = PNDMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(20)
        assert len(scheduler.timesteps) == 20

    def test_ddpm_scheduler(self):
        from src.scheduler import DDPMScheduler

        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(20)
        assert len(scheduler.timesteps) == 20

    def test_scheduler_step(self):
        from src.scheduler import EulerAncestralScheduler

        scheduler = EulerAncestralScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(20)

        dummy_model_output = torch.randn(1, 4, 64, 64)
        dummy_timestep = scheduler.timesteps[0]
        dummy_sample = torch.randn(1, 4, 64, 64)

        result = scheduler.step(dummy_model_output, dummy_timestep, dummy_sample)
        assert result.shape == dummy_sample.shape

    def test_get_scheduler(self):
        from src.scheduler import get_scheduler, DDIMScheduler, EulerAncestralScheduler

        sched = get_scheduler("ddim")
        assert isinstance(sched, DDIMScheduler)

        sched = get_scheduler("euler_a")
        assert isinstance(sched, EulerAncestralScheduler)

    def test_invalid_scheduler(self):
        from src.scheduler import get_scheduler

        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler("nonexistent")


class TestTextEncoder:
    """Test CLIP text encoder."""

    def test_tokenizer(self):
        from src.text_encoder import CLIPTokenizer

        tokenizer = CLIPTokenizer(max_length=77)
        result = tokenizer("A beautiful sunset", return_tensors="pt")

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (1, 77)
        assert result["attention_mask"].shape == (1, 77)

    def test_tokenizer_batch(self):
        from src.text_encoder import CLIPTokenizer

        tokenizer = CLIPTokenizer(max_length=77)
        result = tokenizer(["prompt 1", "prompt 2"], return_tensors="pt")

        assert result["input_ids"].shape == (2, 77)

    def test_tokenizer_truncation(self):
        from src.text_encoder import CLIPTokenizer

        tokenizer = CLIPTokenizer(max_length=10)
        result = tokenizer("A very long prompt that should be truncated", return_tensors="pt")

        assert result["input_ids"].shape == (1, 10)


class TestCLIPTextModel:
    """Test CLIP text model architecture."""

    def test_model_creation(self):
        from src.text_encoder import CLIPTextModel

        model = CLIPTextModel(
            vocab_size=49408,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            intermediate_size=128,
        )

        input_ids = torch.randint(0, 49408, (1, 77))
        output = model(input_ids)

        assert output.shape == (1, 77, 64)


class TestVAEDecoder:
    """Test VAE decoder."""

    def test_res_block(self):
        from src.vae import ResBlock

        block = ResBlock(64, 128)
        x = torch.randn(1, 64, 32, 32)
        out = block(x)
        assert out.shape == (1, 128, 32, 32)

    def test_attention_block(self):
        from src.vae import AttentionBlock

        block = AttentionBlock(64, num_heads=1)
        x = torch.randn(1, 64, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_upsample_block(self):
        from src.vae import UpsampleBlock

        block = UpsampleBlock(64)
        x = torch.randn(1, 64, 8, 8)
        out = block(x)
        assert out.shape == (1, 64, 16, 16)


class TestUNet:
    """Test UNet model components."""

    def test_cross_attention(self):
        from src.unet import CrossAttention

        attn = CrossAttention(query_dim=64, heads=4, dim_head=16, use_flash_attention=False)
        x = torch.randn(1, 10, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_fused_group_norm(self):
        from src.unet import FusedGroupNorm

        norm = FusedGroupNorm(64)
        x = torch.randn(1, 64, 16, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_resnet_block(self):
        from src.unet import ResnetBlock

        block = ResnetBlock(64, 128, time_emb_dim=256)
        x = torch.randn(1, 64, 16, 16)
        time_emb = torch.randn(1, 256)
        out = block(x, time_emb)
        assert out.shape == (1, 128, 16, 16)

    def test_downsample(self):
        from src.unet import Downsample

        down = Downsample(64)
        x = torch.randn(1, 64, 16, 16)
        out = down(x)
        assert out.shape == (1, 64, 8, 8)

    def test_upsample(self):
        from src.unet import Upsample

        up = Upsample(64)
        x = torch.randn(1, 64, 8, 8)
        out = up(x)
        assert out.shape == (1, 64, 16, 16)

    def test_transformer_block(self):
        from src.unet import TransformerBlock

        block = TransformerBlock(64, context_dim=32, num_heads=4, use_flash_attention=False)
        x = torch.randn(1, 64, 8, 8)
        context = torch.randn(1, 10, 32)
        out = block(x, context)
        assert out.shape == x.shape


class TestBenchmark:
    """Test benchmark utilities."""

    def test_benchmark_result(self):
        from src.benchmark import BenchmarkResult

        result = BenchmarkResult(
            component="unet",
            latency_ms=100.0,
            memory_peak_mb=2048.0,
            memory_allocated_mb=1024.0,
            iterations=10,
            batch_size=1,
        )
        assert result.latency_per_iter_ms == 10.0

    def test_benchmark_suite(self):
        from src.benchmark import BenchmarkResult, BenchmarkSuite

        suite = BenchmarkSuite()
        suite.add_result(BenchmarkResult(
            component="unet",
            latency_ms=100.0,
            memory_peak_mb=2048.0,
            memory_allocated_mb=1024.0,
            iterations=10,
        ))

        summary = suite.summary()
        assert summary["total_latency_ms"] == 100.0
        assert "unet" in summary["components"]


class TestTensorRTExport:
    """Test TensorRT export utilities."""

    def test_export_config(self):
        from src.tensorrt_export import ExportConfig

        config = ExportConfig()
        assert config.model_name == "stabilityai/stable-diffusion-xl-base-1.0"
        assert config.precision == "fp16"
        assert config.batch_size == 1
        assert config.opset_version == 18


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
