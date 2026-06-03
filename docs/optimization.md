# ROCm Optimization Guide

## Overview

This document covers optimization techniques for running Stable Diffusion on AMD GPUs with ROCm. These techniques are tuned for MI300X (CDNA3 architecture) but apply broadly to other ROCm-compatible GPUs.

## Hardware Optimizations

### Memory Configuration

```bash
# Enable expandable memory segments for ROCm
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Increase HSA thread pool size
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For MI300X

# Enable memory pool
export PYTORCH_HIP_ARENA_ALLOC_SIZE=2
```

### MI300X Specific

The MI300X has 192GB HBM3 memory. Key optimizations:

- **Flash Attention**: Use `torch.nn.functional.scaled_dot_product_attention` — supported natively in ROCm 6.0+
- **TF32**: Enable TF32 for matmuls: `torch.backends.cuda.matmul.allow_tf32 = True`
- **Channels-Last**: Use `memory_format=torch.channels_last` for all convolutions
- **Large Batch**: Exploit large VRAM with batch sizes 4-8

### MI250X / MI210

- Use fp16 precision (no TF32 on MI250)
- Flash attention supported in ROCm 5.7+
- Typical batch size: 1-2

## Model Optimizations

### UNet Optimizations

1. **Fused GroupNorm + SiLU**: Single kernel instead of two separate ops
2. **Flash Cross-Attention**: Replace naive attention with flash attention
3. **Convolution Kernel Selection**: Use `torch.conv2d` with optimized kernels

### VAE Optimizations

1. **Tiled Decoding**: Process large images in 512x512 tiles with overlap
2. **VAE Tiling Parameters**:
   - Tile size: 512 for balanced quality/speed
   - Overlap: 25% (128px) to avoid seams
   - Buffer: 32px for smooth transitions

3. **FP16 Decode**: VAE in fp16 is sufficient quality with 2× speedup

### Text Encoder

1. **FP16 Quantization**: CLIP encoder runs well in fp16
2. **Cache Embeddings**: Reuse embeddings for same prompts
3. **Batch Encoding**: Encode multiple prompts in parallel

## Runtime Optimizations

### torch.compile

```python
# Compile UNet for kernel fusion
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
```

**Note**: Compile adds 1-2 minute startup but gives 15-30% speedup on subsequent runs.

### Memory-Efficient Inference

```python
# Enable attention slicing for VRAM < 24GB
pipeline.enable_attention_slicing()

# Offload VAE to CPU when not needed
pipeline.vae = pipeline.vae.to("cpu")
```

### Batch Processing

For maximum throughput:
- SD 1.5 (512×512): batch_size=8 on MI300X
- SD 2.1 (768×768): batch_size=4 on MI300X
- SDXL (1024×1024): batch_size=2 on MI300X

## Benchmarking

```bash
# Latency benchmark
python benchmarks/bench_latency.py --model sd-xl --steps 30

# Throughput sweep
python benchmarks/bench_throughput.py --model sd-xl --max-batch 8

# Full profile
python -c "
from src.benchmark import benchmark_model, BenchmarkSuite, get_hardware_info
from src.pipeline import StableDiffusionPipeline, PipelineConfig
pipeline = StableDiffusionPipeline(PipelineConfig())
suite = BenchmarkSuite(hardware_info=get_hardware_info())
# ... run benchmarks and save
"
```

## TensorRT Export

```bash
# Export all components
python scripts/export_model.py --model sd-xl --output-dir ./exported --build-engines

# Verify export
ls -la ./exported/
# unet.onnx, vae.onnx, text_encoder.onnx, unet.engine, vae.engine, text_encoder.engine
```

## Known Issues

| Issue | Workaround | Status |
|-------|-----------|--------|
| ROCm 6.0 flash attention OOM on large heads | Reduce attention head count or use xformers | Fixed in 6.1 |
| TF32 not supported on MI250X | Use fp16 only | Expected |
| torch.compile slow first run | Pre-warm with 1 image | Expected |
| VAE tiled mode slight quality loss | Increase overlap to 30% | Acceptable |

## References

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm](https://pytorch.org/docs/stable/notes/hip.html)
- [MI300X Architecture](https://www.amd.com/en/products/accelerators/instinct/mi300x.html)
- [Stable Diffusion](https://stability.ai/stable-image)
