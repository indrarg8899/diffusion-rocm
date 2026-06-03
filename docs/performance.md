# Performance Tuning

## Flash Attention
Enable ROCm flash attention for 20-40% speedup on attention-heavy models:
```python
config = DiffusionConfig(enable_rocm_flash_attn=True)
```

## VAE Slicing/Tiling
For large images (2048×2048+):
```python
config = DiffusionConfig(vae_tiling=True)
```

## Precision
- FP16: default, best speed
- BF16: slightly slower, better numerical stability
- FP8: experimental, requires MI300X

## Memory Optimization
- `enable_xformers=True`: memory-efficient attention
- `vae_slicing=True`: decode one image at a time
- Reduce resolution before upscaling with a super-resolution model
