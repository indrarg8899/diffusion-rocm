# Getting Started

## Prerequisites

- AMD GPU with ROCm 6.0+ (RDNA2/3 or CDNA2/3)
- Python 3.10+
- PyTorch with ROCm support

## Installation

```bash
pip install diffusion-rocm
```

## First Image

```python
from diffusion_rocm.pipeline import DiffusionPipeline, DiffusionConfig

config = DiffusionConfig(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    num_inference_steps=25,
    width=1024,
    height=1024,
)
pipe = DiffusionPipeline(config)
pipe.load()

image = pipe("A serene lake at dawn, digital art", seed=42)
```

## Supported Models

| Model | Resolution | VRAM |
|---|---|---|
| SD 1.5 | 512×512 | ~4GB |
| SDXL Base | 1024×1024 | ~8GB |
| SD3 Medium | 1024×1024 | ~12GB |
| FLUX.1-dev | 1024×1024 | ~15GB |
