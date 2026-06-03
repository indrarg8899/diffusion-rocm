# Supported Models

## Overview

diffusion-rocm supports Stable Diffusion models that follow the standard UNet + Text Encoder + VAE architecture.

## Model Compatibility Matrix

| Model | ID | Resolution | Steps | Scheduler | Status |
|-------|-----|-----------|-------|-----------|--------|
| SD 1.5 | `runwayml/stable-diffusion-v1-5` | 512×512 | 20 | euler_a | ✅ Full |
| SD 2.0 | `stabilityai/stable-diffusion-2-base` | 512×512 | 25 | dpm++ | ✅ Full |
| SD 2.1 | `stabilityai/stable-diffusion-2-1` | 768×768 | 25 | dpm++ | ✅ Full |
| SD 2.1-v | `stabilityai/stable-diffusion-2-1` | 768×768 | 25 | dpm++ | ✅ Full |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | 1024×1024 | 30 | euler_a | ✅ Full |
| SDXL-Turbo | `stabilityai/stable-diffusion-turbo` | 1024×1024 | 4 | euler_a | ✅ Full |
| SDXL-Lightning | `ByteDance/SDXL-Lightning` | 1024×1024 | 4-8 | euler_a | ⚠️ Partial |

## Model Configurations

### SD 1.5 (512×512)

```yaml
model: runwayml/stable-diffusion-v1-5
unet:
  channels: 320
  attention_resolutions: [1, 2, 4]
  layers_per_block: 2
text_encoder:
  hidden_size: 768
  num_layers: 12
  max_length: 77
vae:
  channels: [128, 256, 512, 512]
  latent_dim: 4
```

### SD 2.1 (768×768)

```yaml
model: stabilityai/stable-diffusion-2-1
unet:
  channels: 320
  attention_resolutions: [1, 2, 4]
  layers_per_block: 2
text_encoder:
  hidden_size: 1024
  num_layers: 23
  max_length: 77
vae:
  channels: [128, 256, 512, 512]
  latent_dim: 4
```

### SDXL (1024×1024)

```yaml
model: stabilityai/stable-diffusion-xl-base-1.0
unet:
  channels: 320
  attention_resolutions: [1, 2, 4]
  layers_per_block: 2
text_encoder:
  encoder_1:
    hidden_size: 1024
    num_layers: 23
  encoder_2:
    hidden_size: 1280
    num_layers: 32
  projection_dim: 1280
vae:
  channels: [128, 256, 512, 512]
  latent_dim: 4
  scale_factor: 0.13025
```

## VRAM Requirements

| Model | Resolution | FP16 VRAM | FP32 VRAM |
|-------|-----------|-----------|-----------|
| SD 1.5 | 512×512 | ~4 GB | ~8 GB |
| SD 2.1 | 768×768 | ~6 GB | ~12 GB |
| SDXL | 1024×1024 | ~8 GB | ~16 GB |
| SDXL | 1024×1024 (tiled) | ~6 GB | N/A |

## Custom Models

To use a custom model:

1. Ensure it follows the standard UNet + Text Encoder + VAE architecture
2. Create a config YAML in `configs/`
3. Pass the model name to `PipelineConfig`

```python
from src.pipeline import StableDiffusionPipeline, PipelineConfig

config = PipelineConfig(
    model_name="your-org/your-model-name",
    dtype=torch.float16,
)
pipeline = StableDiffusionPipeline(config)
result = pipeline("your prompt")
```

## LoRA Support

LoRA weights can be loaded via the standard diffusers API:

```python
from peft import PeftModel

pipeline.unet.model = PeftModel.from_pretrained(
    pipeline.unet.model, "path/to/lora/weights"
)
```

## ONNX Models

For TensorRT export, export with:

```bash
python scripts/export_model.py --model sd-xl --output-dir ./exported
```

Exported models follow ONNX Runtime conventions and can be used with:
- ONNX Runtime (CPU/GPU)
- TensorRT-ROCm
- OpenVINO
