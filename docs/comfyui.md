# ComfyUI Integration

## Setup

```bash
# Install ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git

# Install ROCm backend
pip install diffusion-rocm[comfyui]

# Copy custom nodes
cp -r diffusion_rocm/nodes/ ComfyUI/custom_nodes/rocm_sd/

# Launch
python -m diffusion_rocm.comfyui_server --port 8188
```

## Available Nodes

- **ROCm SD Model Loader** — load SD/SDXL/FLUX models with ROCm optimizations
- **ROCm KSampler** — sampling with ROCm-optimized attention
- **ROCm VAE Decode** — tiled VAE decoding for large images
- **ROCm LoRA Loader** — load and apply LoRA weights
- **ROCm ControlNet** — ControlNet inference with ROCm kernels
