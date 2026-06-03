# Dockerfile for ROCm-optimized Stable Diffusion
# Base image: AMD ROCm 6.2 with PyTorch 2.3

FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_2.3

LABEL maintainer="diffusion-rocm"
LABEL description="Stable Diffusion inference optimized for AMD ROCm GPUs"
LABEL version="1.0.0"

# Environment variables
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV PYTORCH_HIP_ARENA_ALLOC_SIZE=2
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY configs/ configs/
COPY benchmarks/ benchmarks/
COPY docs/ docs/
COPY scripts/ scripts/
COPY tests/ tests/

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm: {torch.version.hip}')"

# Default entrypoint
ENTRYPOINT ["python", "scripts/generate.py"]

# Default arguments
CMD ["--help"]
