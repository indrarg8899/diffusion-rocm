"""ComfyUI integration server for Diffusion-ROCm."""

import json
from typing import Dict, List, Any, Optional


class ComfyUIServer:
    """Serves diffusion pipeline as ComfyUI custom nodes."""

    def __init__(self, port: int = 8188, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.nodes: Dict[str, Any] = {}

    def register_rocm_nodes(self) -> None:
        self.nodes["ROCM_SDLoader"] = {
            "name": "ROCm SD Model Loader",
            "category": "ROCM",
            "inputs": [("model_name", "STRING"), ("dtype", "STRING")],
            "outputs": [("model", "MODEL")],
        }
        self.nodes["ROCM_Sampler"] = {
            "name": "ROCm KSampler",
            "category": "ROCM",
            "inputs": [
                ("model", "MODEL"), ("seed", "INT"), ("steps", "INT"),
                ("cfg", "FLOAT"), ("sampler", "STRING"), ("scheduler", "STRING"),
                ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                ("latent", "LATENT"),
            ],
            "outputs": [("samples", "LATENT")],
        }
        self.nodes["ROCM_VAEDecode"] = {
            "name": "ROCm VAE Decode",
            "category": "ROCM",
            "inputs": [("vae", "VAE"), ("samples", "LATENT")],
            "outputs": [("images", "IMAGE")],
        }
        self.nodes["ROCM_LoRALoader"] = {
            "name": "ROCm LoRA Loader",
            "category": "ROCM",
            "inputs": [("model", "MODEL"), ("lora_name", "STRING"), ("strength", "FLOAT")],
            "outputs": [("model", "MODEL")],
        }

    def get_node_list(self) -> List[Dict]:
        return [
            {"id": name, **info}
            for name, info in self.nodes.items()
        ]

    def serve(self) -> None:
        print(f"ComfyUI ROCm server on {self.host}:{self.port}")
        print(f"Registered nodes: {list(self.nodes.keys())}")
