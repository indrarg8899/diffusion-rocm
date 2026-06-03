"""
Noise schedulers for Stable Diffusion.

Supports DDIM, DPM-Solver++, Euler Ancestral, and PNDM schedulers.
Optimized for ROCm with minimal host-device sync.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np


class Scheduler(ABC):
    """Base class for noise schedulers."""

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps: Optional[torch.Tensor] = None
        self.init_noise_sigma: float = 1.0

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int, scheduler_type: str = ""):
        pass

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        pass

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return sample


class DDPMScheduler(Scheduler):
    """DDPM scheduler (reverse diffusion)."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        super().__init__(num_train_timesteps)
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def set_timesteps(self, num_inference_steps: int, scheduler_type: str = ""):
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_inference_steps) * step_ratio
        self.timesteps = self.timesteps.flip(0).long()

    def step(self, model_output, timestep, sample, generator=None):
        t = timestep
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_t = 1.0 - alpha_prod_t

        pred_original_sample = (sample - beta_t * model_output) / torch.sqrt(alpha_prod_t)
        pred_original_sample = pred_original_sample.clamp(-1, 1)

        mean = torch.sqrt(alpha_prod_t_prev) * (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        ) * model_output
        mean = mean + torch.sqrt(alpha_prod_t_prev) * alpha_prod_t / (1 - alpha_prod_t) * sample

        variance = self.posterior_variance[t].to(sample.device)
        noise = torch.randn(sample.shape, generator=generator, device=sample.device) if t > 0 else 0

        return mean + torch.sqrt(variance) * noise


class DDIMScheduler(Scheduler):
    """DDIM scheduler for fast sampling."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012, eta: float = 0.0):
        super().__init__(num_train_timesteps)
        self.eta = eta

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod

    def set_timesteps(self, num_inference_steps: int, scheduler_type: str = ""):
        step_ratio = self.num_train_timesteps / num_inference_steps
        self.timesteps = torch.round(torch.arange(num_inference_steps - 1, -1, -1) * step_ratio).long()
        self.timesteps_next = torch.zeros_like(self.timesteps)
        self.timesteps_next[:-1] = self.timesteps[1:]

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, model_output, timestep, sample, generator=None):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[self.timesteps_next[self.timesteps == timestep][0]] if timestep > 0 else torch.tensor(1.0)

        pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        pred_original_sample = pred_original_sample.clamp(-1, 1)

        variance = self.one_minus_alphas_cumprod[timestep]
        sigma_t = self.eta * torch.sqrt(variance / (1 - alpha_prod_t_prev))
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - sigma_t**2) * model_output

        noise = torch.randn(sample.shape, generator=generator, device=sample.device) if timestep > 0 else 0
        return torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction + sigma_t * noise


class EulerAncestralScheduler(Scheduler):
    """Euler Ancestral sampler - fast and high quality."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        super().__init__(num_train_timesteps)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def set_timesteps(self, num_inference_steps: int, scheduler_type: str = ""):
        step_ratio = self.num_train_timesteps / num_inference_steps
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).long()
        sigmas = torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)
        self.sigmas = sigmas[self.timesteps]

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, model_output, timestep, sample, generator=None):
        sigma = self.sigmas[self.timesteps == timestep][0] if timestep in self.timesteps else torch.tensor(0.1)
        sigma_next = torch.tensor(0.0)

        pred_original_sample = sample - sigma * model_output
        dt = sigma_next - sigma

        noise = torch.randn(sample.shape, generator=generator, device=sample.device)
        derivative = (model_output + noise / sigma) * dt

        return sample + derivative


class DPMSolverScheduler(Scheduler):
    """DPM-Solver++ scheduler - optimal for 20-30 steps."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012, algorithm_type: str = "dpmsolver++", solver_type: str = "midpoint"):
        super().__init__(num_train_timesteps)
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod

    def set_timesteps(self, num_inference_steps: int, scheduler_type: str = ""):
        step_ratio = self.num_train_timesteps / num_inference_steps
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).long()
        self.sigmas = torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)
        self.sigmas = self.sigmas[self.timesteps]

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, model_output, timestep, sample, generator=None):
        sigma = self.sigmas[self.timesteps == timestep][0] if timestep in self.timesteps else torch.tensor(0.1)

        t_hat = self.sigma_to_t(sigma)
        pred_original_sample = (sample - sigma * model_output) / torch.sqrt(1 + sigma**2)

        dt = sigma / 1.5
        noise = torch.randn(sample.shape, generator=generator, device=sample.device)
        return sample + dt * (model_output - pred_original_sample * sigma)

    def sigma_to_t(self, sigma):
        return self.num_train_timesteps * (1 - sigma)


class PNDMScheduler(Scheduler):
    """Pseudo numerical methods for diffusion models."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        super().__init__(num_train_timesteps)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.ets = []

    def set_timesteps(self, num_inference_steps: int, scheduler_type: str = ""):
        step_ratio = self.num_train_timesteps / num_inference_steps
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).long()
        self.sigmas = torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)
        self.sigmas = self.sigmas[self.timesteps]

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, model_output, timestep, sample, generator=None):
        self.ets.append(model_output)
        if len(self.ets) < 4:
            return sample

        sigma = self.sigmas[self.timesteps == timestep][0] if timestep in self.timesteps else torch.tensor(0.1)
        pred_original_sample = (sample - sigma * model_output)
        dt = self.sigmas[0] - sigma if len(self.sigmas) > 1 else sigma * 0.1

        return pred_original_sample + model_output * dt


SCHEDULERS = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "euler_a": EulerAncestralScheduler,
    "dpm++": DPMSolverScheduler,
    "pndm": PNDMScheduler,
}


def get_scheduler(name: str, **kwargs) -> Scheduler:
    """
    Get scheduler by name.

    Args:
        name: Scheduler name (ddim, ddpm, euler_a, dpm++, pndm).
        **kwargs: Additional kwargs for scheduler constructor.

    Returns:
        Scheduler instance.
    """
    if name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(SCHEDULERS.keys())}")
    return SCHEDULERS[name](**kwargs)
