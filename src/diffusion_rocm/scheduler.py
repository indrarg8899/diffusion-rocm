"""Diffusion schedulers: DDIM, PNDM, Euler."""

import math
import torch
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class SchedulerType(Enum):
    DDIM = "ddim"
    PNDM = "pndm"
    EULER = "euler"
    EULER_A = "euler_ancestral"
    DPM_SOLVER = "dpm_solver"


@dataclass
class SchedulerState:
    timesteps: List[int]
    alphas_cumprod: torch.Tensor
    init_noise_sigma: float = 1.0


class DDIMScheduler:
    """DDIM scheduler with stochastic and deterministic modes."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        steps_offset: int = 1,
        clip_sample: bool = False,
        prediction_type: str = "epsilon",
        eta: float = 0.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.eta = eta
        self.prediction_type = prediction_type

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def set_timesteps(self, num_inference_steps: int) -> SchedulerState:
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = [
            int(i * step_ratio) + self.steps_offset
            for i in range(num_inference_steps)
        ]
        return SchedulerState(
            timesteps=timesteps,
            alphas_cumprod=self.alphas_cumprod,
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[timestep - 1] if timestep > 0
            else torch.tensor(1.0)
        )

        pred_original = (
            (sample - (1 - alpha_prod_t).sqrt() * model_output)
            / alpha_prod_t.sqrt()
        )

        if self.prediction_type == "v_prediction":
            pred_original = alpha_prod_t.sqrt() * sample - (1 - alpha_prod_t).sqrt() * model_output

        pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * model_output

        prev_sample = alpha_prod_t_prev.sqrt() * pred_original + pred_sample_direction

        if self.eta > 0:
            variance = self._get_variance(timestep, alpha_prod_t_prev)
            noise = torch.randn_like(sample) if generator is None else torch.randn(
                sample.shape, generator=generator, device=sample.device
            )
            prev_sample = prev_sample + self.eta * variance.sqrt() * noise

        return prev_sample

    def _get_variance(self, timestep: int, alpha_prod_t: torch.Tensor) -> torch.Tensor:
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)
        return (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)


class EulerScheduler:
    """Euler sampling scheduler."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ):
        self.num_train_timesteps = num_train_timesteps
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod).sqrt()

    def set_timesteps(self, num_inference_steps: int) -> SchedulerState:
        step_size = self.num_train_timesteps // num_inference_steps
        timesteps = list(range(self.num_train_timesteps - 1, 0, -step_size))
        return SchedulerState(timesteps=timesteps, alphas_cumprod=self.alphas_cumprod)

    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor
    ) -> torch.Tensor:
        sigma = self.sigmas[timestep]
        pred_original = sample - sigma * model_output
        step_size = self.sigmas[timestep - 1] - sigma if timestep > 0 else sigma
        return pred_original + step_size * model_output
