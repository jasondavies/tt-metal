# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union
import ttnn
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
import ttnn.device


class TtEulerDiscreteScheduler(nn.Module):
    def __init__(
        self,
        device: ttnn.device.Device,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        timestep_spacing: str = "linspace",
        timestep_type: str = "discrete",  # can be "discrete" or "continuous"
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        final_sigmas_type: str = "zero",  # can be "zero" or "sigma_min"
    ):
        # implements the Euler Discrete Scheduler with default params as in
        # https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/scheduler/scheduler_config.json
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        assert beta_schedule == "scaled_linear", "beta_schedule {beta_schedule} is not supported in this version"
        self.beta_schedule = beta_schedule
        assert trained_betas is None, "trained_betas is not supported in this version"
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        assert prediction_type == "epsilon", "prediction_type {prediction_type} is not supported in this version"
        self.prediction_type = prediction_type
        assert (
            interpolation_type == "linear"
        ), "interpolation_type {interpolation_type} is not supported in this version"
        self.interpolation_type = interpolation_type
        assert use_karras_sigmas == False, "karras sigmas are not supported in this version"
        assert use_exponential_sigmas == False, "exponential sigmas are not supported in this version"
        assert use_beta_sigmas == False, "beta sigmas are not supported in this version"
        assert sigma_min is None, "sigma_min is not supported in this version"
        assert sigma_max is None, "sigma_max is not supported in this version"
        assert timestep_spacing == "leading", "timestep_spacing {timestep_spacing} is not supported in this version"
        self.timestep_spacing = timestep_spacing
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        assert timestep_type == "discrete", "timestep_type {timestep_type} is not supported in this version"
        self.timestep_type = timestep_type
        self.steps_offset = steps_offset
        assert rescale_betas_zero_snr == False, "rescale_betas_zero_snr is not supported in this version"
        assert final_sigmas_type == "zero", "final_sigmas_type {final_sigmas_type} is not supported in this version"

        self.final_sigmas_type = final_sigmas_type
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.num_inference_steps = None
        sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0)
        self.sigmas = torch.cat([sigmas, torch.zeros(1)])
        self.is_scale_input_called = False
        self.step_index = None
        self.begin_index = None
        self.device = device
        self.create_ttnn_timesteps(timesteps)
        self.create_ttnn_sigmas("sigmas")

    def inc_step_index(self):
        self.set_step_index(self.step_index + 1)

    def set_step_index(self, step_index: int):
        self.step_index = step_index

        # Note: For each iteration we copy over 4 tensors to locations expected by trace
        self.update_device_sigmas()
        self.update_device_timestep()
        self.update_device_norm_factor()

    def create_ttnn_sigmas(self, tensor_name):
        array = getattr(self, tensor_name)
        setattr(self, "tt_" + tensor_name, [])
        tt_array = getattr(self, "tt_" + tensor_name)

        for val in array:
            tt_array.append(
                ttnn.from_torch(
                    val,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
            )
        sigma_step = self.tt_sigmas[0]
        self.tt_sigma_step = ttnn.allocate_tensor_on_device(
            sigma_step.shape,
            sigma_step.dtype,
            sigma_step.layout,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        sigma_next_step = self.tt_sigmas[1]
        self.tt_sigma_next_step = ttnn.allocate_tensor_on_device(
            sigma_next_step.shape,
            sigma_next_step.dtype,
            sigma_next_step.layout,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    def update_device_sigmas(self):
        ttnn.copy_host_to_device_tensor(self.tt_sigmas[self.step_index], self.tt_sigma_step)
        ttnn.copy_host_to_device_tensor(self.tt_sigmas[self.step_index + 1], self.tt_sigma_next_step)

    def create_ttnn_timesteps(self, timesteps):
        self.timesteps = []
        for t in timesteps:
            self.timesteps.append(
                ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
            )
        tt_timestep_step = self.timesteps[0]
        self.tt_timestep = ttnn.allocate_tensor_on_device(
            tt_timestep_step.shape,
            tt_timestep_step.dtype,
            tt_timestep_step.layout,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    def update_device_timestep(self):
        ttnn.copy_host_to_device_tensor(self.timesteps[self.step_index], self.tt_timestep)

    def create_ttnn_norm_factor(self, variance_normalization_factor):
        self.variance_normalization_factor = []
        for val in variance_normalization_factor:
            self.variance_normalization_factor.append(
                ttnn.from_torch(
                    val,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
            )
        tt_val_step = self.variance_normalization_factor[0]
        self.tt_norm_factor = ttnn.allocate_tensor_on_device(
            tt_val_step.shape,
            tt_val_step.dtype,
            tt_val_step.layout,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    def update_device_norm_factor(self):
        ttnn.copy_host_to_device_tensor(self.variance_normalization_factor[self.step_index], self.tt_norm_factor)

    # pipeline_stable_diffusion_xl.py __call__() step #4
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        """
        assert timesteps == None, "timesteps is not supported in this version"
        assert sigmas == None, "sigmas is not supported in this version"
        assert num_inference_steps != None, "num_inference_steps cannot be None in this version"

        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        assert self.timestep_spacing == "leading"
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
        timesteps += self.steps_offset

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)
        assert self.interpolation_type == "linear"
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        assert self.final_sigmas_type == "zero"
        sigma_last = 0

        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        assert self.timestep_type == "discrete"
        timesteps = torch.from_numpy(timesteps.astype(np.float32)).to(device=device)
        self.create_ttnn_timesteps(timesteps)

        self.begin_index = 0

        self.sigmas = sigmas
        variance_normalization_factor = (sigmas**2 + 1) ** 0.5
        self.create_ttnn_norm_factor(variance_normalization_factor)

        self.create_ttnn_sigmas("sigmas")
        self.set_step_index(self.begin_index)

    @property
    def init_noise_sigma(self):
        """
        standard deviation of the initial noise distribution.
        """
        max_sigma = self.sigmas.max()
        assert (
            self.timestep_spacing == "leading"
        ), "timestep_spacing {self.timestep_spacing} is not supported in this version"
        return (max_sigma**2 + 1) ** 0.5

    def scale_model_input(
        self, sample: ttnn._ttnn.tensor.Tensor, timestep: Union[float, ttnn._ttnn.tensor.Tensor]
    ) -> ttnn._ttnn.tensor.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
        """
        # timestep is not used in this implementation, step_index is already initialized at set_timesteps()
        # Note: Don't use inplace op here since UNet deallocates its input
        #       Permanent input is required for tracing
        sample = ttnn.div(sample, self.tt_norm_factor)

        self.is_scale_input_called = True
        return sample

    def step(
        self,
        model_output: ttnn._ttnn.tensor.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: ttnn._ttnn.tensor.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ) -> Tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).
        """

        assert timestep is None, "timestep is expected to be None"

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        assert self.step_index is not None, "_init_step_index() should be None before calling step()"
        assert generator is None, "generator is not supported in this version"
        assert return_dict == False, "return_dict==true is not supported in this version"

        # this is a potential accuracy pitfall
        # Upcast to avoid precision issues when computing prev_sample
        # sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
        assert gamma == 0, "gamma > 0 is not supported in this version"

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        assert self.prediction_type == "epsilon"

        # 2. Convert to an ODE derivative
        rec = ttnn.reciprocal(self.tt_sigma_step)

        model_output = ttnn.mul_(model_output, self.tt_sigma_step)
        model_output = ttnn.mul_(model_output, rec)

        dt = self.tt_sigma_next_step - self.tt_sigma_step
        model_output = ttnn.mul_(model_output, dt)

        prev_sample = ttnn.add_(sample, model_output)

        # Note: Step index inc moved out of step func as it is done on host

        # Note: We return None for pred_original_sample since it is never used
        return (prev_sample, None)
