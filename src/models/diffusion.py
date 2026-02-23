"""
Diffusion model wrappers and implementations for watermarking experiments.

This module provides wrappers around diffusion models (Stable Diffusion) using
diffusers' built-in schedulers and sampling methods.
"""

import random
import shutil
from typing import Any, List, Optional

import torch
from diffusers import AutoPipelineForText2Image
from torch import nn

from .vae import decode
from ..utils.utils import cleanup_cuda_memory
from ..utils.config import Config


class DiffusionModel(nn.Module):
    def __init__(self, config: Config, pipe: Optional[AutoPipelineForText2Image] = None):
        super().__init__()
        self.config = config
        if pipe is None:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.config.diffusion.stable_diffusion_model_id,
                torch_dtype=self.config.diffusion.torch_dtype
            )
            print('Loaded Stable Diffusion model:', self.config.diffusion.stable_diffusion_model_id)
        else:
            self.pipe = pipe
        self.device = self.config.get_device()
        self.pipe.to(self.device)
        self.scheduler = self.pipe.scheduler
        
        # Compile model if available
        if shutil.which('gcc') is not None or shutil.which('clang') is not None:
            print("C compiler found. Compiling the model.")
            for attr in ['transformer', 'unet']:
                if hasattr(self.pipe, attr):
                    model = getattr(self.pipe, attr)
                    if hasattr(model, 'compile'):
                        model.compile()
        else:
            print("No C compiler found. Skipping model compilation.")


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, steps: Optional[int] = None, cfg_scale: Optional[float] = None):
        super().__init__()
        self.config = model.config
        self.model = model
        self.steps = steps if steps is not None else self.config.sampling.default_steps
        self.cfg_scale = cfg_scale if cfg_scale is not None else self.config.sampling.cfg_scale
        self.device = self.config.get_device()
        cleanup_cuda_memory()

    def predict_latent(self, positive_prompts: List[str], negative_prompts: List[str], 
                      x_t: Optional[torch.Tensor] = None, x_ts_ret: bool = False) -> torch.Tensor:
        """Generate latent representations from text prompts using the diffusion pipeline."""
        if x_t is None:
            latents = torch.randn(
                (len(positive_prompts), self.config.shapes.latent_channels, 
                 self.config.shapes.latent_height, self.config.shapes.latent_width),
                device=self.device,
                dtype=self.config.diffusion.torch_dtype
            )
        else:
            latents = x_t.to(device=self.device, dtype=self.config.diffusion.torch_dtype)

        result = self.model.pipe(
            prompt=positive_prompts,
            negative_prompt=negative_prompts,
            num_inference_steps=self.steps,
            guidance_scale=self.cfg_scale,
            latents=latents,
            output_type="latent",
            return_dict=True,
        )
        
        return result.images

    def predict_image(self, positive_prompts: List[str], negative_prompts: List[str], **kwargs):
        """Generate and decode images from text prompts."""
        latents = self.predict_latent(positive_prompts, negative_prompts, **kwargs)
        return decode(latents.to(torch.float16))

    def generate_full(self, batch_size: int, noise: Optional[torch.Tensor] = None, 
                      x_ts_ret: bool = False, prompts: Optional[List[str]] = None, 
                      positive_prompts: Optional[List[str]] = None,
                      negative_prompts: Optional[List[str]] = None, ret_prompts: bool = False):
        """Generate latents with optional random prompt selection."""
        if not positive_prompts:
            positive_prompts = [prompts[random.randint(0, len(prompts) - 1)] for _ in range(batch_size)]
        
        if not negative_prompts:
            negative_prompts = [''] * len(positive_prompts)

        latents = self.predict_latent(positive_prompts, negative_prompts, noise, x_ts_ret)
        
        if ret_prompts:
            return latents, positive_prompts, negative_prompts
        return latents
