"""
VAE utilities for encoding and decoding between image and latent space.
Provides functions for converting between PIL images and latent tensors using Stable Diffusion VAE.
"""

from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image

from ..utils.config import Config
from ..utils.utils import setup_device


class VAEWrapper:
    """
    Wrapper class for Stable Diffusion VAE with encode/decode functionality.
    """

    def __init__(self, config: Config, model_id: Optional[str] = None, 
                 torch_dtype: Optional[torch.dtype] = None, device: Optional[str] = None):
        """
        Initialize VAE wrapper.
        
        Args:
            config: Configuration object
            model_id: Model ID for Stable Diffusion pipeline
            torch_dtype: Torch data type
            device: Device to load model on
        """
        self.config = config
        if model_id is None:
            model_id = self.config.diffusion.stable_diffusion_model_id
        if torch_dtype is None:
            torch_dtype = self.config.diffusion.torch_dtype
        if device is None:
            device = self.config.get_device()

        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device = device
        self.vae = None
        self._load_vae()

    def _load_vae(self) -> None:
        """Load and configure the VAE model."""
        print('Loading VAE...')
        self.vae = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype
        ).vae.to(self.device).requires_grad_(False)
        
        self.scaling_factor = self.vae.config.scaling_factor
        print(f'VAE scaling factor: {self.scaling_factor}')

        # Compile for better performance
        if hasattr(self.vae.encoder, 'compile'):
            self.vae.encoder.compile()
        if hasattr(self.vae.decoder, 'compile'):
            self.vae.decoder.compile()

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode image tensor to latent space.
        
        Args:
            img: Image tensor with shape (B, C, H, W) and values in [-1, 1]
            
        Returns:
            Latent tensor with shape (B, 4, H//8, W//8)
        """
        with torch.no_grad():
            img = img.to(self.device)
            return self.vae.encode(img).latent_dist.sample() * self.scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor to image space.
        
        Args:
            latent: Latent tensor with shape (B, 4, H//8, W//8)
            
        Returns:
            Image tensor with shape (B, C, H, W) and values in [-1, 1]
        """
        with torch.no_grad():
            latent = latent.to(self.device)
            return self.vae.decode(latent / self.scaling_factor).sample.clamp(-1, 1)


# Global VAE instance for convenience functions
_vae_instance: Optional[VAEWrapper] = None


def get_vae(config: Config = None) -> VAEWrapper:
    """Get or create global VAE instance."""
    global _vae_instance
    if _vae_instance is None:
        if config is None:
            raise ValueError("Config must be provided for initial VAE creation. Just call get_vae(config) once first.")
        setup_device(config.get_device())
        _vae_instance = VAEWrapper(config)
    return _vae_instance


def encode(img: torch.Tensor) -> torch.Tensor:
    """
    Convenience function for encoding using global VAE instance.
    
    Args:
        img: Image tensor with shape (B, C, H, W) and values in [-1, 1]
        
    Returns:
        Latent tensor with shape (B, 4, H//8, W//8)
    """
    return get_vae().encode(img.half())


def decode(latent: torch.Tensor) -> torch.Tensor:
    """
    Convenience function for decoding using global VAE instance.
    
    Args:
        latent: Latent tensor with shape (B, 4, H//8, W//8)
        
    Returns:
        Image tensor with shape (B, C, H, W) and values in [-1, 1]
    """
    return get_vae().decode(latent.half())
