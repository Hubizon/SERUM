"""
Grid-based watermarking module for latent space manipulation.

This module implements a learnable grid pattern that can be used to inject
watermarks into the latent space of diffusion models.
"""

import glob
import os
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn

from ..utils.config import Config
from ..utils.utils import cleanup_cuda_memory


class Watermark(nn.Module):
    """
    A learnable grid pattern for watermark injection in latent space.
    
    This module maintains a learnable parameter grid that is used to generate
    noise patterns for watermarking. The grid is combined with random noise
    to create watermarked latent representations.
    
    Args:
        config (Config): Configuration object with watermark parameters.
        
    Attributes:
        grid (nn.Parameter): Learnable grid parameter with shape BASE_LATENT_SHAPE.
    """

    def __init__(self, config: Config):
        """
        Initialize the Watermark module.
        
        Args:
            config (Config): Configuration object with watermark parameters.
        """
        super().__init__()

        self.config = config

        self.grid = nn.Parameter(torch.randn(self.config.base_latent_shape))

    def forward(self, batch_size: int, ret_noise: bool = False, alpha: Optional[float] = None) -> torch.Tensor:
        """
        Generate watermarked noise from the grid pattern.
        
        This method combines the learnable grid pattern with random noise to create
        watermarked latent representations. The grid is first clamped to reasonable
        values, then combined with random noise using a weighted average.
        
        Args:
            batch_size (int): Number of samples in the batch.
            digits (torch.Tensor): A tensor of exactly 4 digits to inject.
            ret_noise (bool, optional): If True, also return the original noise.
                Defaults to False.
                
        Returns:
            torch.Tensor: Watermarked noise tensor of shape (batch_size, 4, 64, 64).
            tuple: If ret_noise is True, returns (watermarked_noise, original_noise).
        """
        if alpha is None:
            alpha = self.config.watermark.grid.noise_mix_alpha

        _, C, H, W = self.config.base_latent_shape

        # Generate original random noise
        orig_noise = torch.randn(batch_size, C, H, W).to(self.grid.device)

        # Combine original noise with normalized grid pattern (equal weighting)
        noise = orig_noise * ((1 - alpha) ** 0.5) + (self.grid - self.grid.mean()) * (alpha ** 0.5) / self.grid.std()

        if ret_noise:
            return noise, orig_noise

        return noise

    def save(self, filename: str = None, include_date: bool = True) -> str:
        """
        Save the watermark model to disk.
        
        Args:
            path: Base path for saving
            include_date: Whether to append timestamp to filename
            
        Returns:
            Final path where model was saved
        """
        if filename is None:
            filename = self.config.watermark.watermark_file

        # Create data directory
        os.makedirs(self.config.watermark.model_dir, exist_ok=True)
        filepath = os.path.join(self.config.watermark.model_dir, filename)

        if include_date:
            timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
            base, ext = os.path.splitext(filepath)
            filepath = f"{base}{timestamp}{ext}"

        torch.save(self.state_dict(), filepath)
        print(f"Watermark saved to: {filepath}")
        return filepath

    @staticmethod
    def load(config: Config, filename: Optional[str] = None) -> 'Watermark':
        """
        Load watermark from disk.

        Args:
            filename: Name of the saved model file
            
        Returns:
            Loaded Watermark instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if filename is None:
            filename = config.watermark.watermark_file
        filepath = os.path.join(config.watermark.model_dir, filename)

        if not os.path.exists(filepath):
            # Try to find latest timestamped version
            base, ext = os.path.splitext(filepath)
            pattern = f"{base}_*{ext}"

            files = glob.glob(pattern)
            if files:
                filepath = max(files)  # Get most recent
            else:
                raise FileNotFoundError(f"No watermark found at {filepath}")

        data = torch.load(filepath, map_location=config.get_device())
        
        model = Watermark(config)

        model.load_state_dict(data)
        cleanup_cuda_memory()

        print(f"Watermark loaded from: {filepath}")
        return model
