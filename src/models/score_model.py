"""
Watermark scoring and generation utilities.

This module implements watermarking functionality for diffusion models,
including score calculation, gradient-based guidance, and watermarked
image generation using the LlamaVisionTransformer model.
"""

import glob
import os
from datetime import datetime
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.config import Config
from ..utils.utils import cleanup_cuda_memory


class WatermarkScoreModel(nn.Module):
    def __init__(self, config: Config):
        super(WatermarkScoreModel, self).__init__()
        self.config = config

        # Input: [B, latent_channels, height, width]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.config.shapes.latent_channels, 32, kernel_size=3, stride=1, padding=1),  # -> [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> [B, 32, 32, 32]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> [B, 64, 16, 16]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> [B, 128, 8, 8]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # -> [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> [B, 256, 4, 4]
        )

        # Flatten size = 256 * 4 * 4 = 4096
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)

        return torch.sigmoid(x)

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
            filename = self.config.watermark.score_model_file

        # Create data directory
        os.makedirs(self.config.watermark.model_dir, exist_ok=True)
        filepath = os.path.join(self.config.watermark.model_dir, filename)

        if include_date:
            timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
            base, ext = os.path.splitext(filepath)
            filepath = f"{base}{timestamp}{ext}"

        torch.save(self.state_dict(), filepath)
        print(f"Watermark Score Model saved to: {filepath}")
        return filepath

    @staticmethod
    def load(config: Config, filename: str = None) -> 'WatermarkScoreModel':
        """
        Load watermark model from disk.
        
        Args:
            path: Path to saved model file
            
        Returns:
            Loaded WatermarkScoreModel instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if filename is None:
            filename = config.watermark.score_model_file

        filepath = os.path.join(config.watermark.model_dir, filename)

        if not os.path.exists(filepath):
            # Try to find latest timestamped version
            base, ext = os.path.splitext(filepath)
            pattern = f"{base}_*{ext}"

            files = glob.glob(pattern)
            if files:
                filepath = max(files)  # Get most recent
            else:
                raise FileNotFoundError(f"No Watermark Score Model found at {filepath}")

        data = torch.load(filepath, map_location=config.get_device())
        model = WatermarkScoreModel(config)
        model.load_state_dict(data)
        cleanup_cuda_memory()

        print(f"Watermark Score Model loaded from: {filepath}")
        return model
