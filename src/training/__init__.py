"""
Training module for watermark models.

This module provides the Trainer class and training utilities for both
pretraining and watermark-specific training phases.
"""

from .aug_sampler import AugSampler
from .train import Trainer

__all__ = [
    'Trainer',
    'AugSampler'
]
