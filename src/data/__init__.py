"""
Data handling module for watermarking experiments.

This module provides dataset loading, augmentation, and data preparation utilities.
"""

from .augmentation import Augment
from .dataset import Dataset

__all__ = [
    'Dataset',
    'Augment',
]
