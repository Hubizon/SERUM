"""
Evaluation module for watermark detection models.

This module provides comprehensive evaluation capabilities including metrics,
visualization, and robustness testing.
"""

from . import generate, scores, visualize
from .eval import Evaluation

__all__ = [
    'Evaluation',
    'generate',
    'scores',
    'visualize'
]
