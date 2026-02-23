"""
Exponential Moving Average (EMA) utility for PyTorch models.

This module provides EMA functionality commonly used in deep learning to maintain
a smoothed version of model parameters during training, which often leads to better
performance and more stable inference results.
"""

import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class EMA:
    """
    Exponential Moving Average for PyTorch models.
    
    Maintains a smoothed version of model parameters using exponential moving average.
    This technique is commonly used to improve model stability and performance during
    training and inference.
    
    The EMA update rule is:
        ema_param = beta * ema_param + (1 - beta) * current_param
    
    Attributes:
        step: Current training step counter
        beta: EMA decay factor (higher = more smoothing)
        start_step: Step at which to start EMA updates
        ema_model: The EMA version of the model
        master_model: Reference to the original training model
    """

    def __init__(self,
                 master_model: Optional[nn.Module] = None,
                 beta: float = 0.9985,
                 ema_start_step: int = 2000):
        """
        Initialize EMA tracker.
        
        Args:
            master_model: The model to track with EMA. If None, must be set later.
            beta: EMA decay factor. Higher values (closer to 1) mean more smoothing.
                 Typical values: 0.999, 0.9999, 0.99999
            ema_start_step: Training step at which to start EMA updates.
                          Before this step, EMA copies parameters directly.
        """
        self.step = 0
        self.beta = beta
        self.start_step = ema_start_step

        if master_model is None:
            self.ema_model = None
            self.master_model = None
        else:
            # Create a deep copy for EMA and set to eval mode
            self.ema_model = copy.deepcopy(master_model).eval().requires_grad_(False)
            self.master_model = master_model

    def update(self) -> None:
        """
        Update EMA parameters with current master model parameters.
        
        Should be called after each training step. Before ema_start_step,
        parameters are copied directly. After ema_start_step, uses EMA update rule.
        
        Raises:
            Exception: If master_model is None
        """
        if self.master_model is None:
            raise Exception('Master model is None: cannot update')

        # Use direct copy before start_step, EMA after
        beta = self.beta if self.step > self.start_step else 0

        # Update each parameter using EMA rule
        for current_param, target_param in zip(self.ema_model.parameters(), self.master_model.parameters()):
            current_param.data = beta * current_param.data + (1 - beta) * target_param.data

        self.step += 1

    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for saving EMA state.
        
        Returns:
            Dictionary containing EMA model state and metadata
        """
        return {
            'state_dict': self.ema_model.state_dict(),
            'step': self.step,
            'beta': self.beta,
            'start_step': self.start_step
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load EMA state from state dictionary.
        
        Args:
            state_dict: Dictionary containing EMA state and metadata
        """
        self.ema_model.load_state_dict(state_dict['state_dict'])
        self.step = state_dict['step']
        self.beta = state_dict['beta']
        self.start_step = state_dict['start_step']

    def train(self) -> None:
        """
        Set EMA model to training mode.
        
        Note: EMA models are typically kept in eval mode, but this provides
        flexibility for specific use cases.
        """
        self.ema_model.train()

    def eval(self) -> None:
        """
        Set EMA model to evaluation mode.
        
        This is the typical mode for EMA models during inference.
        """
        self.ema_model.eval()

    def __call__(self, *args, **kwargs):
        """
        Forward pass through EMA model.
        
        Allows using the EMA object directly as a model for inference.
        
        Args:
            *args, **kwargs: Arguments passed to the EMA model
            
        Returns:
            Output from EMA model forward pass
        """
        return self.ema_model(*args, **kwargs)
