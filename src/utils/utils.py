"""
General utility functions for the watermarking project.

This module provides common utility functions including device management,
memory cleanup, and configuration handling.
"""

import gc
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_curve, balanced_accuracy_score

from .config import Config

# Global device variable for memory management
device: Optional[str] = None


def setup_device(_device: str) -> None:
    """Set up the global device for memory management operations."""
    global device
    device = _device


def cleanup_cuda_memory(verbose: bool = False) -> None:
    """
    Clean up CUDA memory by clearing cache and collecting garbage.
    
    Args:
        verbose: Whether to print memory statistics after cleanup
    """
    global device
    if not torch.cuda.is_available() or device == 'cpu':
        return

    if device is None:
        device = 'cuda'

    with torch.cuda.device(device):
        for _ in range(2):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        if verbose:
            allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
            cached = torch.cuda.memory_reserved(device) / 1024 ** 3
            free, total = torch.cuda.mem_get_info(device)
            print(
                f"CUDA Memory - Allocated: {allocated:.2f}GB, "
                f"Cached: {cached:.2f}GB, "
                f"Free: {free / 1024 ** 3:.2f}GB, "
                f"Total: {total / 1024 ** 3:.2f}GB"
            )


def save_config_snapshot(config: Config, 
                         prefix: str = "config", 
                         use_timestamp: bool = True,
                         output_dir: Optional[str] = None) -> str:
    """
    Save a snapshot of the current configuration to YAML file.
    
    Args:
        config: Configuration object to save
        prefix: Prefix for the config filename
        use_timestamp: Whether to include timestamp in filename
        output_dir: Directory to save config to (defaults to config.experiment_dir)
        
    Returns:
        Path where config was saved
    """
    if output_dir is None:
        output_dir = config.experiment_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{prefix}_{timestamp}.yaml"
    else:
        filename = f"{prefix}.yaml"

    filepath = os.path.join(output_dir, filename)

    # Save current config to file
    with open(filepath, 'w') as f:
        yaml.dump(config._raw, f)

    print(f"Config snapshot saved to: {filepath}")
    return filepath


def compute_roc_metrics(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, int, Optional[int]]:
    """
    Compute ROC metrics and find optimal threshold.
    
    Args:
        y_true: True binary labels
        scores: Predicted scores
        target_fpr: Target false positive rate for threshold finding
        
    Returns:
        Tuple containing (best_balanced_accuracy, fpr, tpr, thresholds, idx_at_target_fpr, idx_at_tpr_1)
    """
    if hasattr(y_true, 'detach'):  # Handle torch tensors
        y_true = y_true.detach().cpu().numpy()
    if hasattr(scores, 'detach'):  # Handle torch tensors
        scores = scores.detach().cpu().numpy()

    # Find optimal threshold for balanced accuracy
    sorted_scores = np.sort(np.unique(scores))
    thresholds = np.concatenate([[-np.inf], (sorted_scores[:-1] + sorted_scores[1:]) / 2, [np.inf]])

    best_bal_acc = 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        bal_acc = balanced_accuracy_score(y_true, preds)
        best_bal_acc = max(best_bal_acc, bal_acc)

    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)

    # drop the trivial first point (0,0) if it exists
    mask = ~((fpr == 0) & (tpr == 0))
    fprm, tprm = fpr[mask], tpr[mask]

    # find candidate indices
    leq = np.where(fprm <= target_fpr)[0]
    geq = np.where(fprm >= target_fpr)[0]

    if len(leq) > 0:
        idx = leq[-1]  # largest index with FPR <= target
    elif len(geq) > 0:
        idx = geq[0]  # smallest index with FPR >= target
    else:
        idx = 0  # shouldn't happen if ROC is valid

    # first index where tpr == 1.0 (use isclose to avoid FP equality issues)
    tol = 1e-8
    idx2_candidates = np.where(np.isclose(tprm, 1.0, atol=tol))[0]
    idx2 = int(idx2_candidates[0]) if idx2_candidates.size > 0 else None

    offset = int((~mask).sum())
    idx += offset
    idx2 += offset if idx2 is not None else None

    return best_bal_acc, fpr, tpr, roc_thresholds, idx, idx2
