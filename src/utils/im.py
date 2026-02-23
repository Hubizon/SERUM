"""
Image utilities module for watermark detection.

This module provides comprehensive image processing utilities including:
- Image loading, saving, and format conversion
- Tensor/PIL image transformations
- Visualization and plotting functions

The module handles various tensor formats and provides flexible visualization
options for debugging and analysis of watermark detection models.
"""

import io
import logging
import math
import os
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as PLT

# Suppress matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def prepare(ts: torch.Tensor, v_range: Optional[str] = None) -> torch.Tensor:
    """
    Prepare tensor for visualization by normalizing and reshaping.
    
    This function handles various tensor formats and normalizes values
    to the range [0, 1] for proper visualization.
    
    Args:
        ts: Input tensor of various shapes
        v_range: Value range specification:
            - None: Assumes range (-1, 1)
            - '01': Assumes range (0, 1) 
            - 'any': Uses actual min/max values
            - 'per_image': Normalizes each image individually
            
    Returns:
        Normalized tensor in format (N, 3, H, W) suitable for visualization
        
    Note:
        Automatically converts grayscale to RGB by repeating channels.
        Handles 2D, 3D, and 4D input tensors flexibly.
    """
    # Reshape tensor to (N, C, H, W) format
    if len(ts.shape) == 2:
        h, w = ts.shape
        ts = ts.view(1, 1, h, w)
    elif len(ts.shape) == 3:
        N, h, w = ts.shape
        if N == 3:
            ts = ts.view(1, 3, h, w)
        else:
            ts = ts.view(N, 1, h, w)
    elif len(ts.shape) == 4:
        a, b, h, w = ts.shape
        if a == 1:
            if b not in [1, 3]:
                ts = ts.view(b, 1, h, w)
        elif b == 1:
            pass  # Already in correct format
        elif a == 3 and b not in [3, 1]:
            ts = ts.view(b, 3, h, w)

    # Convert grayscale to RGB
    if ts.shape[1] == 1:
        ts = ts.repeat(1, 3, 1, 1)

    ts = ts.detach().cpu().float()

    # Normalize based on specified range
    min_v = -1
    max_v = 1

    if v_range == '01':
        min_v = 0
        max_v = 1
    elif v_range == 'any':
        min_v = torch.min(ts)
        max_v = torch.max(ts)
    elif v_range == 'per_image':
        max_v = ts.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        min_v = ts.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]

    # Normalize to [0, 1]
    ts = (ts - min_v) / (max_v - min_v + 1e-6)

    return ts


@torch.no_grad()
def show(ts: torch.Tensor,
         title: Optional[str] = None,
         v_range: Optional[str] = None,
         figsize: Union[int, Tuple[int, int]] = 4,
         show_plot: bool = True) -> None:
    """
    Display tensor as image(s) using matplotlib.
    
    Automatically handles single images or batches, creating appropriate
    grid layouts for multiple images.
    
    Args:
        ts: Input tensor to visualize
        title: Optional title for the plot
        v_range: Value range for normalization (see prepare() function)
        figsize: Figure size (int for square, tuple for (width, height))
        show_plot: Whether to display the plot immediately
        
    Note:
        For batches, automatically determines optimal grid layout.
        Supports various tensor formats through the prepare() function.
    """
    ts = prepare(ts, v_range)
    N, _, h, w = ts.shape

    # Handle figsize
    if isinstance(figsize, int):
        fig_width = fig_height = figsize
    else:
        fig_width, fig_height = figsize

    PLT.figure(figsize=(fig_width, fig_height))

    if title is not None:
        PLT.title(title, fontsize=14, fontweight='bold')
        PLT.axis("off")

    if ts.shape[0] > 1:
        # Determine optimal grid layout for multiple images
        if ts.shape[0] <= 16:
            nrow = 4
        elif ts.shape[0] <= 25:
            nrow = 5
        elif ts.shape[0] <= 36:
            nrow = 6
        elif ts.shape[0] <= 49:
            nrow = 7
        else:
            nrow = 8

        grid = vutils.make_grid(ts, padding=2, nrow=nrow)
        if show_plot:
            PLT.imshow(np.transpose(grid, (1, 2, 0)))
            PLT.axis("off")
            PLT.show()
    elif show_plot:
        PLT.imshow(ts.squeeze(dim=0).permute((1, 2, 0)))
        PLT.axis("off")
        PLT.show()


@torch.no_grad()
def show_grid_2x2(images_list: List[torch.Tensor],
                  titles_list: List[str],
                  v_range: Optional[str] = None,
                  figsize: int = 12) -> None:
    """
    Display 4 sets of images in a 2x2 grid layout.
    
    Perfect for comparing different processing stages or augmentations
    of the same images (e.g., original, watermarked, augmented versions).
    
    Args:
        images_list: List of exactly 4 tensor batches to display
        titles_list: List of exactly 4 titles for each subplot
        v_range: Value range for normalization (see prepare() function)
        figsize: Figure size for the entire grid
        
    Raises:
        ValueError: If not exactly 4 image sets and titles provided
    """
    if len(images_list) != 4 or len(titles_list) != 4:
        raise ValueError("show_grid_2x2 requires exactly 4 image sets and 4 titles")

    fig, axes = PLT.subplots(2, 2, figsize=(figsize, figsize))
    axes = axes.flatten()

    for i, (images, title) in enumerate(zip(images_list, titles_list)):
        ts = prepare(images, v_range)

        # Create grid for multiple images or show single image
        if ts.shape[0] > 1:
            # Calculate optimal grid layout for subplot
            num_images = ts.shape[0]
            nrow = int(np.ceil(np.sqrt(num_images)))
            grid = vutils.make_grid(ts, padding=2, nrow=nrow)
            axes[i].imshow(np.transpose(grid, (1, 2, 0)))
        else:
            axes[i].imshow(ts.squeeze(dim=0).permute((1, 2, 0)))

        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].axis('off')

    PLT.tight_layout()
    PLT.show()


def save_evaluation_plots(config, results: Dict[str, Any], eval_type: str, aug_name: str = None,
                         use_timestamp: bool = False) -> None:
    """
    Save evaluation plots to EVAL_OUTPUT_PATH.
    
    Args:
        config: Configuration object containing eval_output_path
        results: Results dictionary containing scores and metrics
        eval_type: Type of evaluation ("eval_tpr" or "augmentation")
        aug_name: Name of augmentation (only for augmentation_robustness_eval)
        use_timestamp: Whether to add timestamp to filenames
    """
    os.makedirs(config.evaluation.eval_output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create filename prefix
    if aug_name:
        prefix = f"{eval_type}_{aug_name}"
    else:
        prefix = f"{eval_type}"
    if use_timestamp:
        prefix += f"_{timestamp}"

    # Save score distribution plot
    plt.figure(figsize=(8, 5))
    plt.title(f'Watermark Score Distribution - {eval_type}' + (f' - {aug_name}' if aug_name else ''))
    log_bins = np.logspace(-9, 0, 50)

    if 'non_watermarked_scores' in results:
        plt.hist(results['non_watermarked_scores'].cpu(), label='generated', alpha=.5, bins=log_bins, color='green')
    if 'watermarked_scores' in results:
        plt.hist(results['watermarked_scores'].cpu(), label='generated + watermark', alpha=.5, bins=log_bins,
                 color='blue')
    if 'non_watermarked_scores_aug' in results:
        plt.hist(results['non_watermarked_scores_aug'].cpu(), label='generated (perturbations)', alpha=.5,
                 bins=log_bins, color='lime')
    if 'watermarked_scores_aug' in results:
        plt.hist(results['watermarked_scores_aug'].cpu(), label='generated (perturbations) + watermark', alpha=.5,
                 bins=log_bins, color='cyan')

    plt.xscale('log')
    plt.legend()
    plt.savefig(os.path.join(config.evaluation.eval_output_path, f"{prefix}_score_distribution.png"), dpi=300,
                bbox_inches='tight')
    plt.close()

    # Save ROC curve if available
    if 'fpr' in results and 'tpr' in results and 'roc_auc' in results:
        plt.figure(figsize=(8, 5))
        plt.plot(results['fpr'], results['tpr'], color='darkorange', lw=2,
                 label=f"ROC curve (AUC = {results['roc_auc']:.5f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {eval_type}" + (f' - {aug_name}' if aug_name else ''))
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(config.evaluation.eval_output_path, f"{prefix}_roc_curve.png"), dpi=300,
                    bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {config.evaluation.eval_output_path} with prefix {prefix}")


def save_augmentation_summary_plot(config, augmentation_results: Dict[str, Tuple], 
                                   compute_roc_fn, use_timestamp: bool = False) -> None:
    """
    Save summary plots showing results for all augmentations.
    
    Args:
        config: Configuration object containing eval_output_path
        augmentation_results: Dictionary mapping augmentation names to (tpr, fpr, roc_auc, scores...)
        compute_roc_fn: Function to compute ROC metrics 
        use_timestamp: Whether to add timestamp to filenames
    """
    os.makedirs(config.evaluation.eval_output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Extract data for plotting
    aug_names = list(augmentation_results.keys())
    tprs = [res[0] for res in augmentation_results.values()]
    fprs = [res[1] for res in augmentation_results.values()]
    aucs = [res[2] for res in augmentation_results.values()]

    # Create bar plot for TPR comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(aug_names)), [tpr * 100 for tpr in tprs], alpha=0.7)
    plt.xlabel('Augmentation Type')
    plt.ylabel('TPR (%)')
    plt.title('TPR by Augmentation Type')
    plt.xticks(range(len(aug_names)), aug_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, tpr) in enumerate(zip(bars, tprs)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{tpr * 100:.1f}%', ha='center', va='bottom')

    # Create bar plot for AUC comparison
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(len(aug_names)), aucs, alpha=0.7, color='orange')
    plt.xlabel('Augmentation Type')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC by Augmentation Type')
    plt.xticks(range(len(aug_names)), aug_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # Add value labels on bars
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{auc:.5f}', ha='center', va='bottom')

    plt.tight_layout()
    filename_base = f"augmentation_summary_{timestamp}" if use_timestamp else "augmentation_summary"
    filename = os.path.join(config.evaluation.eval_output_path, f"{filename_base}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Create combined ROC curves plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(augmentation_results)))

    for i, (aug_name, (tpr, fpr, roc_auc, non_watermarked_scores, watermarked_scores)) in enumerate(
            augmentation_results.items()):
        # Recompute full ROC curve for plotting (we need the full curves, not just the point at target FPR)
        y_true = torch.cat([
            torch.zeros_like(non_watermarked_scores),
            torch.ones_like(watermarked_scores)
        ])
        scores = torch.cat([non_watermarked_scores, watermarked_scores])

        _, full_fpr, full_tpr, _, _, _ = compute_roc_fn(y_true, scores)

        plt.plot(full_fpr, full_tpr, color=colors[i], lw=2,
                 label=f"{aug_name} (AUC = {roc_auc:.5f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Augmentations")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(config.evaluation.eval_output_path, f"augmentation_roc_comparison_{timestamp}.png"),
        dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Augmentation summary plots saved to {config.evaluation.eval_output_path}")


def plot_evaluation_results(results: Dict[str, Any]) -> None:
    """
    Create comprehensive plots for evaluation results.
    
    Args:
        results: Results from evaluation containing scores and metrics
    """
    if results is None:
        print("No evaluation results available.")
        return

    # Score distribution plot
    def visualize_scores():
        """Display histogram of watermark scores for different sample types."""
        plt.figure(None, (8, 5))
        plt.title('Watermark score')
        log_bins = np.logspace(-10, 0, 50)
        plt.hist(results['non_watermarked_scores'].cpu(), label='generated', alpha=.5, bins=log_bins, color='green')
        plt.hist(results['watermarked_scores'].cpu(), label='generated + watermark', alpha=.5, bins=log_bins,
                 color='blue')
        plt.hist(results['non_watermarked_scores_aug'].cpu(), label='generated (perturbations)', alpha=.5,
                 bins=log_bins, color='lime')
        plt.hist(results['watermarked_scores_aug'].cpu(), label='generated (perturbations) + watermark', alpha=.5,
                 bins=log_bins, color='cyan')
        plt.xscale('log')
        plt.legend()
        plt.show()

    # ROC visualization function
    def visualize_roc_auc(fpr, tpr, roc_auc, title_suffix, description):
        """
        Display ROC curve with AUC score.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates  
            roc_auc: Area under curve score
            title_suffix: Suffix for plot title
            description: Description to print
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.5f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {title_suffix}")
        plt.legend(loc="lower right")
        plt.grid(True)
        print(f"{description}")
        plt.show()

    # Show the plots
    visualize_scores()

    # ROC analysis: all categories combined
    visualize_roc_auc(
        results['fpr'],
        results['tpr'],
        results['roc_auc'],
        "All Categories",
        "All clean samples (aug + non-aug) vs all watermarked samples (aug + non-aug)"
    )
