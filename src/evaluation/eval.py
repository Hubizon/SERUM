"""
Evaluation module for watermark detection models.

This module provides comprehensive evaluation capabilities including quick evaluation,
thorough evaluation with ROC analysis, visualization, and augmentation robustness testing.
"""

import argparse
import gc
import os
import pickle
import random
import sys
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_fidelity
from PIL import Image
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from torch import nn
from torchmetrics.multimodal import CLIPScore
from torchvision import transforms
from tqdm import tqdm

from ..utils import config
from ..data import Augment, Dataset
from . import generate
from ..models import WatermarkScoreModel, Watermark, ModelWrapper, DiffusionModel, encode, decode, get_vae
from ..utils.config import Config
from ..utils import im
from ..utils.utils import cleanup_cuda_memory, save_config_snapshot, compute_roc_metrics


class Evaluation:
    """
    Comprehensive evaluation class for watermark detection models.
    
    Provides methods for quick evaluation, thorough evaluation with ROC analysis,
    visualization of results, and augmentation robustness testing.
    """

    def __init__(self,
                 config: Config,
                 diffusion_model: nn.Module,
                 score_model: WatermarkScoreModel,
                 watermark: nn.Module,
                 prompts: List[str],
                 augment: Callable):
        """
        Initialize the evaluation class.
        
        Args:
            score_model: Trained watermark scoring model
            diffusion_model: Diffusion model for generating samples
            device: Device to use for evaluation
        """
        self.config = config
        self.device = self.config.get_device()
        self.diffusion_model = diffusion_model.requires_grad_(False).eval().to(self.device)
        self.score_model = score_model.requires_grad_(False).eval().to(self.device)
        self.watermark = watermark.requires_grad_(False).eval().to(self.device)
        self.prompts = prompts
        self.augment = augment

        # Cache for evaluation results
        self.last_quick_results = None
        self.last_thorough_results = None

        generate.save_pos_neg_prompts(self.config, self.prompts)

    @torch.no_grad()
    def watermark_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get watermark scores for input tensors.
        
        Args:
            x: Input tensor
            
        Returns:
            Watermark scores
        """
        return self.score_model(x.float())

    @torch.no_grad()
    def quick_eval(self,
                   num_samples: int = None,
                   show_images: bool = True,
                   seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Quick evaluation with small sample sizes.
        
        Args:
            batch_size: Number of samples to generate
            show_images: Whether to display generated images
            show_plots: Whether to show score trajectory plots
            W
        Returns:
            Dictionary with evaluation results
        """
        if num_samples is None:
            num_samples = self.config.evaluation.quick_samples

        print("Running quick evaluation...")

        cleanup_cuda_memory()
        if seed:
            torch.manual_seed(seed)

        g_res, orig_noise = self.watermark(num_samples, ret_noise=True)

        samples, positive_prompts, negative_prompts = self.diffusion_model.generate_full(
            batch_size=num_samples, noise=orig_noise.half(), x_ts_ret=False,
            prompts=self.prompts, ret_prompts=True
        )
        samples_w = self.diffusion_model.generate_full(
            batch_size=num_samples, noise=g_res.half(), x_ts_ret=False,
            positive_prompts=positive_prompts, negative_prompts=negative_prompts
        )

        with torch.no_grad():
            clean_scores = self.watermark_score(samples)
            clean_imgs = decode(samples)
            clean_aug_imgs = self.augment(clean_imgs.float())
            clean_aug_scores = self.watermark_score(encode(clean_aug_imgs))

            watermarked_scores = self.watermark_score(samples_w)
            watermarked_imgs = decode(samples_w)
            watermarked_aug_imgs = self.augment(watermarked_imgs.float())
            watermarked_aug_scores = self.watermark_score(encode(watermarked_aug_imgs))

        # Cleanup intermediate tensors
        del samples, samples_w, g_res, orig_noise
        cleanup_cuda_memory()

        if show_images:
            im.show_grid_2x2(
                images_list=[clean_imgs, watermarked_imgs, clean_aug_imgs, watermarked_aug_imgs],
                titles_list=['Clean images', 'Watermarked images', 'Perturbed clean images',
                             'Perturbed watermarked images'],
                figsize=10
            )

        print(f'Watermark score of clean images (non-watermarked): {clean_scores.flatten().tolist()}')
        print(f'  mean: {clean_scores.mean().item()}')
        print(f'  median: {clean_scores.median().item()}')
        print(f'Watermark score of clean images (non-watermarked + augmented): {clean_aug_scores.flatten().tolist()}')
        print(f'  mean: {clean_aug_scores.mean().item()}')
        print(f'  median: {clean_aug_scores.median().item()}')
        print(f'Watermark score of watermarked images (watermarked): {watermarked_scores.flatten().tolist()}')
        print(f'  mean: {watermarked_scores.mean().item()}')
        print(f'  median: {watermarked_scores.median().item()}')
        print(
            f'Watermark score of perturbed watermarked images (watermarked + augmented): {watermarked_aug_scores.flatten().tolist()}')
        print(f'  mean: {watermarked_aug_scores.mean().item()}')
        print(f'  median: {watermarked_aug_scores.median().item()}')

        results = {
            'clean_scores': clean_scores,
            'clean_aug_scores': clean_aug_scores,
            'watermarked_scores': watermarked_scores,
            'watermarked_aug_scores': watermarked_aug_scores
        }

        self.last_quick_results = results
        return results

    @torch.no_grad()
    def eval_tpr(self,
                 num_samples: int = None,
                 batch_size: int = None,
                 clean_path: Optional[str] = None,
                 watermarked_path: Optional[str] = None,
                 seed: Optional[int] = None,
                 force_reload: bool = True) -> Dict[str, Any]:
        """Evaluate watermark detection performance (clean / augmented / combined)."""
        if num_samples is None:
            num_samples = self.config.evaluation.num_samples
        if batch_size is None:
            batch_size = self.config.evaluation.batch_size

        print(f"Running evaluation with {num_samples} samples...")
        cleanup_cuda_memory()
        if seed:
            torch.manual_seed(seed)

        print("Generating samples (fresh set)...")
        clean_path, watermarked_path = generate.generate(
            config=self.config,
            diffusion_model=self.diffusion_model,
            watermark=self.watermark,
            num_samples=num_samples,
            batch_size=batch_size,
            clean_path=clean_path,
            watermarked_path=watermarked_path,
            force_reload=force_reload
        )

        # Collect scores
        non_clean, non_clean_aug = [], []
        wm_clean, wm_clean_aug = [], []

        print("Scoring clean (non-watermarked) images...")
        for imgs in generate.load_images_generator(self.config, clean_path):
            score_c = self.watermark_score(encode(imgs.float()))
            non_clean.append(score_c)
            aug_imgs = self.augment(imgs).half()
            aug_score = self.watermark_score(encode(aug_imgs.float()))
            non_clean_aug.append(aug_score)

        print("Scoring watermarked images...")
        for imgs in generate.load_images_generator(self.config, watermarked_path):
            score_w = self.watermark_score(encode(imgs.float()))
            wm_clean.append(score_w)
            aug_imgs = self.augment(imgs).half()
            aug_score_w = self.watermark_score(encode(aug_imgs.float()))
            wm_clean_aug.append(aug_score_w)

        # Stack
        non_clean = torch.cat(non_clean).squeeze(-1)
        non_clean_aug = torch.cat(non_clean_aug).squeeze(-1)
        wm_clean = torch.cat(wm_clean).squeeze(-1)
        wm_clean_aug = torch.cat(wm_clean_aug).squeeze(-1)

        cleanup_cuda_memory()

        # Combined arrays
        all_negative_scores = torch.cat([non_clean, non_clean_aug])
        all_positive_scores = torch.cat([wm_clean, wm_clean_aug])
        y_comb = torch.cat([torch.zeros_like(all_negative_scores), torch.ones_like(all_positive_scores)])
        scores_comb = torch.cat([all_negative_scores, all_positive_scores])

        comb_best_bal_acc, comb_fpr, comb_tpr, comb_thresholds, comb_idx, comb_idx2 = compute_roc_metrics(
            y_comb.cpu(), scores_comb.cpu(), self.config.evaluation.target_fpr)
        comb_roc_auc = auc(comb_fpr, comb_tpr)

        # Clean only
        y_clean = torch.cat([torch.zeros_like(non_clean), torch.ones_like(wm_clean)])
        scores_clean = torch.cat([non_clean, wm_clean])
        clean_best_bal_acc, clean_fpr, clean_tpr, clean_thresholds, clean_idx, _ = compute_roc_metrics(
            y_clean.cpu(), scores_clean.cpu(), self.config.evaluation.target_fpr)
        clean_roc_auc = auc(clean_fpr, clean_tpr)

        # Aug only
        y_aug = torch.cat([torch.zeros_like(non_clean_aug), torch.ones_like(wm_clean_aug)])
        scores_aug = torch.cat([non_clean_aug, wm_clean_aug])
        aug_best_bal_acc, aug_fpr, aug_tpr, aug_thresholds, aug_idx, _ = compute_roc_metrics(y_aug.cpu(),
                                                                                           scores_aug.cpu(), self.config.evaluation.target_fpr)
        aug_roc_auc = auc(aug_fpr, aug_tpr)

        # Thresholds / TPR@target FPR
        comb_opt_threshold = comb_thresholds[comb_idx];
        comb_tpr_at_target = comb_tpr[comb_idx];
        comb_actual_fpr = comb_fpr[comb_idx]
        clean_opt_threshold = clean_thresholds[clean_idx];
        clean_tpr_at_target = clean_tpr[clean_idx];
        clean_actual_fpr = clean_fpr[clean_idx]
        aug_opt_threshold = aug_thresholds[aug_idx];
        aug_tpr_at_target = aug_tpr[aug_idx];
        aug_actual_fpr = aug_fpr[aug_idx]

        # Print summary here (detailed)
        print("\nEvaluation Summary")
        print(
            f"Clean: {clean_tpr_at_target * 100:.4f}% TPR @ {clean_actual_fpr * 100:.4f}% FPR (ROC AUC: {clean_roc_auc:.4f}) Thr: {clean_opt_threshold:.4f} BalAcc: {clean_best_bal_acc:.4f}")
        print(
            f"Aug: {aug_tpr_at_target * 100:.4f}% TPR @ {aug_actual_fpr * 100:.4f}% FPR (ROC AUC: {aug_roc_auc:.4f}) Thr: {aug_opt_threshold:.4f} BalAcc: {aug_best_bal_acc:.4f}")
        print(
            f"Combined: {comb_tpr_at_target * 100:.4f}% TPR @ {comb_actual_fpr * 100:.4f}% FPR (ROC AUC: {comb_roc_auc:.4f}) Thr: {comb_opt_threshold:.4f} BalAcc: {comb_best_bal_acc:.4f}\n")
        print(
            "Samples: clean={} aug_clean={} wm={} wm_aug={}\n".format(len(non_clean), len(non_clean_aug), len(wm_clean),
                                                                      len(wm_clean_aug)))

        results = {
            # Raw score tensors
            'non_watermarked_scores': non_clean,
            'non_watermarked_scores_aug': non_clean_aug,
            'watermarked_scores': wm_clean,
            'watermarked_scores_aug': wm_clean_aug,
            # Clean metrics
            'clean_fpr': clean_fpr,
            'clean_tpr': clean_tpr,
            'clean_roc_auc': clean_roc_auc,
            'clean_best_balanced_accuracy': clean_best_bal_acc,
            'clean_optimal_threshold': clean_opt_threshold,
            'clean_tpr_at_target_fpr': clean_tpr_at_target,
            'clean_actual_fpr': clean_actual_fpr,
            # Aug metrics
            'aug_fpr': aug_fpr,
            'aug_tpr': aug_tpr,
            'aug_roc_auc': aug_roc_auc,
            'aug_best_balanced_accuracy': aug_best_bal_acc,
            'aug_optimal_threshold': aug_opt_threshold,
            'aug_tpr_at_target_fpr': aug_tpr_at_target,
            'aug_actual_fpr': aug_actual_fpr,
            # Combined
            'fpr': comb_fpr,
            'tpr': comb_tpr,
            'thresholds': comb_thresholds,
            'roc_auc': comb_roc_auc,
            'best_balanced_accuracy': comb_best_bal_acc,
            'optimal_threshold': comb_opt_threshold,
            'tpr_at_target_fpr': comb_tpr_at_target,
            'actual_fpr': comb_actual_fpr,
            'all_negative_scores': all_negative_scores,
            'all_positive_scores': all_positive_scores,
            'num_samples': len(all_negative_scores) + len(all_positive_scores)
        }

        print("Saving evaluation plots...")
        im.save_evaluation_plots(self.config, results, "eval_tpr")
        self.last_thorough_results = results
        return results

    @torch.no_grad()
    def plot_results(self, results: Dict[str, Any] = None):
        """
        Create comprehensive plots for evaluation results.
        
        Args:
            results: Results from thorough_eval (uses last results if None)
        """
        if results is None:
            results = self.last_thorough_results
        im.plot_evaluation_results(results)

    @torch.no_grad()
    def augmentation_robustness_eval(self,
                                     num_samples: int = None,
                                     batch_size: int = None,
                                     clean_path: Optional[str] = None,
                                     watermarked_path: Optional[str] = None,
                                     augmentation_ids: Optional[int] = None,
                                     target_fpr: int = None,
                                     seed: Optional[int] = None,
                                     save_individual_plots: bool = True) -> Dict[str, float]:
        """
        Test robustness against different augmentation types.
        
        Args:
            num_samples: Number of samples per augmentation type
            batch_size: Batch size for testing
            clean_path: Path to clean images (optional)
            watermarked_path: Path to watermarked images (optional)
            target_fpr: Target false positive rate
            save_individual_plots: Whether to save plots for each augmentation (slower)
            
        Returns:
            Dictionary mapping augmentation name to TPR@target_fpr
        """
        if num_samples is None:
            num_samples = self.config.evaluation.num_samples
        if batch_size is None:
            batch_size = self.config.evaluation.batch_size
        if target_fpr is None:
            target_fpr = self.config.evaluation.target_fpr

        print("Testing robustness against different augmentation types...")

        cleanup_cuda_memory()
        if seed:
            torch.manual_seed(seed)

        print("Generating samples without augmentation...")

        clean_path, watermarked_path = generate.generate(
            config=self.config,
            diffusion_model=self.diffusion_model,
            watermark=self.watermark,
            num_samples=num_samples,
            batch_size=batch_size,
            clean_path=clean_path,
            watermarked_path=watermarked_path,
        )

        augmentation_results = {}

        # Test individual augmentation types if the augment function supports it
        for aug_idx in augmentation_ids if augmentation_ids else range(len(self.augment)):
            aug_name = self.augment.transforms[aug_idx].__class__.__name__ + f' ({aug_idx})'
            print(f"Testing augmentation: {aug_name}...")

            non_watermarked_scores_aug = []
            watermarked_scores_aug = []

            print("Scoring non-watermarked samples...")
            for imgs in generate.load_images_generator(self.config, images_path=clean_path, batch_size=batch_size,
                                                       augment=self.augment, augment_idx=aug_idx):
                encoded_imgs = encode(imgs.clamp(-1, 1)).float()
                scores = self.watermark_score(encoded_imgs)
                non_watermarked_scores_aug.append(scores)
                del encoded_imgs, scores

            print("Scoring watermarked samples...")
            for imgs in generate.load_images_generator(self.config, images_path=watermarked_path, batch_size=batch_size,
                                                       augment=self.augment, augment_idx=aug_idx):
                encoded_imgs = encode(imgs.clamp(-1, 1)).float()
                scores = self.watermark_score(encoded_imgs)
                watermarked_scores_aug.append(scores)
                del encoded_imgs, scores

            non_watermarked_scores_aug = torch.cat(non_watermarked_scores_aug).squeeze(-1)
            watermarked_scores_aug = torch.cat(watermarked_scores_aug).squeeze(-1)

            # Clean up memory after processing this augmentation
            cleanup_cuda_memory()

            y_true = torch.cat([
                torch.zeros_like(non_watermarked_scores_aug),
                torch.ones_like(watermarked_scores_aug)
            ])
            scores = torch.cat([
                non_watermarked_scores_aug,
                watermarked_scores_aug
            ])

            best_bal_acc, fpr, tpr, thresholds, idx, idx2 = compute_roc_metrics(y_true, scores, target_fpr)
            roc_auc = auc(fpr, tpr)

            # Save plots for this specific augmentation (optional for speed)
            if save_individual_plots:
                # Create results dict for this augmentation to save plots
                aug_results = {
                    'non_watermarked_scores_aug': non_watermarked_scores_aug,
                    'watermarked_scores_aug': watermarked_scores_aug,
                    'fpr': fpr,
                    'tpr': tpr,
                    'roc_auc': roc_auc
                }
                im.save_evaluation_plots(self.config, aug_results, "augmentation", aug_name)
                del aug_results

            augmentation_results[aug_name] = (
            tpr[idx], fpr[idx], roc_auc, non_watermarked_scores_aug, watermarked_scores_aug)
            print(f"{aug_name}: {tpr[idx] * 100:.4f}% TPR @ {fpr[idx] * 100:.4f}% FPR (ROC AUC: {roc_auc:.4f})")
            if idx2 is not None and idx2 != idx:
                print(f"{aug_name}: {tpr[idx2] * 100:.4f}% TPR @ {fpr[idx2] * 100:.4f}% FPR (ROC AUC: {roc_auc:.4f})")

            # Clean up variables to free memory for next augmentation
            del y_true, scores, fpr, tpr, thresholds
            cleanup_cuda_memory()

        # Summary
        print("\nAugmentation Robustness Summary:")
        print("-" * 50)
        for aug_name, (tpr, fpr, roc_auc, _, _) in augmentation_results.items():
            print(f" {aug_name}: {tpr * 100:.4f}% TPR @ {fpr * 100:.4f}% FPR (ROC AUC: {roc_auc:.4f})")

        print(f"\nAverage TPR: {np.mean([res[0] for res in augmentation_results.values()]) * 100:.4f}%")
        print(f"Average FPR: {np.mean([res[1] for res in augmentation_results.values()]) * 100:.4f}%")
        print(f"Average ROC AUC: {np.mean([res[2] for res in augmentation_results.values()]):.4f}")

        # Save summary plot with all augmentations
        print("\nSaving augmentation summary plots...")
        im.save_augmentation_summary_plot(self.config, augmentation_results, compute_roc_metrics)

        return augmentation_results

    def watermark_detection_time(self, num_samples: Optional[int] = None, batch_size: Optional[int] = None, force_reload: bool = False) -> Dict[str, float]:
        """
        Measure watermark detection time performance.
        
        Args:
            num_samples: Number of samples to evaluate
            batch_size: Batch size for processing
            force_reload: Whether to force regeneration of images
            
        Returns:
            Dictionary containing timing metrics
        """
        if num_samples is None:
            num_samples = self.config.evaluation.num_samples
        if batch_size is None:
            batch_size = self.config.evaluation.batch_size

        print(f"Measuring watermark detection time over {num_samples} samples...")
        cleanup_cuda_memory()

        clean_path, watermarked_path = generate.generate(
            config=self.config,
            diffusion_model=self.diffusion_model,
            watermark=self.watermark,
            num_samples=num_samples,
            force_reload=force_reload
        )

        print("Scoring non-watermarked samples...")
        scores = []
        total_samples = 0
        cleanup_cuda_memory()

        start_time = time.time()

        for imgs in generate.load_images_generator(self.config, images_path=clean_path, 
                                                   batch_size=batch_size, max_num=num_samples):
            scores.append(self.watermark_score(encode(imgs.float())))
            total_samples += imgs.shape[0]

        end_time = time.time()
        detection_time = end_time - start_time

        print("Scoring watermarked samples...")
        start_time_watermarked = time.time()

        for imgs in generate.load_images_generator(self.config, images_path=watermarked_path, 
                                                   batch_size=batch_size, max_num=num_samples):
            scores.append(self.watermark_score(encode(imgs.float())))
            total_samples += imgs.shape[0]

        end_time_watermarked = time.time()
        watermarked_detection_time = end_time_watermarked - start_time_watermarked

        total_detection_time = detection_time + watermarked_detection_time
        avg_time_per_sample = total_detection_time / total_samples

        print(f"\n=== Watermark Detection Time Results ===")
        print(f"Clean samples detection time: {detection_time:.4f} seconds")
        print(f"Watermarked samples detection time: {watermarked_detection_time:.4f} seconds")
        print(f"Total detection time: {total_detection_time:.4f} seconds")
        print(f"Total samples processed: {total_samples}")
        print(f"Average time per sample: {avg_time_per_sample:.6f} seconds")
        print(f"Samples per second: {1. / avg_time_per_sample:.2f}")

        return {
            'total_detection_time': total_detection_time,
            'clean_detection_time': detection_time,
            'watermarked_detection_time': watermarked_detection_time,
            'total_samples': total_samples,
            'avg_time_per_sample': avg_time_per_sample,
            'samples_per_second': 1. / avg_time_per_sample
        }


    @torch.no_grad()
    def run_full_eval(self, num_samples: Optional[int] = None, clean_path: Optional[str] = None, watermarked_path: Optional[str] = None, force_reload: bool = True) -> None:
        """
        Run comprehensive evaluation including TPR and augmentation robustness tests.
        
        Args:
            num_samples: Number of samples to evaluate
            clean_path: Path to clean (non-watermarked) images
            watermarked_path: Path to watermarked images
            force_reload: Whether to force regeneration of images
        """
        save_config_snapshot(self.config, "eval_config")

        if num_samples is None:
            num_samples = self.config.evaluation.num_samples

        # Create timestamp for log file
        torch.set_grad_enabled(False)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(self.config.experiment_dir, f"eval_{timestamp}.txt")
        results_file = os.path.join(self.config.experiment_dir, f"eval_results_{timestamp}.pkl")

        # Ensure directory exists
        os.makedirs(self.config.experiment_dir, exist_ok=True)

        # Redirect stdout to both console and file
        class TeeOutput:
            """Helper class to redirect output to multiple files simultaneously."""
            def __init__(self, *files):
                """Initialize with multiple file objects to write to."""
                self.files = files

            def write(self, obj):
                """Write object to all files."""
                for f in self.files:
                    f.write(obj)
                    f.flush()

            def flush(self):
                """Flush all files."""
                for f in self.files:
                    f.flush()

        # Open log file and set up output redirection
        with open(log_file, 'w') as log_f:
            original_stdout = sys.stdout
            sys.stdout = TeeOutput(sys.stdout, log_f)

            try:
                print("Initializing evaluation...")
                cleanup_cuda_memory(verbose=True)

                # Evaluation
                print("Evaluation...")
                thorough_results = self.eval_tpr(num_samples=num_samples, clean_path=clean_path, watermarked_path=watermarked_path, force_reload=force_reload)
                cleanup_cuda_memory(verbose=True)

                # Augmentation robustness evaluation
                print("Augmentation Robustness Evaluation...")
                augmentation_results = self.augmentation_robustness_eval(num_samples=num_samples, clean_path=clean_path, watermarked_path=watermarked_path)
                cleanup_cuda_memory(verbose=True)

                # Save results
                results_data = {
                    'timestamp': timestamp,
                    'thorough_results': thorough_results,
                    'augmentation_results': augmentation_results
                }

                try:
                    with open(results_file, 'wb') as f:
                        pickle.dump(results_data, f)
                    print(f"\nResults saved to: {results_file}")
                except Exception as e:
                    print(f"Warning: Could not save results file: {e}")

                print(f"Evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Log file: {log_file}")
                print(f"Results file: {results_file}")

            except Exception as e:
                print(f"\nError during evaluation: {str(e)}")
                import traceback
                traceback.print_exc()

            finally:
                # Restore original stdout
                sys.stdout = original_stdout

        print(f"Evaluation complete. Results saved to: {log_file}")

    @torch.no_grad()
    def eval_tpr_clean(self, clean_path: Optional[str] = None, watermarked_path: Optional[str] = None, num_samples: Optional[int] = None,
                        batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate TPR (True Positive Rate) on clean samples without augmentation.
        
        Args:
            clean_path: Path to clean (non-watermarked) images
            watermarked_path: Path to watermarked images
            num_samples: Number of samples to evaluate
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing evaluation results
        """
        if clean_path is None:
            clean_path = self.config.evaluation.clean_path
        if watermarked_path is None:
            watermarked_path = self.config.evaluation.watermarked_path
        if batch_size is None:
            batch_size = self.config.evaluation.batch_size

        print("")

        wm_clean, non_clean = [], []

        print(f"Scoring clean (non-watermarked) images [path: {clean_path}]...")
        for imgs in generate.load_images_generator(self.config, clean_path, 
                                                   max_num=num_samples, batch_size=batch_size):
            score_c = self.watermark_score(encode(imgs.float()))
            non_clean.append(score_c)

        print(f"Scoring watermarked images [path: {watermarked_path}]...")
        for imgs in generate.load_images_generator(self.config, watermarked_path, 
                                                   max_num=num_samples, batch_size=batch_size):
            score_w = self.watermark_score(encode(imgs.float()))
            wm_clean.append(score_w)

        non_clean = torch.cat(non_clean).squeeze(-1)
        wm_clean = torch.cat(wm_clean).squeeze(-1)

        cleanup_cuda_memory()

        y_clean = torch.cat([torch.zeros_like(non_clean), torch.ones_like(wm_clean)])
        scores_clean = torch.cat([non_clean, wm_clean])
        clean_best_bal_acc, clean_fpr, clean_tpr, clean_thresholds, clean_idx, _ = compute_roc_metrics(
            y_clean.cpu(), scores_clean.cpu(), self.config.evaluation.target_fpr)
        clean_roc_auc = auc(clean_fpr, clean_tpr)

        # Thresholds / TPR@target FPR
        clean_opt_threshold = clean_thresholds[clean_idx];
        clean_tpr_at_target = clean_tpr[clean_idx];
        clean_actual_fpr = clean_fpr[clean_idx]

        # Print summary here (detailed)
        print("\nEvaluation Summary")
        print(
            f" {clean_tpr_at_target * 100:.4f}% TPR @ {clean_actual_fpr * 100:.4f}% FPR (ROC AUC: {clean_roc_auc:.4f}) Thr: {clean_opt_threshold:.4f} BalAcc: {clean_best_bal_acc:.4f}")

        results = {
            'non_watermarked_scores': non_clean,
            'watermarked_scores': wm_clean,
            'fpr': clean_fpr,
            'tpr': clean_tpr,
            'roc_auc': clean_roc_auc,
            'best_balanced_accuracy': clean_best_bal_acc,
            'optimal_threshold': clean_opt_threshold,
            'tpr_at_target_fpr': clean_tpr_at_target,
            'actual_fpr': clean_actual_fpr,
        }

        print("Saving evaluation plots...")
        im.save_evaluation_plots(self.config, results, "custom_eval_tpr")
        self.last_thorough_results = results
        return results

    @torch.no_grad()
    def eval_tpr_clean_threshold(self, images_path: str, threshold: float, num_samples: Optional[int] = None, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate TPR using a specific threshold on images from a given path.
        
        Args:
            images_path: Path to images to evaluate
            threshold: Detection threshold to use
            num_samples: Number of samples to evaluate
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing evaluation results
        """
        if batch_size is None:
            batch_size = self.config.evaluation.batch_size

        print("")

        scores = []

        print(f"Scoring images [path: {images_path}]...")
        for imgs in generate.load_images_generator(self.config, images_path, max_num=num_samples,
                                                   batch_size=batch_size):
            score = self.watermark_score(encode(imgs.float()))
            scores.append(score)

        scores = torch.cat(scores).squeeze(-1)

        cleanup_cuda_memory()

        below_thr = sum(scores < threshold).item()
        above_thr = sum(scores >= threshold).item()
        print('Above threshold:', above_thr)
        print('Below threshold:', below_thr)
        print(f'TPR: {above_thr / (below_thr + above_thr)}  /  FPR:', below_thr / (below_thr + above_thr))


@torch.no_grad()
def main() -> None:
    """
    Main evaluation function that runs comprehensive evaluation and saves results.
    """
    parser = argparse.ArgumentParser(description='Evaluate watermark models')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (overrides default)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to generate (overrides default)')
    parser.add_argument('--load-from-checkpoint', action='store_true', default=False,
                        help='Load the model from the latest checkpoint')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Path to the checkpoint file (if not loading from latest)')
    parser.add_argument('--dont-force-reload', action='store_true', default=False,
                        help='Do not force reloading of samples during evaluation')
    parser.add_argument('--evaluate-time', action='store_true', default=False,
                        help='Measure time of watermark injection and detection')
    parser.add_argument('--clean-path', type=str, default=None,
                        help='Path to clean images directory (overrides config default)')
    parser.add_argument('--watermarked-path', type=str, default=None,
                        help='Path to watermarked images directory (overrides config default)')
    parser.add_argument('--eval-clean', action='store_true', default=False,
                        help='Run eval_tpr_clean evaluation (clean vs watermarked without augmentation)')
    parser.add_argument('--eval-threshold', action='store_true', default=False,
                        help='Run eval_tpr_clean_threshold evaluation')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold value for eval_tpr_clean_threshold')
    parser.add_argument('--images-path', type=str, default=None,
                        help='Path to images for threshold evaluation')
    args = parser.parse_args()

    config = Config(args.config)

    print("Initializing evaluation...")

    # Initialize device
    device = config.get_device()

    print("Loading models...")
    diffusion_model = ModelWrapper(DiffusionModel(config).to(device)).to(device)
    augment = Augment()

    if args.load_from_checkpoint:
        from ..training import Trainer
        if args.checkpoint_path is not None:
            print(f"Loading models from specified checkpoint...")
            checkpoint_path = args.checkpoint_path
        else:
            print("Loading models from latest checkpoint...")
            checkpoint_path = Trainer.get_checkpoint_path(config)
        if checkpoint_path is None:
            raise ValueError("No checkpoint found to load from.")
        print(f"Loading from checkpoint: {checkpoint_path}")
        score_model, watermark = Trainer.load_models_from_checkpoint(config, checkpoint_path)
    else:
        score_model = WatermarkScoreModel.load(config)
        watermark = Watermark.load(config)

    score_model = score_model.requires_grad_(False).eval().to(device)
    watermark = watermark.requires_grad_(False).eval().to(device)
    cleanup_cuda_memory(True)

    print("Watermark Score Model:")
    print(f"Total parameters: {sum(p.numel() for p in score_model.parameters()):,}")

    test_prompts = Dataset.get_prompts(config, use_test=True)
    print(f"Sample prompt: {test_prompts[17]}")

    # Initialize evaluator
    evaluator = Evaluation(
        config=config,
        diffusion_model=diffusion_model,
        score_model=score_model,
        watermark=watermark,
        prompts=test_prompts,
        augment=augment
    )

    get_vae(config)
    if args.evaluate_time:
        evaluator.watermark_detection_time(num_samples=args.num_samples)
    elif args.eval_clean:
        print("Running eval_tpr_clean evaluation...")
        results = evaluator.eval_tpr_clean(
            clean_path=args.clean_path,
            watermarked_path=args.watermarked_path,
            num_samples=args.num_samples
        )
        print("eval_tpr_clean evaluation completed.")
    elif args.eval_threshold:
        if args.images_path is None or args.threshold is None:
            raise ValueError("--images-path and --threshold are required when using --eval-threshold")
        print(f"Running eval_tpr_clean_threshold evaluation with threshold {args.threshold}...")
        evaluator.eval_tpr_clean_threshold(
            images_path=args.images_path,
            threshold=args.threshold,
            num_samples=args.num_samples
        )
        print("eval_tpr_clean_threshold evaluation completed.")
    else:
        evaluator.run_full_eval(num_samples=args.num_samples, clean_path=args.clean_path, watermarked_path=args.watermarked_path, force_reload=not args.dont_force_reload)


if __name__ == "__main__":
    main()
