import argparse
import gc
import os
import pickle
import random
import sys
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Callable

import clip
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

from ..data import Augment, Dataset
from . import generate
from ..models import WatermarkScoreModel, Watermark, ModelWrapper, DiffusionModel, encode, decode, get_vae
from ..utils.config import Config
from ..utils import im
from ..utils.utils import cleanup_cuda_memory, save_config_snapshot


@torch.no_grad()
def calculate_fid(config: Config,
                  clean_path: Optional[str] = None,
                  watermarked_path: Optional[str] = None,
                  original_path: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate FID scores between clean, watermarked, and original images.

    Returns:
        dict: Dictionary containing FID scores for all three comparisons
    """
    if clean_path is None:
        clean_path = config.evaluation.clean_path
    if watermarked_path is None:
        watermarked_path = config.evaluation.watermarked_path
    if original_path is None:
        original_path = config.evaluation.original_path

    print("Calculating FID scores...")
    cleanup_cuda_memory()

    # Check if directories exist
    if not os.path.exists(clean_path):
        raise ValueError(f"Clean images directory not found: {clean_path}")
    if not os.path.exists(watermarked_path):
        raise ValueError(f"Watermarked images directory not found: {watermarked_path}")
    if not os.path.exists(original_path) and original_path is not None:
        raise ValueError(f"Original images directory not found: {original_path}")

    results = {}

    if original_path is not None:
        # FID between clean and original images
        print("Calculating FID: Clean vs Original...")
        try:
            metrics_clean_original = torch_fidelity.calculate_metrics(
                input1=clean_path,
                input2=original_path,
                cuda=torch.cuda.is_available(),
                fid=True,
                verbose=False
            )
            results["fid_clean_original"] = metrics_clean_original["frechet_inception_distance"]
            print(f"FID (Clean vs Original): {results['fid_clean_original']:.4f}")
        except Exception as e:
            print(f"Error calculating FID (Clean vs Original): {e}")
            results["fid_clean_original"] = None

        # FID between watermarked and original images
        print("Calculating FID: Watermarked vs Original...")
        try:
            metrics_watermarked_original = torch_fidelity.calculate_metrics(
                input1=watermarked_path,
                input2=original_path,
                cuda=torch.cuda.is_available(),
                fid=True,
                verbose=False
            )
            results["fid_watermarked_original"] = metrics_watermarked_original["frechet_inception_distance"]
            print(f"FID (Watermarked vs Original): {results['fid_watermarked_original']:.4f}")
        except Exception as e:
            print(f"Error calculating FID (Watermarked vs Original): {e}")
            results["fid_watermarked_original"] = None

        

    # FID between clean and watermarked images
    print("Calculating FID: Clean vs Watermarked...")
    try:
        metrics_clean_watermarked = torch_fidelity.calculate_metrics(
            input1=clean_path,
            input2=watermarked_path,
            cuda=torch.cuda.is_available(),
            fid=True,
            verbose=False
        )
        results["fid_clean_watermarked"] = metrics_clean_watermarked["frechet_inception_distance"]
        print(f"FID (Clean vs Watermarked): {results['fid_clean_watermarked']:.4f}")
    except Exception as e:
        print(f"Error calculating FID (Clean vs Watermarked): {e}")
        results["fid_clean_watermarked"] = None

    # Print summary
    print("\n=== FID Results Summary ===")
    for metric_name, value in results.items():
        if value is not None:
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: Error occurred")

    cleanup_cuda_memory()

    return results


@torch.no_grad()
def calculate_clip_score(
        config: Config,
        clip_model_name: str = "ViT-B/32",
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        prompts_path: Optional[str] = None,
        images_clean_path: Optional[str] = None,
        images_watermarked_path: Optional[str] = None,
        plot_histogram: bool = True
) -> Dict[str, float]:
    """
    Calculate CLIP scores for clean and watermarked images against text prompts.
    """
    if batch_size is None:
        batch_size = config.evaluation.batch_size
    if prompts_path is None:
        prompts_path = config.evaluation.positive_prompts_path
    if images_clean_path is None:
        images_clean_path = config.evaluation.clean_path
    if images_watermarked_path is None:
        images_watermarked_path = config.evaluation.watermarked_path

    print("Loading CLIP model...")
    device = config.get_device()
    model, preprocess = clip.load(clip_model_name, device=device)
    model.eval()

    # Load prompts
    prompts, _ = generate.load_pos_neg_prompts(config, prompts_path)
    if num_samples is not None:
        prompts = prompts[:num_samples]

    def load_images(path_list):
        """Load and preprocess images from file paths."""
        images = []
        for p in path_list:
            img = Image.open(p).convert("RGB")
            img = preprocess(img)
            images.append(img)
        return torch.stack(images)

    def compute_scores(images, prompts):
        """Compute CLIP scores between images and prompts."""
        all_scores = []

        for i in tqdm(range(0, len(images), batch_size)):
            batch_imgs = images[i:i + batch_size].to(device)
            batch_prompts = prompts[i:i + batch_size]

            # Encode images
            image_features = model.encode_image(batch_imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Encode text
            text_tokens = clip.tokenize(batch_prompts).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            scores = (image_features @ text_features.T).diag()  # batch-wise similarity
            all_scores.append(scores.cpu())

        return torch.cat(all_scores)

    # Load image paths
    clean_paths = sorted([os.path.join(images_clean_path, f) for f in os.listdir(images_clean_path)])
    if num_samples is not None:
        clean_paths = clean_paths[:num_samples]
    watermarked_paths = sorted([os.path.join(images_watermarked_path, f) for f in os.listdir(images_watermarked_path)])
    if num_samples is not None:
        watermarked_paths = watermarked_paths[:num_samples]

    # Load and preprocess images
    print("Preprocessing images...")
    clean_images = load_images(clean_paths)
    watermarked_images = load_images(watermarked_paths)

    # Compute CLIP scores
    print("Calculating CLIP scores for clean images...")
    clean_scores = compute_scores(clean_images, prompts)
    print("Calculating CLIP scores for watermarked images...")
    watermarked_scores = compute_scores(watermarked_images, prompts)

    mean_clean = clean_scores.mean().item()
    mean_watermarked = watermarked_scores.mean().item()

    print(f"Mean CLIP Score (Clean): {mean_clean:.4f}")
    print(f"Mean CLIP Score (Watermarked): {mean_watermarked:.4f}")

    # Optional histogram
    if plot_histogram:
        plt.figure(figsize=(6, 4))
        plt.hist(clean_scores.numpy(), bins=30, alpha=0.5, label="Clean Images", color="blue")
        plt.hist(watermarked_scores.numpy(), bins=30, alpha=0.5, label="Watermarked Images", color="orange")
        plt.title("CLIP Score Distribution")
        plt.xlabel("CLIP Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "clip_score_clean": clean_scores,
        "clip_score_watermarked": watermarked_scores,
        "clip_score_mean_clean": mean_clean,
        "clip_score_mean_watermarked": mean_watermarked
    }


def visualize_samples(config: Config,
                      num_samples: int = 3,
                      prompts_path: Optional[str] = None,
                      clean_path: Optional[str] = None,
                      watermarked_path: Optional[str] = None,
                      original_path: Optional[str] = None) -> None:
    """
    Visualize samples showing original, clean, and watermarked images with prompts.
    
    Args:
        num_samples: Number of samples to visualize
    """
    if prompts_path is None:
        prompts_path = config.evaluation.positive_prompts_path
    if clean_path is None:
        clean_path = config.evaluation.clean_path
    if watermarked_path is None:
        watermarked_path = config.evaluation.watermarked_path
    if original_path is None:
        original_path = config.evaluation.original_path

    print(f"Visualizing {num_samples} samples...")

    # Check if directories exist
    if not os.path.exists(clean_path):
        print(f"Warning: Clean images directory not found: {clean_path}")
        return
    if not os.path.exists(watermarked_path):
        print(f"Warning: Watermarked images directory not found: {watermarked_path}")
        return
    if not os.path.exists(original_path):
        print(f"Warning: Original images directory not found: {original_path}")
        return

    # Load prompts
    positive_prompts = []
    if os.path.exists(prompts_path):
        with open(prompts_path, 'r', encoding='utf-8') as f:
            positive_prompts = [line.strip() for line in f.readlines()]
    else:
        print(f"Warning: Positive prompts file not found: {prompts_path}")
        positive_prompts = ["No prompt available"] * num_samples

    # Get available image files
    clean_files = sorted([f for f in os.listdir(clean_path) if f.endswith('.png')])
    watermarked_files = sorted([f for f in os.listdir(watermarked_path) if f.endswith('.png')])
    original_files = sorted([f for f in os.listdir(original_path) if f.endswith('.png')])

    # Limit to available samples
    max_available = min(len(clean_files), len(watermarked_files), len(original_files), len(positive_prompts))
    num_samples = min(num_samples, max_available)

    if num_samples == 0:
        print("No samples available for visualization")
        return

    print(f"Displaying {num_samples} samples")

    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))

    # Handle case where num_samples = 1 (axes won't be 2D)
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Load images
        try:
            original_img = Image.open(os.path.join(original_path, original_files[i]))
            clean_img = Image.open(os.path.join(clean_path, clean_files[i]))
            watermarked_img = Image.open(os.path.join(watermarked_path, watermarked_files[i]))

            # Get the prompt for this sample
            prompt = positive_prompts[i] if i < len(positive_prompts) else "No prompt available"

            # Display original image
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title(f"Original\n{prompt}", fontsize=10, wrap=True)
            axes[i, 0].axis('off')

            # Display clean image
            axes[i, 1].imshow(clean_img)
            axes[i, 1].set_title("Clean", fontsize=12)
            axes[i, 1].axis('off')

            # Display watermarked image
            axes[i, 2].imshow(watermarked_img)
            axes[i, 2].set_title("Watermarked", fontsize=12)
            axes[i, 2].axis('off')

        except Exception as e:
            print(f"Error loading images for sample {i}: {e}")
            # Fill with placeholder text
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f"Error loading\nimage {i}",
                                ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Visualization complete for {num_samples} samples")


@torch.no_grad()
def main() -> None:
    """Main function with command line interface for evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Generate images and calculate evaluation metrics')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (overrides default)')
    parser.add_argument('--load-coco', choices=['full', 'images', 'none'], default='none',
                        help='Load COCO dataset: full (prompts+images), images (from existing prompts), none (skip)')
    parser.add_argument('--generate', action='store_true',
                        help='Generate clean and watermarked images')
    parser.add_argument('--calculate-fid', action='store_true',
                        help='Calculate FID score')
    parser.add_argument('--calculate-clip', action='store_true',
                        help='Calculate CLIP score')
    parser.add_argument('--num-samples', type=int, default=None,
                        help=f'Number of samples to process )')
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'Batch size for processing')
    parser.add_argument('--force-reload', action='store_true', default=True,
                        help='Force reload and regenerate existing files (default: True)')
    parser.add_argument('--coco-annotations', type=str,
                        default=None,
                        help='Path to COCO annotations file')
    parser.add_argument('--coco-images', type=str,
                        default=None,
                        help='Path to COCO images directory')
    parser.add_argument('--prompts-path', type=str,
                        help='Path to custom prompts file')
    parser.add_argument('--clean-path', type=str, default=None,
                        help='Output path for clean images')
    parser.add_argument('--watermarked-path', type=str, default=None,
                        help='Output path for watermarked images')
    parser.add_argument('--original-path', type=str, default=None,
                        help='Output path for original COCO images')

    args = parser.parse_args()

    config = Config(args.config)

    # Save config snapshot at the beginning
    save_config_snapshot(config, "scores_config")

    get_vae(config)

    if args.num_samples is None:
        args.num_samples = config.evaluation.fid_num_samples
    if args.batch_size is None:
        args.batch_size = config.evaluation.batch_size
    if args.clean_path is None:
        args.clean_path = config.evaluation.clean_path
    if args.watermarked_path is None:
        args.watermarked_path = config.evaluation.watermarked_path
    if args.original_path is None:
        args.original_path = config.evaluation.original_path

    print(f"Starting evaluation pipeline with arguments:")
    print(f"  Load COCO: {args.load_coco}")
    print(f"  Generate images: {args.generate}")
    print(f"  Calculate FID: {args.calculate_fid}")
    print(f"  Calculate CLIP: {args.calculate_clip}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Force reload: {args.force_reload}")
    print(f"  COCO annotations: {args.coco_annotations}")
    print(f"  COCO images: {args.coco_images}")
    print(f"  Prompts path: {args.prompts_path}")
    print(f"  Clean path: {args.clean_path}")
    print(f"  Watermarked path: {args.watermarked_path}")
    print(f"  Original path: {args.original_path}")

    # Load COCO dataset if requested
    if args.load_coco == 'full':
        print("\n=== Loading COCO prompts and images ===")
        generate.load_coco(
            config=config,
            annotation_path=args.coco_annotations,
            images_path=args.coco_images,
            num_samples=args.num_samples,
            original_path=args.original_path
        )

    elif args.load_coco == 'images':
        print("\n=== Loading COCO images from existing prompts ===")
        generate.load_coco_images_from_prompts(
            config=config,
            annotation_path=args.coco_annotations,
            images_path=args.coco_images,
            num_samples=args.num_samples,
            prompts_path=args.prompts_path,
            original_path=args.original_path
        )

    else:
        print("\n=== Skipping COCO loading - assuming prompts and images are already loaded ===")

    cleanup_cuda_memory()

    # Initialize models if generation is requested
    if args.generate:
        print("\n=== Initializing models ===")
        device = config.get_device()
        print(f"Using device: {device}")

        to_diffusion = DiffusionModel(config).requires_grad_(False).eval().to(device)
        diffusion_model = ModelWrapper(to_diffusion).requires_grad_(False).eval().to(device)
        watermark = Watermark.load(config).requires_grad_(False).eval().to(device)
        cleanup_cuda_memory(True)
        print("Models loaded successfully")

        # Generate images
        print("\n=== Generating clean and watermarked images ===")
        generate.generate(
            config=config,
            diffusion_model=diffusion_model,
            watermark=watermark,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            clean_path=args.clean_path,
            watermarked_path=args.watermarked_path,
            force_reload=args.force_reload
        )
        print("Image generation completed")

    cleanup_cuda_memory(True)

    # Calculate FID if requested
    eval_output_path = config.evaluation.eval_output_path
    if args.calculate_fid:
        print("\n=== Calculating FID score ===")
        fid_results = calculate_fid(
            config=config,
            clean_path=args.clean_path,
            watermarked_path=args.watermarked_path,
            original_path=args.original_path
        )

        # Save FID results to file
        os.makedirs(eval_output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fid_output_file = os.path.join(eval_output_path, f"fid_results_{timestamp}.pkl")
        with open(fid_output_file, 'wb') as f:
            pickle.dump(fid_results, f)
        print(f"FID results saved to: {fid_output_file}")
        print("FID calculation completed")

    # Calculate CLIP score if requested
    if args.calculate_clip:
        print("\n=== Calculating CLIP score ===")
        clip_results = calculate_clip_score(
            config=config,
            batch_size=args.batch_size,
            prompts_path=args.prompts_path,
            images_clean_path=args.clean_path,
            images_watermarked_path=args.watermarked_path
        )

        # Save CLIP results to file
        os.makedirs(eval_output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        clip_output_file = os.path.join(eval_output_path, f"clip_results_{timestamp}.pkl")
        with open(clip_output_file, 'wb') as f:
            pickle.dump(clip_results, f)
        print(f"CLIP results saved to: {clip_output_file}")
        print("CLIP calculation completed")

    print("\n=== Pipeline completed ===")


if __name__ == "__main__":
    main()
