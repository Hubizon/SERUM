"""
Dataset generation and augmentation.

This module provides utilities for downloading prompts, generating images,
and creating augmented datasets for training watermark models.
"""

import argparse
import datetime
import glob
import os
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm, trange

from ..utils.config import Config
from .augmentation import Augment
from ..models import ModelWrapper, DiffusionModel, encode, decode, get_vae
from ..utils.utils import cleanup_cuda_memory, save_config_snapshot


class Dataset:
    def __init__(self, config: Config):
        self.config = config
        self.device = self.config.get_device()

        # Load prompts dataset
        print(f"Loading dataset: {self.config.dataset.name}")
        dataset = load_dataset(self.config.dataset.name)
        prompts = list(dataset['train']['Prompt'])

        # Split prompts
        self.train_prompts = prompts[:self.config.dataset.train_split]
        self.test_prompts = prompts[self.config.dataset.train_split:]

        self.generated_data = None
        self.augmented_data = None

        print(f"Loaded {len(self.train_prompts)} training prompts and {len(self.test_prompts)} test prompts")

    @torch.no_grad()
    def generate_data(self, diffusion_model: Optional[ModelWrapper] = None,
                      start_idx: Optional[int] = None,
                      end_idx: Optional[int] = None,
                      batch_size: Optional[int] = None,
                      use_test_prompts: bool = False) -> List[Tensor]:
        """
        Generate images from text prompts using diffusion model.
        
        Args:
            diffusion_model: Optional pre-initialized model wrapper
            start_idx: Starting index for prompt selection
            end_idx: Ending index for prompt selection (-1 for all)
            batch_size: Number of images to generate per batch
            use_test_prompts: Whether to use test prompts instead of training prompts
            
        Returns:
            List of generated image tensors
        """
        if start_idx is None:
            start_idx = self.config.dataset.img_start
        if end_idx is None:
            end_idx = self.config.dataset.img_end
        if batch_size is None:
            batch_size = self.config.dataset.batch_size

        if diffusion_model is None:
            print("Initializing diffusion model...")
            diffusion_model = ModelWrapper(DiffusionModel(self.config).to(self.device)).to(self.device)

        self.diffusion_model = diffusion_model

        # Select prompt set
        prompts = self.test_prompts if use_test_prompts else self.train_prompts
        if end_idx == -1:
            end_idx = len(prompts)

        print(f"Generating images for prompts {start_idx} to {end_idx}")
        print(f"Using {'test' if use_test_prompts else 'train'} prompts")

        generated_data = []

        # Generate in batches
        for i in trange(start_idx, end_idx, batch_size, desc="Generating images"):
            # Prepare batch
            batch_end = min(i + batch_size, end_idx)
            positive_prompts = prompts[i:batch_end]
            negative_prompts = [''] * len(positive_prompts)

            # Generate images
            try:
                generated_images = self.diffusion_model.predict_latent(positive_prompts, negative_prompts)
                generated_data.append(generated_images.cpu())

            except Exception as e:
                print(f"Error generating batch {i}-{batch_end}: {e}")
                continue

        if len(generated_data) > 0 and len(generated_data[-1]) != batch_size:
            print(f"Last batch size {len(generated_data[-1])} is smaller than {batch_size}, removing it")
            generated_data = generated_data[:-1]

        cleanup_cuda_memory()

        print(f"Generated {len(generated_data)} batches")
        self.generated_data = generated_data
        return generated_data

    @torch.no_grad()
    def generate_augmented_data(self, data: List[Tensor] = None,
                                augmentation_pipeline: Optional[Augment] = None) -> List[Tensor]:
        """
        Apply augmentations to dataset using integrated VAE functions.
        
        Args:
            data: List of latent tensors to augment (uses self.generated_data if None)
            augmentation_pipeline: Optional augmentation pipeline (creates default if None)
            
        Returns:
            List of augmented latent tensors
        """
        if data is None:
            if not hasattr(self, 'generated_data'):
                raise ValueError("No data provided and no generated_data available. Generate data first.")
            data = self.generated_data

        if augmentation_pipeline is None:
            augmentation_pipeline = Augment()

        print("Applying data augmentations...")
        augmented_data = []

        for batch in tqdm(data, desc="Augmenting batches"):
            try:
                # Decode to image space
                images = decode(batch.to(self.device))
                images = images.float()

                # Apply augmentations
                augmented_images = augmentation_pipeline(images)

                # Encode back to latent space
                augmented_latents = encode(augmented_images.half())
                augmented_data.append(augmented_latents.cpu())

            except Exception as e:
                print(f"Error augmenting batch: {e}")
                # Skip this batch and continue
                continue

        cleanup_cuda_memory()

        self.augmented_data = augmented_data
        return augmented_data

    @torch.no_grad()
    def save_dataset(self, include_date: bool = True) -> None:
        """
        Save complete dataset to configured paths.
        
        Args:
            include_date: Whether to append timestamp to filename
        """
        # Create data directory
        os.makedirs(self.config.dataset.paths.data_dir, exist_ok=True)

        if self.generated_data:
            filepath = os.path.join(self.config.dataset.paths.data_dir, self.config.dataset.paths.generated_data_file)
            if include_date:
                timestamp = datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
                base, ext = os.path.splitext(filepath)
                filepath = f"{base}{timestamp}{ext}"

            torch.save(self.generated_data, filepath)
            print(f"Saved generated dataset to {filepath} with length {len(self.generated_data)}")
        else:
            print("No generated data to save")

        if self.augmented_data:
            filepath = os.path.join(self.config.dataset.paths.data_dir, self.config.dataset.paths.augmented_data_file)
            if include_date:
                timestamp = datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
                base, ext = os.path.splitext(filepath)
                filepath = f"{base}{timestamp}{ext}"

            torch.save(self.augmented_data, filepath)
            print(f"Saved augmented dataset to {filepath} with length {len(self.augmented_data)}")
        else:
            print("No augmented data to save")

    @staticmethod
    def load_dataset(config: Config, filename: str) -> Tensor:
        """
        Load complete dataset from configured paths.
        
        Args:
            config: Configuration object containing dataset paths
            filename: Name of the file to load from (e.g., 'data_sd.pt')
            
        Returns:
            Loaded tensor data
        """

        filepath = os.path.join(config.dataset.paths.data_dir, filename)

        if not os.path.exists(filepath):
            # Try to find latest timestamped version
            base, ext = os.path.splitext(filepath)
            pattern = f"{base}_*{ext}"
            files = glob.glob(pattern)
            if files:
                filepath = max(files)  # Get most recent
            else:
                raise FileNotFoundError(f"No dataset found at {filepath}")

        data = torch.load(filepath, map_location='cpu', weights_only=False)
        print(f"Loaded dataset from {filepath} with length {len(data)}")
        return data

    @staticmethod
    def load_data_pair(config: Config) -> Tuple[Tensor, Tensor]:
        """
        Simple function to load data and augmented data from default paths.
            
        Returns:
            Tuple of (original_data, augmented_data)
        """
        data_gen = Dataset.load_dataset(config, config.dataset.paths.generated_data_file)
        data_aug = Dataset.load_dataset(config, config.dataset.paths.augmented_data_file)
        return data_gen, data_aug

    @staticmethod
    def get_prompts(config: Config,
                    use_test: bool = False,
                    start_idx: Optional[int] = None,
                    end_idx: Optional[int] = None) -> List[str]:
        """
        Get prompts for manual use.
            
        Args:
            config: Configuration object containing dataset settings
            use_test: Whether to use test prompts
            start_idx: Starting index
            end_idx: Ending index
                
        Returns:
            List of prompts
        """
        if start_idx is None:
            start_idx = config.dataset.img_start
        if end_idx is None:
            end_idx = config.dataset.img_end

        dataset = Dataset(config)
        prompts = dataset.test_prompts if use_test else dataset.train_prompts

        if end_idx == -1:
            end_idx = len(prompts)
        else:
            end_idx = min(end_idx, len(prompts))

        return prompts[start_idx:end_idx]

    @staticmethod
    def visualize(data_gen: List[Tensor], data_aug: List[Tensor], n_images: int = 2) -> None:
        """
        Visualize n_images randomly sampled pairs from data_gen and data_aug.
        Each row shows a generated and its corresponding augmented image.
        
        Args:
            data_gen: List of generated image tensors
            data_aug: List of augmented image tensors
            n_images: Number of image pairs to visualize
        """
        # Determine the number of batches and batch size
        num_batches = len(data_gen)
        batch_size = data_gen[0].shape[0] if num_batches > 0 else 0
        total_images = num_batches * batch_size
        if total_images == 0:
            print("No images to visualize.")
            return

        # Sample n_images unique indices
        indices = random.sample(range(total_images), min(n_images, total_images))

        def get_img_from_data(data, idx):
            batch_idx = idx // batch_size
            img_idx = idx % batch_size
            img_latent = data[batch_idx][img_idx:img_idx + 1]
            img = decode(img_latent)
            img = ((img + 1) / 2).clamp(0, 1).float()
            return img

        def show_tensor_image(img, ax, title=None):
            # img: (1, 3, H, W) or (3, H, W)
            if img.dim() == 4:
                img = img[0]
            img = img.detach().cpu().float()
            img = img.permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(img)
            ax.axis('off')
            if title:
                ax.set_title(title)

        fig, axs = plt.subplots(n_images, 2, figsize=(8, 4 * n_images), squeeze=False)
        for row, idx in enumerate(indices):
            img_gen = get_img_from_data(data_gen, idx)
            img_aug = get_img_from_data(data_aug, idx)
            show_tensor_image(img_gen, axs[row][0], title='Generated')
            show_tensor_image(img_aug, axs[row][1], title='Augmented')
        plt.tight_layout()
        plt.show()


def main():
    """Main function for command-line interface."""
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Generate and augment datasets for watermarking")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (overrides default)')
    parser.add_argument('--mode', choices=['generate', 'augment', 'both'], default='both',
                        help='Operation mode: generate images, augment existing data, or both')
    parser.add_argument('--use-test-prompts', action='store_true',
                        help='Use test prompts instead of training prompts (default: False)')
    parser.add_argument('--start-idx', type=int, default=None,
                        help='Starting index for prompt selection (overrides config)')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='Ending index for prompt selection, -1 for all (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for generation (overrides config)')

    args = parser.parse_args()

    # Load the provided config
    config = Config(args.config)
    if args.start_idx is not None:
        config.dataset.img_start = args.start_idx
    if args.end_idx is not None:
        config.dataset.img_end = args.end_idx
    if args.batch_size is not None:
        config.dataset.batch_size = args.batch_size

    # Save config snapshot at the beginning
    save_config_snapshot(config, "dataset_config")

    print(f"Running in {args.mode} mode")
    print(f"Device: {config.get_device()}")
    print(f"Parameters: start_idx={args.start_idx}, end_idx={args.end_idx}, batch_size={args.batch_size}")

    get_vae(config)

    # Initialize dataset handler
    dataset = Dataset(config)

    if args.mode in ['generate', 'both']:
        # Generate new data using config values or overrides
        print("Generating dataset...")
        generated_data = dataset.generate_data(
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            batch_size=args.batch_size,
            use_test_prompts=args.use_test_prompts
        )
        print(f"Generated {len(generated_data)} batches")

    if args.mode in ['augment', 'both']:
        # Load or use existing data for augmentation
        try:
            if args.mode == 'augment':
                data = Dataset.load_dataset(config, config.dataset.paths.generated_data_file)
            else:
                # Use generated data from previous step
                if not hasattr(dataset, 'generated_data'):
                    raise ValueError("No generated data available.")
                data = dataset.generated_data

            # Augment the data
            print("Augmenting data...")
            augmented_data = dataset.generate_augmented_data(data)
            print(f"Augmented {len(augmented_data)} batches")
        except Exception as e:
            print(f"Error processing augmentation: {e}")
            return

    # Always save datasets when they are created/processed
    try:
        dataset.save_dataset(include_date=True)
    except Exception as e:
        print(f"Error saving the dataset: {e}")
        return

    print("Dataset processing completed!")


if __name__ == "__main__":
    main()
