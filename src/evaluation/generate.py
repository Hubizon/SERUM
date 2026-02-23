import argparse
import concurrent.futures
import json
import os
import random
import re
import shutil
from collections import defaultdict
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torch import nn
from tqdm import tqdm

from ..utils.config import Config
from ..data import Augment
from . import scores
from ..models import Watermark, DiffusionModel, ModelWrapper, get_vae, decode
from ..utils.utils import cleanup_cuda_memory, save_config_snapshot


@torch.no_grad()
def generate(config: Config,
             diffusion_model: nn.Module,
             watermark: nn.Module,
             num_samples: Optional[int] = None,
             batch_size: Optional[int] = None,
             clean_path: Optional[str] = None,
             watermarked_path: Optional[str] = None,
             positive_prompts_path: Optional[str] = None,
             negative_prompts_path: Optional[str] = None,
             force_reload: bool = False) -> Tuple[str, str]:
    """Generate clean and watermarked images from positive and negative prompts."""
    if num_samples is None:
        num_samples = config.evaluation.num_samples
    if batch_size is None:
        batch_size = config.evaluation.batch_size
    if clean_path is None:
        clean_path = config.evaluation.clean_path
    if watermarked_path is None:
        watermarked_path = config.evaluation.watermarked_path
    if positive_prompts_path is None:
        positive_prompts_path = config.evaluation.positive_prompts_path
    if negative_prompts_path is None:
        negative_prompts_path = config.evaluation.negative_prompts_path

    if force_reload:
        clear_generation_folders([clean_path, watermarked_path])

    os.makedirs(clean_path, exist_ok=True)
    os.makedirs(watermarked_path, exist_ok=True)

    pattern = re.compile(r"^clean_(\d{6})\.png$")
    existing_files = [f for f in os.listdir(clean_path) if pattern.match(f)]
    existing_indices = {int(pattern.match(f).group(1)) for f in existing_files}
    start_idx = max(existing_indices) + 1 if existing_indices else 0
    num_samples = num_samples - start_idx
    if num_samples <= 0:
        print("All samples already generated.")
        return clean_path, watermarked_path

    positive_prompts, negative_prompts = load_pos_neg_prompts(config, positive_prompts_path, negative_prompts_path)

    # Calculate batches
    q, r = divmod(num_samples, batch_size)
    batches = [batch_size] * q + ([r] if r else [])

    sample_idx_clean, sample_idx_watermarked = start_idx, start_idx  # Global sample counter
    for batch_idx, batch_s in enumerate(tqdm(batches, desc="Generating batches")):
        # Clean memory before each batch
        cleanup_cuda_memory()

        # Select specific prompts for this batch
        prompt_start = sample_idx_clean
        prompt_end = min(prompt_start + batch_s, start_idx + num_samples)

        batch_positive_prompts = positive_prompts[prompt_start:prompt_end]
        batch_negative_prompts = negative_prompts[prompt_start:prompt_end]

        # Generate watermark noise and clean noise
        g_res, orig_noise = watermark(batch_s, ret_noise=True)

        # Generate clean samples with specific prompts
        samples = diffusion_model.generate_full(
            batch_size=batch_s, noise=orig_noise.half(), x_ts_ret=False,
            positive_prompts=batch_positive_prompts, negative_prompts=batch_negative_prompts
        )

        # Convert samples to images
        clean_imgs = decode(samples).cpu()

        # Save individual images
        for i in range(batch_s):
            # Save clean image with resizing
            clean_img_path = os.path.join(clean_path, f"clean_{sample_idx_clean:06d}.png")
            clean_img = torchvision.transforms.functional.to_pil_image(
                ((clean_imgs[i].cpu().float() + 1) / 2).clamp(0, 1))
            save_image(config, clean_img, clean_img_path)
            sample_idx_clean += 1

        del samples, clean_imgs
        cleanup_cuda_memory()

        # Generate watermarked samples
        samples_w = diffusion_model.generate_full(
            batch_size=batch_s, noise=g_res.half(), x_ts_ret=False,
            positive_prompts=batch_positive_prompts, negative_prompts=batch_negative_prompts
        )

        # Convert samples to images
        watermarked_imgs = decode(samples_w).cpu()

        # Save individual images
        for i in range(batch_s):
            # Save watermarked image with resizing
            watermarked_img_path = os.path.join(watermarked_path, f"watermarked_{sample_idx_watermarked:06d}.png")
            watermarked_img = torchvision.transforms.functional.to_pil_image(
                ((watermarked_imgs[i].cpu().float() + 1) / 2).clamp(0, 1))
            save_image(config, watermarked_img, watermarked_img_path)
            sample_idx_watermarked += 1

        del samples_w, watermarked_imgs, g_res, orig_noise
        cleanup_cuda_memory()

    print(f"Generated {num_samples} clean and watermarked images.")

    return clean_path, watermarked_path


def generate_based_on_custom_prompts(config: Config,
                                     diffusion_model: nn.Module,
                                     watermark: nn.Module,
                                     batch_size: int = None,
                                     clean_path: str = None,
                                     watermarked_path: str = None,
                                     positive_prompts: List[str] = None,
                                     negative_prompts: Optional[List[str]] = None,
                                     show: bool = False,
                                     preview_max: Optional[int] = None):
    """
    Generate clean and watermarked images based on custom prompts.
    
    Args:
        config: Configuration object
        diffusion_model: Diffusion model for image generation
        watermark: Watermark model
        batch_size: Batch size for generation
        clean_path: Path to save clean images
        watermarked_path: Path to save watermarked images
        positive_prompts: List of positive prompts
        negative_prompts: List of negative prompts
        show: Whether to display generated images
        preview_max: Maximum number of images to preview
        
    Returns:
        Tuple of (clean_path, watermarked_path)
    """
    customize = lambda path: os.path.join(
        os.path.dirname(path),
        "custom_" + os.path.basename(path)
    )

    if batch_size is None:
        batch_size = config.evaluation.batch_size
    if clean_path is None:
        clean_path = customize(config.evaluation.clean_path)
    if watermarked_path is None:
        watermarked_path = customize(config.evaluation.watermarked_path)

    custom_positive_prompts_path = customize(config.evaluation.positive_prompts_path)
    custom_negative_prompts_path = customize(config.evaluation.negative_prompts_path)
    save_pos_neg_prompts(config, positive_prompts, negative_prompts, custom_positive_prompts_path,
                         custom_negative_prompts_path)

    generate(
        config=config,
        diffusion_model=diffusion_model,
        watermark=watermark,
        num_samples=len(positive_prompts),
        batch_size=batch_size,
        clean_path=clean_path,
        watermarked_path=watermarked_path,
        positive_prompts_path=custom_positive_prompts_path,
        negative_prompts_path=custom_negative_prompts_path,
        force_reload=True
    )

    if show:
        clean_files = sorted([f for f in os.listdir(clean_path) if f.lower().endswith(".png")])
        wm_files = sorted([f for f in os.listdir(watermarked_path) if f.lower().endswith(".png")])
        n = min(len(clean_files), len(wm_files))
        if n == 0:
            print("No images found to display.")
            return clean_path, watermarked_path

        if preview_max is not None:
            n = min(n, preview_max)

        cols = 2
        rows = n
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, max(1, rows) * 4))
        if rows == 1:
            axes = [axes]  # make it iterable

        for i in range(n):
            clean_img = Image.open(os.path.join(clean_path, clean_files[i])).convert("RGB")
            wm_img = Image.open(os.path.join(watermarked_path, wm_files[i])).convert("RGB")

            ax_clean, ax_wm = axes[i]
            ax_clean.imshow(clean_img)
            ax_clean.set_title(f"Clean: {clean_files[i]}")
            ax_clean.axis("off")

            ax_wm.imshow(wm_img)
            ax_wm.set_title(f"Watermarked: {wm_files[i]}")
            ax_wm.axis("off")

        plt.tight_layout()
        plt.show()

    return clean_path, watermarked_path


def clear_generation_folders(paths: List[str]) -> None:
    """Clear existing generation folders."""
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed existing directory: {path}")


@torch.no_grad()
def generate_augmentations(config: Config,
                           images_path: str,
                           num_samples: Optional[int] = None,
                           augment: Optional[Augment] = None,
                           augment_idx: Optional[int] = None,
                           augment_path: str = None):
    """Generate and save augmented versions of the original images."""
    if augment_path is None:
        augment_path = config.evaluation.augmented_path
    clear_generation_folders([augment_path])

    os.makedirs(augment_path, exist_ok=True)

    filenames = sorted(os.listdir(images_path))
    if num_samples is not None:
        filenames = filenames[:num_samples]

    for filename in filenames:
        img_path = os.path.join(images_path, filename)
        out_path = os.path.join(augment_path, filename)
        img = Image.open(img_path).convert("RGB")
        img = torchvision.transforms.functional.to_tensor(img) * 2 - 1
        if augment:
            img = augment(img.unsqueeze(0), augment_idx).squeeze(0)

        save_image(config, img, out_path)
    cleanup_cuda_memory()

    return augment_path


@torch.no_grad()
def save_image(config: Config, image: torch.Tensor, path: str, target_size: Optional[Tuple[int, int]] = None) -> None:
    """
    Crop the image to the center, resize to target_size, and save as PNG.
        
    Args:
        image (PIL.Image.Image): Input PIL image.
        path (str): Destination file path.
    """
    if target_size is None:
        target_size = config.evaluation.size

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if isinstance(image, torch.Tensor):
        image = torchvision.transforms.functional.to_pil_image(((image.cpu().float() + 1) / 2).clamp(0, 1))

    # Ensure RGB
    img = image.convert("RGB")

    # Resize to target_size
    if target_size:
        # Center crop to square based on shortest side
        width, height = img.size
        new_side = min(width, height)
        left = (width - new_side) // 2
        top = (height - new_side) // 2
        img = img.crop((left, top, left + new_side, top + new_side))
        img = img.resize(target_size, Image.LANCZOS)

    # Save as PNG (fixed: save the processed img, not the original image)
    img.save(path, "PNG")


@torch.no_grad()
def load_images_generator(config: Config,
                          images_path: str,
                          batch_size: int = None,
                          augment: Optional[Augment] = None,
                          augment_idx: Optional[int] = None,
                          max_num: Optional[int] = None,
                          return_prompts: bool = False):
    """Generator for clean images."""
    if batch_size is None:
        batch_size = config.evaluation.batch_size

    device = config.get_device()
    cleanup_cuda_memory()

    files = sorted(os.listdir(images_path))
    if max_num is not None:
        files = files[:max_num]
    num_samples = len(files)

    if return_prompts:
        positive_prompts, negative_prompts = load_pos_neg_prompts(config)
        if max_num is not None:
            positive_prompts = positive_prompts[:max_num]
            negative_prompts = negative_prompts[:max_num]

    # Calculate batches
    q, r = divmod(num_samples, batch_size)
    batches = [batch_size] * q + ([r] if r else [])

    for i, batch_s in enumerate(batches):
        # cleanup_cuda_memory()

        # Select specific prompts for this batch
        idx_start = i * batch_size
        idx_end = min(idx_start + batch_s, num_samples)

        batch_files = files[idx_start:idx_end]

        imgs = []
        for f in batch_files:
            img_path = os.path.join(images_path, f)
            img = Image.open(img_path).convert("RGB")
            tensor_img = torchvision.transforms.functional.to_tensor(img).unsqueeze(0) * 2 - 1
            imgs.append(tensor_img)

        imgs = torch.cat(imgs, dim=0).to(device)
        if augment:
            imgs = augment(imgs, augment_idx)

        if return_prompts:
            yield imgs, positive_prompts[idx_start:idx_end], negative_prompts[idx_start:idx_end]
        else:
            yield imgs
    cleanup_cuda_memory()


@torch.no_grad()
def save_pos_neg_prompts(config: Config,
                         prompts: List[str],
                         negative_prompts: Optional[List[str]] = None,
                         positive_prompts_path: Optional[str] = None,
                         negative_prompts_path: Optional[str] = None) -> None:
    """Saves positive and negative prompts to a file."""
    if positive_prompts_path is None:
        positive_prompts_path = config.evaluation.positive_prompts_path
    if negative_prompts_path is None:
        negative_prompts_path = config.evaluation.negative_prompts_path

    os.makedirs(os.path.dirname(positive_prompts_path), exist_ok=True)
    os.makedirs(os.path.dirname(negative_prompts_path), exist_ok=True)

    with open(positive_prompts_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(f"{_process_prompt(prompt)}\n")

    with open(negative_prompts_path, 'w', encoding='utf-8') as f:
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)
        for prompt in negative_prompts:
            f.write(f"{_process_prompt(prompt)}\n")

    print(f"Saved {len(prompts)} positive prompts to {positive_prompts_path}")
    print(f"Saved {len(negative_prompts)} negative prompts to {negative_prompts_path}")


def load_pos_neg_prompts(config: Config,
                         positive_prompts_path: Optional[str] = None,
                         negative_prompts_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Loads positive and negative prompts from a file."""
    if positive_prompts_path is None:
        positive_prompts_path = config.evaluation.positive_prompts_path
    if negative_prompts_path is None:
        negative_prompts_path = config.evaluation.negative_prompts_path

    with open(positive_prompts_path, 'r', encoding='utf-8') as f:
        positive_prompts = [line.strip() for line in f]
    with open(negative_prompts_path, 'r', encoding='utf-8') as f:
        negative_prompts = [line.strip() for line in f]
    return positive_prompts, negative_prompts


@torch.no_grad()
def load_coco(config: Config,
              annotation_path: str,
              images_path: Optional[str] = None,
              num_samples: int = None,
              positive_prompts_path: str = None,
              negative_prompts_path: str = None,
              original_path: str = None):
    """"Load COCO prompts and optionally images from the annotations file."""
    if num_samples is None:
        num_samples = config.evaluation.fid_num_samples
    if positive_prompts_path is None:
        positive_prompts_path = config.evaluation.positive_prompts_path
    if negative_prompts_path is None:
        negative_prompts_path = config.evaluation.negative_prompts_path
    if original_path is None:
        original_path = config.evaluation.original_path

    # Load the JSON
    with open(annotation_path, "r") as f:
        captions_data = json.load(f)

    by_image = {}
    for ann in captions_data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann["caption"])

    # Sample one caption per image_id
    coco_prompts = [(f'COCO_val2014_{img_id:012d}.jpg', random.choice(caps)) for img_id, caps in by_image.items()]
    coco_prompts = random.sample(coco_prompts, k=num_samples)

    print(f"Extracted {len(coco_prompts)} unique (image_id, prompt) pairs")
    print(f"Sample prompt ({coco_prompts[0][0]}): {coco_prompts[0][1]}")

    os.makedirs(os.path.dirname(positive_prompts_path), exist_ok=True)
    os.makedirs(os.path.dirname(negative_prompts_path), exist_ok=True)

    # Initialize prompt files (clear if they exist)
    with open(positive_prompts_path, 'w', encoding='utf-8') as f:
        f.write("")
    with open(negative_prompts_path, 'w', encoding='utf-8') as f:
        f.write("")

    positive_prompts = [p[1] for p in coco_prompts]
    negative_prompts = [""] * len(coco_prompts)
    img_filenames = [p[0] for p in coco_prompts]

    with open(positive_prompts_path, 'a', encoding='utf-8') as f:
        for prompt in positive_prompts:
            f.write(f"{_process_prompt(prompt)}\n")

    with open(negative_prompts_path, 'a', encoding='utf-8') as f:
        for prompt in negative_prompts:
            f.write(f"{_process_prompt(prompt)}\n")

    cleanup_cuda_memory()

    # Remove directory if it exists and recreate it
    if os.path.exists(original_path):
        shutil.rmtree(original_path, ignore_errors=True)
        print(f"Removed existing directory: {original_path}")

    os.makedirs(original_path, exist_ok=True)
    print(f"Created directory: {original_path}")

    copied_count = 0

    for filename in tqdm(img_filenames, desc="Copying images"):
        src_path = os.path.join(images_path, filename)
        dst_filename = f"original_{copied_count:06d}.png"
        dst_path = os.path.join(original_path, dst_filename)

        if os.path.exists(src_path):
            try:
                # Open, convert to RGB, resize if needed, and save as PNG
                img = Image.open(src_path).convert("RGB")
                save_image(config, img, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Successfully copied {copied_count}/{num_samples} images")


@torch.no_grad()
def _process_prompt(prompt: str) -> str:
    """Process a prompt by stripping whitespace and removing newlines."""
    return prompt.strip().replace('\n', '\\n')


@torch.no_grad()
def load_coco_images_from_prompts(config: Config,
                                  annotation_path: str,
                                  images_path: str,
                                  num_samples: Optional[int] = None,
                                  prompts_path: str = None,
                                  positive_prompts_path: str = None,
                                  negative_prompts_path: str = None,
                                  original_path: str = None,
                                  *,
                                  ensure_unique: bool = False,
                                  max_workers: Optional[int] = None):
    """
    Load COCO images based on prompts from a file (optimized with parallel processing).
    
    This function reads prompts from a file, matches them to COCO annotations, and loads
    the corresponding images. Images are processed in parallel but saved sequentially
    to maintain deterministic ordering and filenames.
    
    Args:
        config (Config): Configuration object containing evaluation settings.
        annotation_path (str): Path to COCO annotations JSON file.
        images_path (str): Directory containing COCO image files.
        num_samples (Optional[int]): Maximum number of samples to process. If None,
            processes all prompts in the file.
        prompts_path (str, optional): Path to file containing prompts. If None,
            uses config.evaluation.positive_prompts_path.
        positive_prompts_path (str, optional): Output path for processed positive prompts.
            If None, uses config.evaluation.positive_prompts_path.
        negative_prompts_path (str, optional): Output path for negative prompts file.
            If None, uses config.evaluation.negative_prompts_path.
        original_path (str, optional): Output directory for processed images.
            If None, uses config.evaluation.original_path.
        ensure_unique (bool): If True, ensures each image_id is used at most once
            across all prompts to avoid duplicate images. Default is False.
        max_workers (Optional[int]): Maximum number of worker threads for parallel
            image loading. If None, uses min(8, cpu_count()).
    
    Raises:
        ValueError: If a prompt is not found in the annotations file.
        FileNotFoundError: If annotation_path or prompts_path doesn't exist.
    """

    if prompts_path is None:
        prompts_path = config.evaluation.positive_prompts_path
    if positive_prompts_path is None:
        positive_prompts_path = config.evaluation.positive_prompts_path
    if negative_prompts_path is None:
        negative_prompts_path = config.evaluation.negative_prompts_path
    if original_path is None:
        original_path = config.evaluation.original_path

    # Read prompts (trim to num_samples)
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f]
    if num_samples is not None:
        prompts = prompts[:num_samples]

    # Load annotations
    with open(annotation_path, "r", encoding="utf-8") as f:
        captions_data = json.load(f)

    # Recreate original dir (remove + recreate)
    if os.path.exists(original_path):
        shutil.rmtree(original_path, ignore_errors=True)
        print(f"Removed existing directory: {original_path}")
    os.makedirs(original_path, exist_ok=True)

    # Build fast lookup with unique image_ids per variant (preserve order)
    caption_map = defaultdict(list)
    for ann in captions_data.get("annotations", []):
        caption = ann["caption"]
        image_id = ann["image_id"]
        variants = (caption, caption.strip(), _process_prompt(caption))
        for var in variants:
            if not var:
                continue
            # ensure image_id appears at most once for this variant; preserve order
            if image_id not in caption_map[var]:
                caption_map[var].append(image_id)

    img_counter = 0
    duplicates_counter = 0
    used_image_ids = set()

    # Helper to load an image fully and return a safe copy
    def _load_image_copy(path):
        """Load image from path and return a safe copy."""
        img = Image.open(path)
        try:
            img.load()  # force decoding now
            img_copy = img.copy()
        finally:
            img.close()
        return img_copy

    # Determine number of workers
    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 1))

    load_futures = {}
    loaded_images = {}  # prompt_index -> (filename, PIL.Image)

    # Schedule loads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
        for p_idx, prompt in enumerate(prompts):
            matches = caption_map.get(prompt, [])
            if not matches:
                # Keep the original behaviour: raise with current img_counter in message
                raise ValueError(f"[{img_counter:06d}] Prompt not found in annotations: {prompt}")

            if len(matches) > 1:
                # Use p_idx so warning indicates which prompt had the duplicates
                print(f"[{p_idx:06d}] Warning: Multiple entries found for prompt '{prompt}'. Using the first one.")
                duplicates_counter += 1

            # choose image_id, optionally ensuring uniqueness across prompts
            chosen_image_id = None
            if ensure_unique:
                for mid in matches:
                    if mid not in used_image_ids:
                        chosen_image_id = mid
                        break
                if chosen_image_id is None:
                    # fallback to first and warn 
                    print(
                        f"[{p_idx:06d}] Warning: all candidate images already used for prompt '{prompt}'. Using the first one.")
                    chosen_image_id = matches[0]
            else:
                chosen_image_id = matches[0]

            image_id = chosen_image_id
            filename = f'COCO_val2014_{image_id:012d}.jpg'
            img_path = os.path.join(images_path, filename)

            if os.path.exists(img_path):
                # schedule load (decoding) in threadpool
                future = exe.submit(_load_image_copy, img_path)
                load_futures[future] = (p_idx, filename, image_id)
                # mark used_image_ids now when ensure_unique True to avoid race where another prompt picks same id
                if ensure_unique:
                    used_image_ids.add(image_id)
            else:
                print(f"Image file not found: {img_path}")

        # collect loaded images (catch load errors)
        for fut in concurrent.futures.as_completed(list(load_futures.keys())):
            p_idx, filename, image_id = load_futures[fut]
            try:
                img = fut.result()
                loaded_images[p_idx] = (filename, img, image_id)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    # Now save sequentially in prompt order
    for p_idx, prompt in enumerate(prompts):
        if p_idx in loaded_images:
            filename, img, image_id = loaded_images[p_idx]
            try:
                out_path = os.path.join(original_path, f"original_{img_counter:06d}.png")
                save_image(config, img, out_path)
                img_counter += 1
            except Exception as e:
                # match original message text
                print(f"Error loading image {filename}: {e}")

    # Write positive prompts file (processed)
    processed_prompts = [_process_prompt(p) for p in prompts]
    pos_dir = os.path.dirname(positive_prompts_path)
    if pos_dir:
        os.makedirs(pos_dir, exist_ok=True)
    with open(positive_prompts_path, 'w', encoding='utf-8') as f:
        if processed_prompts:
            f.write("\n".join(processed_prompts) + "\n")
        else:
            f.write("")

    # Write negative prompts (same processed empty prompt per sample)
    neg_dir = os.path.dirname(negative_prompts_path)
    if neg_dir:
        os.makedirs(neg_dir, exist_ok=True)
    empty_processed = _process_prompt("")
    with open(negative_prompts_path, 'w', encoding='utf-8') as f:
        if prompts:
            f.write("\n".join([empty_processed] * len(prompts)) + "\n")
        else:
            f.write("")

    print(f"Successfully loaded {img_counter}/{len(prompts)} images")
    print(f"Found {duplicates_counter} duplicate prompts")


@torch.no_grad()
def main() -> None:
    """Main function for image generation."""
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description='Generate clean and watermarked images from prompts')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (overrides default)')
    parser.add_argument('--generate', action='store_true',
                        help='Generate clean and watermarked images')
    parser.add_argument('--load-coco', choices=['full', 'images', 'none'], default='none',
                        help='Load COCO dataset: full (prompts+images), images (from existing prompts), none (skip)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help=f'Number of samples to generate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for generation (default: 32)')
    parser.add_argument('--force-reload', action='store_true', default=True,
                        help='Force reload and regenerate existing files (default: True)')
    parser.add_argument('--coco-annotations', type=str, default=None,
                        help='Path to COCO annotations file')
    parser.add_argument('--coco-images', type=str, default=None,
                        help='Path to COCO images directory')
    parser.add_argument('--prompts-path', type=str, default=None,
                        help='Path to custom prompts file (overrides default)')
    parser.add_argument('--clean-path', type=str, default=None,
                        help='Output path for clean images')
    parser.add_argument('--watermarked-path', type=str, default=None,
                        help='Output path for watermarked images')
    parser.add_argument('--original-path', type=str, default=None,
                        help='Output path for original COCO images')

    args = parser.parse_args()

    config = Config(args.config)

    if args.prompts_path is None:
        args.prompts_path = config.evaluation.positive_prompts_path

    # Save config snapshot at the beginning
    save_config_snapshot(config, "generate_config")

    get_vae(config)

    print(f"Starting image generation with arguments:")
    print(f"  Generate: {args.generate}")
    print(f"  Load COCO: {args.load_coco}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Force reload: {args.force_reload}")

    # Load COCO dataset if requested
    if args.load_coco == 'full':
        print("\n=== Loading COCO prompts and images ===")
        load_coco(
            config=config,
            annotation_path=args.coco_annotations,
            images_path=args.coco_images,
            num_samples=args.num_samples,
            original_path=args.original_path
        )

    elif args.load_coco == 'images':
        print("\n=== Loading COCO images from existing prompts ===")
        load_coco_images_from_prompts(
            config=config,
            annotation_path=args.coco_annotations,
            images_path=args.coco_images,
            prompts_path=args.prompts_path,
            original_path=args.original_path
        )
    else:
        print("\n=== Skipping COCO loading - checking for existing prompts ===")

    # Only generate if --generate flag is True
    if args.generate:
        # Check if prompts exist
        if not os.path.exists(args.prompts_path):
            raise FileNotFoundError(
                f"No prompts file found at {args.prompts_path}. "
                "Please provide prompts using --prompts-path or load COCO dataset with --load-coco."
            )

        print(f"Using prompts from: {args.prompts_path}")

        # Initialize models
        print("\n=== Initializing models ===")
        device = config.get_device()
        print(f"Using device: {device}")

        to_diffusion = DiffusionModel(config).requires_grad_(False).eval().to(device)
        diffusion_model = ModelWrapper(to_diffusion).requires_grad_(False).eval().to(device)
        watermark = Watermark.load(config).requires_grad_(False).eval().to(device)
        print("Models loaded successfully")

        # Generate images
        print("\n=== Generating clean and watermarked images ===")
        clean_path, watermarked_path = generate(
            config=config,
            diffusion_model=diffusion_model,
            watermark=watermark,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            clean_path=args.clean_path,
            watermarked_path=args.watermarked_path,
            positive_prompts_path=args.prompts_path,
            force_reload=args.force_reload
        )

        print(f"\n=== Generation completed ===")
        print(f"Clean images saved to: {clean_path}")
        print(f"Watermarked images saved to: {watermarked_path}")
    else:
        print("\n=== Skipping image generation (use --generate to enable) ===")

    print("\n=== Process completed ===")


if __name__ == "__main__":
    main()
