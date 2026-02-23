"""
Data augmentation utilities for watermarking experiments.
"""

import io
import math
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from kornia.filters import median_blur
from PIL import Image
from torchvision.transforms import InterpolationMode as TIM

from ..utils import utils


# Helper functions for tensor/PIL conversions
def to_tensor(img):
    """Convert PIL image to tensor."""
    if isinstance(img, Image.Image):
        return T.ToTensor()(img)
    elif isinstance(img, torch.Tensor):
        return img
    raise ValueError("Unsupported type")


def to_pil(img):
    """Convert tensor to PIL image."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            return T.ToPILImage()(img)
        raise ValueError("Expected tensor of shape (C, H, W)")
    return img


class Transformation(ABC):
    """
    Abstract base class for image transformations.
    
    This class defines the interface for all image transformations
    to ensure consistency and reusability across different augmentation
    techniques.
    """

    @abstractmethod
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to an image tensor."""
        pass


class Augment:
    """
    A class for applying data augmentations to images in watermarking experiments.
    
    This class provides a flexible framework for applying various image transformations
    including rotation, color jittering, blur, noise, and compression artifacts.
    """

    def __init__(self, transforms: Optional[List[Callable]] = None, postprocess: Optional[List[Callable]] = None):
        """
        Initialize the Augment class with transformation pipelines.
        
        Args:
            transforms: List of image transformations (rotation, color, blur, noise).
                       If None, uses default transformations.
            postprocess: List of post-processing transformations.
                        If None, uses default PIL/tensor conversion.
        """
        if not transforms:  # Use default transformations
            transforms = [
                T.RandomRotation(90, fill=0.5),
                T.Compose([lambda img: to_pil(img), RandomJPEGCompression((25, 25)), lambda img: to_tensor(img), ]),
                T.RandomResizedCrop((512, 512), scale=(0.75, 0.75), ratio=(1, 1)),
                RandomDrop((0.64, 0.64)),
                T.GaussianBlur(kernel_size=15),
                SaltAndPepperNoise(p=0.05),
                AddGaussianNoise((0.1, 0.1)),
                T.ColorJitter(brightness=6.0, contrast=0.0, saturation=0.0, hue=0.0),
            ]
        if not postprocess:
            postprocess = T.Compose([
                lambda img: to_pil(img),
                lambda img: to_tensor(img)
            ])

        self.transforms = transforms
        self.postprocess = postprocess

    def __call__(self, x: torch.Tensor, idx: Optional[int] = None, return_names: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """
        Apply augmentations to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            idx: Optional index to select specific transformation. If None, random choice.
            return_names: Whether to return the names of the applied transformations.
            
        Returns:
            Augmented tensor with same shape as input, or tuple of (tensor, names) if return_names=True
        """
        return self.perturb(x, idx, return_names)

    def perturb(self, x: torch.Tensor, idx: Optional[int] = None, return_names: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """
        Apply perturbations (augmentations) to the input tensor.
        
        Args:
            x: Input tensor in range [-1, 1] of shape (batch_size, channels, height, width)
            idx: Optional index to select specific transformation. If None, random choice.
            return_names: Whether to return the names of the applied transformations.
            
        Returns:
            Perturbed tensor in range [-1, 1] with same shape as input, or tuple of (tensor, names) if return_names=True
        """
        x_cpu = (x.detach().cpu() + 1) / 2

        out = []
        out_names = []
        for img in x_cpu:
            transform = self.transforms[idx] if idx is not None else random.choice(self.transforms)
            transformed = transform(img)
            transformed = self.postprocess(transformed)
            out.append(transformed)
            out_names.append(transform.__class__.__name__)

        out = torch.stack(out)
        out = out.to(x.device) * 2 - 1
        if return_names:
            return out, out_names
        return out

    def visualize(self, x: torch.Tensor, decode: Callable):
        """
        Visualize original and augmented images for debugging.
        
        Args:
            x: Input tensor to visualize
            decode: Function to decode tensor to displayable format
        """
        utils.im.show(decode(x[:1]))
        utils.im.show(decode(x), figsize=12)
        utils.im.show(self.perturb(decode(x)), figsize=12)

    def __len__(self):
        return len(self.transforms)


class FullRotate(Transformation):
    """
    Apply rotation to images while maintaining original dimensions.
    
    This class rotates an image by a random angle within a specified range,
    then crops and resizes it back to the original dimensions to avoid
    black borders while preserving content.
    
    Args:
        degrees: Range of degrees for rotation. Can be a single number (symmetric range)
                or a tuple (min, max).
        interpolation: Interpolation method for rotation (default: BILINEAR).
    """

    def __init__(self, degrees: Union[float, Tuple[float, float]], interpolation: TIM = TIM.BILINEAR):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.interpolation = interpolation

        # Mapping TIM to PIL resampling filters
        self.pil_interpolation = {
            TIM.NEAREST: Image.Resampling.NEAREST,
            TIM.BILINEAR: Image.Resampling.BILINEAR,
            TIM.BICUBIC: Image.Resampling.BICUBIC,
            TIM.BOX: Image.Resampling.BOX,
            TIM.HAMMING: Image.Resampling.HAMMING,
            TIM.LANCZOS: Image.Resampling.LANCZOS
        }[interpolation]

    def get_crop_size(self, w, h, angle_rad):
        """Calculate the crop size to maintain aspect ratio after rotation."""
        cos = abs(math.cos(angle_rad))
        sin = abs(math.sin(angle_rad))
        new_w = w * cos + h * sin
        new_h = w * sin + h * cos
        if w == 0 or h == 0:
            return 0, 0
        return int(w * h / new_w), int(w * h / new_h)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(self.degrees[0], self.degrees[1])
        angle_rad = math.radians(angle)

        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            img = TF.to_pil_image(img)

        orig_w, orig_h = img.size
        rotated = TF.rotate(img, angle, interpolation=self.interpolation, expand=True)

        crop_w, crop_h = self.get_crop_size(orig_w, orig_h, angle_rad)
        rot_w, rot_h = rotated.size

        left = max((rot_w - crop_w) // 2, 0)
        top = max((rot_h - crop_h) // 2, 0)
        right = left + crop_w
        bottom = top + crop_h
        cropped = rotated.crop((left, top, right, bottom))

        resized = cropped.resize((orig_w, orig_h), resample=self.pil_interpolation)

        if is_tensor:
            return TF.to_tensor(resized)
        return resized


class AddGaussianNoise(Transformation):
    """
    Add Gaussian noise to images.
    
    This transformation adds random Gaussian noise to images to simulate
    sensor noise or other image degradation effects commonly encountered
    in real-world scenarios.
    
    Args:
        std: Standard deviation range for the Gaussian noise. Can be a single
            value or a tuple (min, max) for random selection.
    """

    def __init__(self, std: Union[float, Tuple[float, float]]):
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = to_tensor(img)
        return (img + torch.randn_like(img) * random.uniform(*self.std)) % 1


class RandomDrop(Transformation):
    """
    Randomly drop (zero out) a square region of the image.
    
    This augmentation simulates occlusion or missing data by setting
    a random square region of the image to zero (black). Useful for
    testing robustness against partial image corruption.
    
    Args:
        drop_am: Range for the fraction of image area to drop. Can be a tuple
                (min, max) for random selection between bounds.
    """

    def __init__(self, drop_am: Union[float, Tuple[float, float]]):
        self.drop_am = drop_am

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        drop_am = random.uniform(*self.drop_am)
        square_size = int(math.sqrt(drop_am * (img.shape[1] * img.shape[2])))
        img = img.clone()
        xs = random.randint(0, img.shape[1] - square_size)
        ys = random.randint(0, img.shape[2] - square_size)
        img[:, xs:xs + square_size, ys:ys + square_size] = 0
        return img


class SaltAndPepperNoise(Transformation):
    """
    Add salt and pepper noise to images.
    
    This transformation randomly sets pixels to maximum (salt) or minimum
    (pepper) values, simulating impulse noise that can occur in digital
    image acquisition and transmission.
    
    Args:
        p: Probability of noise application. Total noise probability is split
          equally between salt (white) and pepper (black) noise.
    """

    def __init__(self, p: float = 0.05):
        assert 0 <= p <= 1, "p must be between 0 and 1"
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3 or img.shape[0] != 3:
            raise ValueError(f"Expected (3, H, W), got {img.shape}")

        rnd = torch.rand_like(img)

        noisy = img.clone()
        salt_thresh = self.p / 2
        pepper_thresh = 1 - salt_thresh

        noisy[rnd < salt_thresh] = 1.0 if noisy.dtype.is_floating_point else 255
        noisy[rnd > pepper_thresh] = 0.0 if noisy.dtype.is_floating_point else 0

        return noisy


class MedianFilterBlur(Transformation):
    """
    Apply median filter blur to images.
    
    This transformation applies a median filter which replaces each pixel
    with the median value of pixels in its neighborhood. Effective for
    removing salt-and-pepper noise while preserving edges.
    
    Args:
        kernel_size: Size of the median filter kernel. Larger values
                    create more pronounced blurring effects.
    """

    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return median_blur(img.unsqueeze(0), self.kernel_size).squeeze(0)


class RandomJPEGCompression(Transformation):
    """
    Apply random JPEG compression to images.
    
    This transformation simulates JPEG compression artifacts by encoding
    and decoding images with varying quality levels. Useful for testing
    robustness against compression artifacts commonly encountered when
    images are shared or stored.
    
    Args:
        quality_range: Range of JPEG quality values (1-100). Lower values
                      result in more compression artifacts. Can be a tuple
                      (min, max) for random selection.
    """

    def __init__(self, quality_range: Tuple[int, int] = (30, 90)):
        self.quality_range = quality_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = to_pil(img)
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
