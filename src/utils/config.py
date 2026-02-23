import os
import yaml
import torch
from types import SimpleNamespace


class Config:
    """Centralized configuration management for watermarking experiments."""

    def __init__(self, config_path=None):
        # Find the root directory (parent of src/)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_dir = os.path.join(root_dir, "configs")

        if config_path is None:
            config_path = os.path.join(
                config_dir, "config.yaml"
            )
        elif not os.path.exists(config_path):
            # if given path doesn't exist, try configs/{config_path}
            candidate = os.path.join(config_dir, config_path)
            if os.path.exists(candidate):
                config_path = candidate
            else:
                raise FileNotFoundError(
                    f"Config file not found: {config_path} or {candidate}"
                )
        
        with open(config_path, "r") as f:
            self._raw = yaml.safe_load(f)

        print(f"Loaded config from: {config_path}")
        self._parse()

    def _ns(self, d: dict) -> SimpleNamespace:
        """Helper: convert dict to object-like namespace."""
        return SimpleNamespace(**d)

    def _parse(self):
        cfg = self._raw

        # Top-level experiment directory
        self.experiment_dir = cfg["experiment_dir"]

        # Diffusion settings
        self.diffusion = self._ns(cfg["diffusion"])
        self.diffusion.torch_dtype = getattr(torch, self.diffusion.torch_dtype)

        # Sampling
        self.sampling = self._ns(cfg["diffusion"]["sampling"])

        # Shapes
        self.shapes = self._ns(cfg["shapes"])

        # Watermark
        wm = cfg["watermark"]
        self.watermark = self._ns({
            "grid": self._ns(wm["grid"]),
            "buffer": self._ns(wm["buffer"]),
            "model_dir": wm["model_dir"],
            "score_model_file": wm["score_model_file"],
            "watermark_file": wm["watermark_file"],
        })

        # Training
        tr = cfg["training"]
        self.training = self._ns({
            **tr,
            "visualization": self._ns(tr["visualization"]),
        })

        # Evaluation
        self.evaluation = self._ns(cfg["evaluation"])

        # Dataset
        self.dataset = self._ns({
            **cfg["dataset"],
            "paths": self._ns(cfg["dataset"]["paths"]),
        })

        # Derived constants
        self.base_latent_shape = (
            1,
            self.shapes.latent_channels,
            self.shapes.latent_height,
            self.shapes.latent_width,
        )

        # Resolved paths (prepend experiment_dir)
        def _join(p): 
            return os.path.join(self.experiment_dir, p)
        self.watermark.model_dir = _join(self.watermark.model_dir)
        self.training.checkpoint_dir = _join(self.training.checkpoint_dir)
        self.evaluation.original_path = _join(self.evaluation.original_path)
        self.evaluation.clean_path = _join(self.evaluation.clean_path)
        self.evaluation.watermarked_path = _join(self.evaluation.watermarked_path)
        self.evaluation.positive_prompts_path = _join(self.evaluation.positive_prompts_path)
        self.evaluation.negative_prompts_path = _join(self.evaluation.negative_prompts_path)
        self.evaluation.augmented_path = _join(self.evaluation.augmented_path)
        self.evaluation.eval_output_path = _join(self.evaluation.eval_output_path)

    def get_device(self, preferred_device=None):
        """Return the best available device (CUDA if available)."""
        device = preferred_device or self.diffusion.device
        if device == 'cpu' or not torch.cuda.is_available():
            return "cpu"
        return device

