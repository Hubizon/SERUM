# Usage

Run all commands from the project root. Command-line flags override corresponding values in the loaded config.

---
## Dataset Generation
```bash
python -m src.data.dataset --config CONFIG --mode MODE [options]
```

Options:
- `--config PATH` : YAML config file to load
- `--mode generate|augment|both` : Operation mode
- `--batch-size N` : Batch size for generation (overrides config)
- `--start-idx N` / `--end-idx N` : Prompt index range (use `-1` end for all) (overrides config)
- `--use-test-prompts` : Use test prompts instead of training set

---
## Training
```bash
python -m src.training.train --config CONFIG [options]
```

Options:
- `--include-eval` : Run evaluation after training
- `--dont-load-checkpoint` : Force fresh start (ignore existing checkpoint)

---
## Evaluation
```bash
python -m src.evaluation.eval --config CONFIG [options]
```

Core Options:
- `--num-samples N` : Number of samples to evaluate (overrides config)
- `--eval-clean` : Only clean vs watermarked (no augmentation robustness)
- `--eval-threshold --threshold X` : Evaluate fixed threshold performance
- `--evaluate-time` : Measure detection latency
- `--load-from-checkpoint --checkpoint-path PATH` : Use specific checkpoint
- `--dont-force-reload` : Reuse existing generated images if present

Data Paths:
- `--clean-path PATH` : Directory of clean images (overrides config)
- `--watermarked-path PATH` : Directory of watermarked images (overrides config)
- `--images-path PATH` : For threshold evaluation on a single set

---
## Image Generation
```bash
python -m src.evaluation.generate --config CONFIG [options]
```

Core Options:
- `--generate` : Perform generation (required for producing images)
- `--load-coco full|images|none` : COCO loading mode
- `--num-samples N` : Number of images to (re)generate (overrides config)
- `--batch-size N` : Batch size for generation (overrides config)
- `--force-reload` : Regenerate even if files exist

Data Paths:
- `--clean-path PATH` : Output directory for clean images (overrides config)
- `--watermarked-path PATH` : Output directory for watermarked images (overrides config)
- `--coco-annotations PATH` : COCO captions JSON
- `--coco-images PATH` : COCO images directory
- `--prompts-path PATH` : Custom prompts file

---
## Scoring (FID / CLIP)
```bash
python -m src.evaluation.scores --config CONFIG [options]
```

Core Options:
- `--generate` : Trigger generation step before scoring
- `--num-samples N` : Number of samples (overrides config)
- `--batch-size N` : Batch size for feature computation (overrides config)
- `--force-reload` : Regenerate intermediate assets
- `--load-coco full|images|none` : COCO dataset usage
- `--calculate-fid` : Compute FID
- `--calculate-clip` : Compute CLIP text-image alignment

Data Paths:
- `--clean-path PATH` : Output directory for clean images (overrides config)
- `--watermarked-path PATH` : Output directory for watermarked images (overrides config)
- `--coco-annotations PATH` : COCO captions JSON
- `--coco-images PATH` : COCO images directory
- `--prompts-path PATH` : Custom prompts file

---
## Results Layout
Artifacts are written under `results/{experiment_name}` (from `experiment_dir` in config):
- `checkpoints/` : Training states (latest + epoch snapshots)
- `models/` : Final detector + watermark parameters
- `evaluation/` : Metrics, ROC data, plots, histograms
- `gen_images/` : Clean, watermarked, original, augmented images + prompts

---
## Examples

### Dataset
```bash
# Generate only
python -m src.data.dataset --config config.yaml --mode generate --batch-size 16

# Augment existing
python -m src.data.dataset --config config.yaml --mode augment

# Subset slice
python -m src.data.dataset --config config.yaml --mode generate --start-idx 0 --end-idx 100
```

### Training
```bash
# Standard (auto resume)
python -m src.training.train --config config.yaml

# Fresh start (ignore checkpoint)
python -m src.training.train --config config.yaml --dont-load-checkpoint
```

### Evaluation
```bash
# Full eval (robustness + ROC)
python -m src.evaluation.eval --config config.yaml --num-samples 5000

# Threshold eval
python -m src.evaluation.eval --config config.yaml --eval-threshold --threshold 0.5 --images-path path/to/images

# Timing benchmark
python -m src.evaluation.eval --config config.yaml --evaluate-time --num-samples 1000
```

### Scoring & Generation
```bash
# Generate COCO + FID + CLIP
python -m src.evaluation.scores --config config.yaml --load-coco full --generate \
  --calculate-fid --calculate-clip --num-samples 10000 \
  --coco-annotations /path/to/captions.json --coco-images /path/to/coco/images

# Score existing sets
python -m src.evaluation.scores --config config.yaml --calculate-fid --calculate-clip \
  --clean-path results/EXP/gen_images/clean --watermarked-path results/EXP/gen_images/watermarked

# Custom prompts image generation
python -m src.evaluation.generate --config config.yaml --generate --prompts-path prompts.txt \
  --clean-path out/clean --watermarked-path out/watermarked --num-samples 200
```
