# Tiny-TRIBE

A lightweight distilled brain encoding model trained via knowledge distillation from [TRIBE v2](https://github.com/facebookresearch/tribev2). Tiny-TRIBE predicts fMRI brain responses to naturalistic video stimuli using compact multimodal encoders (~14M trainable params).

## Architecture

Tiny-TRIBE v3 uses three frozen backbone encoders:
- **Text**: `all-MiniLM-L6-v2` (22.7M)
- **Audio**: `Whisper-Tiny` encoder (39M)
- **Video**: `MobileViT-S` (5.6M)

These feed into a lightweight fusion Transformer that maps multimodal representations to cortical surface predictions (~20k vertices, fsaverage5).

## Dataset

The distillation dataset (video clips + TRIBE v2 predictions) is too large for GitHub.

> **`distillation_dataset.zip` is available on Google Drive** (link to be added).

The `features/` directory (pre-extracted backbone features) is also hosted externally.

> **`features.zip` is available on Google Drive** (link to be added).

## Quick Start

```bash
pip install -r requirements.txt
```

Run inference:
```bash
python run_inference.py
```

Build distillation dataset:
```bash
python build_distillation_dataset.py
```

## Training

See `LIGHTNING_TRAINING_PLAN.md` for the full training pipeline.

```bash
python tiny_tribe/train_lightning.py
```

## Project Structure

```
tiny_tribe/          # Model code (backbones, model, training)
distillation.py      # KD training utilities
build_distillation_dataset.py  # Dataset generation script
modal_inference.py   # Modal.com inference runner
tribe_colab_inference.ipynb    # Colab notebook for inference
```

## Strategy Docs

- `STRATEGY_C_DEEP_DIVE.md` — Full architecture and training strategy
- `TRIBE_V3_STRATEGY.md` — v3 design decisions
- `DISTILLATION_PATTERNS.md` — KD training patterns
- `TINY_TRIBE_ARCHITECTURE.md` / `TINY_TRIBE_V3_ARCHITECTURE.md` — Architecture diagrams
