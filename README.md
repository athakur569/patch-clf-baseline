# patch-clf-baseline

ResNet-50 patch-level classifier for computational pathology using PyTorch Lightning. Designed as a reproducible baseline for CAMELYON17 (tumor detection) and TCGA binary classification tasks.

## Features
- ResNet-50 with ImageNet pretrain, configurable final layer
- PyTorch Lightning training loop with WandB logging
- Patch dataset class with tile manifest (CSV) input
- AUC + slide-level aggregation evaluation
- Seeded reproducibility (seeds 42, 99, 123)

## Installation
```bash
pip install patch-clf-baseline
```

## Quick start
```bash
# Train
python -m patch_clf.train \
  --manifest data/manifest.csv \
  --label_col label \
  --output_dir results/ \
  --epochs 20 --seed 42

# Evaluate
python -m patch_clf.evaluate \
  --checkpoint results/best.ckpt \
  --manifest data/test_manifest.csv
```

## License
MIT
