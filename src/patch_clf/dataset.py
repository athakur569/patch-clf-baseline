"""Patch dataset from a tile manifest CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def default_train_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def default_val_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class PatchDataset(Dataset):
    """Tile-manifest-driven patch dataset.

    Args:
        manifest_path:  Path to CSV with columns: patch_id, path, label (and optionally slide_id).
        label_col:      Column name for the label (default 'label').
        path_col:       Column name for the patch image path (default 'path').
        transform:      torchvision transform applied to each patch.
        root_dir:       Optional root to prepend to relative paths in the manifest.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        label_col: str = "label",
        path_col: str = "path",
        transform=None,
        root_dir: str | Path | None = None,
    ) -> None:
        self.df = pd.read_csv(manifest_path)
        self.label_col = label_col
        self.path_col = path_col
        self.transform = transform or default_val_transform()
        self.root_dir = Path(root_dir) if root_dir else None

        if label_col not in self.df.columns:
            raise ValueError(f"Column '{label_col}' not in manifest. Available: {list(self.df.columns)}")
        if path_col not in self.df.columns:
            raise ValueError(f"Column '{path_col}' not in manifest. Available: {list(self.df.columns)}")

        self.labels = self.df[label_col].values
        self.paths = self.df[path_col].values
        self.patch_ids = self.df.get("patch_id", pd.RangeIndex(len(self.df))).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(self.paths[idx])
        if self.root_dir and not path.is_absolute():
            path = self.root_dir / path
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label

    @property
    def class_counts(self) -> dict:
        return dict(pd.Series(self.labels).value_counts())

    @property
    def pos_weight(self) -> float:
        counts = self.class_counts
        if 0 not in counts or 1 not in counts:
            return 1.0
        return counts[0] / counts[1]
