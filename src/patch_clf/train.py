"""Training script for PatchClassifier."""

from __future__ import annotations

import argparse
import random

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import PatchDataset, default_train_transform, default_val_transform
from .model import PatchClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    train_ds = PatchDataset(
        args.manifest, label_col=args.label_col, transform=default_train_transform()
    )
    val_manifest = args.val_manifest or args.manifest
    val_ds = PatchDataset(
        val_manifest, label_col=args.label_col, transform=default_val_transform()
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = PatchClassifier(lr=args.lr, pretrained=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir, filename="best-{epoch:02d}-{val/auc:.3f}",
            monitor="val/auc", mode="max", save_top_k=1,
        ),
        pl.callbacks.EarlyStopping(monitor="val/auc", patience=5, mode="max"),
        pl.callbacks.LearningRateMonitor(),
    ]

    logger = None
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project="patch-clf-baseline", name=f"seed{args.seed}")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val_manifest", default=None)
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
