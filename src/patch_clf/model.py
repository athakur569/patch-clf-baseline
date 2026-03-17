"""ResNet-50 classifier Lightning module."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.classification import BinaryAUROC, BinaryF1Score


class PatchClassifier(pl.LightningModule):
    """ResNet-50 binary patch classifier.

    Args:
        num_classes:   Number of output classes (1 for binary sigmoid, >1 for softmax).
        pretrained:    Load ImageNet weights (default True).
        lr:            Learning rate.
        weight_decay:  L2 regularization.
        freeze_backbone: Freeze ResNet layers, fine-tune head only.
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

        self.loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.loss_fn(logits, y.float() if self.hparams.num_classes == 1 else y.long())
        probs = torch.sigmoid(logits) if self.hparams.num_classes == 1 else logits.softmax(1)[:, 1]
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        return loss, probs, y

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss, probs, y = self._shared_step(batch, "train")
        self.train_auc.update(probs, y.int())
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/auc", self.train_auc.compute(), prog_bar=True)
        self.train_auc.reset()

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        _, probs, y = self._shared_step(batch, "val")
        self.val_auc.update(probs, y.int())
        self.val_f1.update(probs, y.int())

    def on_validation_epoch_end(self) -> None:
        self.log("val/auc", self.val_auc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute())
        self.val_auc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate layer embeddings (before classifier head)."""
        feats = []
        def hook(module, input, output):
            feats.append(output)
        handle = self.model.avgpool.register_forward_hook(hook)
        with torch.no_grad():
            self(x)
        handle.remove()
        return feats[0].squeeze(-1).squeeze(-1)
