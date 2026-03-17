"""patch-clf-baseline: ResNet-50 patch classifier for computational pathology."""

from .model import PatchClassifier
from .dataset import PatchDataset

__version__ = "0.1.0"
__all__ = ["PatchClassifier", "PatchDataset"]
