"""Torch datasets and loaders built from a manifest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .channels import LABELS

INPUT_SIZE = 224

# ImageNet statistics, since the backbone is pretrained on it.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def build_transforms(train: bool):
    """Transforms for one split.

    Augmentation is deliberately mild. A horizontal flip would mirror the
    scalp, which turns a leftward saccade into a rightward one and makes the
    negative class incoherent. Colour jitter is left out for the same reason:
    the colormap encodes signal polarity, not appearance.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    INPUT_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)
                ),
                transforms.RandomRotation(7, fill=255),
                transforms.ToTensor(),
                transforms.Normalize(NORM_MEAN, NORM_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


class TopoplotDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.transform = build_transforms(train)
        self.targets = np.array(
            [LABEL_TO_INDEX[label] for label in self.frame["label"]], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), int(self.targets[index])


def _balanced_sampler(targets: np.ndarray) -> WeightedRandomSampler:
    """Sample the minority class more often, so no data has to be discarded."""
    counts = np.bincount(targets, minlength=len(LABELS)).astype(float)
    counts[counts == 0] = 1.0
    weights = (1.0 / counts)[targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(targets),
        replacement=True,
    )


def make_loader(
    frame: pd.DataFrame,
    train: bool,
    batch_size: int = 32,
    num_workers: int = 4,
    balanced: bool = True,
) -> DataLoader:
    dataset = TopoplotDataset(frame, train=train)
    sampler = _balanced_sampler(dataset.targets) if train and balanced else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(train and sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
