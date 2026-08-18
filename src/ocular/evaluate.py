"""Running the trained model over a dataset and scoring it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from . import metrics
from .data import make_loader


@torch.no_grad()
def predict(
    model: nn.Module,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, blink_probability) for every row of frame.

    Row order is preserved, which is what lets the benchmark pair these
    predictions against the ICA baseline segment by segment.
    """
    loader = make_loader(
        frame, train=False, batch_size=batch_size, num_workers=num_workers
    )
    model.eval().to(device)

    y_true, y_score = [], []
    batches = tqdm(loader, desc="predicting", unit="batch", disable=not progress)
    for images, targets in batches:
        logits = model(images.to(device))
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        y_score.append(probabilities.cpu().numpy())
        y_true.append(targets.numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    return y_true, (y_score >= 0.5).astype(np.int64), y_score


def evaluate(
    model: nn.Module,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    y_true, y_pred, y_score = predict(
        model, frame, device, batch_size=batch_size, num_workers=num_workers
    )
    return metrics.compute(y_true, y_pred, y_score)
