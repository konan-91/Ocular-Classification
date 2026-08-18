"""Training the topoplot classifier."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from tqdm import tqdm

from . import manifest, metrics, model as model_module, splits, utils
from .data import make_loader

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    architecture: str = model_module.DEFAULT_ARCHITECTURE
    head_epochs: int = 3
    finetune_epochs: int = 8
    batch_size: int = 32
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    patience: int = 3
    num_workers: int = 4
    seed: int = 91
    pretrained: bool = True
    device: str = "auto"


def _run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer | None,
    description: str,
) -> tuple[float, float]:
    """One pass over a loader. Trains when an optimiser is given."""
    training = optimiser is not None
    model.train(training)

    total_loss, y_true, y_pred = 0.0, [], []
    batches = tqdm(loader, desc=description, unit="batch", leave=False)

    with torch.set_grad_enabled(training):
        for images, targets in batches:
            images, targets = images.to(device), targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * len(targets)
            y_true.append(targets.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
            batches.set_postfix(loss=f"{loss.item():.3f}")

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return total_loss / len(y_true), float(balanced_accuracy_score(y_true, y_pred))


def train(
    manifest_path: Path,
    out_dir: Path,
    config: TrainConfig | None = None,
) -> dict:
    config = config or TrainConfig()
    out_dir = Path(out_dir)
    utils.set_seed(config.seed)

    frame = manifest.read(manifest_path)
    split = splits.make(frame, seed=config.seed)
    frame = split.assign(frame)

    log.info("dataset\n%s", manifest.summarise(frame))
    log.info("splits\n%s", splits.summarise(frame))

    train_frame = frame[frame["split"] == "train"]
    val_frame = frame[frame["split"] == "val"]

    device = model_module.pick_device(config.device)
    log.info("training on %s", device)

    train_loader = make_loader(
        train_frame,
        train=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    val_loader = make_loader(
        val_frame,
        train=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = model_module.build(config.architecture, pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    history: list[dict] = []
    best_score, best_state, epochs_without_gain = -1.0, None, 0
    started = time.time()

    # Two phases. The head is trained against a frozen backbone first, so the
    # randomly initialised layer does not push large gradients into the
    # pretrained weights, then everything is fine tuned at a lower rate.
    phases = (
        ("head", config.head_epochs, config.head_lr, False),
        ("finetune", config.finetune_epochs, config.finetune_lr, True),
    )

    for phase, epochs, lr, unfrozen in phases:
        if epochs <= 0:
            continue

        model_module.set_backbone_trainable(model, unfrozen)
        optimiser = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        for epoch in range(1, epochs + 1):
            train_loss, train_score = _run_epoch(
                model, train_loader, device, criterion, optimiser,
                f"{phase} {epoch}/{epochs} train",
            )
            val_loss, val_score = _run_epoch(
                model, val_loader, device, criterion, None,
                f"{phase} {epoch}/{epochs} val",
            )
            scheduler.step()

            history.append(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_balanced_accuracy": train_score,
                    "val_loss": val_loss,
                    "val_balanced_accuracy": val_score,
                }
            )
            log.info(
                "%s epoch %d/%d  train loss %.4f bal acc %.4f  "
                "val loss %.4f bal acc %.4f",
                phase, epoch, epochs, train_loss, train_score, val_loss, val_score,
            )

            if val_score > best_score:
                best_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= config.patience:
                    log.info("no validation gain in %d epochs, stopping", config.patience)
                    break
        else:
            continue
        break

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "architecture": config.architecture,
        "config": asdict(config),
        "split": {name: list(split.of(name)) for name in splits.SPLITS},
        "best_val_balanced_accuracy": best_score,
        "training_seconds": round(time.time() - started, 1),
    }

    model_path = out_dir / "model.pt"
    model_module.save(model, model_path, metadata)

    from .evaluate import evaluate as run_evaluate

    val_scores = run_evaluate(
        model, val_frame, device,
        batch_size=config.batch_size, num_workers=config.num_workers,
    )

    results = {"metadata": metadata, "history": history, "validation": val_scores}
    utils.write_json(results, out_dir / "training.json")
    metrics.plot_confusion_matrix(
        val_scores, out_dir / "confusion_matrix_val.png", "Validation, held out participants"
    )

    log.info("saved model to %s", model_path)
    log.info("\n%s", metrics.format_report("validation", val_scores))
    return results
