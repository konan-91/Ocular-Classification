"""Scoring and reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .channels import LABELS


def compute(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> dict:
    """Score a set of predictions.

    Balanced accuracy is reported alongside accuracy because the classes are
    uneven.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(LABELS)), zero_division=0
    )

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(LABELS))
        ).tolist(),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "n": int(len(y_true)),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))

    return result


def format_report(name: str, scores: dict) -> str:
    lines = [
        f"{name}  n={scores['n']}",
        f"  accuracy           {scores['accuracy']:.4f}",
        f"  balanced accuracy  {scores['balanced_accuracy']:.4f}",
    ]
    if "roc_auc" in scores:
        lines.append(f"  roc auc            {scores['roc_auc']:.4f}")

    lines.append(f"  {'class':<12}{'prec':>8}{'recall':>8}{'f1':>8}{'n':>8}")
    for label, stats in scores["per_class"].items():
        lines.append(
            f"  {label:<12}{stats['precision']:>8.3f}{stats['recall']:>8.3f}"
            f"{stats['f1']:>8.3f}{stats['support']:>8}"
        )

    matrix = scores["confusion_matrix"]
    lines.append("  confusion matrix (rows true, cols predicted)")
    for label, row in zip(LABELS, matrix):
        lines.append(f"    {label:<12}" + "".join(f"{value:>8}" for value in row))

    return "\n".join(lines)


def plot_confusion_matrix(scores: dict, path: Path, title: str = "Confusion matrix") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(scores["confusion_matrix"], dtype=float)
    normalised = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}\n{normalised[i, j]:.1%}",
                ha="center",
                va="center",
                color="white" if normalised[i, j] > 0.5 else "black",
            )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
