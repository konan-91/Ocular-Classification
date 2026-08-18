"""The dataset manifest.

One row per rendered topoplot. Everything downstream reads this file rather
than walking directories, so the label, the source recording and the split
group all travel with the image.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .channels import LABELS, POSITIVE_EVENTS

COLUMNS = ["key", "path", "label", "event", "study", "participant", "group"]


def label_for_event(event: str) -> str:
    return "blink" if event in POSITIVE_EVENTS else "non-blink"


def group_key(study: str, participant: str) -> str:
    """The unit that splits are drawn over.

    Splitting on anything finer than a recording session lets segments from the
    same person appear in both training and evaluation.
    """
    return f"{study}/{participant}"


def write(rows: list[dict], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def read(path: Path, root: Path | None = None) -> pd.DataFrame:
    """Load a manifest, resolving image paths relative to the manifest itself."""
    path = Path(path)
    frame = pd.read_csv(path)

    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"manifest {path} is missing columns: {sorted(missing)}")

    base = Path(root) if root is not None else path.parent
    frame["path"] = frame["path"].map(lambda p: str((base / p).resolve()))
    return frame


def summarise(frame: pd.DataFrame) -> str:
    counts = frame["label"].value_counts()
    lines = [
        f"{len(frame)} segments from {frame['group'].nunique()} recordings",
        "  by label: "
        + ", ".join(f"{label}={counts.get(label, 0)}" for label in LABELS),
        "  by event: "
        + ", ".join(
            f"{event}={count}" for event, count in frame["event"].value_counts().items()
        ),
    ]
    return "\n".join(lines)
