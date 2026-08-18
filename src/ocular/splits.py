"""Participant level train, validation and test splits.

Segments from one recording share a participant, a montage and a session, and
one recording contributes hundreds of them. Splitting over segments would
therefore place the same participant on both sides of the boundary, so splits
are drawn over recordings instead.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def of(self, name: str) -> tuple[str, ...]:
        return getattr(self, name)

    def assign(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add a "split" column to a manifest, dropping unassigned rows."""
        lookup = {}
        for name in SPLITS:
            for group in self.of(name):
                lookup[group] = name

        out = frame.copy()
        out["split"] = out["group"].map(lookup)
        return out.dropna(subset=["split"]).reset_index(drop=True)


def _stable_order(groups, seed: int) -> list[str]:
    """Order groups by a hash of their name so the split is reproducible.

    Hashing rather than shuffling means adding a new recording does not
    reshuffle the ones already assigned.
    """

    def key(group: str) -> str:
        return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()

    return sorted(groups, key=key)


def make(
    frame: pd.DataFrame,
    seed: int = 91,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Split:
    """Split the recordings in a manifest by group."""
    groups = _stable_order(frame["group"].unique(), seed)
    n = len(groups)
    if n < 3:
        raise ValueError(f"need at least 3 recordings to split, found {n}")

    n_test = max(1, round(n * test_fraction))
    n_val = max(1, round(n * val_fraction))
    if n_test + n_val >= n:
        raise ValueError(
            f"{n} recordings cannot fill a {val_fraction:.0%} validation and "
            f"{test_fraction:.0%} test split"
        )

    return Split(
        test=tuple(groups[:n_test]),
        val=tuple(groups[n_test : n_test + n_val]),
        train=tuple(groups[n_test + n_val :]),
    )


def summarise(frame: pd.DataFrame) -> str:
    lines = []
    for name in SPLITS:
        subset = frame[frame["split"] == name]
        if not len(subset):
            continue
        blinks = int((subset["label"] == "blink").sum())
        lines.append(
            f"  {name:<5} {len(subset):>6} segments  "
            f"{subset['group'].nunique():>3} recordings  "
            f"{blinks / len(subset):.1%} blink"
        )
    return "\n".join(lines)
