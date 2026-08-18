"""Turning the raw dataset into topoplots and a manifest."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import manifest
from .epochs import find_recordings, segments_for_recording
from .topoplots import TopoplotRenderer

log = logging.getLogger(__name__)


def prepare(
    raw_root: Path,
    out_dir: Path,
    manifest_path: Path | None = None,
    limit_per_event: int | None = None,
) -> pd.DataFrame:
    """Render every segment in the raw dataset and write the manifest.

    Images land in out_dir/<label>/<key>.png and the manifest stores paths
    relative to its own location, so the whole data directory can be moved.
    """
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    manifest_path = Path(manifest_path or out_dir.parent / "manifest.csv")

    recordings = find_recordings(raw_root)
    log.info("found %d recordings under %s", len(recordings), raw_root)

    rows: list[dict] = []
    with TopoplotRenderer() as renderer:
        for study, participant, path in tqdm(recordings, desc="recordings", unit="rec"):
            # Counted per event type rather than per recording, so a capped
            # run still contains every class.
            written: Counter[str] = Counter()
            for segment in segments_for_recording(study, participant, path):
                if limit_per_event and written[segment.event] >= limit_per_event:
                    continue

                label = manifest.label_for_event(segment.event)
                image_path = out_dir / label / f"{segment.key}.png"

                try:
                    renderer.render(segment, image_path)
                except ValueError as exc:
                    log.warning("skipping %s: %s", segment.key, exc)
                    break

                rows.append(
                    {
                        "key": segment.key,
                        "path": str(image_path.relative_to(manifest_path.parent)),
                        "label": label,
                        "event": segment.event,
                        "study": study,
                        "participant": participant,
                        "group": manifest.group_key(study, participant),
                    }
                )
                written[segment.event] += 1

    if not rows:
        raise RuntimeError("no segments were rendered, check the dataset layout")

    frame = manifest.write(rows, manifest_path)
    log.info("wrote %s", manifest_path)
    return frame
