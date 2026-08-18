"""Rendering re-epoched segments as topographic scalp maps.

Each segment is averaged over its time window and drawn as a single topoplot.
Collapsing time this way turns a multichannel signal into an image, which lets
a pretrained vision model do the classification.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np

from .channels import scalp_channels
from .epochs import Segment

log = logging.getLogger(__name__)

IMAGE_SIZE = 256
DPI = 100


class TopoplotRenderer:
    """Renders segments to PNG, reusing one figure across calls.

    Matplotlib figure creation dominates the runtime of this stage, so the
    figure and axes are built once and cleared between plots.
    """

    def __init__(self, size: int = IMAGE_SIZE, dpi: int = DPI):
        inches = size / dpi
        self.fig, self.ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._picks_cache: dict[tuple[str, ...], mne.Info] = {}

    def _scalp_info(self, info: mne.Info) -> mne.Info:
        key = tuple(info["ch_names"])
        cached = self._picks_cache.get(key)
        if cached is None:
            keep = scalp_channels(info["ch_names"])
            if not keep:
                raise ValueError("recording has no usable scalp channels")
            cached = mne.pick_info(info, [info["ch_names"].index(ch) for ch in keep])
            if cached.get_montage() is None:
                raise ValueError("recording has no channel positions for topoplots")
            self._picks_cache[key] = cached
        return cached

    def render(self, segment: Segment, path: Path) -> None:
        info = self._scalp_info(segment.info)
        picks = [segment.info["ch_names"].index(ch) for ch in info["ch_names"]]

        # Get data and compute mean over the time window
        mean_activity = segment.data[picks].mean(axis=1)

        # Create "evoked" object with averaged data
        evoked = mne.EvokedArray(mean_activity[:, np.newaxis], info, tmin=0, verbose=False)

        # Generate topomap into the reused axes
        self.ax.clear()
        evoked.plot_topomap(
            times=0,
            ch_type="eeg",
            colorbar=False,
            outlines="head",
            sensors=False,
            contours=6,
            axes=self.ax,
            show=False,
            time_format="",
        )
        self.ax.set_title("")
        self.ax.set_axis_off()

        # Save figure as .png
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=DPI, pad_inches=0, transparent=False)

    def close(self) -> None:
        plt.close(self.fig)

    def __enter__(self) -> "TopoplotRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
