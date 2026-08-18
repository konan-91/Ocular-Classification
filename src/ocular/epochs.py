"""Loading the raw EEGLAB recordings and re-epoching them around eye events.

The source files are pre-epoched, but each epoch holds 6 to 20 events of a
single type. Training needs one event per epoch, so every epoch is cut into
short windows centred on the onsets marked in the eye tracker trigger channels.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mne
import numpy as np

from .channels import EVENT_CODES, TRIGGERS_BY_EVENT

log = logging.getLogger(__name__)

# Window taken around each event onset. Every event type uses the same window
# so that segment duration cannot act as a cue for the classifier.
TMIN = -0.2
TMAX = 0.5


@dataclass(frozen=True)
class Segment:
    """One re-epoched event, tagged with where it came from."""

    study: str
    participant: str
    event: str
    index: int
    data: np.ndarray  # (n_channels, n_times)
    info: mne.Info

    @property
    def key(self) -> str:
        return f"{self.study}_{self.participant}_{self.event}{self.index:05d}"


def participant_id(path: Path) -> str:
    """Derive a stable participant id from a .set filename."""
    stem = path.stem
    match = re.search(r"(?:sub|subj|participant|p)[-_]?(\d+)", stem, re.IGNORECASE)
    if match:
        return f"p{int(match.group(1)):03d}"
    digits = re.findall(r"\d+", stem)
    if digits:
        return f"p{int(digits[0]):03d}"
    return stem


def find_recordings(root: Path) -> list[tuple[str, str, Path]]:
    """Find every .set file under root as (study, participant, path).

    The study name is the directory the file sits in, relative to root. Files
    directly under root are grouped into a single study called "root".
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    recordings = []
    for path in sorted(root.rglob("*.set")):
        relative = path.parent.relative_to(root)
        study = relative.parts[0] if relative.parts else "root"
        recordings.append((study, participant_id(path), path))

    if not recordings:
        raise FileNotFoundError(f"no .set files found under {root}")

    # A participant recorded in two studies would otherwise share an id and
    # leak across the split, so the study is kept as part of the group key.
    return recordings


def load_recording(path: Path) -> mne.Epochs:
    """Read one EEGLAB file. The dataset we use comes pre-epoched."""
    return mne.read_epochs_eeglab(str(path), verbose=False)


def normalise_events(epochs: mne.Epochs) -> mne.Epochs:
    """Rewrite EEGLAB string event ids into the integer form MNE expects."""
    event_dict: dict[str, int] = {}
    event_list = []

    # Extract unique event names from EEGLAB
    for i, event_desc in enumerate(epochs.events[:, -1]):
        # Convert to string and assign ID
        event_label = str(epochs.event_id.get(event_desc, event_desc))
        if event_label not in event_dict:
            event_dict[event_label] = len(event_dict) + 1
        event_list.append([epochs.events[i, 0], 0, event_dict[event_label]])

    # Ensure type Int (else it crashes), and allocate new events
    epochs.events = np.array(event_list, dtype=int)
    epochs.event_id = event_dict
    return epochs


def _select_event(epochs: mne.Epochs, event: str) -> mne.Epochs | None:
    """Pull out the sub-epochs for one event type, or None if absent."""
    code = EVENT_CODES[event]
    if code not in epochs.event_id:
        return None
    selected = epochs[code]
    return selected if len(selected) else None


def split_on_triggers(
    epochs: mne.Epochs, trigger_ch: tuple[str, ...]
) -> list[np.ndarray]:
    """Cut an epoch into one window per trigger onset."""
    present = [ch for ch in trigger_ch if ch in epochs.ch_names]
    if not present:
        return []

    trigger_indices = [epochs.ch_names.index(ch) for ch in present]
    sfreq = epochs.info["sfreq"]  # Sampling frequency (1ms)
    data = epochs.get_data(copy=False)

    # Time window for new epochs
    tmin_samples = int(TMIN * sfreq)  # 200ms before
    tmax_samples = int(TMAX * sfreq)  # 500ms after

    windows = []
    # Keep track of used onsets so the same event is not taken twice
    seen: set[tuple[int, int]] = set()

    for i in range(len(epochs)):
        for trigger_idx in trigger_indices:
            # Get trigger channel data for epoch
            trigger_data = data[i, trigger_idx, :]

            # Find timestamps where events start. The trigger is squared off
            # first: differencing the channel directly picks up every upward
            # sample of whatever noise sits on the baseline.
            peak = float(np.max(trigger_data))
            if peak <= 0:
                continue
            active = (trigger_data > peak / 2).astype(np.int8)
            event_starts = np.flatnonzero(np.diff(active) > 0)
            for event_start in event_starts:
                if (i, int(event_start)) in seen:
                    continue
                seen.add((i, int(event_start)))

                # Calculate window indices
                start_idx = event_start + tmin_samples
                end_idx = event_start + tmax_samples
                # Check if window is within epoch bounds
                if start_idx >= 0 and end_idx < data.shape[2]:
                    # Extract data window around the event
                    windows.append(data[i, :, start_idx:end_idx].copy())

    return windows


def split_rest(epochs: mne.Epochs) -> list[np.ndarray]:
    """Cut resting data into fixed windows.

    Rest is a state rather than an event, so there is no trigger to align to.
    A sliding window of the same length as the event windows keeps segment
    duration constant across classes.
    """
    sfreq = epochs.info["sfreq"]
    window_length = int(TMAX * sfreq) - int(TMIN * sfreq)
    data = epochs.get_data(copy=False)

    windows = []
    for epoch_data in data:
        epoch_length = epoch_data.shape[1]
        current_sample = 0
        # Sliding window creating new segments until end
        while current_sample + window_length <= epoch_length:
            windows.append(
                epoch_data[:, current_sample : current_sample + window_length].copy()
            )
            current_sample += window_length

    return windows


def segments_for_recording(
    study: str, participant: str, path: Path, events: tuple[str, ...] | None = None
) -> Iterator[Segment]:
    """Load one recording and yield its re-epoched segments."""
    events = events or tuple(EVENT_CODES)

    try:
        epochs = normalise_events(load_recording(path))
    except Exception as exc:
        log.warning("skipping %s: %s", path.name, exc)
        return

    for event in events:
        selected = _select_event(epochs, event)
        if selected is None:
            continue

        if event == "rest":
            windows = split_rest(selected)
        else:
            windows = split_on_triggers(selected, TRIGGERS_BY_EVENT[event])

        if not windows:
            log.warning("no %s segments in %s", event, path.name)
            continue

        for index, window in enumerate(windows):
            yield Segment(
                study=study,
                participant=participant,
                event=event,
                index=index,
                data=window,
                info=selected.info.copy(),
            )
