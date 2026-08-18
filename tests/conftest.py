"""Synthetic EEG fixtures.

The real dataset is several gigabytes and lives on OSF, so the tests build
small recordings with the same structure: scalp channels on a standard
montage, eye tracker trigger channels, an EOG channel, and events tagged with
the same codes the source files use.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from ocular.channels import EVENT_CODES

SFREQ = 100.0
SCALP = [
    "Fp1", "Fp2", "F3", "Fz", "F4", "FC1", "FC2", "C3", "Cz", "C4",
    "CP1", "CP2", "P3", "Pz", "P4", "O1", "O2", "T7", "T8",
]
TRIGGERS = ["eye-blink", "eye-u", "eye-d", "eye-l", "eye-r"]
EXTRA = ["VEOG", "HEOG"]

EPOCH_SAMPLES = 1000
EVENTS_PER_EPOCH = 6
PULSE_WIDTH = 10

# Frontal channels carry most of the blink, which gives ICA something real to
# find and separates the classes in the topoplots.
FRONTAL_WEIGHTS = {
    "Fp1": 1.0, "Fp2": 1.0, "F3": 0.8, "Fz": 0.9, "F4": 0.8,
    "FC1": 0.5, "FC2": 0.5, "C3": 0.2, "Cz": 0.2, "C4": 0.2,
}


def _event_onsets(n_events: int = EVENTS_PER_EPOCH) -> np.ndarray:
    """Evenly spaced onsets, kept clear of the epoch edges."""
    return np.linspace(150, EPOCH_SAMPLES - 150, n_events).astype(int)


def _build_epoch(event: str, rng: np.random.Generator) -> np.ndarray:
    """One parent epoch of shape (n_channels, EPOCH_SAMPLES)."""
    ch_names = SCALP + TRIGGERS + EXTRA
    data = rng.normal(0, 1e-6, size=(len(ch_names), EPOCH_SAMPLES))

    # Trigger channels are clean digital lines in the source data
    for trigger in TRIGGERS:
        data[ch_names.index(trigger)] = 0.0

    if event == "rest":
        return data

    trigger_map = {
        "blink": ["eye-blink"],
        "v_saccade": ["eye-u", "eye-d"],
        "h_saccade": ["eye-l", "eye-r"],
    }
    trigger_names = trigger_map[event]
    onsets = _event_onsets()

    for i, onset in enumerate(onsets):
        trigger = trigger_names[i % len(trigger_names)]
        trigger_idx = ch_names.index(trigger)
        # Square pulse marking the onset, which is what the re-epoching reads
        data[trigger_idx, onset : onset + PULSE_WIDTH] = 1.0

        # A deflection in the scalp channels, shaped differently per event type
        span = slice(onset - 20, onset + 50)
        bump = np.hanning(70) * 1e-4

        for channel, weight in FRONTAL_WEIGHTS.items():
            idx = ch_names.index(channel)
            if event == "blink":
                data[idx, span] += bump * weight
            elif event == "v_saccade":
                data[idx, span] += bump * weight * 0.5
            else:
                # Horizontal saccades are lateral, so the sign flips by side
                side = -1.0 if channel.endswith("3") or channel.endswith("1") else 1.0
                data[idx, span] += bump * weight * 0.4 * side

        veog = ch_names.index("VEOG")
        data[veog, span] += bump * (1.5 if event == "blink" else 0.3)

    return data


def make_recording(seed: int = 0, events=("rest", "h_saccade", "v_saccade", "blink")):
    """Build a synthetic recording as an MNE Epochs object."""
    rng = np.random.default_rng(seed)
    ch_names = SCALP + TRIGGERS + EXTRA
    ch_types = ["eeg"] * len(SCALP) + ["misc"] * len(TRIGGERS) + ["eeg"] * len(EXTRA)

    info = mne.create_info(ch_names, SFREQ, ch_types, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore", verbose=False)

    data, event_rows, event_id = [], [], {}
    for event in events:
        code = EVENT_CODES[event]
        event_id[code] = int(code)
        for _ in range(2):  # two parent epochs per event type
            data.append(_build_epoch(event, rng))
            event_rows.append([len(data) * EPOCH_SAMPLES, 0, int(code)])

    return mne.EpochsArray(
        np.stack(data),
        info,
        events=np.array(event_rows),
        event_id=event_id,
        tmin=0,
        verbose=False,
    )


@pytest.fixture
def recording():
    return make_recording(seed=0)


@pytest.fixture
def raw_root(tmp_path, monkeypatch):
    """A fake dataset tree of .set files, with loading monkeypatched.

    Writing real EEGLAB files would pull in another dependency, so the files
    are placeholders and ocular.epochs.load_recording is redirected to the
    synthetic builder. Everything downstream of loading is exercised for real.
    """
    root = tmp_path / "raw"
    paths = {}
    for study in ("study1", "study2"):
        for participant in range(1, 4):
            path = root / study / f"sub-{participant:02d}.set"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")
            paths[path.resolve()] = hash((study, participant)) % 1000

    def fake_load(path):
        return make_recording(seed=paths[path.resolve()])

    monkeypatch.setattr("ocular.epochs.load_recording", fake_load)
    monkeypatch.setattr("ocular.ica_baseline.load_recording", fake_load)
    return root
