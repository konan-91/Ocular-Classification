"""The ICA baseline that the model is measured against.

Independent component analysis is the standard automated route to ocular
artifact handling in EEG. ICA is fit per recording, the component matching the
EOG channel is identified, and each segment is scored by how strongly that
component deflects inside it. Anything above a threshold is called a blink.

Two details keep the comparison honest:

Scores are standardised within a recording before thresholding. ICA component
amplitudes carry an arbitrary sign and scale, so a single raw cutoff across
participants would handicap the baseline for reasons that have nothing to do
with how well it separates blinks.

The threshold is fitted on the validation recordings and applied unchanged to
the test recordings, which is the same information the model gets.

The baseline reads the EOG electrodes, which the model never sees. This
asymmetry favours the baseline and is retained, since the model is intended to
work on recordings without an EOG montage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from tqdm import tqdm

from .channels import pick_eog_channel, scalp_channels
from .epochs import Segment, find_recordings, normalise_events, load_recording
from .epochs import segments_for_recording
from .manifest import group_key

log = logging.getLogger(__name__)

N_COMPONENTS = 20
FIT_HIGHPASS = 1.0
RANDOM_STATE = 91


@dataclass
class RecordingScores:
    group: str
    keys: list[str]
    scores: np.ndarray  # standardised, one per segment
    n_components_found: int


def _robust_standardise(values: np.ndarray) -> np.ndarray:
    """Median and MAD standardisation, within one recording."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad <= 0:
        spread = values.std()
        if spread <= 0:
            return np.zeros_like(values)
        return (values - median) / spread
    # 1.4826 puts MAD on the same scale as a standard deviation for normal data
    return (values - median) / (1.4826 * mad)


def _segments_to_epochs(segments: list[Segment], picks: list[str]) -> mne.EpochsArray:
    """Stack a recording's segments into one EpochsArray of scalp channels."""
    info = segments[0].info
    indices = [info["ch_names"].index(ch) for ch in picks]
    data = np.stack([segment.data[indices] for segment in segments])

    picked_info = mne.pick_info(info, indices)
    events = np.column_stack(
        [np.arange(len(segments)) * data.shape[2], np.zeros(len(segments), int),
         np.ones(len(segments), int)]
    )
    return mne.EpochsArray(data, picked_info, events=events, tmin=0, verbose=False)


def _fit_ica(parent: mne.Epochs, picks: list[str]) -> tuple[ICA, list[int]] | None:
    """Fit ICA on a recording and identify its ocular components."""
    eog_channel = pick_eog_channel(parent.ch_names)
    if eog_channel is None:
        log.warning("no EOG channel available, skipping recording")
        return None

    fit_data = parent.copy().pick(picks + [eog_channel])
    # ICA is unstable on data with slow drifts, so a 1 Hz highpass is standard
    # for the fit. It is applied to the fitting copy only.
    fit_data.filter(l_freq=FIT_HIGHPASS, h_freq=None, verbose=False)
    fit_data.set_channel_types({eog_channel: "eog"}, verbose=False)

    n_components = min(N_COMPONENTS, len(picks) - 1)
    ica = ICA(
        n_components=n_components,
        method="fastica",
        max_iter="auto",
        random_state=RANDOM_STATE,
        verbose=False,
    )

    try:
        ica.fit(fit_data, picks="eeg", verbose=False)
        indices, scores = ica.find_bads_eog(fit_data, ch_name=eog_channel, verbose=False)
    except Exception as exc:
        log.warning("ICA failed: %s", exc)
        return None

    if not indices:
        # Nothing passed the default correlation threshold. Fall back to the
        # single component most correlated with EOG so the recording still
        # produces a decision rather than being dropped.
        scores = np.atleast_1d(np.asarray(scores)).ravel()
        if not scores.size:
            return None
        indices = [int(np.argmax(np.abs(scores)))]

    return ica, list(indices)


def score_recording(
    study: str, participant: str, path: Path
) -> RecordingScores | None:
    """Score every segment of one recording with the ICA baseline."""
    try:
        parent = normalise_events(load_recording(path))
    except Exception as exc:
        log.warning("could not load %s: %s", path.name, exc)
        return None

    picks = scalp_channels(parent.ch_names)
    if len(picks) < 8:
        log.warning("%s has too few scalp channels", path.name)
        return None

    fitted = _fit_ica(parent, picks)
    if fitted is None:
        return None
    ica, components = fitted

    segments = list(segments_for_recording(study, participant, path))
    if not segments:
        return None

    epochs = _segments_to_epochs(segments, picks)
    sources = ica.get_sources(epochs).get_data(copy=False)

    # Peak to peak swing of the ocular components inside each segment. A blink
    # is a large, brief deflection, so range separates it better than variance.
    selected = sources[:, components, :]
    raw_scores = np.abs(selected.max(axis=2) - selected.min(axis=2)).max(axis=1)

    return RecordingScores(
        group=group_key(study, participant),
        keys=[segment.key for segment in segments],
        scores=_robust_standardise(raw_scores),
        n_components_found=len(components),
    )


def score_dataset(raw_root: Path, groups: set[str] | None = None) -> pd.DataFrame:
    """Score every recording, or just the listed groups.

    Returns a frame of key, group and ica_score, ready to be joined onto the
    manifest by key.
    """
    recordings = find_recordings(Path(raw_root))
    if groups is not None:
        recordings = [
            r for r in recordings if group_key(r[0], r[1]) in groups
        ]
        if not recordings:
            raise ValueError("none of the requested recordings were found")

    rows = []
    skipped = []
    for study, participant, path in tqdm(recordings, desc="ICA", unit="rec"):
        result = score_recording(study, participant, path)
        if result is None:
            skipped.append(group_key(study, participant))
            continue
        for key, score in zip(result.keys, result.scores):
            rows.append({"key": key, "group": result.group, "ica_score": float(score)})

    if skipped:
        log.warning("ICA produced no scores for %d recordings: %s", len(skipped), skipped)
    if not rows:
        raise RuntimeError("ICA baseline produced no scores at all")

    return pd.DataFrame(rows)


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Pick the threshold with the best balanced accuracy.

    Returns (threshold, balanced accuracy at that threshold).
    """
    from sklearn.metrics import balanced_accuracy_score

    if len(np.unique(labels)) < 2:
        raise ValueError("threshold calibration needs both classes present")

    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    best_threshold, best_score = float(candidates[0]), -1.0

    for threshold in candidates:
        score = balanced_accuracy_score(labels, (scores >= threshold).astype(int))
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)

    return best_threshold, best_score
