import numpy as np
import pytest

from ocular import ica_baseline
from ocular.channels import pick_eog_channel
from ocular.epochs import find_recordings


def test_pick_eog_prefers_vertical():
    assert pick_eog_channel(["HEOG", "VEOG", "Cz"]) == "VEOG"
    assert pick_eog_channel(["HEOG", "Cz"]) == "HEOG"
    assert pick_eog_channel(["Cz", "Pz"]) is None


def test_pick_eog_never_returns_a_trigger_channel():
    """Trigger channels hold the labels, so using one would leak them."""
    assert pick_eog_channel(["eye-blink", "eye-u", "Cz"]) is None


def test_robust_standardise_centres_the_data():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    standardised = ica_baseline._robust_standardise(values)

    assert np.median(standardised) == pytest.approx(0.0, abs=1e-9)
    # The outlier stays an outlier rather than dragging the scale with it
    assert standardised[-1] > 10


def test_robust_standardise_handles_constant_input():
    standardised = ica_baseline._robust_standardise(np.full(10, 5.0))
    assert np.all(standardised == 0)


def test_calibrate_threshold_finds_a_clean_boundary():
    rng = np.random.default_rng(0)
    negatives = rng.normal(0, 1, 400)
    positives = rng.normal(8, 1, 400)

    scores = np.concatenate([negatives, positives])
    labels = np.concatenate([np.zeros(400, int), np.ones(400, int)])

    threshold, balanced = ica_baseline.calibrate_threshold(scores, labels)

    assert 1 < threshold < 7
    assert balanced > 0.95


def test_calibrate_threshold_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        ica_baseline.calibrate_threshold(np.arange(10.0), np.zeros(10, int))


def test_score_recording_produces_one_score_per_segment(raw_root):
    study, participant, path = find_recordings(raw_root)[0]
    result = ica_baseline.score_recording(study, participant, path)

    assert result is not None
    assert len(result.keys) == len(result.scores)
    assert len(set(result.keys)) == len(result.keys)
    assert result.n_components_found >= 1
    assert np.isfinite(result.scores).all()


def test_score_recording_separates_blinks_from_rest(raw_root):
    """The baseline has to be a real baseline, not noise."""
    study, participant, path = find_recordings(raw_root)[0]
    result = ica_baseline.score_recording(study, participant, path)

    scores = np.asarray(result.scores)
    is_blink = np.array(["blink" in k and "saccade" not in k for k in result.keys])
    is_rest = np.array(["rest" in k for k in result.keys])

    assert scores[is_blink].mean() > scores[is_rest].mean()


def test_score_dataset_covers_the_requested_groups(raw_root):
    frame = ica_baseline.score_dataset(raw_root, groups={"study1/p001", "study1/p002"})

    assert set(frame["group"]) == {"study1/p001", "study1/p002"}
    assert frame["key"].is_unique
    assert {"key", "group", "ica_score"} <= set(frame.columns)


def test_score_dataset_rejects_unknown_groups(raw_root):
    with pytest.raises(ValueError, match="none of the requested"):
        ica_baseline.score_dataset(raw_root, groups={"nope/p999"})
