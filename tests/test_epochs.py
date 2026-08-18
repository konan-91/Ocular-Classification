from pathlib import Path

import pytest

from ocular import epochs
from ocular.channels import scalp_channels


def test_participant_id_parses_common_patterns():
    assert epochs.participant_id(Path("sub-07.set")) == "p007"
    assert epochs.participant_id(Path("subj_12_task.set")) == "p012"
    assert epochs.participant_id(Path("participant3.set")) == "p003"
    assert epochs.participant_id(Path("recording_42.set")) == "p042"


def test_participant_id_falls_back_to_stem():
    assert epochs.participant_id(Path("odd-name.set")) == "odd-name"


def test_find_recordings_groups_by_study(raw_root):
    found = epochs.find_recordings(raw_root)
    assert len(found) == 6
    assert {study for study, _, _ in found} == {"study1", "study2"}
    assert {participant for _, participant, _ in found} == {"p001", "p002", "p003"}


def test_find_recordings_rejects_empty_tree(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        epochs.find_recordings(tmp_path / "empty")


def test_split_on_triggers_finds_every_onset(recording):
    blinks = recording["4"]
    windows = epochs.split_on_triggers(blinks, ("eye-blink",))

    # Two parent epochs, six onsets each, all clear of the boundaries
    assert len(windows) == 12
    assert all(w.shape[0] == len(recording.ch_names) for w in windows)


def test_split_on_triggers_uses_a_constant_window(recording):
    windows = epochs.split_on_triggers(recording["4"], ("eye-blink",))
    expected = int(epochs.TMAX * 100) - int(epochs.TMIN * 100)
    assert {w.shape[1] for w in windows} == {expected}


def test_split_on_triggers_ignores_missing_channels(recording):
    assert epochs.split_on_triggers(recording["4"], ("not-a-channel",)) == []


def test_split_on_triggers_does_not_double_count(recording):
    """Two trigger channels must not both claim the same onset."""
    windows = epochs.split_on_triggers(recording["3"], ("eye-u", "eye-d"))
    assert len(windows) == 12


def test_rest_windows_match_event_windows(recording):
    rest = epochs.split_rest(recording["1"])
    events = epochs.split_on_triggers(recording["4"], ("eye-blink",))

    assert len(rest) > 0
    # Equal duration across classes, so length cannot be a giveaway
    assert rest[0].shape[1] == events[0].shape[1]


def test_segments_carry_provenance(raw_root):
    found = epochs.find_recordings(raw_root)
    study, participant, path = found[0]
    segments = list(epochs.segments_for_recording(study, participant, path))

    assert segments
    assert all(s.study == study and s.participant == participant for s in segments)
    assert {s.event for s in segments} == {"rest", "blink", "h_saccade", "v_saccade"}


def test_segment_keys_are_unique(raw_root):
    found = epochs.find_recordings(raw_root)
    keys = []
    for study, participant, path in found:
        keys.extend(s.key for s in epochs.segments_for_recording(study, participant, path))

    assert len(keys) == len(set(keys))


def test_scalp_channels_drop_triggers_and_eog(recording):
    kept = scalp_channels(recording.ch_names)

    assert "eye-blink" not in kept
    assert "VEOG" not in kept
    assert "Fp1" not in kept  # frontopolar sits on top of the eyes
    assert "Cz" in kept
