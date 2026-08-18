import pandas as pd
import pytest

from ocular import manifest


def test_blinks_are_the_positive_class():
    assert manifest.label_for_event("blink") == "blink"
    for event in ("rest", "h_saccade", "v_saccade"):
        assert manifest.label_for_event(event) == "non-blink"


def test_group_key_includes_the_study():
    """Two studies can number their participants the same way."""
    assert manifest.group_key("study1", "p001") != manifest.group_key("study2", "p001")


def test_write_then_read_round_trips(tmp_path):
    rows = [
        {
            "key": "study1_p001_blink00001",
            "path": "topoplots/blink/study1_p001_blink00001.png",
            "label": "blink",
            "event": "blink",
            "study": "study1",
            "participant": "p001",
            "group": "study1/p001",
        }
    ]
    path = tmp_path / "manifest.csv"
    manifest.write(rows, path)
    frame = manifest.read(path)

    assert len(frame) == 1
    assert frame.loc[0, "key"] == "study1_p001_blink00001"
    # Paths come back absolute, resolved against the manifest location
    assert frame.loc[0, "path"].endswith("topoplots/blink/study1_p001_blink00001.png")
    assert frame.loc[0, "path"].startswith("/")


def test_read_rejects_a_manifest_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"path": "a.png"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        manifest.read(path)


def test_summarise_reports_counts():
    frame = pd.DataFrame(
        [
            {"label": "blink", "event": "blink", "group": "s/p1"},
            {"label": "non-blink", "event": "rest", "group": "s/p1"},
            {"label": "non-blink", "event": "rest", "group": "s/p2"},
        ]
    )
    text = manifest.summarise(frame)

    assert "3 segments from 2 recordings" in text
    assert "blink=1" in text
