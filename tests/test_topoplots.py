import numpy as np
import pytest
from PIL import Image

from ocular.epochs import find_recordings, segments_for_recording
from ocular.topoplots import IMAGE_SIZE, TopoplotRenderer


def first_segment(raw_root, event="blink"):
    study, participant, path = find_recordings(raw_root)[0]
    for segment in segments_for_recording(study, participant, path):
        if segment.event == event:
            return segment
    raise AssertionError(f"no {event} segment found")


def test_renders_a_png_of_the_expected_size(raw_root, tmp_path):
    segment = first_segment(raw_root)
    out = tmp_path / "topo.png"

    with TopoplotRenderer() as renderer:
        renderer.render(segment, out)

    assert out.exists()
    with Image.open(out) as image:
        assert image.size == (IMAGE_SIZE, IMAGE_SIZE)


def test_creates_missing_directories(raw_root, tmp_path):
    segment = first_segment(raw_root)
    out = tmp_path / "a" / "b" / "topo.png"

    with TopoplotRenderer() as renderer:
        renderer.render(segment, out)

    assert out.exists()


def test_blink_and_rest_render_differently(raw_root, tmp_path):
    """If the images were identical there would be nothing to classify."""
    blink = first_segment(raw_root, "blink")
    rest = first_segment(raw_root, "rest")

    with TopoplotRenderer() as renderer:
        renderer.render(blink, tmp_path / "blink.png")
        renderer.render(rest, tmp_path / "rest.png")

    with Image.open(tmp_path / "blink.png") as a, Image.open(tmp_path / "rest.png") as b:
        difference = np.abs(
            np.asarray(a.convert("L"), dtype=float)
            - np.asarray(b.convert("L"), dtype=float)
        )

    assert difference.mean() > 1.0


def test_reuses_one_figure_across_renders(raw_root, tmp_path):
    import matplotlib.pyplot as plt

    segment = first_segment(raw_root)
    with TopoplotRenderer() as renderer:
        before = len(plt.get_fignums())
        for i in range(5):
            renderer.render(segment, tmp_path / f"{i}.png")
        assert len(plt.get_fignums()) == before


def test_recording_without_positions_is_rejected(raw_root, tmp_path):
    import mne

    segment = first_segment(raw_root)
    stripped = mne.create_info(
        segment.info["ch_names"],
        segment.info["sfreq"],
        ["eeg"] * len(segment.info["ch_names"]),
        verbose=False,
    )
    broken = type(segment)(
        study=segment.study, participant=segment.participant, event=segment.event,
        index=segment.index, data=segment.data, info=stripped,
    )

    with TopoplotRenderer() as renderer:
        with pytest.raises(ValueError, match="channel positions"):
            renderer.render(broken, tmp_path / "x.png")
