"""End to end: raw recordings through to the benchmark.

Everything except reading the EEGLAB files themselves runs for real here, on
synthetic recordings small enough to finish in seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ocular import manifest, splits
from ocular.benchmark import run as run_benchmark
from ocular.prepare import prepare
from ocular.train import TrainConfig, train


@pytest.fixture
def prepared(raw_root, tmp_path):
    frame = prepare(
        raw_root,
        tmp_path / "topoplots",
        tmp_path / "manifest.csv",
        limit_per_event=4,
    )
    return frame, tmp_path / "manifest.csv"


def test_prepare_writes_images_and_a_manifest(prepared):
    frame, path = prepared

    assert path.exists()
    assert len(frame) == 6 * 4 * 4  # recordings, event types, capped segments
    assert set(frame["label"]) == {"blink", "non-blink"}
    assert frame["key"].is_unique

    for image in manifest.read(path)["path"]:
        assert image.endswith(".png")


def test_manifest_paths_are_relative_on_disk(prepared):
    _, path = prepared
    raw = pd.read_csv(path)
    assert not raw["path"].str.startswith("/").any()


def test_prepare_covers_every_recording(prepared):
    frame, _ = prepared
    assert frame["group"].nunique() == 6


@pytest.fixture
def trained(prepared, tmp_path):
    _, manifest_path = prepared
    results = train(
        manifest_path,
        tmp_path / "artifacts",
        TrainConfig(
            architecture="resnet18",
            pretrained=False,
            head_epochs=1,
            finetune_epochs=1,
            batch_size=8,
            num_workers=0,
            device="cpu",
            patience=99,
        ),
    )
    return results, tmp_path / "artifacts", manifest_path


def test_training_writes_a_model_and_its_metadata(trained):
    results, artifacts, _ = trained

    assert (artifacts / "model.pt").exists()
    assert (artifacts / "training.json").exists()
    assert (artifacts / "confusion_matrix_val.png").exists()
    assert len(results["history"]) == 2


def test_checkpoint_records_the_split_it_used(trained):
    results, _, _ = trained
    stored = results["metadata"]["split"]

    assert set(stored) == {"train", "val", "test"}
    assert not set(stored["train"]) & set(stored["test"])
    assert not set(stored["train"]) & set(stored["val"])


def test_no_recording_appears_in_two_splits(trained):
    """The leakage guard, checked on the real pipeline rather than a stub."""
    _, _, manifest_path = trained
    frame = manifest.read(manifest_path)
    assigned = splits.make(frame).assign(frame)

    assert (assigned.groupby("group")["split"].nunique() == 1).all()


def test_benchmark_runs_both_methods_and_compares_them(trained, raw_root, tmp_path):
    _, artifacts, manifest_path = trained

    results = run_benchmark(
        manifest_path,
        artifacts / "model.pt",
        raw_root,
        artifacts,
        batch_size=8,
        num_workers=0,
    )

    assert results["test_segments"] > 0
    assert results["test_recordings"] >= 1

    for method in ("model", "baseline"):
        scores = results[method]
        assert 0.0 <= scores["accuracy"] <= 1.0
        assert 0.0 <= scores["balanced_accuracy"] <= 1.0
        assert scores["n"] == results["test_segments"]

    comparison = results["mcnemar"]
    assert 0.0 <= comparison["p_value"] <= 1.0
    assert np.array(comparison["table"]).sum() == results["test_segments"]

    assert (artifacts / "benchmark.json").exists()
    assert (artifacts / "confusion_matrix_ica.png").exists()


def test_benchmark_scores_only_held_out_recordings(trained, raw_root, tmp_path):
    results_train, artifacts, manifest_path = trained
    results = run_benchmark(
        manifest_path, artifacts / "model.pt", raw_root, artifacts,
        batch_size=8, num_workers=0,
    )

    trained_on = set(results_train["metadata"]["split"]["train"])
    assert not trained_on & set(results["held_out_groups"])


def test_benchmark_output_is_valid_json(trained, raw_root):
    _, artifacts, manifest_path = trained
    run_benchmark(
        manifest_path, artifacts / "model.pt", raw_root, artifacts,
        batch_size=8, num_workers=0,
    )

    payload = json.loads((artifacts / "benchmark.json").read_text())
    assert "mcnemar" in payload
    assert "by_event" in payload
    assert set(payload["by_event"]) <= {"rest", "blink", "h_saccade", "v_saccade"}
