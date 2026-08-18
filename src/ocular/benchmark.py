"""The full benchmark: model against the ICA baseline, with a significance test.

Both methods are run over the same held out recordings, matched segment by
segment, and compared with McNemar's test.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import ica_baseline, manifest, metrics, model as model_module, splits, stats, utils
from .data import LABEL_TO_INDEX
from .evaluate import predict

log = logging.getLogger(__name__)


def _split_from_metadata(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Reuse the split the model was trained with.

    Recreating the split here rather than reading it back would risk scoring
    the model on recordings it was trained on.
    """
    stored = metadata.get("split")
    if not stored:
        log.warning("checkpoint has no stored split, recreating from seed")
        seed = metadata.get("config", {}).get("seed", 91)
        return splits.make(frame, seed=seed).assign(frame)

    split = splits.Split(
        train=tuple(stored.get("train", [])),
        val=tuple(stored.get("val", [])),
        test=tuple(stored.get("test", [])),
    )
    return split.assign(frame)


def run(
    manifest_path: Path,
    model_path: Path,
    raw_root: Path,
    out_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    out_dir = Path(out_dir)

    frame = manifest.read(manifest_path)
    model, metadata = model_module.load(Path(model_path))
    device = model_module.pick_device(metadata.get("config", {}).get("device", "auto"))

    frame = _split_from_metadata(frame, metadata)
    val_frame = frame[frame["split"] == "val"].reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)

    if not len(test_frame):
        raise RuntimeError("the test split is empty")

    log.info(
        "benchmarking on %d test segments from %d held out recordings",
        len(test_frame), test_frame["group"].nunique(),
    )

    # The ICA baseline needs the raw recordings for both splits: validation to
    # fit its threshold, test to be scored on.
    needed = set(val_frame["group"]) | set(test_frame["group"])
    ica_scores = ica_baseline.score_dataset(Path(raw_root), groups=needed)

    val_scored = val_frame.merge(ica_scores[["key", "ica_score"]], on="key", how="inner")
    test_scored = test_frame.merge(ica_scores[["key", "ica_score"]], on="key", how="inner")

    if not len(val_scored) or not len(test_scored):
        raise RuntimeError(
            "ICA scores did not match any manifest rows, check that the raw "
            "dataset is the same one the manifest was built from"
        )

    dropped = len(test_frame) - len(test_scored)
    if dropped:
        log.warning(
            "%d test segments had no ICA score and are excluded from the "
            "paired comparison", dropped,
        )

    val_labels = val_scored["label"].map(LABEL_TO_INDEX).to_numpy()
    threshold, val_balanced = ica_baseline.calibrate_threshold(
        val_scored["ica_score"].to_numpy(), val_labels
    )
    log.info(
        "ICA threshold %.3f fitted on validation, balanced accuracy %.4f",
        threshold, val_balanced,
    )

    # Score both methods on exactly the same rows, in the same order.
    y_true, model_pred, model_score = predict(
        model, test_scored, device, batch_size=batch_size, num_workers=num_workers
    )
    baseline_score = test_scored["ica_score"].to_numpy()
    baseline_pred = (baseline_score >= threshold).astype(np.int64)

    model_metrics = metrics.compute(y_true, model_pred, model_score)
    baseline_metrics = metrics.compute(y_true, baseline_pred, baseline_score)
    comparison = stats.compare(model_pred == y_true, baseline_pred == y_true)

    # Per event breakdown, since saccades are the hard negatives and rest is not.
    by_event = {}
    for event, subset in test_scored.groupby("event"):
        mask = test_scored["event"].to_numpy() == event
        by_event[event] = {
            "n": int(mask.sum()),
            "model_accuracy": float(np.mean(model_pred[mask] == y_true[mask])),
            "baseline_accuracy": float(np.mean(baseline_pred[mask] == y_true[mask])),
        }

    results = {
        "test_segments": int(len(test_scored)),
        "test_recordings": int(test_scored["group"].nunique()),
        "held_out_groups": sorted(test_scored["group"].unique()),
        "ica_threshold": threshold,
        "ica_val_balanced_accuracy": val_balanced,
        "model": model_metrics,
        "baseline": baseline_metrics,
        "mcnemar": comparison,
        "by_event": by_event,
        "segments_without_ica_score": int(dropped),
    }

    utils.write_json(results, out_dir / "benchmark.json")
    metrics.plot_confusion_matrix(
        model_metrics, out_dir / "confusion_matrix_test.png", "Model, held out participants"
    )
    metrics.plot_confusion_matrix(
        baseline_metrics, out_dir / "confusion_matrix_ica.png", "ICA baseline, held out participants"
    )

    log.info("\n%s", metrics.format_report("model (test)", model_metrics))
    log.info("\n%s", metrics.format_report("ICA baseline (test)", baseline_metrics))
    log.info("\n%s", stats.format_report(comparison))
    return results


def format_summary(results: dict) -> str:
    """A compact table, ready to paste into the README."""
    model, baseline = results["model"], results["baseline"]
    lines = [
        f"| Method | Accuracy | Balanced accuracy | Blink recall | Non-blink recall |",
        f"| --- | --- | --- | --- | --- |",
    ]
    for name, scores in (("Model", model), ("ICA baseline", baseline)):
        lines.append(
            f"| {name} | {scores['accuracy']:.3f} | {scores['balanced_accuracy']:.3f} | "
            f"{scores['per_class']['blink']['recall']:.3f} | "
            f"{scores['per_class']['non-blink']['recall']:.3f} |"
        )

    comparison = results["mcnemar"]
    lines.append("")
    lines.append(
        f"McNemar's test: p = {comparison['p_value']:.3g} "
        f"({comparison['test']}), "
        f"difference {comparison['accuracy_difference']:+.3f} "
        f"(95% CI {comparison['accuracy_difference_ci95'][0]:+.3f} to "
        f"{comparison['accuracy_difference_ci95'][1]:+.3f})"
    )
    return "\n".join(lines)
