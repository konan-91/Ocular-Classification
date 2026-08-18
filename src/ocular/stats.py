"""Testing whether the model and the baseline actually differ.

Both methods are scored on the same segments, so their errors are paired.
McNemar's test is the right test for that: it looks only at the segments the
two methods disagree on and asks whether the disagreement is lopsided.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.contingency_tables import mcnemar

# Below this many disagreements the chi squared approximation is unreliable
# and the exact binomial test is used instead.
EXACT_THRESHOLD = 25


def contingency(model_correct: np.ndarray, baseline_correct: np.ndarray) -> np.ndarray:
    """2x2 table with model outcome on the rows, baseline on the columns."""
    model_correct = np.asarray(model_correct).astype(bool)
    baseline_correct = np.asarray(baseline_correct).astype(bool)

    if model_correct.shape != baseline_correct.shape:
        raise ValueError("paired outcomes must have the same length")

    return np.array(
        [
            [
                int(np.sum(~model_correct & ~baseline_correct)),
                int(np.sum(~model_correct & baseline_correct)),
            ],
            [
                int(np.sum(model_correct & ~baseline_correct)),
                int(np.sum(model_correct & baseline_correct)),
            ],
        ]
    )


def paired_difference_ci(
    model_correct: np.ndarray, baseline_correct: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Confidence interval for the accuracy difference, model minus baseline.

    Uses the standard error of the paired difference, which accounts for the
    two methods being scored on the same segments.
    """
    model_correct = np.asarray(model_correct).astype(float)
    baseline_correct = np.asarray(baseline_correct).astype(float)

    differences = model_correct - baseline_correct
    n = len(differences)
    estimate = float(differences.mean())

    if n < 2:
        return estimate, float("nan"), float("nan")

    standard_error = float(differences.std(ddof=1) / np.sqrt(n))
    margin = scipy_stats.norm.ppf(0.5 + confidence / 2) * standard_error
    return estimate, estimate - margin, estimate + margin


def compare(model_correct: np.ndarray, baseline_correct: np.ndarray) -> dict:
    """Run McNemar's test on two sets of paired outcomes."""
    table = contingency(model_correct, baseline_correct)

    baseline_only = int(table[0, 1])  # baseline right, model wrong
    model_only = int(table[1, 0])  # model right, baseline wrong
    discordant = baseline_only + model_only

    exact = discordant < EXACT_THRESHOLD
    if discordant == 0:
        statistic, p_value = 0.0, 1.0
    else:
        result = mcnemar(table, exact=exact, correction=not exact)
        statistic, p_value = float(result.statistic), float(result.pvalue)

    difference, low, high = paired_difference_ci(model_correct, baseline_correct)

    return {
        "table": table.tolist(),
        "both_wrong": int(table[0, 0]),
        "baseline_only_correct": baseline_only,
        "model_only_correct": model_only,
        "both_correct": int(table[1, 1]),
        "discordant": discordant,
        "test": "exact binomial" if exact else "chi squared with continuity correction",
        "statistic": statistic,
        "p_value": p_value,
        "model_accuracy": float(np.mean(model_correct)),
        "baseline_accuracy": float(np.mean(baseline_correct)),
        "accuracy_difference": difference,
        "accuracy_difference_ci95": [low, high],
        "significant_at_05": bool(p_value < 0.05),
    }


def format_report(result: dict) -> str:
    lines = [
        "McNemar's test, model against ICA baseline",
        f"  model accuracy     {result['model_accuracy']:.4f}",
        f"  baseline accuracy  {result['baseline_accuracy']:.4f}",
        f"  difference         {result['accuracy_difference']:+.4f} "
        f"(95% CI {result['accuracy_difference_ci95'][0]:+.4f} to "
        f"{result['accuracy_difference_ci95'][1]:+.4f})",
        "",
        "  paired outcomes",
        f"    both correct            {result['both_correct']:>7}",
        f"    model only correct      {result['model_only_correct']:>7}",
        f"    baseline only correct   {result['baseline_only_correct']:>7}",
        f"    both wrong              {result['both_wrong']:>7}",
        "",
        f"  {result['test']}",
        f"  statistic {result['statistic']:.4f}   p = {result['p_value']:.3g}",
    ]

    if result["discordant"] == 0:
        lines.append("  the two methods never disagreed")
    elif result["significant_at_05"]:
        better = "model" if result["accuracy_difference"] > 0 else "baseline"
        lines.append(f"  the difference is significant, favouring the {better}")
    else:
        lines.append("  no significant difference between the two methods")

    return "\n".join(lines)
