import numpy as np
import pytest

from ocular import stats


def test_contingency_counts_each_cell():
    model = np.array([1, 1, 0, 0, 1])
    baseline = np.array([1, 0, 1, 0, 0])
    table = stats.contingency(model, baseline)

    assert table.tolist() == [[1, 1], [2, 1]]
    assert table.sum() == 5


def test_contingency_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        stats.contingency(np.array([1, 0]), np.array([1]))


def test_identical_methods_give_no_difference():
    outcomes = np.random.default_rng(0).integers(0, 2, 200)
    result = stats.compare(outcomes, outcomes)

    assert result["discordant"] == 0
    assert result["p_value"] == 1.0
    assert not result["significant_at_05"]


def test_clearly_better_model_is_significant():
    # Model right everywhere, baseline wrong on a third of the segments
    model = np.ones(300, dtype=int)
    baseline = np.ones(300, dtype=int)
    baseline[:100] = 0

    result = stats.compare(model, baseline)

    assert result["model_only_correct"] == 100
    assert result["baseline_only_correct"] == 0
    assert result["p_value"] < 0.001
    assert result["significant_at_05"]
    assert result["accuracy_difference"] > 0


def test_small_samples_use_the_exact_test():
    model = np.array([1] * 10 + [0] * 2)
    baseline = np.array([0] * 10 + [1] * 2)
    result = stats.compare(model, baseline)

    assert result["discordant"] < stats.EXACT_THRESHOLD
    assert result["test"] == "exact binomial"


def test_large_samples_use_the_chi_squared_test():
    model = np.array([1] * 60 + [0] * 40)
    baseline = np.array([0] * 60 + [1] * 40)
    result = stats.compare(model, baseline)

    assert result["discordant"] >= stats.EXACT_THRESHOLD
    assert "chi squared" in result["test"]


def test_accuracies_match_the_raw_outcomes():
    model = np.array([1, 1, 1, 0])
    baseline = np.array([1, 0, 0, 0])
    result = stats.compare(model, baseline)

    assert result["model_accuracy"] == pytest.approx(0.75)
    assert result["baseline_accuracy"] == pytest.approx(0.25)
    assert result["accuracy_difference"] == pytest.approx(0.5)


def test_confidence_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    model = rng.integers(0, 2, 500)
    baseline = rng.integers(0, 2, 500)

    estimate, low, high = stats.paired_difference_ci(model, baseline)
    assert low < estimate < high


def test_report_mentions_the_winner():
    model = np.ones(200, dtype=int)
    baseline = np.zeros(200, dtype=int)
    text = stats.format_report(stats.compare(model, baseline))

    assert "favouring the model" in text
