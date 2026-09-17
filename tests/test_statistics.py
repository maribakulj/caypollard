from __future__ import annotations

import numpy as np
import pytest

from caypollard.statistics import (
    bootstrap_mean_ci,
    cliffs_delta,
    compare_methods,
    paired_bootstrap_difference,
    paired_cohens_d,
    paired_permutation_test,
)


def _noisy(mean: float, *, n: int = 200, seed: int = 7) -> list[float]:
    generator = np.random.default_rng(seed)
    return list(np.clip(generator.normal(mean, 0.15, size=n), 0.0, 1.0))


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = _noisy(0.4)
    first = bootstrap_mean_ci(values, seed=42)
    assert first["ci_low"] < first["mean"] < first["ci_high"]
    assert first == bootstrap_mean_ci(values, seed=42)
    assert first["n_queries"] == 200


def test_bootstrap_ci_narrows_as_queries_accumulate():
    narrow = bootstrap_mean_ci(_noisy(0.4, n=2000, seed=1), n_resamples=2000)
    wide = bootstrap_mean_ci(_noisy(0.4, n=50, seed=1), n_resamples=2000)
    assert (narrow["ci_high"] - narrow["ci_low"]) < (wide["ci_high"] - wide["ci_low"])


def test_bootstrap_ci_collapses_on_a_constant_metric():
    result = bootstrap_mean_ci([0.5] * 40, n_resamples=500)
    assert result["ci_low"] == result["ci_high"] == 0.5


def test_bootstrap_ci_rejects_bad_input():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.1, float("nan")])
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.1, 0.2], confidence=1.0)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.1, 0.2], n_resamples=0)


def test_paired_difference_detects_a_real_gain():
    baseline = _noisy(0.40, seed=3)
    improved = [value + 0.08 for value in baseline]
    result = paired_bootstrap_difference(improved, baseline, n_resamples=2000)
    assert result["excludes_zero"]
    assert result["ci_low"] > 0.0
    assert result["mean_difference"] == pytest.approx(0.08)


def test_paired_difference_stays_inconclusive_on_noise():
    generator = np.random.default_rng(11)
    baseline = list(generator.normal(0.4, 0.2, size=120))
    shuffled = list(generator.normal(0.4, 0.2, size=120))
    result = paired_bootstrap_difference(shuffled, baseline, n_resamples=2000)
    assert not result["excludes_zero"]


def test_paired_routines_require_matching_lengths():
    for routine in (paired_bootstrap_difference, paired_permutation_test, paired_cohens_d):
        with pytest.raises(ValueError, match="one observation per query"):
            routine([0.1, 0.2, 0.3], [0.1, 0.2])


def test_permutation_p_value_is_small_for_a_consistent_shift():
    baseline = _noisy(0.4, seed=5)
    improved = [value + 0.05 for value in baseline]
    result = paired_permutation_test(improved, baseline, n_permutations=2000)
    assert result["p_value"] < 0.01


def test_permutation_p_value_is_large_for_identical_methods():
    values = _noisy(0.4, seed=5)
    result = paired_permutation_test(values, values, n_permutations=1000)
    assert result["p_value"] == pytest.approx(1.0)


def test_permutation_p_value_is_never_exactly_zero():
    baseline = _noisy(0.2, seed=9)
    improved = [value + 0.5 for value in baseline]
    result = paired_permutation_test(improved, baseline, n_permutations=500)
    assert 0.0 < result["p_value"] <= 1.0 / 501 + 1e-12


def test_cohens_d_is_zero_without_variation():
    assert paired_cohens_d([0.3, 0.3, 0.3], [0.1, 0.1, 0.1]) == 0.0


def test_cohens_d_sign_follows_the_direction_of_the_gain():
    baseline = _noisy(0.4, seed=2)
    improved = [value + 0.05 for value in baseline]
    assert paired_cohens_d(improved, baseline) > 0
    assert paired_cohens_d(baseline, improved) < 0


def test_cliffs_delta_spans_its_full_range():
    assert cliffs_delta([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1.0)
    assert cliffs_delta([0.0, 0.0], [1.0, 1.0]) == pytest.approx(-1.0)
    assert cliffs_delta([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_compare_methods_bundles_interval_test_and_effect_sizes():
    baseline = _noisy(0.35, seed=13)
    improved = [value + 0.06 for value in baseline]
    report = compare_methods(
        improved,
        baseline,
        first_name="fusion",
        second_name="visual",
        n_resamples=2000,
        n_permutations=2000,
    )
    assert report["first"] == "fusion"
    assert report["second"] == "visual"
    assert report["mean_difference"] > 0
    assert report["excludes_zero"]
    assert report["p_value"] < 0.01
    assert report["cohens_d"] > 0
    assert set(report) >= {"ci_low", "ci_high", "cliffs_delta", "n_queries", "seed"}
