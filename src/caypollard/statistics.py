"""Deterministic inferential statistics over per-query retrieval metrics.

The evaluation plan asks for bootstrap confidence intervals, paired comparisons,
and effect sizes rather than bare point estimates. A retrieval metric reported as
a single number hides whether a gap between two methods exceeds the noise of the
particular query sample, which is exactly the question a fusion claim turns on.

Every routine here resamples *queries*, not ranked items: the query is the
independent unit of observation, and two methods are always compared on the same
queries so their shared difficulty cancels out. All randomness is seeded, so a
reported interval is reproducible from the artifact alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(list(values), dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} must contain at least one observation")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _check_confidence(confidence: float) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    tail = (1.0 - confidence) / 2.0
    return 100.0 * tail, 100.0 * (1.0 - tail)


def _paired(
    first: Sequence[float], second: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    left = _as_vector(first, name="first")
    right = _as_vector(second, name="second")
    if left.size != right.size:
        raise ValueError(
            "Paired comparisons need one observation per query in both methods; "
            f"got {left.size} and {right.size}"
        )
    return left, right


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Percentile bootstrap interval for the mean of a per-query metric."""
    vector = _as_vector(values, name="values")
    low_pct, high_pct = _check_confidence(confidence)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    generator = np.random.default_rng(seed)
    indices = generator.integers(0, vector.size, size=(n_resamples, vector.size))
    means = vector[indices].mean(axis=1)

    return {
        "mean": float(vector.mean()),
        "ci_low": float(np.percentile(means, low_pct)),
        "ci_high": float(np.percentile(means, high_pct)),
        "confidence": confidence,
        "n_queries": int(vector.size),
        "n_resamples": int(n_resamples),
        "seed": seed,
    }


def paired_bootstrap_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap interval for the mean per-query difference ``first - second``.

    Queries are resampled jointly so the pairing is preserved; resampling the two
    methods independently would inflate the interval with between-query variance
    that the paired design removes.
    """
    left, right = _paired(first, second)
    differences = left - right
    result = bootstrap_mean_ci(
        differences, confidence=confidence, n_resamples=n_resamples, seed=seed
    )
    result["mean_difference"] = result.pop("mean")
    result["mean_first"] = float(left.mean())
    result["mean_second"] = float(right.mean())
    # An interval excluding zero is the condition the fusion decision gate needs;
    # it is reported rather than turned into a verdict here.
    result["excludes_zero"] = bool(result["ci_low"] > 0.0 or result["ci_high"] < 0.0)
    return result


def paired_permutation_test(
    first: Sequence[float],
    second: Sequence[float],
    *,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Two-sided sign-flip permutation test on paired per-query differences.

    Under the null hypothesis that the two methods are exchangeable per query,
    the sign of each difference is arbitrary. Flipping signs at random builds the
    null distribution without assuming normality, which retrieval metrics bounded
    in [0, 1] rarely satisfy.

    The p-value uses the add-one correction, so it is never reported as exactly
    zero: with a finite number of permutations, unobserved is not impossible.
    """
    left, right = _paired(first, second)
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    differences = left - right
    observed = float(differences.mean())

    generator = np.random.default_rng(seed)
    signs = generator.choice((-1.0, 1.0), size=(n_permutations, differences.size))
    null = (signs * differences).mean(axis=1)
    extreme = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))

    return {
        "mean_difference": observed,
        "p_value": (extreme + 1) / (n_permutations + 1),
        "n_queries": int(differences.size),
        "n_permutations": int(n_permutations),
        "seed": seed,
    }


def paired_cohens_d(first: Sequence[float], second: Sequence[float]) -> float:
    """Standardised mean paired difference. Returns 0.0 when no variation exists."""
    left, right = _paired(first, second)
    differences = left - right
    if differences.size < 2:
        raise ValueError("Cohen's d needs at least two paired observations")
    spread = float(differences.std(ddof=1))
    if spread == 0.0:
        return 0.0
    return float(differences.mean() / spread)


def cliffs_delta(first: Sequence[float], second: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1]: P(first > second) - P(first < second).

    Reported alongside Cohen's d because retrieval metrics are bounded and often
    heavily tied — many queries score exactly 0 or 1 — which makes a
    standard-deviation-scaled effect size hard to read on its own.
    """
    left = _as_vector(first, name="first")
    right = _as_vector(second, name="second")
    comparisons = np.sign(left[:, None] - right[None, :])
    return float(comparisons.mean())


def compare_methods(
    first: Sequence[float],
    second: Sequence[float],
    *,
    first_name: str = "first",
    second_name: str = "second",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Full paired comparison of two methods on the same queries.

    Bundles the interval, the test, and both effect sizes into one artifact-ready
    record so a results table cannot quietly report a p-value without the effect
    size that gives it meaning.
    """
    difference = paired_bootstrap_difference(
        first, second, confidence=confidence, n_resamples=n_resamples, seed=seed
    )
    test = paired_permutation_test(
        first, second, n_permutations=n_permutations, seed=seed
    )
    return {
        "first": first_name,
        "second": second_name,
        "mean_first": difference["mean_first"],
        "mean_second": difference["mean_second"],
        "mean_difference": difference["mean_difference"],
        "ci_low": difference["ci_low"],
        "ci_high": difference["ci_high"],
        "confidence": confidence,
        "excludes_zero": difference["excludes_zero"],
        "p_value": test["p_value"],
        "cohens_d": paired_cohens_d(first, second),
        "cliffs_delta": cliffs_delta(first, second),
        "n_queries": difference["n_queries"],
        "n_resamples": int(n_resamples),
        "n_permutations": int(n_permutations),
        "seed": seed,
    }
