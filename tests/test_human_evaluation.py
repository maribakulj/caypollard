"""Tests for the expert-evaluation agreement measures."""

from __future__ import annotations

import pytest

from caypollard.human_evaluation import (
    compare_with_metric,
    disagreements,
    kendall_tau_b,
    krippendorff_alpha,
    summarise_dimensions,
)


def test_perfect_agreement_gives_alpha_one() -> None:
    judgements = {
        "q1": {"a": 5, "b": 5},
        "q2": {"a": 1, "b": 1},
        "q3": {"a": 3, "b": 3},
    }
    assert krippendorff_alpha(judgements)["alpha"] == pytest.approx(1.0)


def test_systematic_opposition_gives_alpha_at_or_below_zero() -> None:
    judgements = {
        "q1": {"a": 1, "b": 5},
        "q2": {"a": 5, "b": 1},
        "q3": {"a": 1, "b": 5},
        "q4": {"a": 5, "b": 1},
    }
    assert krippendorff_alpha(judgements)["alpha"] <= 0.0


def test_ordinal_weighting_punishes_a_far_disagreement_more_than_a_near_one() -> None:
    near = {f"q{i}": {"a": 3, "b": 4} for i in range(6)} | {"q6": {"a": 1, "b": 1}}
    far = {f"q{i}": {"a": 1, "b": 5} for i in range(6)} | {"q6": {"a": 1, "b": 1}}
    # Unweighted agreement would call these identical: one disagreement each time.
    assert krippendorff_alpha(near)["alpha"] > krippendorff_alpha(far)["alpha"]


def test_units_with_a_single_rater_are_reported_not_silently_dropped() -> None:
    result = krippendorff_alpha({"q1": {"a": 3, "b": 4}, "q2": {"a": 5}, "q3": {"b": 2}})
    assert result["n_units"] == 1
    assert result["n_units_skipped_single_rater"] == 2


def test_no_variation_is_reported_as_undefined_rather_than_as_agreement() -> None:
    result = krippendorff_alpha({"q1": {"a": 4, "b": 4}, "q2": {"a": 4, "b": 4}})
    assert result["alpha"] is None
    assert "variation" in result["reason"]


def test_empty_input_is_handled() -> None:
    result = krippendorff_alpha({})
    assert result["alpha"] is None
    assert result["n_units"] == 0


def test_kendall_tau_b_on_identical_and_reversed_orders() -> None:
    assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_kendall_tau_b_accounts_for_ties() -> None:
    # Ordinal ratings tie constantly; an implementation ignoring ties would
    # report a correlation of 1 here.
    tau = kendall_tau_b([1, 1, 2, 2], [1, 2, 1, 2])
    assert tau is not None
    assert abs(tau) < 0.5


def test_kendall_tau_b_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        kendall_tau_b([1, 2], [1, 2, 3])


def test_compare_with_metric_uses_only_shared_items() -> None:
    result = compare_with_metric({"a": 5, "b": 1, "c": 3}, {"a": 0.9, "b": 0.1, "d": 0.5})
    assert result["n"] == 2
    assert result["tau_b"] == pytest.approx(1.0)


def test_disagreements_are_returned_worst_first_as_items() -> None:
    rows = disagreements(
        {
            "mild": {"a": 3, "b": 4},
            "severe": {"a": 1, "b": 5},
            "moderate": {"a": 2, "b": 4},
        },
        threshold=2.0,
    )
    assert [row["unit"] for row in rows] == ["severe", "moderate"]
    assert rows[0]["ratings"] == {"a": 1, "b": 5}


def test_dimensions_are_summarised_separately_with_no_overall_score() -> None:
    result = summarise_dimensions(
        {
            "visuel": {"q1": {"a": 5, "b": 5}},
            "iconographique": {"q1": {"a": 1, "b": 5}},
        }
    )
    assert set(result) == {"visuel", "iconographique"}
    assert result["iconographique"]["n_disagreements"] == 1
    assert result["visuel"]["n_disagreements"] == 0
