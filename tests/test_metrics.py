from __future__ import annotations

import pytest

from navjev.eval.metrics import (
    calibration_curve,
    exact_match,
    expected_calibration_error,
    normalize_answer,
    paired_bootstrap,
    section_recall,
    token_f1,
)


def test_section_recall_is_exact_node_match() -> None:
    assert section_recall(["a", "b"], ["a"]) == 1.0
    assert section_recall(["a"], ["a", "b"]) == 0.5
    assert section_recall(["parent"], ["child"]) == 0.0
    with pytest.raises(ValueError):
        section_recall(["a"], [])


def test_exact_match_and_f1_use_squad_normalization() -> None:
    assert normalize_answer("The $412 Million.") == "412 million"
    assert exact_match("$412 million", "412 Million") == 1.0
    assert exact_match("412", ["413", "412 million"]) == 0.0
    assert token_f1("capex was 412 million", "412 million") == pytest.approx(
        2 * (0.5 * 1) / 1.5
    )
    assert token_f1("", "") == 1.0
    assert token_f1("nothing", "412") == 0.0


def test_paired_bootstrap_interval_covers_true_difference() -> None:
    a = [1.0] * 50 + [0.0] * 50
    b = [0.0] * 20 + [1.0] * 80
    diff, (lo, hi) = paired_bootstrap(a, b, n_resamples=2000, seed=1)
    assert diff == pytest.approx(-0.3)
    assert lo < -0.3 < hi
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    a, b = [0.2, 0.4, 0.9, 0.1], [0.3, 0.3, 0.5, 0.2]
    assert paired_bootstrap(a, b, 500, seed=7) == paired_bootstrap(a, b, 500, seed=7)


def test_calibration_curve_buckets() -> None:
    predicted = [0.05, 0.15, 0.95, 0.92, 0.5]
    outcomes = [False, False, True, True, False]
    curve = calibration_curve(predicted, outcomes, n_buckets=10)
    assert curve[0] == (pytest.approx(0.05), 0.0, 1)
    assert curve[-1] == (pytest.approx(0.935), 1.0, 2)
    assert expected_calibration_error(curve) == pytest.approx(
        (0.05 + 0.15 + 0.5 + 0.065 * 2) / 5
    )
    with pytest.raises(ValueError):
        calibration_curve([1.2], [True])
