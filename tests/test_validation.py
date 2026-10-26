"""Tests for the independent validation suite and the reasoning checks."""

import pytest

import bayesian_genetic_inference.enumeration as enumeration
import bayesian_genetic_inference.model as model
from bayesian_genetic_inference.pedigree import load_pedigree_csv
from bayesian_genetic_inference.validation import (
    SPEC_EXAMPLE_JOINT, count_checks, published_output_check, reasoning_checks, reference_child_distribution,
    reference_example_joint, reference_founder_posterior, run_validation, verify_example_fixture,
)

SMALL = load_pedigree_csv("data/small_family_3.csv")
MAIN = load_pedigree_csv("data/synthetic_pedigree_8.csv")


def test_reference_functions_do_not_use_the_engine(monkeypatch):
    """Break every engine function; the reference values must still be computable."""
    def broken(*args, **kwargs):
        raise RuntimeError("engine called from a reference function")
    for name in ["trait_likelihood", "transmission_probability", "child_gene_distribution",
                 "gene_probability", "joint_probability", "normalize"]:
        monkeypatch.setattr(model, name, broken)
    monkeypatch.setattr(enumeration, "enumerate_posteriors", broken)
    assert reference_example_joint() == pytest.approx(SPEC_EXAMPLE_JOINT, abs=1e-15)
    assert sum(reference_child_distribution(1, 2).values()) == pytest.approx(1.0)
    assert sum(reference_founder_posterior(True).values()) == pytest.approx(1.0)


def test_closed_form_formula_reproduces_published_example():
    assert abs(reference_example_joint() - 0.0026643247488) < 1e-15


def test_twelve_closed_form_and_nine_inheritance_checks_pass():
    results = run_validation(SMALL.parent_map())
    assert count_checks(results["closed_form"]) == 12
    assert count_checks(results["inheritance"]) == 9
    assert len(results["inheritance"]) == 27
    for row in results["closed_form"] + results["inheritance"]:
        assert row.passed, row


def test_fixture_must_match_the_spec_example():
    with pytest.raises(ValueError, match="does not match"):
        verify_example_fixture({"Harry": ("James", "Lily"), "James": None, "Lily": None})   # parents swapped


def test_published_course_output_matches_within_rounding():
    result = enumeration.enumerate_posteriors(SMALL)
    rows = published_output_check(result.posteriors)
    assert len(rows) == 15 and all(row.passed for row in rows)


def test_reasoning_checks_all_pass():
    rows = reasoning_checks(MAIN, SMALL)
    assert all(row.passed for row in rows), [r for r in rows if not r.passed]
    categories = {row.category for row in rows}
    assert {"total probability", "prior predictive", "conditional independence", "independence",
            "symmetry", "monotonicity", "evidence consistency"} <= categories
