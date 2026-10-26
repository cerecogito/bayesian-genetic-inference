"""Tests for the scaling and sensitivity helpers, and for the generated reports."""

import csv
import json
from pathlib import Path

import pytest

from bayesian_genetic_inference.analysis import (
    carrier_prior, evidence_ablation, prefix_pedigree, scaling_cases, total_variation,
)
from bayesian_genetic_inference.enumeration import enumerate_posteriors, expected_configuration_count
from bayesian_genetic_inference.model import ModelParameters
from bayesian_genetic_inference.pedigree import load_pedigree_csv

MAIN = load_pedigree_csv("data/synthetic_pedigree_8.csv")
REPORTS = Path("reports")


def test_prefixes_are_valid_pedigrees_with_correct_counts():
    for case in scaling_cases(MAIN, extra_children=1):
        result = enumerate_posteriors(case.pedigree) if len(case.pedigree.people) <= 6 else None
        if result is not None:
            assert result.configurations_evaluated == expected_configuration_count(case.pedigree)
    assert len(prefix_pedigree(MAIN, 5).people) == 5


def test_carrier_prior_keeps_course_ratio_and_baseline():
    assert carrier_prior(0.04) == pytest.approx({0: 0.96, 1: 0.03, 2: 0.01})
    assert sum(carrier_prior(0.2).values()) == pytest.approx(1.0)


def test_total_variation_bounds():
    p = {0: 0.2, 1: 0.5, 2: 0.3}
    assert total_variation(p, p) == 0.0
    assert total_variation({0: 1.0, 1: 0.0, 2: 0.0}, {0: 0.0, 1: 0.0, 2: 1.0}) == 1.0


def test_removing_negative_evidence_matters_less_than_flipping_it():
    rows = evidence_ablation(load_pedigree_csv("data/small_family_3.csv"), ModelParameters())
    shift = {(r["value"], r["person"]): r["tv_from_baseline"] for r in rows}
    assert shift[("remove Lily=0", "Lily")] < shift[("flip Lily=0", "Lily")]


needs_reports = pytest.mark.skipif(not (REPORTS / "cv_evidence.json").exists(),
                                   reason="run scripts/run_experiment.py first")


@needs_reports
def test_reported_metrics_match_output_files():
    evidence = {m["metric"]: m for m in json.loads((REPORTS / "cv_evidence.json").read_text())["metrics"]}
    posteriors = json.loads((REPORTS / "posteriors_8_person.json").read_text())
    assert evidence["configurations_evaluated_8_person"]["value"] == posteriors["configurations_evaluated"] == 209_952
    with open(REPORTS / "validation_table.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    exact = [r for r in rows if r["category"] not in ("published output", "monotonicity")
             and not r["check_id"].startswith("R")]
    assert max(float(r["abs_error"]) for r in exact) == evidence["max_abs_error_closed_form_and_inheritance"]["value"]
    assert all(r["passed"] == "True" for r in rows)
    summary = (REPORTS / "results_summary.md").read_text(encoding="utf-8")
    assert "209,952" in summary
