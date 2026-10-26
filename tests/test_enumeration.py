"""Tests for exact enumeration: counts, normalisation, evidence handling, repeatability."""

from itertools import product

import pytest

from bayesian_genetic_inference.enumeration import (
    ImpossibleEvidenceError, enumerate_posteriors, expected_configuration_count,
)
from bayesian_genetic_inference.model import ModelParameters, joint_probability
from bayesian_genetic_inference.pedigree import Person, build_pedigree, load_pedigree_csv

MAIN = load_pedigree_csv("data/synthetic_pedigree_8.csv")


@pytest.fixture(scope="module")
def main_result():
    return enumerate_posteriors(MAIN)


def test_exact_configuration_count(main_result):
    assert main_result.configurations_evaluated == 3 ** 8 * 2 ** 5 == 209_952
    assert main_result.configurations_evaluated == expected_configuration_count(MAIN)


def test_posteriors_sum_to_one(main_result):
    for post in main_result.posteriors.values():
        assert sum(post.gene.values()) == pytest.approx(1.0, abs=1e-12)
        assert sum(post.trait.values()) == pytest.approx(1.0, abs=1e-12)


def test_observed_traits_match_evidence(main_result):
    for name, value in MAIN.observed_traits.items():
        assert main_result.posteriors[name].trait[value] == 1.0
        assert main_result.posteriors[name].trait[not value] == 0.0


def test_repeatable(main_result):
    again = enumerate_posteriors(MAIN)
    assert again.posteriors == main_result.posteriors
    assert again.evidence_probability == main_result.evidence_probability


def test_matches_brute_force_over_model_joint_probability():
    """The enumeration shortcut (gene factors reused across trait loops) equals a plain brute force."""
    family = load_pedigree_csv("data/small_family_3.csv")
    params = ModelParameters()
    names, parents = family.names, family.parent_map()
    unknown = family.unobserved_trait_names
    gene_totals = {n: {0: 0.0, 1: 0.0, 2: 0.0} for n in names}
    total = 0.0
    for genes in product((0, 1, 2), repeat=len(names)):
        for traits in product((False, True), repeat=len(unknown)):
            trait_states = {**family.observed_traits, **dict(zip(unknown, traits))}
            p = joint_probability(parents, dict(zip(names, genes)), trait_states, params)
            total += p
            for n, g in zip(names, genes):
                gene_totals[n][g] += p
    result = enumerate_posteriors(family, params)
    assert result.evidence_probability == pytest.approx(total, rel=1e-13)
    for n in names:
        for g in (0, 1, 2):
            assert result.posteriors[n].gene[g] == pytest.approx(gene_totals[n][g] / total, rel=1e-12)


def test_fully_observed_traits_enumerate_only_genes():
    people = [Person("M", None, None, True), Person("F", None, None, False), Person("K", "M", "F", True)]
    result = enumerate_posteriors(build_pedigree(people))
    assert result.configurations_evaluated == 3 ** 3
    assert result.unobserved_traits == 0


def test_no_evidence_gives_the_prior_for_founders():
    result = enumerate_posteriors(build_pedigree([Person("Solo", None, None, None)]))
    assert result.posteriors["Solo"].gene == pytest.approx({0: 0.96, 1: 0.03, 2: 0.01})
    assert result.evidence_probability == pytest.approx(1.0)


def test_impossible_evidence_raises():
    never = ModelParameters(trait_given_genes={0: 0.0, 1: 0.0, 2: 0.0})   # nobody can have the trait
    with pytest.raises(ImpossibleEvidenceError):
        enumerate_posteriors(build_pedigree([Person("A", None, None, True)]), never)


def test_zero_probability_configurations_are_counted_and_skipped():
    # with no mutation, two 0-copy parents cannot have a carrier child
    params = ModelParameters(mutation_probability=0.0)
    people = [Person("M", None, None, None), Person("F", None, None, None), Person("K", "M", "F", None)]
    result = enumerate_posteriors(build_pedigree(people), params)
    assert result.zero_probability_configurations > 0
    assert result.configurations_evaluated == 3 ** 3 * 2 ** 3
    assert sum(result.posteriors["K"].gene.values()) == pytest.approx(1.0)
