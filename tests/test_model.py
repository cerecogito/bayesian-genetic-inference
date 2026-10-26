"""Tests for the probability model (priors, inheritance, trait likelihoods, joint probability)."""

import pytest

from bayesian_genetic_inference.model import (
    ModelParameters, child_gene_distribution, gene_probability, joint_probability, normalize,
    trait_likelihood, transmission_probability,
)

PARAMS = ModelParameters()


def test_default_parameters_are_the_course_values():
    assert PARAMS.founder_gene_prior == {0: 0.96, 1: 0.03, 2: 0.01}
    assert PARAMS.trait_given_genes == {0: 0.01, 1: 0.56, 2: 0.65}
    assert PARAMS.mutation_probability == 0.01


def test_trait_likelihoods_are_complementary():
    for g in (0, 1, 2):
        assert trait_likelihood(g, True, PARAMS) + trait_likelihood(g, False, PARAMS) == pytest.approx(1.0)


def test_transmission_probabilities():
    assert transmission_probability(0, PARAMS) == 0.01
    assert transmission_probability(1, PARAMS) == 0.5
    assert transmission_probability(2, PARAMS) == 0.99
    with pytest.raises(ValueError):
        transmission_probability(3, PARAMS)


def test_child_distribution_sums_to_one_for_all_nine_parent_pairs():
    for m in (0, 1, 2):
        for f in (0, 1, 2):
            assert sum(child_gene_distribution(m, f, PARAMS).values()) == pytest.approx(1.0, abs=1e-15)


def test_founder_and_non_founder_gene_probability():
    assert gene_probability(2, None, None, PARAMS) == 0.01                    # founder prior
    assert gene_probability(2, 2, 2, PARAMS) == pytest.approx(0.99 * 0.99)    # inherits both
    assert gene_probability(0, 0, 0, PARAMS) == pytest.approx(0.99 * 0.99)    # no mutation either side
    with pytest.raises(ValueError):
        gene_probability(1, 1, None, PARAMS)                                  # one parent only


def test_spec_worked_example_joint_probability():
    parents = {"Harry": ("Lily", "James"), "James": None, "Lily": None}
    genes = {"Harry": 1, "James": 2, "Lily": 0}
    traits = {"Harry": False, "James": True, "Lily": False}
    assert joint_probability(parents, genes, traits, PARAMS) == pytest.approx(0.0026643247488, abs=1e-15)


def test_normalize_keeps_ratios_and_rejects_all_zero():
    assert normalize({"a": 1.0, "b": 3.0}) == {"a": 0.25, "b": 0.75}
    with pytest.raises(ValueError):
        normalize({"a": 0.0, "b": 0.0})


@pytest.mark.parametrize("kwargs", [
    {"founder_gene_prior": {0: 0.5, 1: 0.3, 2: 0.3}},          # does not sum to one
    {"founder_gene_prior": {0: 1.0, 1: 0.0}},                  # missing gene count
    {"trait_given_genes": {0: 0.1, 1: 1.2, 2: 0.5}},           # probability above one
    {"mutation_probability": -0.1},
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelParameters(**kwargs)
