"""The probability model: priors, inheritance and trait likelihoods.

This module only does arithmetic on probabilities. It knows nothing about
CSV files, enumeration or reports.

Model (CS50 AI Heredity):
* every person has a hidden gene count G in {0, 1, 2} and a binary trait T;
* a founder (no recorded parents) draws G from the founder prior;
* a child gets one copy from each parent independently, and G is the sum
  of the two inherited indicators;
* a parent passes the gene on with probability
      mutation             if the parent has 0 copies (only by mutation)
      0.5                  if the parent has 1 copy
      1 - mutation         if the parent has 2 copies
* the trait depends only on the person's own gene count: P(T | G).

The default numbers below are the course's ILLUSTRATIVE assumptions. They are
not medical estimates and this is not a clinical tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose

GENE_COUNTS = (0, 1, 2)


def _default_gene_prior() -> dict[int, float]:
    return {0: 0.96, 1: 0.03, 2: 0.01}


def _default_trait_given_genes() -> dict[int, float]:
    # probability that the trait is PRESENT; absent = 1 - this
    return {0: 0.01, 1: 0.56, 2: 0.65}


@dataclass(frozen=True)
class ModelParameters:
    """All the probabilities the model needs. Defaults = CS50 course values."""

    founder_gene_prior: dict[int, float] = field(default_factory=_default_gene_prior)
    trait_given_genes: dict[int, float] = field(default_factory=_default_trait_given_genes)
    mutation_probability: float = 0.01

    def __post_init__(self) -> None:
        if set(self.founder_gene_prior) != set(GENE_COUNTS):
            raise ValueError("founder_gene_prior needs entries for gene counts 0, 1 and 2")
        if set(self.trait_given_genes) != set(GENE_COUNTS):
            raise ValueError("trait_given_genes needs entries for gene counts 0, 1 and 2")
        for name, value in [*self.founder_gene_prior.items(), *self.trait_given_genes.items(),
                            ("mutation", self.mutation_probability)]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"probability for {name!r} must be in [0, 1], got {value}")
        if not isclose(sum(self.founder_gene_prior.values()), 1.0, abs_tol=1e-12):
            raise ValueError("founder gene prior must sum to 1")


def trait_likelihood(gene_count: int, has_trait: bool, params: ModelParameters) -> float:
    """P(T = has_trait | G = gene_count)."""
    probability_present = params.trait_given_genes[gene_count]
    return probability_present if has_trait else 1.0 - probability_present


def transmission_probability(parent_gene_count: int, params: ModelParameters) -> float:
    """Probability that a parent with this many copies passes ONE copy to the child."""
    if parent_gene_count == 0:
        return params.mutation_probability           # only possible through a mutation
    if parent_gene_count == 1:
        return 0.5                                    # passes either copy with equal chance
    if parent_gene_count == 2:
        return 1.0 - params.mutation_probability     # passes a copy unless it mutates away
    raise ValueError(f"gene count must be 0, 1 or 2, got {parent_gene_count}")


def child_gene_distribution(mother_genes: int, father_genes: int,
                            params: ModelParameters) -> dict[int, float]:
    """P(child G = g | mother's and father's gene counts), for g = 0, 1, 2.

    The two transmissions are independent Bernoulli events with success
    probabilities p_m and p_f, so the child's count is their sum:
        P(0) = (1 - p_m)(1 - p_f)
        P(1) = p_m (1 - p_f) + (1 - p_m) p_f
        P(2) = p_m p_f
    """
    p_mother = transmission_probability(mother_genes, params)
    p_father = transmission_probability(father_genes, params)
    return {
        0: (1.0 - p_mother) * (1.0 - p_father),
        1: p_mother * (1.0 - p_father) + (1.0 - p_mother) * p_father,
        2: p_mother * p_father,
    }


def gene_probability(gene_count: int, mother_genes: int | None, father_genes: int | None,
                     params: ModelParameters) -> float:
    """P(G = gene_count | parents' gene counts), or the founder prior if there are no parents."""
    if mother_genes is None and father_genes is None:
        return params.founder_gene_prior[gene_count]
    if mother_genes is None or father_genes is None:
        raise ValueError("a person must have both parents recorded or neither")
    return child_gene_distribution(mother_genes, father_genes, params)[gene_count]


def joint_probability(parents: dict[str, tuple[str, str] | None], gene_counts: dict[str, int],
                      trait_states: dict[str, bool], params: ModelParameters) -> float:
    """Joint probability of ONE complete assignment of everybody's genes and traits.

    parents maps each name to (mother, father) or None for a founder.
    The joint factorises along the Bayesian network:
        P(all G, all T) = prod_person P(G_person | parents' G) * P(T_person | G_person)
    """
    probability = 1.0
    for name, parent_names in parents.items():
        genes = gene_counts[name]
        if parent_names is None:
            probability *= gene_probability(genes, None, None, params)
        else:
            mother, father = parent_names
            probability *= gene_probability(genes, gene_counts[mother], gene_counts[father], params)
        probability *= trait_likelihood(genes, trait_states[name], params)
    return probability


def normalize(weights: dict) -> dict:
    """Rescale non-negative weights so they sum to 1 (keeps their ratios).

    Raises ValueError when every weight is zero, because then there is no
    valid distribution to report.
    """
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("cannot normalise: all weights are zero (evidence is impossible)")
    return {key: value / total for key, value in weights.items()}
