"""Exact inference by enumerating every configuration of the hidden variables.

Hidden variables: every person's gene count (3 values each) and every
UNOBSERVED trait (2 values each). Observed traits are fixed to their evidence,
so assignments that contradict the evidence are never generated. For n people
with k unobserved traits there are exactly 3^n * 2^k configurations.

For each configuration we compute the joint probability (product of the
gene and trait factors of every person), then
* add it to the evidence total  P(evidence) = sum of all joints,
* add it to every person's matching gene-count and trait bucket (marginalisation),
and finally divide each person's buckets by the total (normalisation), giving
    P(G_person = g | evidence)  and  P(T_person = t | evidence).
No sampling or approximation is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from bayesian_genetic_inference.model import (
    GENE_COUNTS, ModelParameters, gene_probability, normalize, trait_likelihood,
)
from bayesian_genetic_inference.pedigree import Pedigree

TRAIT_STATES = (False, True)


class ImpossibleEvidenceError(ValueError):
    """Every configuration consistent with the evidence has probability zero."""


@dataclass(frozen=True)
class PersonPosterior:
    name: str
    gene: dict[int, float]        # P(G = 0/1/2 | evidence)
    trait: dict[bool, float]      # P(T = True/False | evidence)
    observed_trait: bool | None


@dataclass(frozen=True)
class InferenceResult:
    posteriors: dict[str, PersonPosterior]
    evidence_probability: float          # P(observed traits), the normalising constant
    configurations_evaluated: int        # counted while enumerating, not computed from a formula
    zero_probability_configurations: int
    people: int
    unobserved_traits: int


def expected_configuration_count(pedigree: Pedigree) -> int:
    """3^(number of people) * 2^(number of unobserved traits) - used only as a cross-check."""
    return len(GENE_COUNTS) ** len(pedigree.people) * len(TRAIT_STATES) ** len(pedigree.unobserved_trait_names)


def enumerate_posteriors(pedigree: Pedigree, params: ModelParameters | None = None) -> InferenceResult:
    """Exact posterior gene-count and trait distributions for everyone in the pedigree."""
    params = params or ModelParameters()
    people = pedigree.people                       # parents always come before children
    position = {person.name: i for i, person in enumerate(people)}
    parent_positions = [None if p.is_founder else (position[p.mother], position[p.father]) for p in people]
    unknown_positions = [i for i, p in enumerate(people) if p.observed_trait is None]
    observed_positions = [i for i, p in enumerate(people) if p.observed_trait is not None]

    # trait factor table: trait_factor[g][t] = P(T = t | G = g)
    trait_factor = {g: {t: trait_likelihood(g, t, params) for t in TRAIT_STATES} for g in GENE_COUNTS}

    gene_weight = [{g: 0.0 for g in GENE_COUNTS} for _ in people]
    trait_weight = [{t: 0.0 for t in TRAIT_STATES} for _ in people]
    evidence_total = 0.0
    configurations = 0
    zero_configurations = 0

    for genes in product(GENE_COUNTS, repeat=len(people)):
        # factors that depend only on the gene assignment: inheritance/prior terms
        # and the likelihood of the OBSERVED traits
        gene_part = 1.0
        for i, parents in enumerate(parent_positions):
            if parents is None:
                gene_part *= gene_probability(genes[i], None, None, params)
            else:
                gene_part *= gene_probability(genes[i], genes[parents[0]], genes[parents[1]], params)
        for i in observed_positions:
            gene_part *= trait_factor[genes[i]][people[i].observed_trait]

        for unknown_traits in product(TRAIT_STATES, repeat=len(unknown_positions)):
            configurations += 1
            joint = gene_part
            for i, trait in zip(unknown_positions, unknown_traits):
                joint *= trait_factor[genes[i]][trait]
            if joint == 0.0:
                zero_configurations += 1
                continue                              # contributes nothing to any sum

            evidence_total += joint
            for i, g in enumerate(genes):
                gene_weight[i][g] += joint
            for i in observed_positions:
                trait_weight[i][people[i].observed_trait] += joint
            for i, trait in zip(unknown_positions, unknown_traits):
                trait_weight[i][trait] += joint

    if evidence_total == 0.0:
        raise ImpossibleEvidenceError(
            "the observed traits have probability zero under these model parameters")

    posteriors = {
        person.name: PersonPosterior(
            name=person.name,
            gene=normalize(gene_weight[i]),
            trait=normalize(trait_weight[i]),
            observed_trait=person.observed_trait,
        )
        for i, person in enumerate(people)
    }
    return InferenceResult(
        posteriors=posteriors,
        evidence_probability=evidence_total,
        configurations_evaluated=configurations,
        zero_probability_configurations=zero_configurations,
        people=len(people),
        unobserved_traits=len(unknown_positions),
    )
