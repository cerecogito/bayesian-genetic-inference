"""Independent validation of the inference engine against closed-form results.

Two kinds of functions live here:

* reference_* functions compute EXPECTED values from the published CS50 numbers
  (copied below as literals) with explicit arithmetic. They never import or
  call the model / enumeration code, so a bug in the engine cannot leak into
  the expected values. tests/test_validation.py checks this by breaking the
  engine and showing the reference functions still work.
* run_validation() computes the CALCULATED values with the engine and compares.

Formulas used for the reference values are written next to each function and
copied into the validation table so the report documents them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# --- published CS50 AI Heredity values, typed in from the specification --------------
SPEC_FOUNDER_PRIOR = {0: 0.96, 1: 0.03, 2: 0.01}
SPEC_TRAIT_TRUE = {0: 0.01, 1: 0.56, 2: 0.65}
SPEC_MUTATION = 0.01
# worked joint-probability example from the specification
SPEC_EXAMPLE_PARENTS = {"Harry": ("Lily", "James"), "James": None, "Lily": None}
SPEC_EXAMPLE_ONE_GENE = {"Harry"}
SPEC_EXAMPLE_TWO_GENES = {"James"}
SPEC_EXAMPLE_HAVE_TRAIT = {"James"}
SPEC_EXAMPLE_JOINT = 0.0026643247488
# published program output for the 3-person example family (rounded to 4 dp in the spec)
SPEC_FAMILY_OUTPUT = {
    "Harry": {"gene": {2: 0.0092, 1: 0.4557, 0: 0.5351}, "trait": {True: 0.2665, False: 0.7335}},
    "James": {"gene": {2: 0.1976, 1: 0.5106, 0: 0.2918}, "trait": {True: 1.0, False: 0.0}},
    "Lily": {"gene": {2: 0.0036, 1: 0.0136, 0: 0.9827}, "trait": {True: 0.0, False: 1.0}},
}

FLOAT_TOLERANCE = 1e-12          # for exact closed-form comparisons
PUBLISHED_ROUNDING_TOLERANCE = 5e-5 + 1e-12   # published output is rounded to 4 decimals


# --- independent reference calculations -------------------------------------------------

def reference_trait_probability(gene_count: int, has_trait: bool) -> float:
    """P(T | G): the published value, or its complement for an absent trait."""
    present = SPEC_TRAIT_TRUE[gene_count]
    return present if has_trait else 1.0 - present


def reference_pass_probability(parent_genes: int) -> float:
    """Probability a parent passes one copy: 0 copies -> m, 1 copy -> 1/2, 2 copies -> 1 - m."""
    return {0: SPEC_MUTATION, 1: 0.5, 2: 1.0 - SPEC_MUTATION}[parent_genes]


def reference_child_distribution(mother_genes: int, father_genes: int) -> dict[int, float]:
    """Enumerate the four (mother passes?, father passes?) outcomes and add up by copy count.

    P(child = c) = sum over a, b in {0, 1} with a + b = c of
                   P(mother passes = a) * P(father passes = b)
    """
    p_m, p_f = reference_pass_probability(mother_genes), reference_pass_probability(father_genes)
    distribution = {0: 0.0, 1: 0.0, 2: 0.0}
    for mother_passes in (0, 1):
        for father_passes in (0, 1):
            probability = (p_m if mother_passes else 1.0 - p_m) * (p_f if father_passes else 1.0 - p_f)
            distribution[mother_passes + father_passes] += probability
    return distribution


def reference_example_joint() -> float:
    """Closed-form product for the spec's worked example (written out factor by factor).

    Lily:  P(G=0) * P(no trait | 0)            = 0.96 * 0.99
    James: P(G=2) * P(trait | 2)               = 0.01 * 0.65
    Harry: P(G=1 | Lily 0, James 2) * P(no trait | 1)
           = [0.99*0.99 + 0.01*0.01] * 0.44
    """
    lily = SPEC_FOUNDER_PRIOR[0] * (1.0 - SPEC_TRAIT_TRUE[0])
    james = SPEC_FOUNDER_PRIOR[2] * SPEC_TRAIT_TRUE[2]
    harry_gets_one = ((1 - SPEC_MUTATION) * (1 - SPEC_MUTATION)    # Lily keeps hers, James passes
                      + SPEC_MUTATION * SPEC_MUTATION)              # Lily mutates, James fails
    harry = harry_gets_one * (1.0 - SPEC_TRAIT_TRUE[1])
    return lily * james * harry


def reference_founder_posterior(has_trait: bool) -> dict[int, float]:
    """Bayes' rule for a lone founder: P(G=g | T) = P(G=g) P(T|G=g) / sum_h P(G=h) P(T|G=h)."""
    numerators = {g: SPEC_FOUNDER_PRIOR[g] * reference_trait_probability(g, has_trait) for g in (0, 1, 2)}
    evidence = sum(numerators.values())
    return {g: numerators[g] / evidence for g in (0, 1, 2)}


# --- comparison with the engine ---------------------------------------------------------

@dataclass(frozen=True)
class CheckRow:
    check_id: str
    category: str
    description: str
    formula: str
    expected: float
    calculated: float
    abs_error: float
    tolerance: float
    passed: bool


def _row(check_id, category, description, formula, expected, calculated, tolerance=FLOAT_TOLERANCE) -> CheckRow:
    error = abs(expected - calculated)
    return CheckRow(check_id, category, description, formula, expected, calculated, error, tolerance,
                    error <= tolerance)


def verify_example_fixture(parent_map: dict[str, tuple[str, str] | None]) -> None:
    """Refuse to use the published joint value unless the fixture is exactly the spec's family."""
    if parent_map != SPEC_EXAMPLE_PARENTS:
        raise ValueError(f"fixture does not match the specification example: {parent_map}")


def run_validation(example_parent_map: dict[str, tuple[str, str] | None]) -> dict[str, list[CheckRow]]:
    """Run the 12 closed-form checks and the 9 inheritance checks against the engine."""
    # engine imports are local so that importing this module never pulls in the engine
    from bayesian_genetic_inference.enumeration import enumerate_posteriors
    from bayesian_genetic_inference.model import (
        ModelParameters, child_gene_distribution, joint_probability, trait_likelihood,
    )
    from bayesian_genetic_inference.pedigree import Person, build_pedigree

    params = ModelParameters()
    closed_form: list[CheckRow] = []

    # checks 1-3: founder prior, read off the engine's posterior for a lone founder with no evidence
    lone = enumerate_posteriors(build_pedigree([Person("Founder", None, None, None)]), params)
    for number, g in enumerate((0, 1, 2), start=1):
        closed_form.append(_row(f"C{number:02d}", "founder prior",
                                f"P(G={g}) for a founder with no evidence",
                                f"published prior P(G={g}) = {SPEC_FOUNDER_PRIOR[g]}",
                                SPEC_FOUNDER_PRIOR[g], lone.posteriors["Founder"].gene[g]))

    # checks 4-9: trait likelihoods
    number = 4
    for g in (0, 1, 2):
        for has_trait in (True, False):
            formula = (f"published P(T=1|G={g}) = {SPEC_TRAIT_TRUE[g]}" if has_trait
                       else f"1 - P(T=1|G={g}) = 1 - {SPEC_TRAIT_TRUE[g]}")
            closed_form.append(_row(f"C{number:02d}", "trait likelihood",
                                    f"P(T={has_trait}|G={g})", formula,
                                    reference_trait_probability(g, has_trait),
                                    trait_likelihood(g, has_trait, params)))
            number += 1

    # check 10: the specification's worked joint-probability example
    verify_example_fixture(example_parent_map)
    genes = {name: 2 if name in SPEC_EXAMPLE_TWO_GENES else 1 if name in SPEC_EXAMPLE_ONE_GENE else 0
             for name in example_parent_map}
    traits = {name: name in SPEC_EXAMPLE_HAVE_TRAIT for name in example_parent_map}
    closed_form_value = reference_example_joint()
    if abs(closed_form_value - SPEC_EXAMPLE_JOINT) > FLOAT_TOLERANCE:
        raise AssertionError("closed-form example product does not reproduce the published value")
    closed_form.append(_row("C10", "joint probability",
                            "spec example: Harry 1 copy no trait, James 2 copies trait, Lily 0 copies no trait",
                            "(0.96*0.99)*(0.01*0.65)*((0.99*0.99+0.01*0.01)*0.44); published 0.0026643247488",
                            SPEC_EXAMPLE_JOINT,
                            joint_probability(example_parent_map, genes, traits, params)))

    # checks 11-12: founder posterior after observing the trait (3 components each)
    for check_id, has_trait in (("C11", True), ("C12", False)):
        observed = enumerate_posteriors(build_pedigree([Person("Founder", None, None, has_trait)]), params)
        expected = reference_founder_posterior(has_trait)
        for g in (0, 1, 2):
            closed_form.append(_row(f"{check_id}.g{g}", "founder posterior",
                                    f"P(G={g} | T={has_trait}) for a lone founder",
                                    "P(G=g)P(T|G=g) / sum_h P(G=h)P(T|G=h)",
                                    expected[g], observed.posteriors["Founder"].gene[g]))

    # nine inheritance checks: every (mother, father) gene-count pair, 3 components each
    inheritance: list[CheckRow] = []
    for mother_genes in (0, 1, 2):
        for father_genes in (0, 1, 2):
            expected = reference_child_distribution(mother_genes, father_genes)
            calculated = child_gene_distribution(mother_genes, father_genes, params)
            for child in (0, 1, 2):
                inheritance.append(_row(f"I{mother_genes}{father_genes}.c{child}", "inheritance",
                                        f"P(child G={child} | mother {mother_genes}, father {father_genes})",
                                        "sum over 4 transmission outcomes of P(mother passes)*P(father passes)",
                                        expected[child], calculated[child]))
    return {"closed_form": closed_form, "inheritance": inheritance}


def published_output_check(posteriors) -> list[CheckRow]:
    """Extra cross-check: the spec's printed output for the 3-person family (rounded to 4 dp)."""
    rows = []
    for name, published in SPEC_FAMILY_OUTPUT.items():
        for g, value in published["gene"].items():
            rows.append(_row(f"P-{name}-g{g}", "published output", f"{name}: P(G={g} | evidence)",
                             "printed in the CS50 specification (4 dp)", value,
                             posteriors[name].gene[g], PUBLISHED_ROUNDING_TOLERANCE))
        for t, value in published["trait"].items():
            rows.append(_row(f"P-{name}-t{int(t)}", "published output", f"{name}: P(T={t} | evidence)",
                             "printed in the CS50 specification (4 dp)", value,
                             posteriors[name].trait[t], PUBLISHED_ROUNDING_TOLERANCE))
    return rows


def reference_founder_evidence_probability(has_trait: bool) -> float:
    """Law of total probability for one founder: P(T) = sum_g P(G=g) P(T | G=g)."""
    return sum(SPEC_FOUNDER_PRIOR[g] * reference_trait_probability(g, has_trait) for g in (0, 1, 2))


def reference_child_prior_predictive() -> dict[int, float]:
    """Child of two founders, no evidence: P(G=c) = sum_{m,f} P(m) P(f) P(c | m, f)."""
    totals = {0: 0.0, 1: 0.0, 2: 0.0}
    for m in (0, 1, 2):
        for f in (0, 1, 2):
            child = reference_child_distribution(m, f)
            for c in (0, 1, 2):
                totals[c] += SPEC_FOUNDER_PRIOR[m] * SPEC_FOUNDER_PRIOR[f] * child[c]
    return totals


def reasoning_checks(main_pedigree, small_family) -> list[CheckRow]:
    """Probability identities and invariants that any correct inference must satisfy.

    These test the REASONING (independence structure, Bayes' rule, total
    probability), not just whether a formula was typed correctly.
    """
    from dataclasses import replace

    from bayesian_genetic_inference.enumeration import enumerate_posteriors
    from bayesian_genetic_inference.pedigree import Person, build_pedigree

    rows: list[CheckRow] = []
    main = enumerate_posteriors(main_pedigree)

    # R1 total probability of the evidence in the 3-person family. James and Lily are
    # independent founders and Harry's trait is unobserved, so P(E) factorises.
    small = enumerate_posteriors(small_family)
    expected = reference_founder_evidence_probability(True) * reference_founder_evidence_probability(False)
    rows.append(_row("R1", "total probability", "P(James=1, Lily=0) in the 3-person family",
                     "[sum_g P(g)P(T=1|g)] * [sum_g P(g)P(T=0|g)] (independent founders)",
                     expected, small.evidence_probability))

    # R2 prior predictive of a child when nothing is observed
    trio = build_pedigree([Person("Mum", None, None, None), Person("Dad", None, None, None),
                           Person("Kid", "Mum", "Dad", None)])
    kid = enumerate_posteriors(trio).posteriors["Kid"].gene
    predictive = reference_child_prior_predictive()
    for c in (0, 1, 2):
        rows.append(_row(f"R2.g{c}", "prior predictive", f"P(child G={c}) with no evidence",
                         "sum_{m,f} P(m)P(f)P(c|m,f)", predictive[c], kid[c]))

    # R3 trait depends on the rest of the family only through the person's own gene count:
    #    P(T=1 | E) = sum_g P(G=g | E) P(T=1 | G=g) for every unobserved trait
    for name in main_pedigree.unobserved_trait_names:
        post = main.posteriors[name]
        mixture = sum(post.gene[g] * SPEC_TRAIT_TRUE[g] for g in (0, 1, 2))
        rows.append(_row(f"R3.{name}", "conditional independence",
                         f"{name}: P(T=1|E) equals the gene-posterior mixture",
                         "sum_g P(G=g|E) P(T=1|G=g)", mixture, post.trait[True]))

    # R4 observed traits are reproduced exactly
    for name, value in main_pedigree.observed_traits.items():
        rows.append(_row(f"R4.{name}", "evidence consistency", f"{name}: P(T={value} | E) for observed trait",
                         "observed evidence has posterior probability 1", 1.0,
                         main.posteriors[name].trait[value]))

    # R5 an unrelated founder with an observed trait is d-separated from the family
    with_stranger = build_pedigree(list(main_pedigree.people) + [Person("Stranger", None, None, True)])
    stranger_run = enumerate_posteriors(with_stranger)
    worst = max(abs(stranger_run.posteriors[n].gene[g] - main.posteriors[n].gene[g])
                for n in main_pedigree.names for g in (0, 1, 2))
    rows.append(_row("R5", "independence", "adding an unrelated observed founder changes nobody's posterior",
                     "max |difference| over people and gene counts = 0", 0.0, worst))

    # R6 the model treats mother and father symmetrically
    swapped = build_pedigree([p if p.is_founder else replace(p, mother=p.father, father=p.mother)
                              for p in main_pedigree.people])
    swapped_run = enumerate_posteriors(swapped)
    worst = max(abs(swapped_run.posteriors[n].gene[g] - main.posteriors[n].gene[g])
                for n in main_pedigree.names for g in (0, 1, 2))
    rows.append(_row("R6", "symmetry", "swapping every mother/father leaves all posteriors unchanged",
                     "max |difference| = 0", 0.0, worst))

    # R7 exchangeable founders (same role, no evidence of their own) get identical posteriors
    exchangeable = _exchangeable_founder_pairs(main_pedigree)
    for left, right in exchangeable:
        worst = max(abs(main.posteriors[left].gene[g] - main.posteriors[right].gene[g]) for g in (0, 1, 2))
        rows.append(_row(f"R7.{left}-{right}", "symmetry", f"{left} and {right} are exchangeable",
                         "max |difference| = 0", 0.0, worst))

    # R8 Bayes' rule direction: observing a child WITH the trait raises each parent's carrier probability
    for person in main_pedigree.people:
        if person.observed_trait is True and not person.is_founder:
            hidden = build_pedigree([replace(p, observed_trait=None) if p.name == person.name else p
                                     for p in main_pedigree.people])
            hidden_run = enumerate_posteriors(hidden)
            for parent in (person.mother, person.father):
                before = 1.0 - hidden_run.posteriors[parent].gene[0]
                after = 1.0 - main.posteriors[parent].gene[0]
                rows.append(CheckRow(f"R8.{person.name}->{parent}", "monotonicity",
                                     f"observing {person.name}=1 raises P({parent} carrier)",
                                     "P(G>=1 | E) > P(G>=1 | E without the child's trait)",
                                     before, after, abs(after - before), 0.0, after > before))
    return rows


def _exchangeable_founder_pairs(pedigree) -> list[tuple[str, str]]:
    """Founder couples with no own evidence whose child is the only link (e.g. Carl & Dana)."""
    pairs = []
    for person in pedigree.people:
        if person.is_founder:
            continue
        mother = next(p for p in pedigree.people if p.name == person.mother)
        father = next(p for p in pedigree.people if p.name == person.father)
        other_children = [p for p in pedigree.people
                          if p.name != person.name and (p.mother in (mother.name, father.name)
                                                        or p.father in (mother.name, father.name))]
        if (mother.is_founder and father.is_founder and mother.observed_trait is None
                and father.observed_trait is None and not other_children):
            pairs.append((mother.name, father.name))
    return pairs


def rows_as_dicts(rows: list[CheckRow]) -> list[dict]:
    return [asdict(row) for row in rows]


def count_checks(rows: list[CheckRow]) -> int:
    """Number of distinct checks (a multi-component check like C11.g0/g1/g2 counts once)."""
    return len({row.check_id.split(".")[0] for row in rows})
