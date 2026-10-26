"""Experiments on top of the engine: computational scaling and model sensitivity.

Scaling: exact enumeration visits 3^n * 2^k configurations (n people, k unobserved
traits), so the work grows exponentially. We MEASURE runtime and peak memory on
pedigrees of increasing size instead of assuming a complexity.

Sensitivity: the posterior is always a valid, normalised distribution, but it is
only as good as the model's assumptions. We vary the founder prior, the mutation
probability and the 0-copy trait probability, and we remove or flip each
observation, then measure how far each person's gene-count posterior moves.
Distance used: total variation TV(p, q) = 1/2 * sum_g |p(g) - q(g)|, which is the
largest possible difference in probability that the two distributions assign
to any event.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, replace
from statistics import median

from bayesian_genetic_inference.enumeration import (
    InferenceResult, enumerate_posteriors, expected_configuration_count,
)
from bayesian_genetic_inference.model import ModelParameters
from bayesian_genetic_inference.pedigree import Pedigree, Person, build_pedigree

# ---------------------------------------------------------------- pedigrees for scaling

def prefix_pedigree(pedigree: Pedigree, size: int) -> Pedigree:
    """The first `size` people in parents-before-children order.

    Every prefix of a topological order is itself a valid pedigree, because a
    person's parents always appear before them.
    """
    if not 1 <= size <= len(pedigree.people):
        raise ValueError(f"size must be between 1 and {len(pedigree.people)}")
    return build_pedigree(list(pedigree.people[:size]))


def with_extra_children(pedigree: Pedigree, mother: str, father: str, count: int) -> Pedigree:
    """Add `count` synthetic children of (mother, father) with unknown traits (benchmark only)."""
    extra = [Person(f"ExtraChild{i + 1}", mother, father, None) for i in range(count)]
    return build_pedigree(list(pedigree.people) + extra)


@dataclass(frozen=True)
class ScalingCase:
    label: str
    pedigree: Pedigree


def scaling_cases(base: Pedigree, extra_children: int = 2) -> list[ScalingCase]:
    """Prefixes of the base pedigree (1..n people) plus up to `extra_children` larger cases."""
    cases = [ScalingCase(f"prefix_{size}", prefix_pedigree(base, size))
             for size in range(1, len(base.people) + 1)]
    last = base.people[-1]
    mother, father = (last.mother, last.father) if not last.is_founder else (None, None)
    if mother is not None:
        for count in range(1, extra_children + 1):
            cases.append(ScalingCase(f"extended_{len(base.people) + count}",
                                     with_extra_children(base, mother, father, count)))
    return cases


def time_enumeration(pedigree: Pedigree, min_total_seconds: float = 0.5, max_repeats: int = 25) -> dict:
    """Median wall-clock time of enumerate_posteriors (repeat small cases to beat timer noise)."""
    timings: list[float] = []
    started = time.perf_counter()
    while len(timings) < max_repeats:
        tic = time.perf_counter()
        enumerate_posteriors(pedigree)
        timings.append(time.perf_counter() - tic)
        if time.perf_counter() - started >= min_total_seconds and len(timings) >= 3:
            break
    return {"median_seconds": median(timings), "min_seconds": min(timings),
            "max_seconds": max(timings), "repeats": len(timings)}


def peak_memory_bytes(pedigree: Pedigree) -> int:
    """Peak Python heap allocation during one enumeration (tracemalloc; separate from timing)."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    enumerate_posteriors(pedigree)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


# ---------------------------------------------------------------- sensitivity analysis

def total_variation(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p[key] - q[key]) for key in p)


def carrier_prior(carrier_mass: float) -> dict[int, float]:
    """Founder prior with P(G >= 1) = carrier_mass, keeping the course's 3:1 ratio of 1 to 2 copies."""
    if not 0.0 <= carrier_mass <= 1.0:
        raise ValueError("carrier_mass must be in [0, 1]")
    return {0: 1.0 - carrier_mass, 1: 0.75 * carrier_mass, 2: 0.25 * carrier_mass}


def posterior_table(result: InferenceResult) -> dict[str, dict[int, float]]:
    return {name: dict(post.gene) for name, post in result.posteriors.items()}


def assumption_sweep(pedigree: Pedigree, baseline: ModelParameters,
                     carrier_masses: list[float], mutations: list[float],
                     zero_copy_trait_rates: list[float]) -> list[dict]:
    """One row per (assumption value, person): posterior and its shift from the baseline."""
    base_posteriors = posterior_table(enumerate_posteriors(pedigree, baseline))
    variants: list[tuple[str, float, ModelParameters]] = []
    for mass in carrier_masses:
        variants.append(("founder carrier prior P(G>=1)", mass,
                         replace(baseline, founder_gene_prior=carrier_prior(mass))))
    for mutation in mutations:
        variants.append(("mutation probability", mutation, replace(baseline, mutation_probability=mutation)))
    for rate in zero_copy_trait_rates:
        trait = dict(baseline.trait_given_genes)
        trait[0] = rate
        variants.append(("P(trait | 0 copies)", rate, replace(baseline, trait_given_genes=trait)))

    rows = []
    for assumption, value, params in variants:
        posteriors = posterior_table(enumerate_posteriors(pedigree, params))
        for name in pedigree.names:
            rows.append(_shift_row(assumption, value, name, posteriors[name], base_posteriors[name]))
    return rows


def evidence_ablation(pedigree: Pedigree, params: ModelParameters) -> list[dict]:
    """Remove or flip each observed trait in turn and measure the posterior shift."""
    base_posteriors = posterior_table(enumerate_posteriors(pedigree, params))
    rows = []
    for observed_name, observed_value in pedigree.observed_traits.items():
        for action in ("remove", "flip"):
            new_value = None if action == "remove" else not observed_value
            people = [replace(p, observed_trait=new_value) if p.name == observed_name else p
                      for p in pedigree.people]
            posteriors = posterior_table(enumerate_posteriors(build_pedigree(people), params))
            scenario = f"{action} {observed_name}={int(observed_value)}"
            for name in pedigree.names:
                rows.append(_shift_row("evidence", scenario, name, posteriors[name], base_posteriors[name]))
    return rows


def _shift_row(assumption: str, value, person: str, posterior: dict, baseline: dict) -> dict:
    return {
        "assumption": assumption, "value": value, "person": person,
        "p_gene_0": posterior[0], "p_gene_1": posterior[1], "p_gene_2": posterior[2],
        "p_carrier": posterior[1] + posterior[2],
        "carrier_shift": (posterior[1] + posterior[2]) - (baseline[1] + baseline[2]),
        "tv_from_baseline": total_variation(posterior, baseline),
    }


def summarise_shifts(rows: list[dict]) -> list[dict]:
    """Largest movement per assumption (or per evidence scenario) and who moved most."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["assumption"],) if row["assumption"] != "evidence" else ("evidence", row["value"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, members in groups.items():
        worst = max(members, key=lambda r: r["tv_from_baseline"])
        least = min({r["person"]: r for r in members}.values(),
                    key=lambda r: max(m["tv_from_baseline"] for m in members if m["person"] == r["person"]))
        least_tv = max(m["tv_from_baseline"] for m in members if m["person"] == least["person"])
        summary.append({
            "assumption": key[0], "scenario": key[1] if len(key) > 1 else "full sweep",
            "max_tv": worst["tv_from_baseline"], "most_affected_person": worst["person"],
            "at_value": worst["value"], "max_abs_carrier_shift": max(abs(r["carrier_shift"]) for r in members),
            "least_affected_person": least["person"], "least_affected_max_tv": least_tv,
        })
    return sorted(summary, key=lambda s: -s["max_tv"])
