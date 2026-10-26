"""Run every experiment and write all reports.

Usage:
    uv run python scripts/run_experiment.py              # default (~1 minute)
    uv run python scripts/run_experiment.py --extended   # also benchmarks a 10-person case (~5 minutes)

Deterministic outputs (identical on every re-run):
    reports/posteriors_8_person.csv / .json, reports/posteriors_small_family.csv
    reports/validation_table.csv, reports/scaling_configurations.csv
    reports/sensitivity_assumptions.csv, reports/sensitivity_evidence.csv, reports/sensitivity_summary.csv
    reports/results_summary.md, reports/cv_evidence.json, reports/figures/posterior_gene_counts.png
Runtime metadata (changes from run to run - wall-clock time, memory, machine):
    reports/runtime_metadata.json, reports/runtime_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bayesian_genetic_inference.analysis import (  # noqa: E402
    assumption_sweep, evidence_ablation, peak_memory_bytes, scaling_cases, summarise_shifts, time_enumeration,
)
from bayesian_genetic_inference.enumeration import (  # noqa: E402
    enumerate_posteriors, expected_configuration_count,
)
from bayesian_genetic_inference.model import ModelParameters  # noqa: E402
from bayesian_genetic_inference.pedigree import load_pedigree_csv  # noqa: E402
from bayesian_genetic_inference.validation import (  # noqa: E402
    count_checks, published_output_check, reasoning_checks, rows_as_dicts, run_validation,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
COMMAND = "uv run python scripts/run_experiment.py"

CARRIER_MASSES = [0.01, 0.02, 0.04, 0.08, 0.16]      # baseline 0.04 = 0.03 + 0.01
MUTATIONS = [0.0, 0.001, 0.01, 0.05, 0.1]            # baseline 0.01
ZERO_COPY_TRAIT_RATES = [0.001, 0.01, 0.05, 0.1]     # baseline 0.01


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def posterior_rows(result) -> list[dict]:
    rows = []
    for name, post in result.posteriors.items():
        rows.append({
            "person": name,
            "observed_trait": "" if post.observed_trait is None else int(post.observed_trait),
            "p_gene_0": post.gene[0], "p_gene_1": post.gene[1], "p_gene_2": post.gene[2],
            "p_trait_true": post.trait[True], "p_trait_false": post.trait[False],
        })
    return rows


def draw_posterior_chart(result, pedigree, path: Path) -> None:
    """Heat map of P(G = 0/1/2 | evidence) for every person, with each value printed."""
    names = pedigree.names
    values = [[result.posteriors[n].gene[g] for g in (0, 1, 2)] for n in names]
    labels = []
    for person in pedigree.people:
        tag = "trait unknown" if person.observed_trait is None else f"observed trait = {int(person.observed_trait)}"
        labels.append(f"{person.name}  ({tag})")

    figure, axes = plt.subplots(figsize=(7.2, 4.8), dpi=130)
    image = axes.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for row, row_values in enumerate(values):
        for col, value in enumerate(row_values):
            axes.text(col, row, f"{value:.3f}", ha="center", va="center",
                      color="white" if value > 0.55 else "#0b0b0b", fontsize=9)
    axes.set_xticks(range(3), ["0 copies", "1 copy", "2 copies"])
    axes.set_yticks(range(len(names)), labels)
    axes.set_title("Exact posterior gene-count probabilities, 8-person synthetic pedigree\n"
                   "(CS50 illustrative probabilities; not a medical estimate)", fontsize=10, loc="left")
    colourbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    colourbar.set_label("P(G = g | observed traits)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"Software": None})   # no version/time stamp -> byte-stable
    plt.close(figure)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extended", action="store_true",
                        help="add a 10-person benchmark case (7.5 million configurations)")
    args = parser.parse_args()
    params = ModelParameters()
    main_pedigree = load_pedigree_csv(DATA / "synthetic_pedigree_8.csv")
    small_family = load_pedigree_csv(DATA / "small_family_3.csv")

    # ---------------------------------------------------------------- exact inference
    main_result = enumerate_posteriors(main_pedigree, params)
    small_result = enumerate_posteriors(small_family, params)
    expected_count = expected_configuration_count(main_pedigree)
    if main_result.configurations_evaluated != expected_count:
        raise SystemExit(f"enumerated {main_result.configurations_evaluated} configurations, "
                         f"design requires {expected_count}")
    write_csv(REPORTS / "posteriors_8_person.csv", posterior_rows(main_result))
    write_csv(REPORTS / "posteriors_small_family.csv", posterior_rows(small_result))
    (REPORTS / "posteriors_8_person.json").write_text(json.dumps({
        "evidence_probability": main_result.evidence_probability,
        "configurations_evaluated": main_result.configurations_evaluated,
        "zero_probability_configurations": main_result.zero_probability_configurations,
        "posteriors": posterior_rows(main_result),
    }, indent=1) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- validation
    validation = run_validation(small_family.parent_map())
    published = published_output_check(small_result.posteriors)
    reasoning = reasoning_checks(main_pedigree, small_family)
    all_rows = validation["closed_form"] + validation["inheritance"] + published + reasoning
    write_csv(REPORTS / "validation_table.csv", rows_as_dicts(all_rows))
    exact_rows = validation["closed_form"] + validation["inheritance"]
    max_exact_error = max(r.abs_error for r in exact_rows)
    max_closed_error = max(r.abs_error for r in validation["closed_form"])
    max_inheritance_error = max(r.abs_error for r in validation["inheritance"])
    max_published_error = max(r.abs_error for r in published)
    equality_reasoning = [r for r in reasoning if r.category != "monotonicity"]
    max_reasoning_error = max(r.abs_error for r in equality_reasoning)
    all_passed = all(r.passed for r in all_rows)

    # ---------------------------------------------------------------- scaling (deterministic part)
    # extra synthetic children (unknown traits) measure growth past the 8-person design
    cases = scaling_cases(main_pedigree, extra_children=2 if args.extended else 1)
    scaling_rows = []
    for case in cases:
        run = enumerate_posteriors(case.pedigree, params)
        scaling_rows.append({
            "case": case.label, "people": run.people, "unobserved_traits": run.unobserved_traits,
            "configurations_evaluated": run.configurations_evaluated,
            "formula_3^n_x_2^k": expected_configuration_count(case.pedigree),
            "matches_formula": run.configurations_evaluated == expected_configuration_count(case.pedigree),
        })
    write_csv(REPORTS / "scaling_configurations.csv", scaling_rows)

    # ---------------------------------------------------------------- sensitivity
    assumption_rows = assumption_sweep(main_pedigree, params, CARRIER_MASSES, MUTATIONS, ZERO_COPY_TRAIT_RATES)
    evidence_rows = evidence_ablation(main_pedigree, params)
    summary_rows = summarise_shifts(assumption_rows + evidence_rows)
    write_csv(REPORTS / "sensitivity_assumptions.csv", assumption_rows)
    write_csv(REPORTS / "sensitivity_evidence.csv", evidence_rows)
    write_csv(REPORTS / "sensitivity_summary.csv", summary_rows)

    # ---------------------------------------------------------------- figure
    draw_posterior_chart(main_result, main_pedigree, REPORTS / "figures" / "posterior_gene_counts.png")

    # ---------------------------------------------------------------- deterministic reports
    write_results_summary(main_pedigree, main_result, small_result, validation, published, reasoning,
                          scaling_rows, summary_rows, assumption_rows, max_exact_error, all_passed)
    write_cv_evidence(main_result, validation, exact_rows, max_closed_error, max_inheritance_error,
                      max_exact_error, max_published_error, max_reasoning_error, reasoning, all_rows, summary_rows)

    # ---------------------------------------------------------------- runtime metadata (non-deterministic)
    timings = []
    for case in cases:
        timing = time_enumeration(case.pedigree)
        configurations = next(r["configurations_evaluated"] for r in scaling_rows if r["case"] == case.label)
        timings.append({"case": case.label, "people": len(case.pedigree.people),
                        "configurations": configurations, **timing,
                        "microseconds_per_configuration": 1e6 * timing["median_seconds"] / configurations,
                        "peak_memory_bytes": peak_memory_bytes(case.pedigree)})
        print(f"  timed {case.label}: {timing['median_seconds']:.4f}s median over {timing['repeats']} runs")
    write_runtime_reports(timings)
    print(f"done: {main_result.configurations_evaluated:,} configurations, all checks passed = {all_passed}")
    if not all_passed:
        raise SystemExit("some validation checks failed - see reports/validation_table.csv")


def write_results_summary(pedigree, result, small, validation, published, reasoning, scaling_rows,
                          summary_rows, assumption_rows, max_exact_error, all_passed) -> None:
    lines = [
        "# Results summary (deterministic)",
        "",
        f"Generated by `{COMMAND}`. Every number below is computed by the script; re-running gives an "
        "identical file. Timing and memory are in `runtime_report.md` and `runtime_metadata.json` "
        "because they change between runs.",
        "",
        "> Educational probability model. The probabilities are the CS50 AI Heredity course's "
        "illustrative assumptions, not medically validated estimates. All pedigrees are synthetic.",
        "",
        "## 1. Exact enumeration on the 8-person pedigree",
        "",
        f"* People: {result.people} (all gene counts latent); observed traits: "
        f"{len(pedigree.observed_traits)} "
        f"({', '.join(f'{n}={int(v)}' for n, v in pedigree.observed_traits.items())}); "
        f"unobserved traits: {result.unobserved_traits}",
        f"* Configurations evaluated (counted during enumeration): **{result.configurations_evaluated:,}**; "
        f"design 3^{result.people} × 2^{result.unobserved_traits} = "
        f"{3 ** result.people * 2 ** result.unobserved_traits:,}",
        f"* Configurations with zero probability: {result.zero_probability_configurations:,}",
        f"* P(observed traits) = {result.evidence_probability:.12g} (the normalising constant)",
        "",
        markdown_table(["person", "observed trait", "P(G=0)", "P(G=1)", "P(G=2)", "P(T=1)", "P(T=0)"],
                       [[n, "" if p.observed_trait is None else int(p.observed_trait), fmt(p.gene[0]),
                         fmt(p.gene[1]), fmt(p.gene[2]), fmt(p.trait[True]), fmt(p.trait[False])]
                        for n, p in result.posteriors.items()]),
        "",
        "![posterior gene counts](figures/posterior_gene_counts.png)",
        "",
        "## 2. Small worked example (3-person family, same structure as the CS50 example)",
        "",
        f"Configurations: {small.configurations_evaluated} (3^3 × 2^1). The CS50 specification prints these "
        "values to 4 decimal places; our exact values agree within rounding.",
        "",
        markdown_table(["person", "observed trait", "P(G=0)", "P(G=1)", "P(G=2)", "P(T=1)", "P(T=0)"],
                       [[n, "" if p.observed_trait is None else int(p.observed_trait), fmt(p.gene[0]),
                         fmt(p.gene[1]), fmt(p.gene[2]), fmt(p.trait[True]), fmt(p.trait[False])]
                        for n, p in small.posteriors.items()]),
        "",
        "## 3. Validation",
        "",
        f"* Closed-form checks: **{count_checks(validation['closed_form'])}** "
        f"({len(validation['closed_form'])} scalar comparisons), max |error| = "
        f"{max(r.abs_error for r in validation['closed_form']):.3e}",
        f"* Inheritance checks: **{count_checks(validation['inheritance'])}** parent combinations "
        f"({len(validation['inheritance'])} scalar comparisons), max |error| = "
        f"{max(r.abs_error for r in validation['inheritance']):.3e}",
        f"* Max |error| over all closed-form and inheritance comparisons: **{max_exact_error:.3e}** "
        "(tolerance 1e-12)",
        f"* Published CS50 output cross-check: {len(published)} values, max |error| = "
        f"{max(r.abs_error for r in published):.3e} (published values are rounded to 4 dp, tolerance 5e-5)",
        f"* Reasoning checks (probability identities and invariants): {count_checks(reasoning)} checks, "
        f"{len(reasoning)} comparisons, all passed = {all(r.passed for r in reasoning)}",
        f"* **All checks passed: {all_passed}**. Full table: `validation_table.csv`.",
        "",
        markdown_table(["id", "check", "expected", "calculated", "abs error"],
                       [[r.check_id, r.description, f"{r.expected:.12g}", f"{r.calculated:.12g}",
                         f"{r.abs_error:.2e}"] for r in validation["closed_form"]]),
        "",
        "Inheritance checks: reference values come from enumerating the four (mother passes?, father "
        "passes?) outcomes. Calculated values come from the model's closed form "
        "P(0)=(1-p_m)(1-p_f), P(1)=p_m(1-p_f)+(1-p_m)p_f, P(2)=p_m p_f.",
        "",
        markdown_table(["mother", "father", "P(child 0)", "P(child 1)", "P(child 2)", "max abs error"],
                       [[m, f] + [fmt(next(r.calculated for r in validation["inheritance"]
                                           if r.check_id == f"I{m}{f}.c{c}"), 6) for c in (0, 1, 2)]
                        + [f"{max(r.abs_error for r in validation['inheritance'] if r.check_id.startswith(f'I{m}{f}')):.1e}"]
                        for m in (0, 1, 2) for f in (0, 1, 2)]),
        "",
        "Reasoning checks:",
        "",
        markdown_table(["id", "identity / invariant", "expected", "calculated", "passed"],
                       [[r.check_id, r.description, f"{r.expected:.10g}", f"{r.calculated:.10g}", r.passed]
                        for r in reasoning]),
        "",
        "## 4. Computational scaling (configuration counts; timings in runtime_report.md)",
        "",
        markdown_table(["case", "people", "unobserved traits", "configurations evaluated", "3^n × 2^k"],
                       [[r["case"], r["people"], r["unobserved_traits"], f"{r['configurations_evaluated']:,}",
                         f"{r['formula_3^n_x_2^k']:,}"] for r in scaling_rows]),
        "",
        "Each extra person multiplies the work by 3 (gene count), or by 6 if their trait is also "
        "unobserved. The `extended_*` cases add synthetic children with unknown traits to measure growth "
        "past the 8-person design (`extended_10` only with `--extended`).",
        "",
        "## 5. Sensitivity of the posterior (same 8-person evidence throughout)",
        "",
        "Shift = total-variation distance between a person's gene-count posterior and the baseline "
        "posterior (0 = unchanged, 1 = completely different). Ranked by the largest shift:",
        "",
        markdown_table(["assumption / evidence change", "scenario", "max shift", "most affected",
                        "at value", "least affected (its max shift)"],
                       [[s["assumption"], s["scenario"], fmt(s["max_tv"], 3), s["most_affected_person"],
                         "-" if s["assumption"] == "evidence" else s["at_value"],
                         f"{s['least_affected_person']} ({s['least_affected_max_tv']:.3f})"]
                        for s in summary_rows]),
        "",
        *sensitivity_findings(summary_rows),
        "Carrier probability P(G>=1 | evidence) for each person, by assumption value:",
        "",
    ]
    for assumption in dict.fromkeys(r["assumption"] for r in assumption_rows):
        members = [r for r in assumption_rows if r["assumption"] == assumption]
        values = list(dict.fromkeys(r["value"] for r in members))
        lines.append(f"**{assumption}**")
        lines.append("")
        lines.append(markdown_table(["person"] + [str(v) for v in values],
                                    [[name] + [fmt(next(r["p_carrier"] for r in members
                                                        if r["person"] == name and r["value"] == v), 3)
                                               for v in values] for name in pedigree.names]))
        lines.append("")
    (REPORTS / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sensitivity_findings(summary_rows: list[dict]) -> list[str]:
    """Plain-English findings built from the computed shifts (no hand-typed numbers)."""
    assumptions = [s for s in summary_rows if s["assumption"] != "evidence"]
    removals = sorted((s for s in summary_rows if s["scenario"].startswith("remove")), key=lambda s: -s["max_tv"])
    strongest, weakest = assumptions[0], assumptions[-1]
    lines = [
        "**Findings (computed):**",
        "",
        f"* Most influential assumption: **{strongest['assumption']}**. Over the sweep it moves "
        f"{strongest['most_affected_person']}'s gene posterior by up to {strongest['max_tv']:.3f} (at value "
        f"{strongest['at_value']}), but {strongest['least_affected_person']} by at most "
        f"{strongest['least_affected_max_tv']:.3f}.",
        f"* Least influential of the swept assumptions: **{weakest['assumption']}** (largest shift "
        f"{weakest['max_tv']:.3f}, on {weakest['most_affected_person']}). "
        f"{weakest['least_affected_person']} is the least sensitive person to it "
        f"(at most {weakest['least_affected_max_tv']:.3f}).",
        f"* Most informative observation: removing **{removals[0]['scenario'].removeprefix('remove ')}** "
        f"shifts the posterior by up to {removals[0]['max_tv']:.3f}. Least informative: removing "
        f"**{removals[-1]['scenario'].removeprefix('remove ')}** shifts it by at most {removals[-1]['max_tv']:.3f}, "
        "so the posterior is nearly insensitive to that observation.",
        "* Every posterior above is correctly normalised, yet the numbers change with the assumptions. "
        "Normalisation only guarantees a coherent distribution *given the model*; it says nothing about "
        "whether the model's priors and likelihoods are right.",
        "",
    ]
    return lines


def write_cv_evidence(result, validation, exact_rows, max_closed_error, max_inheritance_error, max_exact_error,
                      max_published_error, max_reasoning_error, reasoning, all_rows, summary_rows) -> None:
    evidence = {
        "project": "bayesian-genetic-inference",
        "reproduce_command": COMMAND,
        "test_command": "uv run pytest",
        "disclaimer": "Educational model with CS50 illustrative probabilities; synthetic data; not medical.",
        "deterministic": True,
        "metrics": [
            {"metric": "configurations_evaluated_8_person", "value": result.configurations_evaluated,
             "calculation": "counter incremented once per enumerated configuration in enumerate_posteriors",
             "design_value": "3^8 * 2^5", "output_file": "reports/posteriors_8_person.json",
             "source": "src/bayesian_genetic_inference/enumeration.py"},
            {"metric": "closed_form_checks", "value": count_checks(validation["closed_form"]),
             "scalar_comparisons": len(validation["closed_form"]),
             "calculation": "3 founder priors, 6 trait likelihoods, 1 spec joint probability, 2 founder posteriors",
             "output_file": "reports/validation_table.csv", "source": "src/bayesian_genetic_inference/validation.py"},
            {"metric": "inheritance_checks", "value": count_checks(validation["inheritance"]),
             "scalar_comparisons": len(validation["inheritance"]),
             "calculation": "all 9 (mother, father) gene-count pairs vs enumeration of transmission outcomes",
             "output_file": "reports/validation_table.csv", "source": "src/bayesian_genetic_inference/validation.py"},
            {"metric": "max_abs_error_closed_form", "value": max_closed_error,
             "calculation": "max |expected - calculated| over closed-form comparisons",
             "output_file": "reports/validation_table.csv"},
            {"metric": "max_abs_error_inheritance", "value": max_inheritance_error,
             "calculation": "max |expected - calculated| over the 27 inheritance comparisons",
             "output_file": "reports/validation_table.csv"},
            {"metric": "max_abs_error_closed_form_and_inheritance", "value": max_exact_error,
             "calculation": "max over both sets above", "output_file": "reports/validation_table.csv"},
            {"metric": "spec_joint_probability_calculated",
             "value": next(r.calculated for r in validation["closed_form"] if r.check_id == "C10"),
             "expected": 0.0026643247488, "output_file": "reports/validation_table.csv"},
            {"metric": "max_abs_error_vs_published_cs50_output", "value": max_published_error,
             "calculation": "published values rounded to 4 dp, so errors up to 5e-5 are rounding",
             "output_file": "reports/validation_table.csv"},
            {"metric": "reasoning_checks", "value": count_checks(reasoning),
             "max_abs_error_equality_identities": max_reasoning_error,
             "all_passed": all(r.passed for r in reasoning), "output_file": "reports/validation_table.csv"},
            {"metric": "all_validation_checks_passed", "value": all(r.passed for r in all_rows),
             "output_file": "reports/validation_table.csv"},
            {"metric": "most_influential_assumption", "value": next(s for s in summary_rows
                                                                    if s["assumption"] != "evidence"),
             "calculation": "largest total-variation shift of any person's gene posterior over the sweep",
             "output_file": "reports/sensitivity_summary.csv"},
            {"metric": "runtime_and_memory", "value": "see output_file (runtime metadata, varies by run)",
             "calculation": "median wall-clock of repeated enumerate_posteriors calls; tracemalloc peak",
             "output_file": "reports/runtime_metadata.json"},
        ],
    }
    (REPORTS / "cv_evidence.json").write_text(json.dumps(evidence, indent=1) + "\n", encoding="utf-8")


def write_runtime_reports(timings: list[dict]) -> None:
    largest = timings[-1]
    per_config = largest["median_seconds"] / largest["configurations"]
    extrapolation = []
    for people in (10, 12, 14, 16, 20):
        configurations = 6 ** people          # worst case: every trait unobserved
        extrapolation.append({"people_all_traits_unknown": people, "configurations": configurations,
                              "estimated_seconds": configurations * per_config})
    metadata = {
        "note": "Runtime metadata: varies between runs and machines. Not part of the deterministic outputs.",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
        "benchmarks": timings,
        "extrapolation_basis": f"median seconds per configuration measured at {largest['case']}",
        "extrapolation": extrapolation,
    }
    (REPORTS / "runtime_metadata.json").write_text(json.dumps(metadata, indent=1) + "\n", encoding="utf-8")

    lines = ["# Runtime report (runtime metadata - varies between runs)", "",
             f"Machine: {metadata['platform']}, Python {metadata['python']}. Generated {metadata['generated_utc']}.",
             "", "## Measured", "",
             markdown_table(["case", "people", "configurations", "median s", "runs", "µs per configuration",
                             "peak memory (KiB)"],
                            [[t["case"], t["people"], f"{t['configurations']:,}", f"{t['median_seconds']:.4f}",
                              t["repeats"], f"{t['microseconds_per_configuration']:.2f}",
                              f"{t['peak_memory_bytes'] / 1024:.1f}"] for t in timings]),
             "", "Peak memory is the tracemalloc peak of one separate run. Enumeration streams over "
             "configurations and keeps only per-person running totals, so memory stays small while "
             "time grows with the configuration count.", "",
             "## Extrapolated (NOT measured)", "",
             f"Basis: {metadata['extrapolation_basis']} ({per_config * 1e6:.2f} µs). Worst case 6^n "
             "configurations (every trait unobserved):", "",
             markdown_table(["people", "configurations", "estimated time"],
                            [[e["people_all_traits_unknown"], f"{e['configurations']:.3e}",
                              _human_seconds(e["estimated_seconds"])] for e in extrapolation])]
    (REPORTS / "runtime_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _human_seconds(seconds: float) -> str:
    for unit, size in (("years", 365 * 86400), ("days", 86400), ("hours", 3600), ("minutes", 60)):
        if seconds >= size:
            return f"{seconds / size:.1f} {unit}"
    return f"{seconds:.1f} s"


if __name__ == "__main__":
    main()
