# Bayesian Genetic Inference

Exact Bayesian inference of hidden gene counts and traits across a family pedigree. It enumerates
every configuration of the latent variables, conditions on observed traits, marginalises and
normalises. No sampling or approximation is used.

> **Educational probability model, not a medical or clinical tool.** The model and its
> probabilities follow the CS50 AI *Heredity* project
> (https://cs50.harvard.edu/ai/projects/2/heredity/). The numbers are the course's
> **illustrative assumptions, not medically validated estimates**. All pedigrees are synthetic.

## What it shows

- **Exact enumeration** over 3^n gene assignments × 2^k unobserved traits. On the 8-person
  pedigree with 3 observed traits that is 3^8 × 2^5 = 209,952 configurations. The count is
  recorded by a counter during enumeration, not computed from the formula.
- **Joint probability**, **Bayesian conditioning**, **marginalisation** and **normalisation**.
- **Independent validation:**
  - 12 closed-form checks, including the published spec example 0.0026643247488;
  - all 9 parent gene-count inheritance cases;
  - the course's published output;
  - 8 groups of probability identities and invariants.

  None of the reference values call the engine.
- **Computational scaling:** measured runtime and peak memory from 1 to 9 people (10 with
  `--extended`), plus a labelled extrapolation.
- **Sensitivity:** the founder prior, mutation probability and 0-copy trait rate are swept,
  and each observation is removed or flipped, all on identical evidence.

## Quick start

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the locked
dependencies (matplotlib; pytest for tests).

```bash
cd bayesian-genetic-inference
uv sync
uv run python scripts/run_experiment.py      # about 1 minute; add --extended for a 10-person benchmark
uv run pytest
```

## Outputs (`reports/`)

| File | Content | Deterministic? |
|---|---|---|
| `results_summary.md` | posteriors, validation tables, scaling counts, sensitivity findings | yes |
| `posteriors_8_person.csv` / `.json`, `posteriors_small_family.csv` | posterior distributions | yes |
| `validation_table.csv` | every check: expected, calculated, absolute error, formula | yes |
| `scaling_configurations.csv` | configurations per pedigree size | yes |
| `sensitivity_*.csv` | assumption sweeps, evidence ablation, summary | yes |
| `figures/posterior_gene_counts.png` | posterior gene-count heat map | yes |
| `cv_evidence.json` | source, calculation, output file and command for each reported metric | yes |
| `runtime_metadata.json`, `runtime_report.md` | timings, peak memory, machine, extrapolation | **no** (runtime metadata) |

Re-running the experiment reproduces every deterministic file byte for byte. Only the two
runtime files change.

## Layout

```
src/bayesian_genetic_inference/
  model.py         probability model: priors, inheritance, trait likelihoods, joint probability
  pedigree.py      CSV loading and pedigree validation (missing/single parents, cycles, ...)
  enumeration.py   exact enumeration, marginalisation, normalisation
  validation.py    independent closed-form references and reasoning checks
  analysis.py      scaling benchmark and sensitivity analysis
data/              synthetic pedigrees (see data/README.md)
scripts/run_experiment.py
tests/             pytest suite
docs/              model_and_validation.md, interview_preparation.md
```