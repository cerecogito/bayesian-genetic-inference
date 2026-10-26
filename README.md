# Bayesian Genetic Inference

I wanted to see exact Bayesian inference actually work end to end, with no sampling and no
approximations. So I built an engine that works out, for every person in a family tree, the
probability that they carry 0, 1 or 2 copies of a gene and whether they show a trait, given
only a few observed traits.

It's built on the model from Harvard's CS50 AI *Heredity* project. I structured it my own way and
pushed it further with validation, scaling and sensitivity experiments.

> **This is an educational probability model, not a medical tool.** The probabilities are the
> course's illustrative numbers, not real medical estimates, and every family here is made up.

## How it works

Every person has a hidden gene count (0, 1 or 2) and a trait (yes or no):

- Founders with no recorded parents draw their gene count from a prior.
- Children inherit one copy from each parent independently. A parent passes the gene on with
  probability 0.01 if they have no copies (mutation only), 0.5 with one copy, and 0.99 with two.
- The chance of showing the trait depends only on a person's own gene count.

To get the posteriors I enumerate **every** combination of hidden gene counts and unknown traits.
I compute the joint probability of each combination, add them up per person (marginalisation),
and divide by the total (normalisation).

## Results

On my 8-person synthetic family with 3 observed traits:

- The engine evaluates exactly **3^8 × 2^5 = 209,952 configurations**, counted while it runs
  rather than taken from the formula, in about half a second on my laptop.
- It matches the course's published worked example (joint probability **0.0026643247488**). It
  also matches the course's printed output for its example family, to the 4 decimals they show.
- It passes 12 closed-form checks and all 9 parent gene-count inheritance cases, with a max
  absolute error of **8.7e-19**. The expected values come from formulas that never call the
  engine, and a test proves that by breaking the engine on purpose.
- It passes 8 groups of reasoning checks: total probability, conditional independence,
  d-separation, mother/father symmetry, and the fact that seeing a child with the trait raises
  each parent's chance of carrying the gene.

**Scaling:** each extra person multiplies the work by 3, or by 6 if their trait is unknown. I
measured it rather than guessing. Time per configuration stays at a few microseconds:

- 9 people take about 4 seconds;
- memory stays under 9 KiB, because nothing is stored;
- 20 people with no observed traits would take centuries. That's why real pedigree software uses
  smarter algorithms such as variable elimination.

**Sensitivity:** the most interesting part for me.

- The assumption that matters most is the chance of showing the trait **without** the gene.
  Moving it from 0.01 to 0.10 shifts one person's gene posterior by 0.59 (on a 0 to 1 scale),
  because the observed trait can then be explained without the gene.
- The founder prior matters much less: at most 0.17.
- Removing one person's "no trait" observation barely changes anything (at most 0.08). The prior
  already says they almost certainly don't carry the gene.

The posterior is always perfectly normalised, but that only means it's consistent with the model,
not that the model is right.

## Running it

You need [uv](https://docs.astral.sh/uv/), which sets up Python 3.12 and the locked dependencies.

```bash
cd bayesian-genetic-inference
uv sync
uv run python scripts/run_experiment.py      # about a minute; --extended adds a 10-person benchmark
uv run pytest                                # 38 tests
```

The experiment writes everything to `reports/`:
- posterior CSV and JSON files;
- a full validation table with expected values, calculated values and errors;
- the scaling and sensitivity tables;
- a heat map of the posteriors;
- `results_summary.md`;
- `cv_evidence.json`, which records where every number comes from.

Re-running reproduces those files byte for byte. Only the timing files change.

## Layout

```
src/bayesian_genetic_inference/
  model.py         priors, inheritance, trait likelihoods, joint probability
  pedigree.py      loading families from CSV and validating them (missing parents, cycles, ...)
  enumeration.py   exact enumeration, marginalisation, normalisation
  validation.py    independent closed-form checks and reasoning checks
  analysis.py      scaling benchmark and sensitivity analysis
data/              the synthetic families
scripts/run_experiment.py
tests/
docs/model_and_validation.md   the maths and every validation formula
```
