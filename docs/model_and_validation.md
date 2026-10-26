# Model, algorithm and validation

## Source

The model follows the CS50 AI *Heredity* project specification
(https://cs50.harvard.edu/ai/projects/2/heredity/). This is an independently structured
implementation, not a course solution. The probabilities are the course's **illustrative
assumptions, not medically validated estimates**, and every pedigree here is synthetic.

## Bayesian network

For each person $i$ there is a hidden gene count $G_i \in \{0,1,2\}$ and a trait $T_i \in \{0,1\}$.

- Founder: $P(G_i = g) = \pi_g$, with $\pi = (0.96, 0.03, 0.01)$.
- Child of mother $M$ and father $F$: each parent independently passes one copy, with
  $t(0) = \mu$, $t(1) = \tfrac12$, $t(2) = 1-\mu$ and $\mu = 0.01$. With $p_m = t(G_M)$ and $p_f = t(G_F)$:
  $P(G_i=0\mid\cdot) = (1-p_m)(1-p_f)$,
  $P(G_i=1\mid\cdot) = p_m(1-p_f) + (1-p_m)p_f$,
  $P(G_i=2\mid\cdot) = p_m p_f$.
- Trait: $P(T_i = 1 \mid G_i = g) = \tau_g$, with $\tau = (0.01, 0.56, 0.65)$.

**Joint probability** of one complete assignment (the product follows the network):
$$P(\mathbf G, \mathbf T) = \prod_i P(G_i \mid G_{M(i)}, G_{F(i)})\, P(T_i \mid G_i).$$

## Exact inference

Let $E$ be the observed traits. Enumerate every assignment of all gene counts and every
**unobserved** trait. There are $3^n 2^k$ of them, and observed traits are fixed rather than
enumerated. Then:

- **Marginalise:** $P(G_i = g, E) = \sum_{\text{configs with } G_i = g} P(\mathbf G,\mathbf T)$.
- **Evidence:** $P(E) = \sum_{\text{all configs}} P(\mathbf G, \mathbf T)$.
- **Normalise:** $P(G_i = g \mid E) = P(G_i = g, E) / P(E)$, and the same for traits.

Implementation detail (`enumeration.py`): the gene factors depend only on the gene
assignment, so they are computed once per gene assignment and reused for its $2^k$ trait
assignments. Every configuration is still visited and counted. `tests/test_enumeration.py`
checks the result against a plain brute force over `model.joint_probability`.

If $P(E) = 0$ (impossible evidence under the chosen parameters) an `ImpossibleEvidenceError`
is raised instead of dividing by zero.

## Deliberate extensions beyond the course specification

- Configurable `ModelParameters` with validation, needed for the sensitivity study.
- Pedigree validation: duplicate names, missing parent, single parent, self-parent, cycles.
- Explicit configuration and zero-probability counters, and impossible-evidence handling.
- The evidence probability $P(E)$ is reported as well as the posteriors.
- Scaling benchmark and sensitivity analysis (`analysis.py`).

## Validation (reference values never call the engine)

Reference values in `validation.py` use only the published constants and explicit arithmetic.
`tests/test_validation.py` breaks every engine function and shows the reference functions
still run.

| Group | Checks | Reference formula |
|---|---|---|
| Founder prior | C01–C03 | $\pi_g$ as published; the engine value is a lone founder's posterior with no evidence |
| Trait likelihood | C04–C09 | $\tau_g$ and $1-\tau_g$ |
| Spec joint probability | C10 | $(0.96\cdot0.99)(0.01\cdot0.65)\big((0.99^2+0.01^2)\cdot0.44\big) = 0.0026643247488$. The fixture is checked against the spec family and event sets before use |
| Founder posterior | C11 (T=1), C12 (T=0) | $P(G=g\mid T) = \pi_g P(T\mid g) / \sum_h \pi_h P(T\mid h)$ |
| Inheritance | 9 parent pairs × 3 child counts | enumerate the four (mother passes?, father passes?) outcomes |
| Published output | 15 values | CS50's printed output for its 3-person family (4 dp; tolerance 5e-5) |
| Reasoning | R1–R8 | total probability; prior predictive; $P(T\mid E)=\sum_g P(G=g\mid E)\tau_g$; d-separation of an unrelated founder; mother/father symmetry; exchangeable founders; monotonicity (a child observed with the trait raises each parent's carrier probability) |

## Why a correct, normalised posterior can still be wrong

Normalisation guarantees the posterior is a coherent probability distribution *under the
model*. It cannot check whether the prior, the transmission model or the trait likelihoods
are true. The sensitivity study shows how much the answer moves when these assumptions move.
For example, the 0-copy trait rate controls how much an observed trait points to a gene at
all. `reports/results_summary.md` §5 lists the measured shifts, including the cases where the
posterior is nearly insensitive.
