# Methodology and Integrity Statement

This document explains what the pipeline does, how the demonstration data are
generated, and the rule that keeps the demonstration honest. Read it before
presenting any output.

## The integrity rule

The pipeline is shipped with a **simulated** dataset produced by a declared
generative model. The distinction that governs every use of this repository:

- **Structural assumptions are inputs.** The discipline-by-dimension means, the
  item and rater noise, the factor-correlation structure, and the criterion and
  retest correlations are set by hand in `cdm/model_core.py`, on grounds of
  measurement-theoretic realism.
- **The reported statistics are outputs.** Internal consistency, inter-rater
  reliability, factor structure, criterion validity, and test-retest reliability
  are *not* set; they emerge from the generative model plus sampling noise.

Consequently, the outputs demonstrate that the **pipeline computes each statistic
correctly and that, under the agreed design, the instrument behaves as intended**.
They are **not** empirical findings. Every generated artifact is stamped
`SIMULATED`, and the dataset carries a `DATA_PROVENANCE = SIMULATED` column.

The honest one-line framing: *a validated analysis pipeline, demonstrated on
simulated data, ready for the pilot.*

### One caution worth stating explicitly

The inter-dimension correlation figure is the easiest artifact to over-read. Its
structure is an **input assumption** to the generator, so the figure shows only
that the pipeline recovers and visualizes the injected correlations correctly. It
should never be described as evidence that the CDM dimensions "are related but
distinct" — on simulated data that would be circular.

## How the synthetic data are generated

For each simulated student, the model in `cdm/model_core.py`:

1. Draws a latent ability on each of the five CDM dimensions from the
   discipline-specific means and standard deviations, with the declared
   inter-dimension correlation structure.
2. Expands each dimension into a five-item rubric, adding item difficulty and
   item-specific noise (which drives internal consistency).
3. Simulates three expert coders with systematic leniency or severity plus random
   error (which drives inter-rater reliability).
4. Generates external criterion measures (a discipline grade-point average and an
   AI-task grade) and a re-administration, each correlated with latent ability at
   the declared strength (which drives criterion and test-retest statistics).

## What the pipeline computes

Running `cdm/pipeline.py` on that dataset produces, in order:

- **MANOVA** of discipline against the five dimensions, with partial eta-squared
  effect sizes per dimension.
- **Cronbach's alpha** for internal consistency of each dimension.
- **Quadratic-weighted Cohen's kappa** and **Krippendorff's alpha** for
  inter-rater reliability, the ordinal-appropriate statistics for rubric scores.
- **Confirmatory factor analysis** fit indices at several sample sizes.
- **Criterion validity** correlations against the external measures.
- **Test-retest reliability** via the intraclass correlation.

## Moving to real data

Only one file changes: `cdm/model_core.py`. Replace `simulate()` with a loader
that returns the same columns from the real, IRB-approved collection. The analysis
code in `cdm/pipeline.py` and `cdm/agreement.py` is intended to run unchanged.
Real data collection requires Institutional Review Board approval and is out of
scope for this repository, which distributes only simulated data.
