# Cognitive Diversity Metrics (CDM) — Analysis Pipeline

A reproducible pipeline for **validating the Cognitive Diversity Metrics (CDM)
instrument**: a five-dimension rubric for assessing how students engage
cognitively with AI-assisted work. The repository provides the full
measurement-validation workflow — a declared generative model, ordinal
inter-rater agreement statistics, and a complete psychometric analysis — together
with a **simulated demonstration dataset** so the machinery can be run and
inspected before any real data are collected.

> [!IMPORTANT]
> **This repository contains simulated data only.** Every statistic produced here
> emerges from a declared generative model plus sampling noise; the structural
> assumptions are inputs, not results. Outputs are **a demonstration that the
> pipeline is correct and ready**, not empirical findings. All artifacts are
> stamped `SIMULATED`. See [`docs/methodology.md`](docs/methodology.md) for the
> full integrity statement.

---

## The CDM instrument

CDM scores five dimensions of AI-assisted cognitive work:

| Dimension | What it captures |
|---|---|
| Problem Decomposition | Breaking a task into tractable sub-problems |
| Prompt Engineering | Eliciting useful output from an AI system |
| Output Validation | Checking and correcting AI output |
| Ethical Integration | Applying ethical judgment to AI use |
| Collaborative Sensemaking | Building shared understanding with AI in the loop |

Each dimension is measured by a five-item rubric scored 1–5, coded by three expert
raters. The validation study design is a convergent mixed-methods pilot across
three disciplines (business, health sciences, humanities).

## What the pipeline does

Given a dataset (simulated here, real once collected), the pipeline computes a
complete psychometric validation:

- **Group differences** — MANOVA of discipline against the five dimensions, with
  partial eta-squared effect sizes.
- **Internal consistency** — Cronbach's alpha per dimension.
- **Inter-rater reliability** — quadratic-weighted Cohen's kappa and
  Krippendorff's alpha, the ordinal-appropriate statistics for rubric scores.
- **Construct validity** — confirmatory factor analysis fit across sample sizes.
- **Criterion validity** — correlations against external measures.
- **Test-retest reliability** — intraclass correlation across a re-administration.

## Repository structure

```
cdm-framework/
├── cdm/                        # the package
│   ├── model_core.py           # generative model — the ONLY file to edit for real data
│   ├── agreement.py            # ordinal inter-rater statistics
│   └── pipeline.py             # full analysis pipeline + output writer
├── docs/
│   ├── methodology.md          # how it works + the integrity statement (read first)
│   ├── results_demonstration.md# narrative of the simulated demonstration results
│   └── data_dictionary.md      # every column of the dataset, documented
├── examples/
│   └── expected_output/        # a reference run (report, dataset, figures)
├── tests/
│   └── test_pipeline.py        # sanity tests
├── run_demo.py                 # convenience entry point
├── requirements.txt
├── Makefile
├── CITATION.cff
└── LICENSE
```

## Installation

Requires Python 3.9 or newer.

```bash
git clone <repository-url>
cd cdm-framework
python -m pip install -r requirements.txt
```

## Quickstart

Run the demonstration (writes stamped outputs to `results/`):

```bash
python -m cdm.pipeline --outdir results
# or, equivalently:
python run_demo.py --outdir results
# or:
make demo
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--n-per-group` | `10` | Simulated students per discipline (total = 3 × this) |
| `--seed` | `20260617` | Random seed for reproducibility |
| `--outdir` | `.` | Directory for the report, dataset, and figures |

### Outputs

Each run writes:

- `cdm_pipeline_report.txt` — the full stamped statistical report
- `cdm_simulated_dataset.csv` — the generated dataset (`DATA_PROVENANCE = SIMULATED`)
- `fig_dimension_means.png`, `fig_dimension_correlations.png` — figures

A reference copy of these is in [`examples/expected_output/`](examples/expected_output/).

## Demonstration results (simulated)

On the pilot design (N = 30), the pipeline recovers the injected discipline
pattern (MANOVA), and the emergent reliability and validity statistics fall within
their intended ranges. The most useful result is the **confirmatory factor
analysis as a function of sample size** — the same model, varying only N:

| Sample size | Comparative Fit Index |
|---|---|
| N = 30 (pilot) | 0.48 — underpowered |
| N = 120 | 0.92 — acceptable |
| N = 240 | 0.99 — clean fit |

This provides the quantitative case for scaling the confirmed study to at least
120 participants. Full narrative in
[`docs/results_demonstration.md`](docs/results_demonstration.md). As above, these
are pipeline-demonstration outputs on simulated data, not empirical findings.

## Adapting to real data

The analysis code is designed to run unchanged on real data. Only
`cdm/model_core.py` changes: replace `simulate()` with a loader that returns the
same columns from the real, IRB-approved collection. Real data collection is out
of scope for this repository and requires Institutional Review Board approval.

## Testing

```bash
python -m pytest -q      # or: make test
```

## Data availability

This repository distributes **only simulated data**. No participant data —
real or identifiable — are included, and none should ever be committed (see
`.gitignore`).

## Citation

If you use this pipeline, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## Team

Developed in the High-Performance Computing and AI (HPC-AI) Laboratory,
The University of Texas at Tyler.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
