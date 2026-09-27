# Cognitive Diversity Metrics (CDM) — Analysis Pipeline

Software for building and validating **CDM**, a five-dimension rubric that
measures how students conduct AI-assisted work — here, writing, debugging and
optimizing parallel and high-performance code (HPC/PDC coursework) with a
conversational model in the loop. The repository carries three things:

1. **A transcript path.** Converts exported student–AI transcripts into a turn
   log, computes 25 features, scores the 25 rubric items, and calibrates those
   scores against human coding.
2. **A psychometric report.** Reliability, inter-coder agreement, factor
   structure, criterion correlations and test-retest stability from a table of
   item scores.
3. **Design-time evidence.** A declared generative model plus negative controls,
   power curves and a fit-by-sample-size sweep, run before any real data exist.

> [!IMPORTANT]
> **This repository contains simulated data only.** Every statistic produced by
> the demonstrations emerges from a declared generative model plus sampling
> noise; the structural assumptions are inputs, not results. Simulated
> artifacts are stamped `SIMULATED`, and automated scores carry their
> calibration state (uncalibrated, norm-referenced, or calibrated). See
> [`docs/methodology.md`](docs/methodology.md) for the integrity statement.

---

## The CDM instrument

Five dimensions, five items each, every item scored 1–5 on a session (one
student's continuous work on one task). The coder manual, with general and
HPC/PDC anchors for every item, is [`docs/rubric.md`](docs/rubric.md).

| Dimension | Decision it concerns | HPC/PDC instance |
|---|---|---|
| Problem Decomposition | What to hand the model, in what units and order | Decomposing a routine into parallelizable units |
| Prompt Engineering | How to specify usable work | Specifying an MPI/OpenMP/CUDA task precisely |
| Output Validation | Whether and how to check output | Testing generated code for correctness and performance |
| Ethical Integration | What may be sent and disclosed | Attribution and AI-use disclosure; handling credentials and shared resources |
| Collaborative Sensemaking | Reconciling output with other sources | Reconciling output with profiler traces and documentation |

Two items, OV4 (calibrated skepticism) and CS2 (explanation quality), are
ideal-point items: the top score is a middle band, not the maximum. They are
excluded from the dominance factor model and scored with an interior-optimum
mapping.

## Status

- **Feasibility deployment.** One section of 15 students in Computer
  Engineering at The University of Texas at Tyler, debugging a matrix
  multiplication implementation with a conversational model. Transcripts were
  collected; human coding is not yet complete, so no measurement statistic is
  reported from it.
- **Design-time evaluation.** Complete on simulated data (see below).
- **Revised pilot protocol.** Crossed common/domain task design, several
  sections per cohort, silent-log vs think-aloud conditions; pending review
  board approval.

Transcripts collected during regular instruction are not automatically
available for research use. Settle review-board status before analysing them
(see [`docs/pilot_workflow.md`](docs/pilot_workflow.md)).

## Repository structure

```
cdm_framework/
├── cdm/
│   ├── ingest.py          # ChatGPT / Claude / text / CSV / directory exports -> turn log
│   ├── logs.py            # turn-log loading, validation, PII screen
│   ├── lexicons.py        # marker lexicons as profiles: general, hpc
│   ├── features.py        # 25 session features, one per rubric item
│   ├── autoscore.py       # features -> item scores; calibration; agreement
│   ├── thresholds.py      # feature distributions, flat-feature diagnosis, threshold provenance
│   ├── agreement.py       # weighted kappa, Krippendorff's alpha, ICC
│   ├── ordinal.py         # polychoric correlations, ordinal alpha/omega, lavaan WLSMV script
│   ├── model_core.py      # declared generative model (simulated item scores)
│   ├── simulate_logs.py   # simulated transcripts driven by the same model
│   ├── pipeline.py        # psychometric report + figures
│   └── controls.py        # negative controls, power curve, fit-by-N sweep
├── docs/
│   ├── rubric.md                  # coder manual (25 items, general + HPC/PDC anchors)
│   ├── pilot_workflow.md          # processing collected transcripts, step by step
│   ├── methodology.md             # generative model and integrity statement
│   ├── data_dictionary.md         # columns of the item-score dataset
│   └── results_demonstration.md   # narrative of the simulated demonstration
├── examples/expected_output/      # reference run of the psychometric demo
├── tests/                         # test_pipeline, test_log_path, test_pilot_path
├── run_pilot.py      # collected transcripts -> scores, thresholds, agreement
├── run_controls.py   # design-time evidence, with LaTeX fragments for the paper
├── run_log_demo.py   # transcript path on simulated logs
├── run_demo.py       # psychometric report on simulated item scores
├── Makefile, requirements.txt, requirements-dev.txt, CITATION.cff, LICENSE
```

## Installation

Requires Python 3.9 or newer.

```bash
git clone <repository-url>
cd cdm_framework
python -m pip install -r requirements-dev.txt
python -m pytest -q          # or: make test
```

## Processing collected transcripts

The main path for real data. No code changes are needed.

```bash
# 1. Convert exports to the turn schema. Read the dry-run report first.
python -m cdm.ingest --format dir --input transcripts/ \
    --model "gpt-4o-2024-11-20" --roster roster.csv --dry-run
python -m cdm.ingest --format dir --input transcripts/ \
    --model "gpt-4o-2024-11-20" --roster roster.csv --out student_logs.jsonl

# 2. Score, derive norm-referenced thresholds, diagnose flat features.
python run_pilot.py --logs student_logs.jsonl --profile hpc --outdir results_pilot

# 3. Once a double-coded subset exists, calibrate and report agreement.
python run_pilot.py --logs student_logs.jsonl --profile hpc \
    --human human_items.csv --outdir results_pilot
```

- Every turn needs `session_id`, `student`, `role`, `turn` and `content`, plus a
  pinned `model` identifier; model version is a confound, so a corpus without
  it cannot support cohort comparison. Use `--model` if the export lacks one.
- `--roster` pseudonymises students and writes a crosswalk file. Keep that file
  **outside** the repository; it maps pseudonyms back to real students.
- Use `--profile hpc` for HPC/PDC coursework. The `general` lexicon looks for
  patients, clients and citations and will report flat ethics features on HPC
  transcripts.
- Without `--human`, scores are norm-referenced: they rank students within
  the corpus and mean nothing outside it. Thresholds do not transfer between
  corpora, so derivation is a required step on every new corpus.
- An item counts as automated only if it reaches weighted κ ≥ 0.80 against
  human coding, the same standard applied to the coders.

Full walkthrough: [`docs/pilot_workflow.md`](docs/pilot_workflow.md).

## Running the psychometric report on real item scores

`cdm/pipeline.py` currently runs only on simulated data from
`model_core.simulate()`; there is no command-line option to load real item
scores yet. To run it on coded data, replace `simulate()` with a loader that
returns the same three frames (item scores, per-coder scores, think-aloud
codes) with the columns documented in
[`docs/data_dictionary.md`](docs/data_dictionary.md). The simulated groups are
three placeholder disciplines (Business, HealthSciences, Humanities) inherited
from the original cross-discipline design; the grouping variable for the
revised HPC protocol is defined by the approved pilot design.

## Simulated demonstrations

```bash
make demo                        # psychometric report -> results/
python run_log_demo.py           # transcript path on simulated logs -> results_logs/
make controls-quick              # smoke test of the design-time controls
make controls                    # full controls: 500 reps, 50 sweep reps per N
```

The psychometric demo reproduces [`examples/expected_output/`](examples/expected_output/)
exactly. Its confirmatory factor analysis at pilot scale (N = 30) gives
CFI 0.477 and RMSEA 0.192 (not estimable), against CFI 0.991 and RMSEA 0.018
at N = 240. `run_controls.py` gives the full fit-by-N curve with replication
bands, which is the version to report.

The single-factor control compares the five-factor model against a one-factor
model using the χ² difference test and χ²-based AIC/BIC, not CFI: the
five-factor model nests the one-factor model, so both fit one-factor data
equally well and a CFI comparison cannot discriminate them.

None of these outputs is evidence about students.

## Data availability

This repository distributes **only simulated data**. No participant data — real
or identifiable — are included, and none should ever be committed (see
`.gitignore`).

## Citation

If you use this pipeline, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## Team

Developed in the High-Performance Computing and AI (HPC-AI) Laboratory,
The University of Texas at Tyler.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
