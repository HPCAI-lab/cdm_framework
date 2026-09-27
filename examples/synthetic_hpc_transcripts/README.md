# Synthetic HPC/PDC transcripts for testing the CDM pipeline

> [!IMPORTANT]
> **Synthetic test data.** All 120 transcripts were assembled from templates by
> `generate_hpc_transcripts.py`. No student wrote any of it. They exist to
> exercise the pipeline and are not evidence about students or about the
> validity of the instrument. The login names, hostnames and account IDs in
> them are fictitious (`example.edu` / `example.com` domains).

## Design

- **120 sessions**, one per synthetic student (`student` = 1–120), 20 per task:

  | Files | Task |
  |---|---|
  | 001–020 | MPI row-block matrix multiplication (N not divisible by ranks) |
  | 021–040 | OpenMP matrix multiplication with a race on `sum`, `j`, `k` |
  | 041–060 | CUDA tiled matmul (missing `__syncthreads`, partial tiles) |
  | 061–080 | MPI halo exchange that deadlocks at large messages |
  | 081–100 | OpenMP partial sums with false sharing |
  | 101–120 | Strong-scaling analysis of an OpenMP matmul (bandwidth plateau) |

- Each student has a latent level (1–5) per CDM dimension, correlated at 0.4.
  Each of the 25 items gets a target score from its dimension's level plus item
  noise, and the transcript is rendered so each item shows the behaviour its
  target describes in `docs/rubric.md`.
- **OV4 and CS2 are ideal-point.** OV4 follows one of four patterns: accept
  everything, uniform caution, proportional scrutiny (top score), or
  over-scrutiny (scored low). CS2 follows one of four restatement patterns:
  verbatim copy, too loose, close paraphrase, or own words (top score).
- **Assistant behaviour varies** so the jointness and validation rules are
  exercised. In 61% of sessions the first answer is flawed; 25% of sessions have
  the assistant raise the course AI-use policy itself; every session includes
  one unsupported performance claim and one overgeneralized claim for the
  student to accept or challenge.
- 42% are team sessions (CS5 is scorable); in the rest CS5 is `NA`.
- 49% of sessions paste code in ``` fences; the rest paste it bare.
- 44 sessions contain planted login/email identifiers (low EI2 scores), for
  testing the PII screen.

## Files

| File | Contents |
|---|---|
| `transcripts/NNN_<task>.txt` | 120 speaker-labelled transcripts (`Student:` / `Assistant:`). Leading digits = student id. |
| `hpc_test_logs.jsonl` | The same 2,028 turns already in the CDM turn schema. |
| `design_key.csv` | Per session: task, latent levels, OV4/CS2 pattern, and which conditions were planted. |
| `generator_item_labels.csv` | The generator's **target** score for each of the 25 items (`NA` for CS5 in solo sessions). |
| `generate_hpc_transcripts.py`, `tasks.py` | Generator and task content. `python generate_hpc_transcripts.py --out <dir>` reproduces the set exactly (seed 20260924). |

`generator_item_labels.csv` is **not human coding**. The transcripts were
written to match those targets, so agreement against them is partly circular:
it shows whether the features pick up the rubric behaviours as the generator
phrased them, not whether they would pick them up in real student writing.

## Using them

```bash
# from the cdm_framework root
python -m cdm.ingest --format dir --input examples/synthetic_hpc_transcripts/transcripts \
    --model synthetic-template-assistant-v1 --dry-run
python run_pilot.py --logs examples/synthetic_hpc_transcripts/hpc_test_logs.jsonl \
    --profile hpc --outdir results_synthetic
```

To exercise calibration, pass the generator targets as `--human`. The current
pipeline cannot read `NA`, so replace the CS5 `NA` values first (for example
with 1) and treat that item's result accordingly.

## Limits

The phrasing comes from a fixed set of templates, so the language is far more
regular than real student writing, and the assistant turns are templated rather
than produced by a model. The `model` field is a placeholder. Use this set to
test the machinery, never to set thresholds for real data.
