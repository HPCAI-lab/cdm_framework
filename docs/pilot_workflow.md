# Processing collected transcripts

The path from a folder of student transcripts to scored items, and what each
step is allowed to claim.

> **Before anything else.** Transcripts collected during regular instruction
> are not automatically available for research use. Whether analysing them
> requires review board approval, and whether approval can cover data already
> collected, is the board's determination and not the research team's. Settle
> that before running any step below.

## 1. Convert to the turn schema

```bash
python -m cdm.ingest --format dir --input transcripts/ \
    --model "gpt-4o-2024-11-20" --roster roster.csv --dry-run
```

Adapters exist for ChatGPT exports (`openai`), Claude exports (`anthropic`),
speaker-labelled text or Markdown (`text`, `dir`) and spreadsheets (`csv`).
None of them is a specification of your export; run `--dry-run` and read the
report before writing anything.

Two lines of that report matter more than the rest. The student-to-assistant
turn ratio should sit near 1.0, and a ratio far from it means the parser has
merged turns or lost a speaker. The model-recoverable count should equal the
turn count, because model version is a confound and a corpus that cannot
recover it per turn will not support comparison against a later cohort.

`--roster` maps sessions to students and pseudonymises by default, writing the
crosswalk to a separate file. Keep that file out of the repository.

## 2. Score and derive thresholds

```bash
python run_pilot.py --logs student_logs.jsonl --profile hpc --outdir results_pilot
```

Choose the lexicon profile deliberately. The `general` profile looks for
patients, clients and citations; the `hpc` profile looks for race conditions,
profiler output, correctness against a serial baseline, and academic-integrity
attribution. Running HPC coursework under the general profile will report
that students never consider ethics, which is a fact about the lexicon.

Read `flat_features.csv` before the scores. A feature that never fires has two
possible causes and they call for opposite responses: the behaviour is
genuinely rare in this cohort, which is a finding, or the lexicon was written
for another domain, which is a defect. The `lexicon_suspect` column flags the
families where the second is likely.

## 3. Understand what the thresholds mean

Two quantities are both called thresholds and they are not the same.

The **scale** of a feature is the range a real corpus produces. Transcripts
alone establish it, and `feature_distributions.csv` reports it. The shipped a
priori cutpoints are guesses about this scale that no corpus is obliged to
honour.

The **cutpoints** are where that scale is divided so a 4 means what a human
coder's 4 means. Only human scores on the same sessions establish them.
Norm-referencing produces spread and rank order inside one sample and says
nothing about what a score means outside it.

`thresholds.json` records which of the two you have, how many cases it rests
on, the lexicon profile, and every caveat that applies. A cutpoint file
without that record is unusable six months later because nobody can tell
whether the numbers were fitted, norm-referenced, or guessed.

## 4. Calibrate once coding exists

```bash
python run_pilot.py --logs student_logs.jsonl --profile hpc \
    --human human_items.csv --outdir results_pilot
```

`human_items.csv` carries a `student` column and the twenty-five item columns
named `<Dimension>_i<n>`. Calibration fits cutpoints so the automated score
distribution matches the human one. That is not agreement. Agreement is
reported separately, per item, under the same weighted kappa and the same 0.80
threshold the human coders are held to, and an item that cannot meet the
standard demanded of a coder has not been automated.

Sixty or more double-coded sessions is a working minimum. Below twenty the
quantile estimates are unstable at the tails, which is exactly where scores of
1 and 5 are decided.

The same run reports the length baseline: the same items scored from the
amount the student wrote, nothing else. If that matches the twenty-five
feature scorer, the feature engineering is not doing the work.

## 5. Design-time evidence for the paper

```bash
python run_controls.py --outdir results_controls --reps 500 --sweep-reps 50
```

Produces the negative controls table, the power curve, the fit-by-sample-size
sweep with replication bands and its figure, and the ordinal reliability
table, each also written as a LaTeX fragment. Use `--quick` for a smoke test;
the replication counts in a smoke test are too small to report.

The WLSMV re-estimation leaves Python. `results_controls/lavaan/cfa_wlsmv.R`
reads the exported item data and reports scaled fit indices beside the
maximum-likelihood ones, which is what separates the ordinality component of
the small-sample misfit from the identification component.
