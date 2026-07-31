# CDM Simulated Dataset — Data Dictionary

**File:** `cdm_simulated_dataset.csv`  ·  **Rows:** one per student (N = 30 at pilot scale)  ·  **Columns:** 41

> All values are **SIMULATED**. They are generated from the declared model in
> `model_core.py` (latent level → item → coder → round) and demonstrate the
> analysis pipeline; they are not empirical findings.

Every column exists because a specific pipeline statistic consumes it. The
groups below are organized by that purpose.

---

## Identity (cols 0–2)

| Column | Type | Meaning |
|--------|------|---------|
| `DATA_PROVENANCE` | string | Literal `SIMULATED` on every row — the provenance stamp. |
| `student` | int | Student ID, 1–30. |
| `discipline` | categorical | `Business` / `HealthSciences` / `Humanities`. The grouping variable for MANOVA and ANOVA. |

## Dimension scores — consumed by MANOVA, ANOVA/Tukey, criterion validity (cols 5,7,9,11,13)

| Column | Range | Meaning |
|--------|-------|---------|
| `PromptEngineering` | 1–5 | Consensus score on the dimension (mean over 5 items × 3 coders). Higher = stronger. |
| `ProblemDecomposition` | 1–5 | " |
| `OutputValidation` | 1–5 | " |
| `EthicalIntegration` | 1–5 | " |
| `CollaborativeSensemaking` | 1–5 | " |

These are **not** integers — they are averages (e.g. 2.333). They are the headline
"how did this student score" numbers per dimension.

## Retest scores — consumed by test–retest ICC (cols 6,8,10,12,14)

| Column | Range | Meaning |
|--------|-------|---------|
| `PromptEngineering_retest` (and the other four `*_retest`) | 1–5 | The same five dimensions scored a second time (re-administration). Generated as `ρ·(first) + √(1−ρ²)·noise`, ρ = 0.82. ICC measures how close each `*_retest` is to its first administration. |

## Item-level ratings — consumed by Cronbach's α and CFA (cols 15–39)

| Column pattern | Range | Meaning |
|----------------|-------|---------|
| `<Dimension>_i1 … _i5` (25 columns) | integer 1–5 | One expert coder's rating on a single rubric item within the dimension. Integers because they are individual ratings, not averages. Five items per dimension → α (internal consistency); all 25 → CFA (5-factor structure). |

## Criterion & composite — consumed by criterion validity (cols 3,4,40)

| Column | Range | Meaning |
|--------|-------|---------|
| `gpa` | 0–4 | Discipline GPA. Built to carry a target population correlation ρ = 0.42 with the student's latent ability. |
| `ai_task_grade` | 0–100 | AI-task grade. Built to carry ρ = 0.52 with latent ability. |
| `CDM_total` | 1–5 | Mean of the five dimension scores. Convenience column for the criterion correlations. |

---

## How one value is produced (tracing `PromptEngineering_i1`)

1. **Latent dimension level.** Start from the discipline's injected mean for the
   dimension (e.g. Business Prompt Engineering = 3.8, from the slides). Add a
   correlated random draw across the five dimensions (so related dimensions move
   together), scaled by the dimension's SD → the student's *true* level (1–5).
2. **Item.** Add a fixed item-difficulty offset + item-specific noise (SD 0.65).
3. **Coder.** Add a small coder bias + random scoring error (SD 0.35).
4. **Discretize.** Round to the nearest integer in [1, 5] → the stored value.

The **dimension score** is the mean of all 3 coders across all 5 items. The
**`*_retest`** column repeats steps 1–4 with a fresh draw correlated 0.82 with the
first administration. **`gpa` / `ai_task_grade`** are generated separately to carry
their target correlation with the student's overall latent ability.

## Design logic (why these columns)

The column list is the union of what the seven pipeline statistics need as input:
group label + dimension scores (group comparison), item scores (reliability and
factor structure), a retest copy (stability), and two external measures (criterion
validity). Nothing is entered as a result; every number emerges from the
latent → item → coder → round process.

## Caveat (honest)

In real data the items would be the raw measurements and the dimension score would
be computed from them. The simulation instead generates the latent level first and
derives both the items and the consensus score from it — the standard way to
simulate a measurement model (you need a known ground truth to generate around).
Consequence: the items and the dimension score share a common origin by
construction, which real data would not have.
