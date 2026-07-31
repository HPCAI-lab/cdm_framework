# CDM Analysis Pipeline — Demonstration on Simulated Data

**What this is:** illustrative outputs from the Cognitive Diversity Metrics (CDM)
analysis pipeline, run on a **simulated** dataset built from the agreed design
assumptions. It shows the analysis machinery runs end to end and is ready for the
Spring pilot.

**What this is not:** empirical findings. Every value below comes from simulated
data; the structural assumptions are inputs, and the statistics emerge from them
plus random noise. Real findings will come from the pilot data collection.

Design simulated: mixed-methods convergent design, N = 30 (10 per discipline) across
business, health sciences, and humanities; five CDM dimensions; five-item rubric per
dimension scored 1–5; expert coding by three coders; think-aloud subset (n = 18).

---

## 1. Discipline differences (MANOVA)

The pipeline recovers the hypothesized discipline-specific pattern:
Wilks' λ = 0.11, F(10, 46) = 9.39, p < 0.001.

| Dimension | partial η² | significant? |
|---|---|---|
| Prompt Engineering | 0.44 | yes (p < 0.001) |
| Output Validation | 0.40 | yes (p < 0.001) |
| Collaborative Sensemaking | 0.45 | yes (p < 0.001) |
| Ethical Integration | 0.26 | yes (p = 0.02) |
| Problem Decomposition | 0.12 | no (p = 0.18) — modeled as similar across disciplines |

## 2. Internal consistency (Cronbach's α) — target > 0.75

| Dimension | α |
|---|---|
| Prompt Engineering | 0.78 |
| Problem Decomposition | 0.87 |
| Output Validation | 0.79 |
| Ethical Integration | 0.85 |
| Collaborative Sensemaking | 0.86 |

All five dimensions clear the 0.75 target.

## 3. Inter-rater reliability (3 coders) — target > 0.80

Quadratic-weighted Cohen's κ and Krippendorff's α (interval metric), the
ordinal-appropriate statistics for rubric scores:

| Dimension | weighted κ | Krippendorff α |
|---|---|---|
| Prompt Engineering | 0.88 | 0.88 |
| Problem Decomposition | 0.82 | 0.82 |
| Output Validation | 0.81 | 0.81 |
| Ethical Integration | 0.88 | 0.88 |
| Collaborative Sensemaking | 0.86 | 0.86 |

All clear the 0.80 target.

## 4. Construct validity (confirmatory factor analysis) — the sample-size result

Same five-factor model, same pipeline, only the sample size changes:

| Sample | CFI | RMSEA | interpretation |
|---|---|---|---|
| N = 30 (pilot) | 0.48 | 0.19 | underpowered — too few cases for a 25-indicator model |
| N = 120 | 0.92 | 0.055 | acceptable |
| N = 240 | 0.99 | 0.018 | clean fit (targets: CFI > 0.95, RMSEA < 0.06) |

This is the quantitative case for scaling to N ≥ 120 in the NSF study: the
confirmatory factor analysis is not stably estimable at pilot size.

## 5. Criterion validity

CDM total score against external measures (simulated):
discipline GPA, r = 0.38 (p = 0.04); AI-task grade, r = 0.50 (p = 0.005).

## 6. Test–retest reliability (ICC(2,1))

Across a re-administration: ICC ≈ 0.68–0.90 across dimensions, indicating stable
scores.

---

## How to read this

The discipline *pattern* (Section 1) is the hypothesis we put in. Everything in
Sections 2–6 — reliability, factor structure, validity — is **not** set; it emerges
from the model and demonstrates that the pipeline computes each statistic correctly
and that, under the agreed design, the instrument behaves as intended. The honest
framing for the proposal: a **validated analysis pipeline, demonstrated on simulated
data, ready for the pilot** — not preliminary findings.
