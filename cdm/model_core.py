# Generative-model core for the CDM pipeline demonstration (SIMULATED data).
# Design parameters are grounded in the slides' stated study profile and in
# standard measurement theory. Output statistics EMERGE from these inputs;
# none are reverse-fitted to the slide figures.
import numpy as np

DIMENSIONS = ["PromptEngineering", "ProblemDecomposition", "OutputValidation",
              "EthicalIntegration", "CollaborativeSensemaking"]
DISCIPLINES = ["Business", "HealthSciences", "Humanities"]

# --- Injected discipline x dimension MEANS (1-5). Slide-stated cells are marked;
#     others are documented interpolations consistent with the slide narrative. ---
#                       PromptEng  Decomp  OutValid  Ethics  Collab
MEANS = {
    "Business":       [3.8,       3.4,    2.7,      2.3,    3.0],   # PromptEng,Ethics slide-stated; OutValid=4.1-1.4(slide gap)
    "HealthSciences": [3.2,       3.4,    4.1,      3.7,    3.2],   # OutValid,Ethics slide-stated
    "Humanities":     [2.1,       3.0,    3.1,      3.0,    3.9],   # PromptEng(=3.8-1.7 slide gap),Collab slide-stated
}
# --- Injected per-cell SDs (slide-stated where given, else 0.6). ---
SDS = {
    "Business":       [0.6, 0.6, 0.6, 0.7, 0.6],   # PromptEng .6, Ethics .7 (slide)
    "HealthSciences": [0.6, 0.6, 0.5, 0.6, 0.6],   # OutValid .5 (slide)
    "Humanities":     [0.8, 0.6, 0.6, 0.6, 0.5],   # PromptEng .8, Collab .5 (slide)
}
# --- Structured factor correlation: procedural cluster (PromptEng,Decomp) and
#     critical-engagement cluster (OutValid,Ethics) correlate within; Collab and
#     cross-cluster moderate. Reflects construct affinity, not a flat 0.30. ---
PHI = np.array([
    [1.00, 0.45, 0.25, 0.25, 0.30],
    [0.45, 1.00, 0.25, 0.25, 0.30],
    [0.25, 0.25, 1.00, 0.45, 0.30],
    [0.25, 0.25, 0.45, 1.00, 0.30],
    [0.30, 0.30, 0.30, 0.30, 1.00],
])

ITEMS_PER_DIM   = 5
ITEM_DIFFICULTY = np.array([-0.30, -0.15, 0.0, 0.15, 0.30])  # items vary in difficulty
ITEM_UNIQUE_SD  = 0.65    # item-specific (unique) SD; realistic human-rated rubric noise -> drives alpha
N_RATERS        = 3
RATER_BIAS      = np.array([-0.12, 0.0, 0.12])               # systematic coder leniency/severity
RATER_NOISE_SD  = 0.35    # trained-coder random error -> drives weighted kappa / Kripp. alpha
RHO_GPA         = 0.42    # injected POPULATION corr: ability -> discipline GPA
RHO_AIGRADE     = 0.52    # injected POPULATION corr: ability -> AI task grade
RHO_RETEST      = 0.82    # injected latent stability (3-week) -> drives ICC
TA_FRACTION     = 0.60    # think-aloud subset (0.60*30 = 18, per slides)


def _ordinal(x):
    """Threshold a continuous latent onto ordinal 1..5 via half-integer cutpoints
    (ordered-categorical discretization)."""
    return np.clip(np.rint(x), 1, 5).astype(int)


def simulate(n_per_group, seed=20260617):
    rng = np.random.default_rng(seed)
    nd = len(DIMENSIONS)
    L = np.linalg.cholesky(PHI)   # validates PHI is positive-definite

    rows, raters_long = [], []
    sid = 0
    for disc in DISCIPLINES:
        mu = np.array(MEANS[disc]); sd = np.array(SDS[disc])
        for _ in range(n_per_group):
            sid += 1
            # (1) correlated standardized factors -> per-cell mean/SD (corr preserved = PHI)
            g = L @ rng.standard_normal(nd)
            F = np.clip(mu + sd * g, 1, 5)                       # true dimension level (1-5)

            # ability composite (standardized mean of factor z-scores) for criterion paths
            ability = g.mean()

            # (2) crossed item x rater measurement model
            #     true item score = F + item difficulty + item-unique error (shared across raters)
            #     each rater observes it with bias + random error, then discretized
            cons_items = {}                                     # consensus (rater-mean) item scores
            rater_dim = np.zeros((N_RATERS, nd))                # each rater's per-dimension score
            for j, dim in enumerate(DIMENSIONS):
                item_true = F[j] + ITEM_DIFFICULTY + rng.normal(0, ITEM_UNIQUE_SD, ITEMS_PER_DIM)
                rater_item = np.zeros((N_RATERS, ITEMS_PER_DIM))
                for m in range(N_RATERS):
                    obs = item_true + RATER_BIAS[m] + rng.normal(0, RATER_NOISE_SD, ITEMS_PER_DIM)
                    rater_item[m] = _ordinal(obs)
                cons = rater_item.mean(axis=0)                  # consensus (for dimension score)
                for i in range(ITEMS_PER_DIM):
                    # alpha / CFA use a single coder's ordinal item responses (the instrument
                    # as administered) — using rater-averaged items would inflate internal consistency
                    cons_items[f"{dim}_i{i+1}"] = int(rater_item[0, i])
                # rater's dimension score = rounded mean of that rater's item ratings
                rater_dim[:, j] = _ordinal(rater_item.mean(axis=1))
                for m in range(N_RATERS):
                    raters_long.append(dict(student=sid, discipline=disc, rater=m+1,
                                            dimension=dim, score=int(rater_dim[m, j])))

            dim_score = rater_dim.mean(axis=0)                  # consensus dimension score

            # (3) criterion variables: inject POPULATION correlation with ability
            def crit(rho):  # standardized criterion correlated rho with ability
                return rho * ability + np.sqrt(1 - rho**2) * rng.standard_normal()
            gpa = float(np.clip(3.2 + 0.45 * crit(RHO_GPA), 0, 4.0))
            ai_grade = float(np.clip(82 + 8.0 * crit(RHO_AIGRADE), 0, 100))

            # (4) test-retest: re-administration with injected latent stability
            g2 = RHO_RETEST * g + np.sqrt(1 - RHO_RETEST**2) * (L @ rng.standard_normal(nd))
            F2 = np.clip(mu + sd * g2, 1, 5)
            retest_dim = np.zeros(nd)
            for j in range(nd):
                obs = F2[j] + ITEM_DIFFICULTY.mean() + rng.normal(0, RATER_NOISE_SD/np.sqrt(N_RATERS))
                retest_dim[j] = float(np.clip(obs, 1, 5))

            row = dict(student=sid, discipline=disc,
                       gpa=round(gpa, 3), ai_task_grade=round(ai_grade, 2))
            for j, dim in enumerate(DIMENSIONS):
                row[dim] = round(float(dim_score[j]), 3)
                row[f"{dim}_retest"] = round(float(retest_dim[j]), 3)
            row.update(cons_items)
            rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    raters_df = pd.DataFrame(raters_long)

    # (5) think-aloud codes (qualitative stream), n=18 subset, slide-aligned dominant codes
    code_p = {
        "Business":       dict(PROMPT_REFINE=.38, EVAL_DOUBT=.15, COLLAB_EXPLAIN=.20, ETHICAL_FLAG=.07, DECOMPOSE=.20),
        "HealthSciences": dict(PROMPT_REFINE=.17, EVAL_DOUBT=.41, COLLAB_EXPLAIN=.14, ETHICAL_FLAG=.14, DECOMPOSE=.14),
        "Humanities":     dict(PROMPT_REFINE=.13, EVAL_DOUBT=.17, COLLAB_EXPLAIN=.35, ETHICAL_FLAG=.10, DECOMPOSE=.25),
    }
    n_ta = int(round(TA_FRACTION * len(df)))
    ta_ids = rng.choice(df.student.values, size=n_ta, replace=False)
    ta_rows = []
    for s_ in ta_ids:
        disc = df.loc[df.student == s_, "discipline"].iloc[0]
        codes = list(code_p[disc]); p = np.array(list(code_p[disc].values())); p = p/p.sum()
        counts = rng.multinomial(rng.integers(25, 60), p)
        for c, k in zip(codes, counts):
            ta_rows.append(dict(student=int(s_), discipline=disc, code=c, count=int(k)))
    ta_df = pd.DataFrame(ta_rows)
    return df, raters_df, ta_df
