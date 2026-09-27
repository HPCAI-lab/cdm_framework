"""Negative controls: what the analysis does when it ought to fail.

Recovering injected structure establishes that estimators are implemented
correctly. It does not establish that the model is correctly specified, since
a misspecified model evaluated against data generated from itself also passes.
The controls here generate data under conditions where the planned analysis
should *not* fire, and measure whether it fires anyway.

Four conditions, each answering a question the pilot cannot afford to answer
empirically:

``null``          groups with identical means. The rejection rate over
                  replications estimates the false-positive rate of the
                  planned MANOVA at the planned sample size.
``single_factor`` one general factor rather than five. Whether the five-factor
                  model can be distinguished from a one-factor alternative at
                  a given N is a property of the design, not of the data.
``contaminated``  one of three coders replaced by a careless rater. Whether
                  the agreement statistics degrade detectably determines
                  whether the holdout double-coding plan can catch drift while
                  it is still correctable.
``power``         rejection rate as a function of sample size and injected
                  separation, giving the N required to detect a stated effect.

Every result here is a property of the design under the declared generative
model. None is evidence about students.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from cdm import agreement as ag
from cdm import model_core as mc
from cdm.model_core import DIMENSIONS, DISCIPLINES, ITEMS_PER_DIM

ITEM_COLS = [f"{d}_i{i+1}" for d in DIMENSIONS for i in range(ITEMS_PER_DIM)]


def _manova_p(df: pd.DataFrame) -> float:
    from statsmodels.multivariate.manova import MANOVA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mv = MANOVA.from_formula(" + ".join(DIMENSIONS) + " ~ C(discipline)", data=df)
        wl = mv.mv_test().results["C(discipline)"]["stat"].loc["Wilks' lambda"]
    return float(wl["Pr > F"])


def _simulate_flat(n_per_group: int, seed: int, spread: float = 0.0):
    """Simulate with group means collapsed toward the grand mean.

    ``spread`` scales the injected between-group separation: 0.0 removes it
    entirely (the null), 1.0 leaves the declared design untouched.
    """
    grand = np.mean([mc.MEANS[d] for d in DISCIPLINES], axis=0)
    saved = {d: list(mc.MEANS[d]) for d in DISCIPLINES}
    try:
        for d in DISCIPLINES:
            mc.MEANS[d] = list(grand + spread * (np.array(saved[d]) - grand))
        return mc.simulate(n_per_group, seed)
    finally:
        for d in DISCIPLINES:
            mc.MEANS[d] = saved[d]


def null_control(reps: int = 200, n_per_group: int = 10, alpha: float = 0.05,
                 seed: int = 20260915) -> dict:
    """False-positive rate of the planned MANOVA when no group difference exists."""
    rng = np.random.default_rng(seed)
    rejects = 0
    for _ in range(reps):
        df, _, _ = _simulate_flat(n_per_group, int(rng.integers(1, 2**31)), spread=0.0)
        rejects += int(_manova_p(df) < alpha)
    rate = rejects / reps
    lo, hi = _binom_ci(rejects, reps)
    return dict(condition="Null group structure", reps=reps, n=n_per_group * 3,
                observed=f"{rate:.3f} [{lo:.3f}, {hi:.3f}]", nominal=alpha,
                calibrated=bool(lo <= alpha <= hi),
                implication=("Test is calibrated at the nominal rate"
                             if lo <= alpha <= hi else
                             "Rejection rate departs from nominal; p-values from this "
                             "design are not trustworthy at this N"))


def _fit(data: pd.DataFrame, spec: str) -> dict:
    """Fit one CFA and return fit indices plus what a nested comparison needs.

    AIC and BIC are recomputed from the model chi-square. semopy's own AIC is
    built from an unscaled log-likelihood that does not grow with N, so its
    AIC difference between two models is almost entirely the parameter
    penalty and prefers the smaller model whatever generated the data. Using
    chi-square (which is scaled by N) gives the standard SEM information
    criteria, up to a constant that cancels in any comparison.
    """
    import semopy
    nan = dict(cfi=np.nan, rmsea=np.nan, chi2=np.nan, dof=np.nan, aic=np.nan, bic=np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = semopy.Model(spec)
            m.fit(data)
            s = semopy.calc_stats(m).T
        chi2 = float(s.loc["chi2"].iloc[0])
        dof = float(s.loc["DoF"].iloc[0])
        p = data.shape[1]
        npar = p * (p + 1) / 2 - dof
        n = len(data)
        # CFI is a normed index: values above one are estimation noise and are
        # conventionally truncated. Reporting 1.005 in a paper invites a query.
        return dict(cfi=float(np.clip(s.loc["CFI"].iloc[0], 0.0, 1.0)),
                    rmsea=float(max(s.loc["RMSEA"].iloc[0], 0.0)),
                    chi2=chi2, dof=dof,
                    aic=chi2 + 2 * npar, bic=chi2 + npar * np.log(n))
    except Exception:
        return nan


def _compare(df: pd.DataFrame) -> tuple:
    """Five-factor vs one-factor on the same data.

    Returns (cfi5, cfi1, aic prefers 5, bic prefers 5, chi-square difference
    p-value). The one-factor model is the five-factor model with every factor
    correlation fixed at one, so the models are nested and the chi-square
    difference test applies. The restriction sits on the boundary of the
    parameter space, which makes the naive test somewhat conservative.
    """
    five = _fit(df[ITEM_COLS], FIVE_FACTOR)
    one = _fit(df[ITEM_COLS], ONE_FACTOR)
    d_chi2, d_df = one["chi2"] - five["chi2"], one["dof"] - five["dof"]
    p = float(stats.chi2.sf(d_chi2, d_df)) if d_df > 0 and np.isfinite(d_chi2) else np.nan
    return (five["cfi"], one["cfi"], float(five["aic"] < one["aic"]),
            float(five["bic"] < one["bic"]), p)


FIVE_FACTOR = "\n".join(
    f"{d} =~ " + " + ".join(f"{d}_i{i+1}" for i in range(ITEMS_PER_DIM)) for d in DIMENSIONS)
ONE_FACTOR = "G =~ " + " + ".join(ITEM_COLS)


def _simulate_one_factor(n: int, seed: int) -> pd.DataFrame:
    """Generate 25 items from a single general factor.

    Every item loads on one latent with the item difficulty offsets and
    unique-error variance already declared in :mod:`cdm.model_core`, so the
    only structural difference from the five-factor generator is the number of
    latents. This is the alternative the instrument's dimensional claim has to
    beat.
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n)
    rows = []
    for k in range(n):
        row = {"student": k + 1, "discipline": DISCIPLINES[k % len(DISCIPLINES)]}
        for d in DIMENSIONS:
            for i in range(ITEMS_PER_DIM):
                x = (3.0 + 0.8 * g[k] + mc.ITEM_DIFFICULTY[i]
                     + rng.normal(0, mc.ITEM_UNIQUE_SD) + rng.normal(0, mc.RATER_NOISE_SD))
                row[f"{d}_i{i+1}"] = int(np.clip(round(x), 1, 5))
        rows.append(row)
    return pd.DataFrame(rows)


def single_factor_control(reps: int = 20, n_per_group: int = 40,
                          seed: int = 20260915, alpha: float = 0.05) -> dict:
    """Can the design tell a five-factor structure from a general factor?

    Data are generated from one latent, and the five-factor and one-factor
    models are compared three ways: CFI, information criteria, and the
    chi-square difference test. A usable comparison must prefer the one-factor
    model here *and* prefer the five-factor model when five factors generated
    the data, so the same comparison is also run on data from the declared
    five-factor model at the same N (the sensitivity column).

    CFI is not a usable comparison. The five-factor model nests the one-factor
    model, so on one-factor data both fit equally well and the CFI difference
    is near zero whatever the sample size.
    """
    rng = np.random.default_rng(seed)
    n = n_per_group * len(DISCIPLINES)
    null_rows, alt_rows = [], []
    for _ in range(reps):
        null_rows.append(_compare(_simulate_one_factor(n, int(rng.integers(1, 2**31)))))
        five_df, _, _ = mc.simulate(n_per_group=n_per_group, seed=int(rng.integers(1, 2**31)))
        alt_rows.append(_compare(five_df))
    null = np.array(null_rows, float)
    alt = np.array(alt_rows, float)
    null = null[~np.isnan(null).any(axis=1)]
    alt = alt[~np.isnan(alt).any(axis=1)]

    five_cfi, one_cfi = float(np.mean(null[:, 0])), float(np.mean(null[:, 1]))
    delta = five_cfi - one_cfi
    false_reject = float(np.mean(null[:, 4] < alpha))
    true_reject = float(np.mean(alt[:, 4] < alpha)) if len(alt) else float("nan")
    aic_null, bic_null = float(np.mean(null[:, 2])), float(np.mean(null[:, 3]))

    # Over-rejection is judged by a one-sided binomial test against the nominal
    # rate, so one false rejection in a five-rep smoke test is not a verdict.
    n_false = int(np.sum(null[:, 4] < alpha))
    over_rejects = (len(null) > 0 and
                    stats.binomtest(n_false, len(null), alpha, alternative="greater").pvalue < 0.05)

    if not over_rejects and true_reject >= 0.80:
        verdict = (f"The chi-square difference test discriminates the two structures at N={n}: "
                   f"it rejects the one-factor model in {false_reject:.0%} of one-factor draws "
                   f"and {true_reject:.0%} of five-factor draws. CFI does not (difference "
                   f"{delta:+.3f} on one-factor data), so the dimensional claim must rest on the "
                   "difference test or information criteria, not on global fit")
    elif over_rejects:
        verdict = ("The chi-square difference test rejects the one-factor model too often when "
                   "it is true; the five-factor structure cannot be claimed from this comparison")
    else:
        verdict = (f"The chi-square difference test detects a true five-factor structure in only "
                   f"{true_reject:.0%} of draws at N={n}; the comparison is underpowered here")

    return dict(condition="Single-factor truth", reps=int(len(null)), n=n,
                observed=(f"CFI 5f {five_cfi:.3f} vs 1f {one_cfi:.3f} (difference {delta:+.3f}); "
                          f"AIC prefers 5f in {aic_null:.0%}, BIC in {bic_null:.0%}; "
                          f"chi-square difference rejects 1f in {false_reject:.0%} "
                          f"(sensitivity on 5f data: {true_reject:.0%})"),
                implication=verdict)


def contaminated_coder_control(reps: int = 100, n_per_group: int = 10,
                               seed: int = 20260915) -> dict:
    """Does replacing one coder with a careless rater show up in the statistics?"""
    rng = np.random.default_rng(seed)
    clean, dirty = [], []
    for _ in range(reps):
        s = int(rng.integers(1, 2**31))
        _, raters, _ = mc.simulate(n_per_group, s)
        r = np.random.default_rng(s)
        for label, contaminate in (("clean", False), ("dirty", True)):
            rr = raters.copy()
            if contaminate:
                mask = rr.rater == 3
                rr.loc[mask, "score"] = r.integers(1, 6, int(mask.sum()))
            ks = []
            for dim in DIMENSIONS:
                wide = rr[rr.dimension == dim].pivot(index="student", columns="rater",
                                                     values="score")
                ks.append(ag.mean_pairwise_weighted_kappa(wide))
            (clean if label == "clean" else dirty).append(float(np.mean(ks)))
    c, d = np.array(clean), np.array(dirty)
    sep = float(np.mean(d < 0.80))
    return dict(condition="Contaminated coder", reps=reps, n=n_per_group * 3,
                observed=f"mean weighted kappa {c.mean():.3f} clean vs {d.mean():.3f} "
                         f"contaminated; contaminated run falls below 0.80 in {sep:.0%} of draws",
                implication=("Holdout double-coding detects a careless rater"
                             if sep >= 0.90 else
                             "A careless rater is not reliably detected by the agreement "
                             "threshold alone; increase holdout fraction or inspect per-coder"))


def power_curve(sizes=(8, 10, 15, 20, 30, 40), spreads=(0.5, 1.0), reps: int = 100,
                alpha: float = 0.05, seed: int = 20260915) -> pd.DataFrame:
    """Rejection rate against sample size and injected between-group separation."""
    rng = np.random.default_rng(seed)
    rows = []
    for spread in spreads:
        for n in sizes:
            rejects = 0
            for _ in range(reps):
                df, _, _ = _simulate_flat(n, int(rng.integers(1, 2**31)), spread=spread)
                rejects += int(_manova_p(df) < alpha)
            lo, hi = _binom_ci(rejects, reps)
            rows.append(dict(spread=spread, n_per_group=n, N=n * 3,
                             power=round(rejects / reps, 3),
                             ci=f"[{lo:.3f}, {hi:.3f}]"))
    return pd.DataFrame(rows)


def _binom_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson interval, which behaves at rates near zero and one."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - h) / d, (c + h) / d)


def fit_sweep(sizes=(30, 60, 90, 120, 180, 240, 360, 480), reps: int = 10,
              seed: int = 20260915) -> pd.DataFrame:
    """Confirmatory fit as a function of sample size, with replications.

    The existing repository reports one draw per sample size. A single draw
    cannot show how variable the estimate is, and at N = 30 the variability is
    the point. This returns every replication so the figure can carry an
    interquartile band rather than a line through three points.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for N in sizes:
        per = max(N // len(DISCIPLINES), 1)
        for r in range(reps):
            df, _, _ = mc.simulate(per, int(rng.integers(1, 2**31)))
            f = _fit(df[ITEM_COLS], FIVE_FACTOR)
            rows.append(dict(N=per * len(DISCIPLINES), rep=r,
                             cfi=f["cfi"], rmsea=f["rmsea"]))
    return pd.DataFrame(rows)


def fit_sweep_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """Median and interquartile range per sample size, with convergence count."""
    g = sweep.groupby("N")
    out = pd.DataFrame({
        "reps": g.size(),
        "converged": g.cfi.apply(lambda s: int(s.notna().sum())),
        "cfi_median": g.cfi.median().round(3),
        "cfi_q25": g.cfi.quantile(0.25).round(3),
        "cfi_q75": g.cfi.quantile(0.75).round(3),
        "rmsea_median": g.rmsea.median().round(3),
        "rmsea_q25": g.rmsea.quantile(0.25).round(3),
        "rmsea_q75": g.rmsea.quantile(0.75).round(3),
    }).reset_index()
    out["reading"] = np.where(out.cfi_median >= 0.95, "clean",
                     np.where(out.cfi_median >= 0.90, "acceptable", "not estimable"))
    return out


def run_all(reps_fast: int = 200, reps_slow: int = 20, n_per_group: int = 10,
            seed: int = 20260915) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the four controls. Returns the summary table and the power curve."""
    results = [
        null_control(reps=reps_fast, n_per_group=n_per_group, seed=seed),
        single_factor_control(reps=reps_slow, n_per_group=40, seed=seed),
        contaminated_coder_control(reps=max(reps_fast // 2, 20), n_per_group=n_per_group, seed=seed),
    ]
    curve = power_curve(reps=max(reps_fast // 2, 50), seed=seed)
    at_pilot = curve[(curve.n_per_group == n_per_group) & (curve.spread == 1.0)]
    results.append(dict(
        condition="Power at designed separation", reps=int(max(reps_fast // 2, 50)),
        n=n_per_group * 3,
        observed=f"{float(at_pilot.power.iloc[0]):.3f} {at_pilot.ci.iloc[0]}",
        implication="Detection rate for the separation the design assumes, at pilot size"))
    cols = ["condition", "reps", "n", "observed", "implication"]
    return pd.DataFrame(results)[cols], curve
