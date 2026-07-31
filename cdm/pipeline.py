#!/usr/bin/env python3
# =============================================================================
#  CDM SCORING PIPELINE — DEMONSTRATION ON SIMULATED DATA
# =============================================================================
#  Generates synthetic CDM data from a DECLARED generative model whose design
#  parameters are grounded in the proposal slides (discipline x dimension means
#  and SDs, target reliabilities) and in standard measurement theory, then runs
#  the full analysis pipeline on it.
#
#  *** EVERY NUMBER IS FROM SIMULATED DATA — NOT AN EMPIRICAL FINDING. ***
#  Structural parameters (means, SDs, loadings, rater noise, criterion/retest
#  correlations) are INPUTS. The reported statistics (MANOVA, eta^2, alpha,
#  weighted kappa, Krippendorff's alpha, CFA fit, criterion r, ICC) EMERGE from
#  those inputs plus sampling noise; none are reverse-fitted to slide figures.
#  The demonstration is that the PIPELINE recovers known injected structure.
# =============================================================================
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats

from cdm import model_core as mc
from cdm.model_core import DIMENSIONS, DISCIPLINES, ITEMS_PER_DIM
from cdm import agreement as ag

STAMP = "SIMULATED DATA — PIPELINE DEMONSTRATION (not empirical findings)"


def cronbach_alpha(M):
    M = np.asarray(M, float); k = M.shape[1]
    iv = M.var(axis=0, ddof=1); tv = M.sum(axis=1).var(ddof=1)
    return (k/(k-1))*(1 - iv.sum()/tv) if tv > 0 else np.nan

def partial_eta_sq(groups):
    allv = np.concatenate(groups); g = allv.mean()
    ssb = sum(len(x)*(np.mean(x)-g)**2 for x in groups)
    sst = ((allv-g)**2).sum()
    return ssb/sst if sst > 0 else np.nan

def icc_2_1(t1, t2):
    Y = np.column_stack([t1, t2]).astype(float); n, k = Y.shape
    g = Y.mean()
    msr = k*((Y.mean(1)-g)**2).sum()/(n-1)
    msc = n*((Y.mean(0)-g)**2).sum()/(k-1)
    sse = ((Y-g)**2).sum() - k*((Y.mean(1)-g)**2).sum() - n*((Y.mean(0)-g)**2).sum()
    mse = sse/((n-1)*(k-1))
    den = msr + (k-1)*mse + (k/n)*(msc-mse)
    return (msr-mse)/den if den > 0 else np.nan


def run_pipeline(df, raters_df, ta_df, out):
    w = out.append
    w(f"\n{'='*78}\n{STAMP}\n{'='*78}")
    w(f"\nSimulated sample: N = {len(df)}  ({df.discipline.value_counts().to_dict()})")
    w("Design parameters grounded in slides + measurement theory; statistics below")
    w("are emergent properties of the SIMULATED data, demonstrating the pipeline.\n")

    # 1. MANOVA
    from statsmodels.multivariate.manova import MANOVA
    mv = MANOVA.from_formula(" + ".join(DIMENSIONS) + " ~ C(discipline)", data=df)
    wl = mv.mv_test().results["C(discipline)"]["stat"].loc["Wilks' lambda"]
    w("-"*78 + "\n1. MANOVA  (discipline -> 5 CDM dimensions)\n" + "-"*78)
    w(f"   Wilks' lambda = {wl['Value']:.3f}   F({wl['Num DF']:.0f},{wl['Den DF']:.0f}) "
      f"= {wl['F Value']:.2f}   p = {wl['Pr > F']:.4g}")

    # 2. ANOVA + Tukey + eta^2
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    w("\n" + "-"*78 + "\n2. Per-dimension ANOVA + Tukey HSD + partial eta^2\n" + "-"*78)
    for dim in DIMENSIONS:
        gs = [df.loc[df.discipline == d, dim].values for d in DISCIPLINES]
        F, p = stats.f_oneway(*gs)
        w(f"   {dim:<26} F = {F:6.2f}  p = {p:9.4g}  partial_eta^2 = {partial_eta_sq(gs):.3f}")
        tk = pairwise_tukeyhsd(df[dim].values, df.discipline.values)
        for r in tk.summary().data[1:]:
            flag = "*" if (isinstance(r[3], float) and r[3] < 0.05) else " "
            w(f"        {r[0]:>14} vs {r[1]:<14} diff={r[2]:+.2f}  p_adj={r[3]:.3g} {flag}")

    # 3. Cronbach's alpha (over consensus item scores)
    w("\n" + "-"*78 + "\n3. Internal consistency (Cronbach's alpha, per dimension)\n" + "-"*78)
    for dim in DIMENSIONS:
        cols = [f"{dim}_i{i+1}" for i in range(ITEMS_PER_DIM)]
        a = cronbach_alpha(df[cols].values)
        w(f"   {dim:<26} alpha = {a:.3f}   [{'OK >=0.75' if a >= 0.75 else 'below 0.75'}]")

    # 4. Inter-rater reliability (ordinal-appropriate)
    w("\n" + "-"*78 + "\n4. Inter-rater reliability (3 coders; quadratic-weighted kappa + "
      "Krippendorff alpha, interval)\n" + "-"*78)
    for dim in DIMENSIONS:
        wide = raters_df[raters_df.dimension == dim].pivot(
            index="student", columns="rater", values="score")
        wk = ag.mean_pairwise_weighted_kappa(wide)
        ka = ag.krippendorff_interval(wide.values)
        w(f"   {dim:<26} weighted kappa = {wk:.3f}   Krippendorff alpha = {ka:.3f}")

    # 5. CFA (dual-N: shows N=30 underpowered, fit resolves at proposed scale)
    w("\n" + "-"*78 + "\n5. Confirmatory Factor Analysis (5-factor model)\n" + "-"*78)
    run_cfa(df, w)

    # 6. Criterion validity
    w("\n" + "-"*78 + "\n6. Criterion validity (CDM total vs external outcomes)\n" + "-"*78)
    df = df.copy(); df["CDM_total"] = df[DIMENSIONS].mean(axis=1)
    for c, lab in [("gpa", "discipline GPA"), ("ai_task_grade", "AI task grade")]:
        r, p = stats.pearsonr(df["CDM_total"], df[c])
        w(f"   CDM_total vs {lab:<16} r = {r:+.3f}   p = {p:.4g}")

    # 7. Test-retest ICC
    w("\n" + "-"*78 + "\n7. Test-retest reliability (ICC(2,1))\n" + "-"*78)
    for dim in DIMENSIONS:
        w(f"   {dim:<26} ICC(2,1) = {icc_2_1(df[dim].values, df[dim+'_retest'].values):.3f}")

    # 8. Think-aloud
    if len(ta_df):
        w("\n" + "-"*78 + "\n8. Think-aloud code distribution (qualitative stream)\n" + "-"*78)
        tot = ta_df.groupby(["discipline", "code"])["count"].sum().reset_index()
        for disc in DISCIPLINES:
            d = tot[tot.discipline == disc].set_index("code")["count"]
            d = (d/d.sum()*100).sort_values(ascending=False)
            w(f"   {disc:<16} dominant: {d.index[0]} ({d.iloc[0]:.0f}%)   "
              + " ".join(f"{c}:{v:.0f}%" for c, v in d.items()))

    w(f"\n{'='*78}\n{STAMP}\nReplace model_core.simulate() with real collected data (same columns) to "
      f"produce\nactual results — the pipeline is unchanged.\n" + "="*78)
    return df


def run_cfa(df, w):
    try:
        import semopy
    except Exception:
        w("   [semopy unavailable — use R lavaan for CFI/RMSEA]"); return
    spec = "\n".join(f"{dim} =~ " + " + ".join(f"{dim}_i{i+1}" for i in range(ITEMS_PER_DIM))
                     for dim in DIMENSIONS)
    item_cols = [f"{dim}_i{i+1}" for dim in DIMENSIONS for i in range(ITEMS_PER_DIM)]
    def fit(data, tag):
        try:
            m = semopy.Model(spec); m.fit(data)
            s = semopy.calc_stats(m).T
            w(f"   [{tag}, N={len(data)}]  CFI = {float(s.loc['CFI'].iloc[0]):.3f}   "
              f"RMSEA = {float(s.loc['RMSEA'].iloc[0]):.3f}")
        except Exception as e:
            w(f"   [{tag}, N={len(data)}]  CFA unstable: {type(e).__name__} "
              f"(expected when N is small vs free parameters)")
    fit(df[item_cols], "pilot scale")
    w("   A 25-indicator 5-factor CFA needs N well above 30; this is why the")
    w("   proposal scales to N=120+. Same pipeline on a larger SIMULATED draw:")
    big, _, _ = mc.simulate(n_per_group=80, seed=99)
    fit(big[item_cols], "proposed scale (simulated)")


def make_figures(df, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = df.groupby("discipline")[DIMENSIONS].mean()
    sems = df.groupby("discipline")[DIMENSIONS].sem()
    fig, ax = plt.subplots(figsize=(10, 5.2)); x = np.arange(len(DIMENSIONS)); wd = 0.25
    for i, disc in enumerate(DISCIPLINES):
        ax.bar(x+(i-1)*wd, means.loc[disc], wd, yerr=sems.loc[disc], capsize=3, label=disc)
    ax.set_xticks(x); ax.set_xticklabels(
        [d.replace("Engineering"," Eng").replace("Decomposition"," Decomp")
          .replace("Validation"," Valid").replace("Integration"," Integ")
          .replace("Sensemaking"," Sense") for d in DIMENSIONS], rotation=20, ha="right")
    ax.set_ylabel("Mean CDM score (1-5)"); ax.set_ylim(0, 5)
    ax.set_title("CDM dimension means by discipline  (simulated data)", fontsize=11)
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(outdir,"fig_dimension_means.png"), dpi=130)

    corr = df[DIMENSIONS].corr()
    fig2, ax2 = plt.subplots(figsize=(6.2, 5.2))
    im = ax2.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax2.set_xticks(range(5)); ax2.set_yticks(range(5))
    sh = [d[:10] for d in DIMENSIONS]
    ax2.set_xticklabels(sh, rotation=45, ha="right", fontsize=8); ax2.set_yticklabels(sh, fontsize=8)
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax2.set_title("Inter-dimension correlations  (simulated data)", fontsize=11)
    fig2.colorbar(im, fraction=0.046); fig2.tight_layout()
    fig2.savefig(os.path.join(outdir,"fig_dimension_correlations.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser(description="CDM pipeline demonstration on SIMULATED data.")
    ap.add_argument("--n-per-group", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    df, raters_df, ta_df = mc.simulate(a.n_per_group, a.seed)
    out = []; df = run_pipeline(df, raters_df, ta_df, out)
    report = "\n".join(out); print(report)
    os.makedirs(a.outdir, exist_ok=True)
    df.insert(0, "DATA_PROVENANCE", "SIMULATED")
    df.to_csv(os.path.join(a.outdir, "cdm_simulated_dataset.csv"), index=False)
    open(os.path.join(a.outdir, "cdm_pipeline_report.txt"), "w").write(report + "\n")
    try: make_figures(df, a.outdir); print("[figures written]")
    except Exception as e: print(f"[figure step skipped: {e}]")


if __name__ == "__main__":
    main()
