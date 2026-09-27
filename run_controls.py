#!/usr/bin/env python3
"""Produce the design-time evidence the paper draft is waiting on.

    python run_controls.py --outdir results_controls

Writes, in a form ready to paste into the draft:

  negative_controls.csv / .tex   Table II, the four controls
  power_curve.csv                rejection rate by sample size and separation
  fit_sweep.csv                  every replication of the fit-by-N sweep
  fit_sweep_summary.csv / .tex   Table III, medians with interquartile ranges
  fig_fit_by_n.png / .pdf        the fit-as-a-function-of-sample-size figure
  reliability_ordinal.csv / .tex Cronbach alpha beside ordinal alpha and omega
  lavaan/cfa_wlsmv.R             the WLSMV re-estimation this cannot do in Python

Everything here runs on simulated data from the declared generative model.
These are properties of the design. None of them is evidence about students.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cdm import controls as ctl
from cdm import model_core as mc
from cdm import ordinal as od

STAMP = "SIMULATED DATA — DESIGN-TIME EVIDENCE (not empirical findings)"


def _tex(df: pd.DataFrame, path: Path, caption: str, label: str,
         widths: str | None = None) -> None:
    """Write a booktabs table. Column widths are given explicitly because IEEE
    two-column measure is unforgiving of automatic sizing."""
    cols = list(df.columns)
    spec = widths or ("l" * len(cols))
    lines = [r"\begin{table}[t]", r"\caption{" + caption + "}",
             r"\label{" + label + "}", r"\centering", r"\small",
             r"\begin{tabular}{@{}" + spec + r"@{}}", r"\toprule",
             " & ".join(r"\textbf{" + str(c).replace("_", " ") + "}" for c in cols) + r" \\",
             r"\midrule"]
    for _, r in df.iterrows():
        cells = [str(v).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
                 for v in r.tolist()]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(sweep: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = ctl.fit_sweep_summary(sweep)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)

    ax1.fill_between(s.N, s.cfi_q25, s.cfi_q75, alpha=0.25, color="tab:blue")
    ax1.plot(s.N, s.cfi_median, "o-", color="tab:blue", label="CFI (median, IQR)")
    ax1.axhline(0.95, ls="--", lw=1, color="0.4")
    ax1.text(s.N.max(), 0.955, "0.95", ha="right", va="bottom", fontsize=8, color="0.3")
    ax1.set_ylabel("CFI")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc="lower right")

    ax2.fill_between(s.N, s.rmsea_q25, s.rmsea_q75, alpha=0.25, color="tab:red")
    ax2.plot(s.N, s.rmsea_median, "o-", color="tab:red", label="RMSEA (median, IQR)")
    ax2.axhline(0.06, ls="--", lw=1, color="0.4")
    ax2.text(s.N.max(), 0.063, "0.06", ha="right", va="bottom", fontsize=8, color="0.3")
    ax2.set_ylabel("RMSEA")
    ax2.set_xlabel("Sample size (N)")
    ax2.set_xscale("log")
    ax2.set_xticks(list(s.N))
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle("Confirmatory fit by sample size (simulated data, ML estimator)",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_fit_by_n.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="results_controls")
    ap.add_argument("--reps", type=int, default=200, help="replications for the fast controls")
    ap.add_argument("--reps-slow", type=int, default=20, help="replications for CFA controls")
    ap.add_argument("--sweep-reps", type=int, default=10)
    ap.add_argument("--n-per-group", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--quick", action="store_true", help="small replication counts, for a smoke test")
    a = ap.parse_args()

    if a.quick:
        a.reps, a.reps_slow, a.sweep_reps = 25, 5, 3

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78, f"\n{STAMP}\n", "=" * 78, sep="")
    t0 = time.time()

    print(f"\n1. Negative controls  ({a.reps} reps fast, {a.reps_slow} slow) ...")
    table, curve = ctl.run_all(reps_fast=a.reps, reps_slow=a.reps_slow,
                               n_per_group=a.n_per_group, seed=a.seed)
    table.to_csv(out / "negative_controls.csv", index=False)
    curve.to_csv(out / "power_curve.csv", index=False)
    _tex(table, out / "negative_controls.tex",
         "Negative controls and power, simulated data", "tab:negcontrols",
         widths=r"p{0.20\columnwidth}p{0.06\columnwidth}p{0.05\columnwidth}"
                r"p{0.28\columnwidth}p{0.30\columnwidth}")
    print(table.to_string(index=False))

    print(f"\n2. Fit sweep  ({a.sweep_reps} reps per sample size) ...")
    sweep = ctl.fit_sweep(reps=a.sweep_reps, seed=a.seed)
    sweep.to_csv(out / "fit_sweep.csv", index=False)
    summary = ctl.fit_sweep_summary(sweep)
    summary.to_csv(out / "fit_sweep_summary.csv", index=False)
    _tex(summary[["N", "converged", "cfi_median", "cfi_q25", "cfi_q75",
                  "rmsea_median", "reading"]],
         out / "fit_sweep_summary.tex",
         "Confirmatory fit by sample size with replication, simulated data, "
         "maximum likelihood estimator", "tab:cfabyn")
    print(summary.to_string(index=False))

    print("\n3. Figure ...")
    try:
        make_figure(sweep, out)
        print("   fig_fit_by_n.png and .pdf written")
    except Exception as e:
        print(f"   figure skipped: {type(e).__name__}: {e}")

    print("\n4. Ordinal reliability ...")
    df, _, _ = mc.simulate(a.n_per_group, a.seed)
    rel = od.reliability_table(df, mc.DIMENSIONS)
    rel.to_csv(out / "reliability_ordinal.csv", index=False)
    _tex(rel, out / "reliability_ordinal.tex",
         "Internal consistency under continuous and ordinal treatment of the "
         "five-point items, simulated data", "tab:ordinalalpha")
    print(rel.to_string(index=False))

    print("\n5. lavaan script for WLSMV re-estimation ...")
    csv_path, r_path = od.write_lavaan_script(df, mc.DIMENSIONS, out / "lavaan")
    print(f"   {r_path.relative_to(out)} and {csv_path.relative_to(out)}")
    print("   Python cannot do WLSMV reliably; run that script in R and paste the")
    print("   scaled fit indices beside the ML columns in the draft's fit table.")

    print(f"\n{'=' * 78}\n{STAMP}\nartifacts in {out}/   elapsed {time.time() - t0:.0f}s\n{'=' * 78}")


if __name__ == "__main__":
    main()
