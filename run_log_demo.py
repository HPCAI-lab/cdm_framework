#!/usr/bin/env python3
"""Demonstration of the log -> feature -> score path on SIMULATED logs.

    python run_log_demo.py --outdir results_logs

What this shows: the path runs end to end, the features respond to the
behaviours they are meant to respond to, the degeneracy check catches bad
cutpoints, and the calibration and agreement machinery works.

What this does NOT show: that the features measure the CDM constructs. The
"human" scores here are generated from the same latent levels that drove the
text markers, so agreement between them is circular by construction. Real
validity evidence requires human coders scoring real transcripts with
docs/rubric.md, and is the point of the calibration subset in the pilot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cdm.autoscore import (CUTPOINTS, DIMENSIONS, ITEMS_PER_DIM, agreement,
                           calibrate, degeneracy_report, quantile_cutpoints,
                           save_cutpoints, score_items)
from cdm.features import extract
from cdm.logs import load_jsonl, screen_log, validate
from cdm.model_core import ITEM_DIFFICULTY, ITEM_UNIQUE_SD, RATER_NOISE_SD
from cdm.simulate_logs import simulate_logs, write_jsonl

STAMP = "SIMULATED LOGS — PATH DEMONSTRATION (not validity evidence)"


def simulated_human_items(latents: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """A stand-in expert coder, scoring from the same latent levels.

    Uses the item-difficulty and noise parameters already declared in
    model_core, so the "human" stream is consistent with the rest of the repo.
    Circular against the log features by design — see module docstring.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in latents.iterrows():
        row = {"student": int(r.student)}
        for dim in DIMENSIONS:
            for i in range(ITEMS_PER_DIM):
                x = (r[dim] + ITEM_DIFFICULTY[i]
                     + rng.normal(0, ITEM_UNIQUE_SD) + rng.normal(0, RATER_NOISE_SD))
                row[f"{dim}_i{i+1}"] = int(np.clip(round(x), 1, 5))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-group", type=int, default=10)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--outdir", default="results_logs")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    print("=" * 78, f"\n{STAMP}\n", "=" * 78, sep="")

    records, latents = simulate_logs(a.n_per_group, a.sessions, a.seed, return_latents=True)
    log_path = write_jsonl(records, out / "cdm_simulated_logs.jsonl")
    df = load_jsonl(log_path)
    print(f"\n1. Log         {len(df)} turns, {df.session_id.nunique()} sessions, "
          f"{df.student.nunique()} students  [{log_path.name}]")
    problems = validate(df)
    print(f"   validation   {'clean' if not problems else problems}")

    scr = screen_log(df)
    scr.to_csv(out / "pii_screen.csv", index=False)
    print(f"2. PII screen  {int((scr.total_hits > 0).sum())}/{len(scr)} sessions contain apparent "
          f"identifiers — manual review before any release")

    feats = extract(df, level="student")
    feats.to_csv(out / "cdm_log_features.csv", index=False)
    print(f"3. Features    {len(feats)} students x 25 features (backend: {feats.nlp_backend.iloc[0]})")

    human = simulated_human_items(latents)
    human.to_csv(out / "simulated_human_items.csv", index=False)

    # --- a priori cutpoints
    items_prior = score_items(feats, CUTPOINTS)
    items_prior.to_csv(out / "auto_items_apriori.csv", index=False)
    deg = degeneracy_report(items_prior)
    deg.to_csv(out / "degeneracy_apriori.csv", index=False)
    bad = deg[deg.degenerate]
    print(f"4. A priori    {len(bad)}/25 items degenerate"
          + ("".join(f"\n                 {r['item']} — {r['modal_share']:.0%} score {r['modal_score']}"
                     for _, r in bad.iterrows()) if len(bad) else ""))

    # --- norm-referenced cutpoints
    cps_norm = quantile_cutpoints(feats)
    save_cutpoints(cps_norm, out / "cutpoints_norm.json")
    items_norm = score_items(feats, cps_norm, tag="AUTOMATED_NORM_REFERENCED")
    items_norm.to_csv(out / "auto_items_norm.csv", index=False)
    deg_n = degeneracy_report(items_norm)
    print(f"5. Norm-ref    {int(deg_n.degenerate.sum())}/25 items degenerate "
          f"(spread restored; still no criterion validity)")

    # --- calibration against the (simulated) human stream
    cps_cal, report = calibrate(feats, human)
    save_cutpoints(cps_cal, out / "cutpoints_calibrated.json")
    report.to_csv(out / "cutpoint_calibration.csv", index=False)
    items_cal = score_items(feats, cps_cal, tag="AUTOMATED_CALIBRATED")
    items_cal.to_csv(out / "auto_items_calibrated.csv", index=False)

    rep = agreement(items_cal, human)
    rep.to_csv(out / "auto_vs_human_agreement.csv", index=False)
    print(f"6. Calibrated  cutpoints fitted to the human marginals; "
          f"{int(rep['meets_0.80'].sum())}/25 items reach weighted kappa >= 0.80")
    print("\n   per-item agreement (automated vs simulated human coder):")
    print("   " + rep.to_string(index=False).replace("\n", "\n   "))

    med = float(rep.weighted_kappa.median())
    print(f"\n   median weighted kappa = {med:.3f}")
    print("   Read this as a working agreement harness, not a result: the two")
    print("   streams share a latent origin, so the ceiling here is set by")
    print("   simulation noise rather than by whether the features track the")
    print("   constructs. On real double-coded transcripts this number is the")
    print("   one that decides whether any item can be automated at all.")

    print(f"\n{'=' * 78}\n{STAMP}\nArtifacts in {out}/\n{'=' * 78}")


if __name__ == "__main__":
    main()
