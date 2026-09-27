#!/usr/bin/env python3
"""Process collected student transcripts and derive scoring thresholds.

    python run_pilot.py --logs student_logs.jsonl --profile hpc --outdir results_pilot
    python run_pilot.py --logs student_logs.jsonl --profile hpc \\
                        --human human_items.csv --outdir results_pilot

Convert transcripts to the turn schema first with ``python -m cdm.ingest``.

Without ``--human`` this reports what the corpus looks like and derives
norm-referenced thresholds, which rank cases inside this sample and mean
nothing outside it. With ``--human``, a CSV of coder scores keyed on
``student`` with the twenty-five item columns, thresholds are calibrated and
agreement against the coders is reported.

Read the flat-feature report before the scores. A feature that never fires is
a lexicon problem, and no choice of thresholds repairs it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cdm import features as ft
from cdm import lexicons
from cdm.autoscore import agreement, degeneracy_report
from cdm.logs import load_jsonl, screen_log, validate
from cdm.thresholds import (ThresholdSet, compare_to_prior, derive,
                            distribution_report, flat_features, length_baseline)

BANNER = "COLLECTED STUDENT TRANSCRIPTS — scores are provisional until calibrated"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", required=True, help="JSONL turn log")
    ap.add_argument("--profile", default="hpc", choices=sorted(lexicons.PROFILES),
                    help="marker lexicon profile (default: hpc)")
    ap.add_argument("--human", help="CSV of human item scores keyed on student")
    ap.add_argument("--level", default="student", choices=["student", "session"])
    ap.add_argument("--corpus", default=None, help="name recorded in the threshold file")
    ap.add_argument("--outdir", default="results_pilot")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78, f"\n{BANNER}\n", "=" * 78, sep="")

    df = load_jsonl(a.logs)
    problems = validate(df)
    print(f"\n1. Log          {len(df)} turns, {df.session_id.nunique()} sessions, "
          f"{df.student.nunique()} students")
    if problems:
        print("   validation   PROBLEMS — fix these before trusting anything below:")
        for p in problems:
            print(f"      - {p}")
    else:
        print("   validation   clean")

    scr = screen_log(df)
    scr.to_csv(out / "pii_screen.csv", index=False)
    flagged = int((scr.total_hits > 0).sum())
    print(f"\n2. PII screen   {flagged}/{len(scr)} sessions contain apparent identifiers")
    if flagged:
        print("   Review those transcripts by hand. The screen has no capacity to detect")
        print("   identifying narrative and does not make this corpus releasable.")

    ft.use_profile(a.profile)
    feats = ft.extract(df, level=a.level)
    feats.to_csv(out / "features.csv", index=False)
    print(f"\n3. Features     {len(feats)} rows, lexicon profile '{lexicons.active()}', "
          f"nlp backend '{feats.nlp_backend.iloc[0]}'")

    dist = distribution_report(feats)
    dist.to_csv(out / "feature_distributions.csv", index=False)
    flat = flat_features(feats)
    flat.to_csv(out / "flat_features.csv", index=False)
    print(f"\n4. Coverage     {len(flat)}/25 features carry no usable signal in this corpus")
    if len(flat):
        for _, r in flat.iterrows():
            tag = "  <-- lexicon suspect" if r.lexicon_suspect else ""
            print(f"      {r.feature:<28} zero in {r.zero_share:.0%} of cases, "
                  f"{r.distinct} distinct value(s){tag}")
        if bool(flat.lexicon_suspect.any()):
            print("\n   Features marked lexicon suspect are domain-bound. If this corpus is")
            print("   HPC coursework scored under the general profile, rerun with")
            print("   --profile hpc before concluding the behaviour is absent.")

    human = pd.read_csv(a.human) if a.human else None
    ts, items = derive(feats, human, corpus=a.corpus or Path(a.logs).stem)
    ts.save(out / "thresholds.json")
    items.to_csv(out / "auto_items.csv", index=False)
    print(f"\n5. Thresholds\n{ts.summary()}")

    shift = compare_to_prior(ts)
    shift.to_csv(out / "threshold_shift.csv", index=False)
    print(f"\n   Median shift from the shipped a priori cutpoints: "
          f"{shift.mean_rel_shift.median():.2f}x")
    print("   Large movement is expected. It is the quantitative form of the claim")
    print("   that absolute thresholds do not transfer between corpora.")

    deg = degeneracy_report(items)
    deg.to_csv(out / "degeneracy.csv", index=False)
    bad = deg[deg.degenerate]
    print(f"\n6. Degeneracy   {len(bad)}/25 items still pile on one score after derivation")
    for _, r in bad.iterrows():
        print(f"      {r['item']:<32} {r['modal_share']:.0%} score {r['modal_score']}")

    if human is not None:
        rep = agreement(items, human)
        rep.to_csv(out / "auto_vs_human_agreement.csv", index=False)
        n_ok = int(rep["meets_0.80"].sum())
        print(f"\n7. Agreement    {n_ok}/25 items reach weighted kappa >= 0.80 against "
              "human coding")
        print(f"   median kappa {rep.weighted_kappa.median():.3f}, "
              f"{int((rep.weighted_kappa < 0.40).sum())} items below 0.40")
        worst = rep.nsmallest(3, "weighted_kappa")
        for _, r in worst.iterrows():
            print(f"      {r['item']:<32} kappa {r.weighted_kappa:+.3f}")
        print("   An item that cannot meet the standard demanded of a human coder has")
        print("   not been automated, whatever its other properties.")

        base = length_baseline(feats, human)
        base.to_csv(out / "length_baseline.csv", index=False)
        print("\n8. Length baseline")
        print("   " + base.to_string(index=False).replace("\n", "\n   "))
        print("   If the single length feature matches the twenty-five, the feature")
        print("   engineering is not doing the work. The draft commits to discarding it.")
    else:
        print("\n7. Agreement    not computed: no human scores supplied")
        print("   Scores above are norm-referenced. They rank students within this")
        print("   corpus and carry no criterion meaning. Code a double-coded subset")
        print("   and rerun with --human to calibrate.")

    manifest = dict(logs=str(a.logs), profile=lexicons.active(),
                    nlp_backend=str(feats.nlp_backend.iloc[0]), level=a.level,
                    n_rows=int(len(feats)), n_sessions=int(df.session_id.nunique()),
                    n_turns=int(len(df)), provenance=ts.provenance,
                    human_coded=int(ts.n_human_coded), flat_features=ts.flat_features,
                    degenerate_items=ts.degenerate_items)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 78}\n{BANNER}\nartifacts in {out}/\n{'=' * 78}")


if __name__ == "__main__":
    main()
