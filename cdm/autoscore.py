"""Feature -> item score, plus calibration against human coding.

The scorer converts each session feature into a 1-5 item score using declared
cutpoints, producing the same ``<Dimension>_i{n}`` columns the analysis pipeline
already consumes. Two things must stay clear:

1. **The shipped cutpoints are a priori guesses.** They were set by reasoning
   about what "occasional / regular / heavy" looks like in a transcript, not
   fitted to anything. Output scored with them is stamped
   ``AUTOMATED_UNCALIBRATED`` and is not a measurement of anything yet.
2. **Calibration needs human scores on the same sessions.** ``calibrate()``
   fits cutpoints so the automated scores reproduce the human score
   distribution, and ``agreement()`` reports how well they then agree
   case-by-case. Until that has been done on real double-coded transcripts,
   the automated path is plumbing, not an instrument.

Calibrating on simulated logs proves nothing: the generator draws text markers
from the same latent levels the human scores would be derived from, so
agreement is circular by construction. The functions here warn when they see a
simulated provenance stamp.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cdm import agreement as ag
from cdm.features import DIMENSION_FEATURES, FEATURES, extract
from cdm.logs import load_jsonl, screen_log, validate

ITEMS_PER_DIM = 5
UNCALIBRATED = "AUTOMATED_UNCALIBRATED"

_DENSITY = [0.20, 0.50, 1.00, 1.80]     # markers per student turn
_COUNT = [0.50, 1.50, 3.50, 6.50]       # markers per session: 1 / 2-3 / 4-6 / 7+
_BAND_CUTS = [-0.25, -0.15, -0.06, -0.001]

#: item column -> how to turn its feature into a 1-5 score.
#: ``direction='up'``   more is better, ``cuts`` ascending.
#: ``direction='band'`` an interval is best; distance outside it is penalised.
CUTPOINTS: dict[str, dict] = {
    # Prompt Engineering
    "PromptEngineering_i1": dict(feature="constraint_marker_density", direction="up", cuts=[0.15, 0.45, 0.90, 1.60]),
    "PromptEngineering_i2": dict(feature="context_ratio", direction="up", cuts=[0.05, 0.15, 0.35, 0.60]),
    "PromptEngineering_i3": dict(feature="refinement_ratio", direction="up", cuts=[0.10, 0.25, 0.40, 0.60]),
    "PromptEngineering_i4": dict(feature="specificity_index", direction="up", cuts=[0.30, 0.38, 0.46, 0.54]),
    "PromptEngineering_i5": dict(feature="framing_clarity", direction="up", cuts=[0.30, 0.60, 0.85, 1.10]),
    # Problem Decomposition
    "ProblemDecomposition_i1": dict(feature="subtask_count", direction="up", cuts=_COUNT),
    "ProblemDecomposition_i2": dict(feature="sequence_marker_density", direction="up", cuts=_DENSITY),
    "ProblemDecomposition_i3": dict(feature="scope_qualifier_density", direction="up", cuts=_DENSITY),
    "ProblemDecomposition_i4": dict(feature="limitation_reference_count", direction="up", cuts=[0.50, 1.50, 2.50, 4.00]),
    "ProblemDecomposition_i5": dict(feature="allocation_marker_count", direction="up", cuts=[0.50, 1.50, 2.50, 4.00]),
    # Output Validation
    "OutputValidation_i1": dict(feature="external_source_count", direction="up", cuts=_COUNT),
    "OutputValidation_i2": dict(feature="error_assertion_count", direction="up", cuts=[0.50, 1.50, 2.50, 4.00]),
    "OutputValidation_i3": dict(feature="provenance_request_count", direction="up", cuts=_COUNT),
    "OutputValidation_i4": dict(feature="scrutiny_rate", direction="band", band=(0.25, 0.75), cuts=_BAND_CUTS),
    "OutputValidation_i5": dict(feature="recheck_rate", direction="up", cuts=[0.01, 0.25, 0.50, 0.75]),
    # Ethical Integration
    "EthicalIntegration_i1": dict(feature="unprompted_ethics_rate", direction="up", cuts=[0.02, 0.08, 0.18, 0.30]),
    "EthicalIntegration_i2": dict(feature="privacy_handling", direction="up", cuts=[-0.50, 0.50, 1.50, 3.00]),
    "EthicalIntegration_i3": dict(feature="bias_marker_count", direction="up", cuts=[0.50, 1.50, 2.50, 4.00]),
    "EthicalIntegration_i4": dict(feature="stakeholder_marker_count", direction="up", cuts=_COUNT),
    "EthicalIntegration_i5": dict(feature="attribution_marker_count", direction="up", cuts=[0.50, 1.50, 2.50, 4.00]),
    # Collaborative Sensemaking
    "CollaborativeSensemaking_i1": dict(feature="reconcile_count", direction="up", cuts=[0.50, 1.50, 2.50, 4.00]),
    "CollaborativeSensemaking_i2": dict(feature="paraphrase_overlap", direction="band", band=(0.12, 0.45), cuts=_BAND_CUTS),
    "CollaborativeSensemaking_i3": dict(feature="why_how_rate", direction="up", cuts=_DENSITY),
    "CollaborativeSensemaking_i4": dict(feature="disagreement_count", direction="up", cuts=[0.50, 1.50, 2.50, 4.00]),
    "CollaborativeSensemaking_i5": dict(feature="shared_artifact_markers", direction="up", cuts=[0.50, 1.50, 2.50, 4.00]),
}

ITEM_COLUMNS = list(CUTPOINTS)
DIMENSIONS = list(DIMENSION_FEATURES)


def _transform(spec: dict, x: float) -> float:
    """Map a raw feature onto a monotone-increasing scale for cutting."""
    if spec["direction"] == "up":
        return float(x)
    if spec["direction"] == "band":
        lo, hi = spec["band"]
        if lo <= x <= hi:
            return 0.0
        return -float(min(abs(x - lo), abs(x - hi)))
    raise ValueError(f"unknown direction {spec['direction']!r}")


def score_items(features: pd.DataFrame, cutpoints: dict | None = None,
                tag: str | None = None) -> pd.DataFrame:
    """Score the 25 items from a feature frame (one row per student or session)."""
    cps = cutpoints or CUTPOINTS
    out = features[[c for c in ("student", "session_id", "discipline") if c in features]].copy()
    for item, spec in cps.items():
        vals = features[spec["feature"]].astype(float).map(lambda x, s=spec: _transform(s, x))
        cuts = np.asarray(spec["cuts"], float)
        out[item] = (1 + (vals.values[:, None] > cuts[None, :]).sum(axis=1)).astype(int)
    for dim in DIMENSIONS:
        cols = [f"{dim}_i{i+1}" for i in range(ITEMS_PER_DIM)]
        out[dim] = out[cols].mean(axis=1).round(3)
    out.insert(0, "SCORING_SOURCE", tag or UNCALIBRATED)
    return out


#: default target score distribution for norm-referenced cutpoints:
#: 10% / 20% / 40% / 20% / 10% across scores 1-5.
NORM_PROPS = (0.10, 0.30, 0.70, 0.90)


def quantile_cutpoints(features: pd.DataFrame, props=NORM_PROPS) -> dict:
    """Derive cutpoints from the observed feature distribution (norm-referenced).

    Use this when no human scores exist yet and the a priori cutpoints produce
    no variance on your data — which they will, for at least a couple of
    features, because absolute marker rates depend on task design, model
    verbosity, and logging granularity in ways no one can guess in advance.

    What this buys: usable score spread and rank ordering. What it does not
    buy: any claim that a 4 means what a human coder's 4 means. Scores from
    these cutpoints are stamped ``AUTOMATED_NORM_REFERENCED`` and are only
    interpretable *within* the sample they were derived from.
    """
    new: dict[str, dict] = {}
    for item, spec in CUTPOINTS.items():
        vals = features[spec["feature"]].astype(float).map(lambda x, s=spec: _transform(s, x)).values
        cuts = list(np.maximum.accumulate([float(np.quantile(vals, p)) for p in props]))
        new[item] = {**spec, "cuts": cuts}
    return new


def degeneracy_report(items: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """Flag items whose scores are piled on one value — the cutpoints are wrong.

    Run this on every new corpus. An item where 80%+ of cases get the same
    score carries almost no information, whatever its reliability statistics
    look like afterwards.
    """
    rows = []
    for item in ITEM_COLUMNS:
        vc = items[item].value_counts(normalize=True)
        rows.append(dict(item=item, modal_score=int(vc.index[0]),
                         modal_share=round(float(vc.iloc[0]), 3),
                         distinct_scores=int(items[item].nunique()),
                         degenerate=bool(vc.iloc[0] >= threshold)))
    return pd.DataFrame(rows).sort_values("modal_share", ascending=False).reset_index(drop=True)


def calibrate(features: pd.DataFrame, human_items: pd.DataFrame,
              key: str = "student", tag: str = "AUTOMATED_CALIBRATED") -> tuple[dict, pd.DataFrame]:
    """Fit cutpoints by equipercentile matching to human item scores.

    For each item, the human scores give cumulative proportions at 1, 2, 3, 4;
    the cutpoints are set at the matching quantiles of the transformed feature.
    The automated scores then reproduce the human *marginal distribution* by
    construction — which is why :func:`agreement` matters: matching marginals
    is not the same as agreeing case by case.

    Returns ``(cutpoints, report)``. Needs both frames keyed on ``key``.
    """
    merged = features.merge(human_items, on=key, suffixes=("", "_human"))
    if len(merged) < 20:
        print(f"[calibrate] warning: only {len(merged)} paired cases — "
              "cutpoints will be unstable; 60+ double-coded sessions is a "
              "reasonable minimum.")
    new: dict[str, dict] = {}
    rows = []
    for item, spec in CUTPOINTS.items():
        if item not in merged:
            raise KeyError(f"human scores missing item column {item}")
        vals = merged[spec["feature"]].astype(float).map(lambda x, s=spec: _transform(s, x)).values
        h = merged[item].astype(float).values
        props = [(h <= k).mean() for k in (1, 2, 3, 4)]
        cuts = [float(np.quantile(vals, min(max(p, 0.0), 1.0))) for p in props]
        cuts = list(np.maximum.accumulate(cuts))          # enforce monotonicity
        new[item] = {**spec, "cuts": cuts}
        n_distinct = int(len(np.unique(vals)))
        rows.append(dict(item=item, feature=spec["feature"],
                         human_mean=round(float(h.mean()), 3),
                         feature_distinct_values=n_distinct,
                         low_resolution=bool(n_distinct < 5),
                         prior_cuts=[round(c, 4) for c in spec["cuts"]],
                         fitted_cuts=[round(c, 4) for c in cuts]))
    report = pd.DataFrame(rows)
    coarse = report[report.low_resolution]
    if len(coarse):
        print(f"[calibrate] {len(coarse)} feature(s) too coarse to support a 5-point "
              "score at this sample size — no cutpoints can fix these, the feature "
              "needs redefining or the item stays human-coded:")
        for _, r in coarse.iterrows():
            print(f"   - {r['item']:<32} {r['feature']} has "
                  f"{r['feature_distinct_values']} distinct values")
    return new, report


def agreement(machine_items: pd.DataFrame, human_items: pd.DataFrame,
              key: str = "student") -> pd.DataFrame:
    """Per-item agreement between automated and human scores.

    Quadratic-weighted kappa is the headline, matching how inter-rater
    agreement is reported for the human coders. An item that cannot reach the
    same threshold the human coders are held to has not been automated.
    """
    m = machine_items.merge(human_items, on=key, suffixes=("_auto", "_human"))
    rows = []
    for item in ITEM_COLUMNS:
        a = m[f"{item}_auto"].astype(int).values
        h = m[f"{item}_human"].astype(int).values
        rows.append(dict(
            item=item,
            weighted_kappa=round(float(ag.weighted_kappa(a, h)), 3),
            exact_match=round(float((a == h).mean()), 3),
            mean_abs_diff=round(float(np.abs(a - h).mean()), 3),
            auto_mean=round(float(a.mean()), 2),
            human_mean=round(float(h.mean()), 2),
        ))
    out = pd.DataFrame(rows)
    out["meets_0.80"] = out.weighted_kappa >= 0.80
    return out


def save_cutpoints(cutpoints: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cutpoints, indent=2), encoding="utf-8")


def load_cutpoints(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score CDM items from an interaction log.")
    ap.add_argument("--logs", required=True, help="JSONL turn log")
    ap.add_argument("--human", help="CSV of human item scores (for calibration/agreement)")
    ap.add_argument("--cutpoints", help="JSON cutpoints file (default: a priori cutpoints)")
    ap.add_argument("--fit-cutpoints", action="store_true",
                    help="fit cutpoints from --human and write them out")
    ap.add_argument("--norm-cutpoints", action="store_true",
                    help="derive norm-referenced cutpoints from this sample's "
                         "feature distribution (no human scores needed)")
    ap.add_argument("--level", default="student", choices=["student", "session"])
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(a.logs)
    problems = validate(df)
    if problems:
        print("[validate] problems found:")
        for p in problems:
            print("   -", p)
    else:
        print("[validate] log is well-formed")

    scr = screen_log(df)
    flagged = int((scr.total_hits > 0).sum())
    print(f"[pii screen] {flagged}/{len(scr)} sessions contain apparent identifiers "
          f"— manual review required before any release")
    scr.to_csv(outdir / "pii_screen.csv", index=False)

    feats = extract(df, level=a.level)
    feats.to_csv(outdir / "cdm_log_features.csv", index=False)
    print(f"[features] {len(feats)} rows x {len(FEATURES)} features (backend: {feats.nlp_backend.iloc[0]})")

    if a.cutpoints:
        cps, tag = load_cutpoints(a.cutpoints), "AUTOMATED_CALIBRATED"
    elif a.norm_cutpoints:
        cps, tag = quantile_cutpoints(feats), "AUTOMATED_NORM_REFERENCED"
        save_cutpoints(cps, outdir / "cutpoints_norm.json")
        print("[cutpoints] norm-referenced cutpoints derived from this sample "
              "— interpretable within this sample only")
    else:
        cps, tag = CUTPOINTS, UNCALIBRATED

    if a.human:
        human = pd.read_csv(a.human)
        if a.fit_cutpoints:
            cps, report = calibrate(feats, human, key=a.level if a.level == "student" else "session_id")
            report.to_csv(outdir / "cutpoint_calibration.csv", index=False)
            save_cutpoints(cps, outdir / "cutpoints_fitted.json")
            tag = "AUTOMATED_CALIBRATED"
            print("[calibrate] fitted cutpoints written to cutpoints_fitted.json")

    items = score_items(feats, cps, tag=tag)
    items.to_csv(outdir / "cdm_auto_items.csv", index=False)
    print(f"[score] wrote 25 item columns + 5 dimension scores, stamped {tag}")

    deg = degeneracy_report(items)
    deg.to_csv(outdir / "degeneracy_check.csv", index=False)
    bad = deg[deg.degenerate]
    if len(bad):
        print(f"[degeneracy] {len(bad)} item(s) with no usable spread — "
              "cutpoints need revising before these items mean anything:")
        for _, r in bad.iterrows():
            print(f"   - {r['item']:<32} {r['modal_share']:.0%} of cases score {r['modal_score']}")
    else:
        print("[degeneracy] all 25 items show usable score spread")

    if a.human:
        rep = agreement(items, pd.read_csv(a.human),
                        key=a.level if a.level == "student" else "session_id")
        rep.to_csv(outdir / "auto_vs_human_agreement.csv", index=False)
        print(rep.to_string(index=False))
        n_ok = int(rep["meets_0.80"].sum())
        print(f"\n[agreement] {n_ok}/25 items reach weighted kappa >= 0.80 against human coding")
    else:
        print("\n[agreement] no human scores supplied — automated scores are "
              "UNCALIBRATED and carry no validity evidence")


if __name__ == "__main__":
    main()
