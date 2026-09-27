"""Deriving scoring thresholds from a corpus, and reporting what they rest on.

Two quantities get called "thresholds" and they are not the same thing.

The *scale* of a feature is the range a real corpus produces. It is obtainable
from transcripts alone, with no human coding, and the shipped a priori
cutpoints are guesses about it that no corpus is obliged to honour.

The *cutpoints* are where that scale is divided so a 4 means what a human
coder's 4 means. They require human scores on the same sessions. Nothing else
substitutes: norm-referencing produces spread and rank order within one sample
and says nothing about what a score means outside it.

This module makes the distinction operational. :func:`distribution_report`
answers the first question from transcripts. :func:`flat_features` says which
features carry no signal in this corpus, which is a lexicon problem rather
than a threshold problem. :func:`derive` produces cutpoints at a stated
provenance level and records what it was given.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cdm.autoscore import (CUTPOINTS, ITEM_COLUMNS, NORM_PROPS, _transform,
                           agreement, calibrate, degeneracy_report,
                           quantile_cutpoints, score_items)
from cdm.features import FEATURES

MIN_CALIBRATION_CASES = 60
MIN_USABLE_CASES = 20


@dataclass
class ThresholdSet:
    """Cutpoints plus the evidence that produced them.

    A cutpoint file without this record is unusable six months later: nobody
    can tell whether the numbers were fitted, norm-referenced, or guessed.
    """
    provenance: str
    cutpoints: dict
    n_cases: int
    corpus: str
    lexicon_profile: str
    nlp_backend: str
    derived_on: str = field(default_factory=lambda: date.today().isoformat())
    n_human_coded: int = 0
    flat_features: list = field(default_factory=list)
    degenerate_items: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> "ThresholdSet":
        return ThresholdSet(**json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        lines = [f"provenance       {self.provenance}",
                 f"derived on       {self.derived_on}",
                 f"corpus           {self.corpus}  (n = {self.n_cases})",
                 f"lexicon / nlp    {self.lexicon_profile} / {self.nlp_backend}"]
        if self.provenance == "AUTOMATED_CALIBRATED":
            lines.append(f"human-coded      {self.n_human_coded} cases")
        if self.flat_features:
            lines.append(f"flat features    {len(self.flat_features)}: "
                         + ", ".join(self.flat_features[:6])
                         + (" ..." if len(self.flat_features) > 6 else ""))
        if self.degenerate_items:
            lines.append(f"degenerate items {len(self.degenerate_items)}")
        lines += [f"note             {n}" for n in self.notes]
        return "\n".join(lines)


def distribution_report(features: pd.DataFrame) -> pd.DataFrame:
    """Per-feature distribution on this corpus.

    ``zero_share`` and ``distinct`` are the two columns to read first. A
    feature that is zero for most sessions has no low end to cut, and a
    feature with fewer distinct values than the scale has categories cannot
    support a five-point score under any cutpoints.
    """
    rows = []
    for f in FEATURES:
        v = features[f].astype(float).values
        rows.append(dict(
            feature=f,
            n=len(v),
            distinct=int(len(np.unique(v))),
            zero_share=round(float((v == 0).mean()), 3),
            min=round(float(np.min(v)), 4),
            q25=round(float(np.quantile(v, 0.25)), 4),
            median=round(float(np.median(v)), 4),
            q75=round(float(np.quantile(v, 0.75)), 4),
            max=round(float(np.max(v)), 4),
            iqr=round(float(np.quantile(v, 0.75) - np.quantile(v, 0.25)), 4),
        ))
    return pd.DataFrame(rows)


def flat_features(features: pd.DataFrame, zero_threshold: float = 0.90,
                  min_distinct: int = 3) -> pd.DataFrame:
    """Features carrying no usable signal in this corpus.

    Distinguish the two causes before acting. A feature that is flat because
    the behaviour is genuinely rare is a finding about the cohort. A feature
    that is flat because its lexicon was written for another domain is a bug
    in the lexicon, and no choice of cutpoints repairs it. The
    ``lexicon_suspect`` column flags the families most likely to be the second
    case when a general profile is run on domain-specific coursework.
    """
    rep = distribution_report(features)
    flat = rep[(rep.zero_share >= zero_threshold) | (rep.distinct < min_distinct)].copy()
    domain_bound = {"unprompted_ethics_rate", "privacy_handling", "bias_marker_count",
                    "stakeholder_marker_count", "attribution_marker_count",
                    "external_source_count", "provenance_request_count", "reconcile_count"}
    flat["lexicon_suspect"] = flat.feature.isin(domain_bound)
    return flat.reset_index(drop=True)


def derive(features: pd.DataFrame, human_items: pd.DataFrame | None = None,
           corpus: str = "unnamed", props=NORM_PROPS) -> tuple[ThresholdSet, pd.DataFrame]:
    """Produce the strongest cutpoints the available evidence supports.

    With human scores, cutpoints are fitted by equipercentile matching and the
    result is stamped calibrated. Without them, cutpoints come from the
    observed feature distribution and the result is stamped norm-referenced,
    which is interpretable only inside this sample. The returned frame is the
    scored item table.
    """
    notes: list[str] = []
    flat = flat_features(features)
    n = len(features)
    profile = str(features.get("lexicon_profile", pd.Series(["unknown"])).iloc[0])
    backend = str(features.get("nlp_backend", pd.Series(["unknown"])).iloc[0])

    if n < MIN_USABLE_CASES:
        notes.append(f"n = {n} is below {MIN_USABLE_CASES}; quantile estimates are unstable "
                     "at the tails, which is exactly where scores of 1 and 5 are decided")

    if human_items is None:
        cps = quantile_cutpoints(features, props=props)
        prov = "AUTOMATED_NORM_REFERENCED"
        notes.append("no human scores supplied; cutpoints rank cases within this sample "
                     "and carry no criterion meaning outside it")
        n_human = 0
    else:
        paired = features.merge(human_items, on="student")
        n_human = len(paired)
        if n_human < MIN_CALIBRATION_CASES:
            notes.append(f"calibrated on {n_human} cases, below the {MIN_CALIBRATION_CASES} "
                         "double-coded sessions that make fitted cutpoints stable")
        cps, _ = calibrate(features, human_items)
        prov = "AUTOMATED_CALIBRATED"

    items = score_items(features, cps, tag=prov)
    deg = degeneracy_report(items)
    degenerate = deg[deg.degenerate].item.tolist()
    if degenerate:
        notes.append(f"{len(degenerate)} item(s) still degenerate after deriving cutpoints; "
                     "these are feature problems, not threshold problems")

    ts = ThresholdSet(provenance=prov, cutpoints=cps, n_cases=n, corpus=corpus,
                      lexicon_profile=profile, nlp_backend=backend,
                      n_human_coded=n_human,
                      flat_features=flat.feature.tolist(),
                      degenerate_items=degenerate, notes=notes)
    return ts, items


def compare_to_prior(ts: ThresholdSet) -> pd.DataFrame:
    """How far the derived cutpoints sit from the shipped a priori guesses.

    Large movement is expected and is the point: it is the quantitative form
    of the claim that absolute thresholds do not transfer between corpora.
    """
    rows = []
    for item, spec in CUTPOINTS.items():
        prior = np.asarray(spec["cuts"], float)
        fitted = np.asarray(ts.cutpoints[item]["cuts"], float)
        denom = np.where(np.abs(prior) > 1e-9, np.abs(prior), 1.0)
        rows.append(dict(item=item, feature=spec["feature"],
                         prior=[round(c, 3) for c in prior],
                         derived=[round(c, 3) for c in fitted],
                         mean_abs_shift=round(float(np.mean(np.abs(fitted - prior))), 4),
                         mean_rel_shift=round(float(np.mean(np.abs(fitted - prior) / denom)), 3)))
    return pd.DataFrame(rows).sort_values("mean_rel_shift", ascending=False).reset_index(drop=True)


def length_baseline(features: pd.DataFrame, human_items: pd.DataFrame,
                    length_col: str = "n_student_turns") -> pd.DataFrame:
    """The comparison that decides whether the feature set earns its complexity.

    Scores every item from one quantity, the amount the student wrote, using
    the same equipercentile procedure, and reports agreement against the same
    human scores. If this matches the twenty-five-feature scorer, the feature
    engineering is not doing the work and should be discarded in favour of the
    simpler measure. The falsification conditions in the paper commit to that.
    """
    stub = features.copy()
    for f in FEATURES:
        stub[f] = features[length_col].astype(float)
    cps, _ = calibrate(stub, human_items)
    base_items = score_items(stub, cps, tag="LENGTH_BASELINE")
    full_cps, _ = calibrate(features, human_items)
    full_items = score_items(features, full_cps, tag="AUTOMATED_CALIBRATED")

    rows = []
    for label, items in (("student length only", base_items), ("25 features", full_items)):
        rep = agreement(items, human_items)
        rows.append(dict(method=label,
                         median_kappa=round(float(rep.weighted_kappa.median()), 3),
                         items_at_080=int(rep["meets_0.80"].sum()),
                         items_below_040=int((rep.weighted_kappa < 0.40).sum())))
    return pd.DataFrame(rows)
