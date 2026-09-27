"""Tests for the log -> feature -> score path.

These assert that the machinery behaves correctly: the schema check catches
broken logs, features respond to the behaviours they target, the score mapping
stays in range, and calibration reproduces the distribution it is fitted to.
None of them assert that a feature measures its construct — that is a question
about real transcripts, not about code.
"""
import json

import numpy as np
import pandas as pd
import pytest

from cdm import autoscore as asc
from cdm import features as ft
from cdm import logs as lg
from cdm.simulate_logs import simulate_logs, write_jsonl


def _turns(*texts, session="T1", student=1, discipline="Business"):
    """Build a turn frame alternating student / AI turns."""
    rows = []
    for i, t in enumerate(texts):
        rows.append(dict(session_id=session, student=student, discipline=discipline,
                         turn=i, role="user" if i % 2 == 0 else "assistant",
                         content=t, model="test-model-0"))
    return pd.DataFrame(rows)


# --- schema and loading -----------------------------------------------------

def test_load_jsonl_roundtrip(tmp_path):
    recs = simulate_logs(n_per_group=2, sessions_per_student=1, seed=1)
    p = write_jsonl(recs, tmp_path / "l.jsonl")
    df = lg.load_jsonl(p)
    assert len(df) == len(recs)
    assert set(lg.REQUIRED_FIELDS) <= set(df.columns)


def test_load_jsonl_rejects_missing_field(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"session_id": "a", "student": 1, "role": "user"}) + "\n")
    with pytest.raises(ValueError, match="missing field"):
        lg.load_jsonl(p)


def test_load_jsonl_rejects_malformed_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        lg.load_jsonl(p)


def test_validate_catches_duplicate_turns_and_missing_model():
    df = _turns("hello", "hi there")
    df.loc[1, "turn"] = 0                      # duplicate index
    df = df.drop(columns=["model"])
    problems = lg.validate(df)
    assert any("duplicate turn" in p for p in problems)
    assert any("model" in p for p in problems)


def test_validate_clean_log_has_no_problems():
    recs = simulate_logs(n_per_group=1, sessions_per_student=1, seed=3)
    assert lg.validate(pd.DataFrame(recs)) == []


# --- PII screening ----------------------------------------------------------

def test_pii_hits_finds_common_identifiers():
    hits = lg.pii_hits("Contact jane.doe@example.com or 903-555-0134, DOB 03/14/1991")
    assert hits.get("email") == 1
    assert hits.get("phone") == 1
    assert hits.get("dob_like") == 1


def test_pii_hits_clean_text():
    assert lg.pii_hits("Summarise the second chapter in three bullets.") == {}


def test_screen_log_orders_by_severity():
    df = pd.concat([
        _turns("clean prompt about hypertension", "ok", session="A", student=1),
        _turns("email me at a@b.com and call 903-555-0134", "ok", session="B", student=2),
    ], ignore_index=True)
    rep = lg.screen_log(df)
    assert rep.iloc[0].session_id == "B"
    assert rep.iloc[0].total_hits >= 2


# --- feature response -------------------------------------------------------

def test_constraints_raise_constraint_density():
    bare = ft.session_features(_turns("Write about hypertension", "Here is a draft."))
    rich = ft.session_features(_turns(
        "Write about hypertension in under 200 words, as a table, for a lay reader. "
        "Don't include dosing.", "Here is a draft."))
    assert rich["constraint_marker_density"] > bare["constraint_marker_density"]


def test_provenance_and_source_features_respond():
    f = ft.session_features(_turns(
        "Summarise the guideline", "Here is a summary.",
        "Cite your sources. According to the textbook it says otherwise.", "Revised."))
    assert f["provenance_request_count"] >= 1
    assert f["external_source_count"] >= 1


def test_unprompted_ethics_excludes_primed_mentions():
    primed = ft.session_features(_turns(
        "Draft the handout", "One privacy consideration: avoid identifiers.",
        "Right, the privacy issue matters here.", "Understood."))
    spontaneous = ft.session_features(_turns(
        "Draft the handout. I've de-identified the names first.", "Here you go."))
    assert spontaneous["unprompted_ethics_rate"] > primed["unprompted_ethics_rate"]


def test_pii_emission_penalises_privacy_handling():
    leaky = ft.session_features(_turns("Patient ID 44812, DOB 03/14/1991", "Noted."))
    careful = ft.session_features(_turns("I've de-identified the record; no real names.", "Noted."))
    assert leaky["privacy_handling"] < careful["privacy_handling"]


def test_verbatim_reproduction_scores_outside_paraphrase_band():
    ai = "The framing depends on the audience and the evidence is mixed on this point."
    copied = ft.session_features(_turns("Draft it", ai, ai, "ok"))
    ignored = ft.session_features(_turns("Draft it", ai, "Now do the second section please", "ok"))
    spec = asc.CUTPOINTS["CollaborativeSensemaking_i2"]
    lo, hi = spec["band"]
    assert copied["paraphrase_overlap"] > hi        # reproduction
    assert ignored["paraphrase_overlap"] < lo       # no engagement


def test_enumeration_across_lines_counts_subtasks():
    f = ft.session_features(_turns("I need three things:\n1. structure\n2. evidence\n3. wording",
                                   "Here you go."))
    assert f["subtask_count"] >= 3


def test_extract_levels():
    recs = simulate_logs(n_per_group=2, sessions_per_student=2, seed=5)
    df = pd.DataFrame(recs)
    per_session = ft.extract(df, level="session")
    per_student = ft.extract(df, level="student")
    assert len(per_session) == df.session_id.nunique()
    assert len(per_student) == df.student.nunique()
    assert set(ft.FEATURES) <= set(per_student.columns)
    with pytest.raises(ValueError):
        ft.extract(df, level="nonsense")


# --- scoring ----------------------------------------------------------------

def test_band_transform_peaks_inside_band():
    spec = dict(direction="band", band=(0.25, 0.75), cuts=asc._BAND_CUTS)
    assert asc._transform(spec, 0.50) == 0.0
    assert asc._transform(spec, 0.10) < 0.0
    assert asc._transform(spec, 5.00) < asc._transform(spec, 1.00)


def test_score_items_shape_and_range():
    recs = simulate_logs(n_per_group=3, sessions_per_student=2, seed=11)
    feats = ft.extract(pd.DataFrame(recs))
    items = asc.score_items(feats)
    assert len(asc.ITEM_COLUMNS) == 25
    assert set(asc.ITEM_COLUMNS) <= set(items.columns)
    vals = items[asc.ITEM_COLUMNS].values
    assert vals.min() >= 1 and vals.max() <= 5
    for dim in asc.DIMENSIONS:
        assert items[dim].between(1, 5).all()
    assert (items.SCORING_SOURCE == asc.UNCALIBRATED).all()


def test_degeneracy_report_flags_constant_item():
    items = pd.DataFrame({c: [3] * 20 for c in asc.ITEM_COLUMNS})
    items[asc.ITEM_COLUMNS[0]] = list(range(1, 6)) * 4
    rep = asc.degeneracy_report(items).set_index("item")
    assert not rep.loc[asc.ITEM_COLUMNS[0], "degenerate"]
    assert rep.loc[asc.ITEM_COLUMNS[1], "degenerate"]


def test_quantile_cutpoints_remove_degeneracy():
    recs = simulate_logs(n_per_group=10, sessions_per_student=3, seed=13)
    feats = ft.extract(pd.DataFrame(recs))
    prior = asc.degeneracy_report(asc.score_items(feats, asc.CUTPOINTS))
    normed = asc.degeneracy_report(asc.score_items(feats, asc.quantile_cutpoints(feats)))
    assert int(prior.degenerate.sum()) > 0          # a priori cutpoints are guesses
    assert int(normed.degenerate.sum()) == 0


def test_calibration_matches_human_marginals():
    recs, latents = simulate_logs(n_per_group=10, sessions_per_student=3, seed=17,
                                  return_latents=True)
    feats = ft.extract(pd.DataFrame(recs))
    rng = np.random.default_rng(0)
    human = pd.DataFrame({"student": feats.student})
    for item in asc.ITEM_COLUMNS:
        human[item] = rng.integers(1, 6, len(feats))
    cps, report = asc.calibrate(feats, human)
    items = asc.score_items(feats, cps, tag="AUTOMATED_CALIBRATED")
    # equipercentile fitting matches the marginal distribution -- but only for
    # features with enough distinct values to support five ordered categories
    resolved = report[~report.low_resolution].item.tolist()
    assert len(resolved) >= 15
    for item in resolved:
        assert abs(items[item].mean() - human[item].mean()) < 0.75
    assert len(report) == 25
    assert (items.SCORING_SOURCE == "AUTOMATED_CALIBRATED").all()


def test_calibrate_requires_item_columns():
    recs = simulate_logs(n_per_group=2, sessions_per_student=1, seed=19)
    feats = ft.extract(pd.DataFrame(recs))
    with pytest.raises(KeyError):
        asc.calibrate(feats, pd.DataFrame({"student": feats.student}))


def test_agreement_perfect_when_streams_identical():
    recs = simulate_logs(n_per_group=5, sessions_per_student=2, seed=23)
    feats = ft.extract(pd.DataFrame(recs))
    items = asc.score_items(feats)
    human = items[["student"] + asc.ITEM_COLUMNS].copy()
    rep = asc.agreement(items, human)
    nondegenerate = rep[items[asc.ITEM_COLUMNS].nunique().values > 1]
    assert (nondegenerate.weighted_kappa >= 0.999).all()
    assert (rep.exact_match == 1.0).all()


def test_cutpoints_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    asc.save_cutpoints(asc.CUTPOINTS, p)
    back = asc.load_cutpoints(p)
    assert set(back) == set(asc.CUTPOINTS)
    assert back["PromptEngineering_i1"]["cuts"] == asc.CUTPOINTS["PromptEngineering_i1"]["cuts"]


# --- handoff to the existing pipeline ---------------------------------------

def test_auto_items_are_pipeline_compatible():
    """The 25 auto item columns and 5 dimension columns match what pipeline.py reads."""
    from cdm.model_core import DIMENSIONS as MC_DIMS, ITEMS_PER_DIM as MC_IPD
    expected = [f"{d}_i{i+1}" for d in MC_DIMS for i in range(MC_IPD)]
    assert sorted(asc.ITEM_COLUMNS) == sorted(expected)
    recs = simulate_logs(n_per_group=3, sessions_per_student=1, seed=29)
    items = asc.score_items(ft.extract(pd.DataFrame(recs)))
    from cdm.pipeline import cronbach_alpha
    for dim in MC_DIMS:
        cols = [f"{dim}_i{i+1}" for i in range(MC_IPD)]
        a = cronbach_alpha(items[cols].values)
        assert np.isnan(a) or -2.0 <= a <= 1.0


def test_bare_pasted_code_counts_as_context():
    code = ("My loop gives wrong results:\n"
            "#pragma omp parallel for\n"
            "for (i = 0; i < N; i++) { sum = 0; for (k = 0; k < N; k++) sum += A[i][k]*B[k][j]; }")
    f = ft.session_features(_turns(code, "Declare sum private."))
    assert f["context_ratio"] > 0.5
    prose = ft.session_features(_turns("Why is my speedup low (see the notes)? It crashes after sleep(1).", "ok"))
    assert prose["context_ratio"] == 0


def test_inline_enumeration_counts_subtasks():
    f = ft.session_features(_turns("I'll (1) check indexing, (2) check tiling, (3) check bounds.", "ok"))
    assert f["subtask_count"] == 3
    after_colon = ft.session_features(_turns("plan: 1) serial baseline, 2) OpenMP, 3) compare", "ok"))
    assert after_colon["subtask_count"] == 3
    code = ft.session_features(_turns("it hangs at sleep(1) and f(a) in main", "ok"))
    assert code["subtask_count"] == 0
    backref = ft.session_features(_turns("For (2) the docs say to sync threads.", "ok"))
    assert backref["subtask_count"] == 0
