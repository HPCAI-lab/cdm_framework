"""Tests for the transcript-processing and design-time evidence modules.

These check machinery, not measurement: that adapters parse, that the
threshold provenance is recorded honestly, that a flat feature is detected,
and that the controls return the shape the paper's tables need.
"""
import json

import numpy as np
import pandas as pd
import pytest

from cdm import controls as ctl
from cdm import features as ft
from cdm import ingest, lexicons
from cdm import ordinal as od
from cdm import thresholds as th
from cdm.autoscore import ITEM_COLUMNS
from cdm.model_core import DIMENSIONS
from cdm.simulate_logs import simulate_logs

TRANSCRIPT = """User: Parallelise this loop with OpenMP, in under 40 lines.
Assistant: Here is a draft using collapse(2).
User: Why collapse(2)? The docs say the inner loop carries a dependency.
Assistant: Collapsing k would race on the accumulator.
User: That's wrong, I ran it and got a mismatch against the serial baseline.
Assistant: You are right, the reduction clause was missing.
"""


# --- ingest -----------------------------------------------------------------

def test_text_adapter_splits_turns(tmp_path):
    p = tmp_path / "01_s1.txt"
    p.write_text(TRANSCRIPT)
    df = ingest.from_text_transcript(p, model="m-1")
    assert len(df) == 6
    assert list(df.role) == ["user", "assistant"] * 3
    assert (df.model == "m-1").all()


def test_directory_adapter_takes_student_from_stem(tmp_path):
    for i in (3, 7):
        (tmp_path / f"{i:02d}_s1.txt").write_text(TRANSCRIPT)
    df = ingest.from_directory(tmp_path, model="m-1")
    assert set(df.student) == {3, 7}
    assert df.session_id.nunique() == 2


def test_openai_export_adapter(tmp_path):
    conv = {"conversation_id": "c1", "mapping": {
        "a": {"message": {"author": {"role": "user"}, "create_time": 1.0,
                          "content": {"parts": ["hello"]}, "metadata": {"model_slug": "gpt-x"}}},
        "b": {"message": {"author": {"role": "assistant"}, "create_time": 2.0,
                          "content": {"parts": ["hi"]}, "metadata": {}}}}}
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps([conv]))
    df = ingest.from_openai_export(p)
    assert list(df.role) == ["user", "assistant"]
    assert df.model.iloc[0] == "gpt-x"


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="unrecognised role"):
        ingest._role("narrator")


def test_assign_students_pseudonymises(tmp_path):
    p = tmp_path / "01_s1.txt"
    p.write_text(TRANSCRIPT)
    df = ingest.from_text_transcript(p)
    roster = tmp_path / "roster.csv"
    pd.DataFrame({"session_id": ["01_s1"], "student": ["Real Name"]}).to_csv(roster, index=False)
    out, crosswalk = ingest.assign_students(df, roster)
    assert set(out.student) == {1}
    assert "Real Name" not in set(out.student)
    assert "Real Name" in set(crosswalk.student)


def test_ingest_report_flags_lopsided_roles():
    df = pd.DataFrame([dict(session_id="a", student=1, turn=i, role="user",
                            content="x", model="m", discipline=None, phase=None,
                            timestamp=None) for i in range(6)])
    assert "inspect" in ingest.ingest_report(df)


# --- lexicons ---------------------------------------------------------------

def test_profiles_share_marker_families():
    assert set(lexicons.GENERAL) == set(lexicons.HPC)


def test_hpc_profile_fires_on_hpc_language():
    ft.use_profile("hpc")
    try:
        hpc_hits = ft._count(ft.M["error_assert"], "I got a segfault and a race condition")
    finally:
        ft.use_profile("general")
    general_hits = ft._count(lexicons.GENERAL["error_assert"],
                             "I got a segfault and a race condition")
    assert hpc_hits > general_hits


def test_profile_recorded_in_features():
    recs = simulate_logs(n_per_group=2, sessions_per_student=1, seed=3)
    ft.use_profile("hpc")
    try:
        feats = ft.extract(pd.DataFrame(recs))
        assert (feats.lexicon_profile == "hpc").all()
    finally:
        ft.use_profile("general")


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        lexicons.use("klingon")


# --- thresholds -------------------------------------------------------------

def _feats(n_per_group=10, seed=5):
    recs = simulate_logs(n_per_group=n_per_group, sessions_per_student=3, seed=seed)
    return ft.extract(pd.DataFrame(recs))


def test_distribution_report_covers_every_feature():
    rep = th.distribution_report(_feats(4))
    assert len(rep) == len(ft.FEATURES)
    assert {"zero_share", "distinct", "median"} <= set(rep.columns)


def test_flat_features_detects_a_constant_feature():
    feats = _feats(4)
    feats["bias_marker_count"] = 0.0
    flat = th.flat_features(feats)
    assert "bias_marker_count" in set(flat.feature)
    assert bool(flat.set_index("feature").loc["bias_marker_count", "lexicon_suspect"])


def test_derive_without_human_scores_is_norm_referenced(tmp_path):
    ts, items = th.derive(_feats(10), None, corpus="test")
    assert ts.provenance == "AUTOMATED_NORM_REFERENCED"
    assert ts.n_human_coded == 0
    assert any("no human scores" in n for n in ts.notes)
    assert (items.SCORING_SOURCE == "AUTOMATED_NORM_REFERENCED").all()
    p = ts.save(tmp_path / "t.json")
    assert th.ThresholdSet.load(p).provenance == ts.provenance


def test_derive_with_human_scores_is_calibrated_and_warns_on_small_n():
    feats = _feats(10)
    rng = np.random.default_rng(0)
    human = pd.DataFrame({"student": feats.student})
    for c in ITEM_COLUMNS:
        human[c] = rng.integers(1, 6, len(feats))
    ts, items = th.derive(feats, human, corpus="test")
    assert ts.provenance == "AUTOMATED_CALIBRATED"
    assert ts.n_human_coded == len(feats)
    assert any("below the" in n for n in ts.notes)


def test_small_corpus_is_flagged_as_unstable():
    ts, _ = th.derive(_feats(4), None)
    assert any("unstable" in n for n in ts.notes)


def test_compare_to_prior_reports_every_item():
    ts, _ = th.derive(_feats(10), None)
    shift = th.compare_to_prior(ts)
    assert len(shift) == 25
    assert (shift.mean_abs_shift >= 0).all()


def test_length_baseline_returns_both_methods():
    feats = _feats(10)
    rng = np.random.default_rng(1)
    human = pd.DataFrame({"student": feats.student})
    for c in ITEM_COLUMNS:
        human[c] = rng.integers(1, 6, len(feats))
    out = th.length_baseline(feats, human)
    assert set(out.method) == {"student length only", "25 features"}
    assert out.median_kappa.between(-1, 1).all()


# --- ordinal ----------------------------------------------------------------

def test_polychoric_exceeds_pearson_on_ordinal_data():
    rng = np.random.default_rng(2)
    g = rng.standard_normal(300)
    x = np.clip(np.round(3 + g), 1, 5).astype(int)
    y = np.clip(np.round(3 + 0.9 * g + 0.4 * rng.standard_normal(300)), 1, 5).astype(int)
    assert od.polychoric(x, y) > np.corrcoef(x, y)[0, 1]


def test_polychoric_returns_nan_without_variance():
    assert np.isnan(od.polychoric(np.full(20, 3), np.arange(20) % 5 + 1))


def test_reliability_table_orders_alpha_below_ordinal_alpha():
    from cdm.model_core import simulate
    df, _, _ = simulate(10, 20260617)
    tab = od.reliability_table(df, DIMENSIONS)
    assert len(tab) == 5
    assert (tab.ordinal_alpha >= tab.cronbach_alpha - 1e-9).all()


def test_lavaan_script_written(tmp_path):
    from cdm.model_core import simulate
    df, _, _ = simulate(4, 1)
    csv_path, r_path = od.write_lavaan_script(df, DIMENSIONS, tmp_path)
    assert csv_path.exists() and r_path.exists()
    text = r_path.read_text()
    assert "WLSMV" in text and "ordered" in text


# --- controls ---------------------------------------------------------------

def test_null_control_returns_a_rate_and_interval():
    res = ctl.null_control(reps=8, n_per_group=6, seed=1)
    assert res["condition"] == "Null group structure"
    assert "[" in res["observed"]


def test_one_factor_generator_has_expected_shape():
    df = ctl._simulate_one_factor(20, seed=3)
    assert len(df) == 20
    assert set(ctl.ITEM_COLS) <= set(df.columns)
    vals = df[ctl.ITEM_COLS].values
    assert vals.min() >= 1 and vals.max() <= 5


def test_power_curve_is_monotone_in_sample_size():
    curve = ctl.power_curve(sizes=(6, 30), spreads=(1.0,), reps=12, seed=4)
    assert len(curve) == 2
    assert curve.iloc[1].power >= curve.iloc[0].power - 0.2


def test_fit_sweep_summary_shapes():
    sweep = ctl.fit_sweep(sizes=(30, 120), reps=2, seed=5)
    s = ctl.fit_sweep_summary(sweep)
    assert set(s.N) == {30, 120}
    assert {"cfi_median", "cfi_q25", "reading"} <= set(s.columns)
    assert (s.cfi_median <= 1.0).all()


def test_openai_export_tool_turns_are_kept_as_system(tmp_path):
    conv = {"id": "c1", "mapping": {
        "a": {"message": {"author": {"role": "user"}, "create_time": 1,
                          "content": {"parts": ["help with mpi"]}, "metadata": {}}},
        "b": {"message": {"author": {"role": "tool"}, "create_time": 2,
                          "content": {"parts": ["search results"]}, "metadata": {}}},
        "c": {"message": {"author": {"role": "assistant"}, "create_time": 3,
                          "content": {"parts": ["use MPI_Bcast"]},
                          "metadata": {"model_slug": "gpt-x"}}}}}
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps([conv]))
    df = ingest.from_openai_export(p)
    assert list(df.role) == ["user", "system", "assistant"]


def test_model_comparison_uses_scaled_information_criteria():
    # On data from the five-factor model, the chi-square difference test and
    # chi-square-based AIC must prefer five factors. semopy's native AIC is not
    # scaled by N and would prefer the smaller one-factor model here.
    from cdm import model_core as mc
    df, _, _ = mc.simulate(n_per_group=40, seed=11)
    cfi5, cfi1, aic5, bic5, p = ctl._compare(df)
    assert aic5 == 1.0 and bic5 == 1.0
    assert p < 0.05
