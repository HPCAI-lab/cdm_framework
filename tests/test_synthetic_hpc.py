"""Smoke tests on the synthetic HPC transcript set in examples/.

These check that the ingest and feature path handle a realistic-sized HPC
corpus; they say nothing about measurement quality.
"""
from pathlib import Path

import pandas as pd
import pytest

from cdm import features as ft
from cdm import ingest, logs

ROOT = Path(__file__).resolve().parents[1] / "examples" / "synthetic_hpc_transcripts"
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="synthetic HPC set not present")


def test_text_ingest_matches_shipped_jsonl():
    parsed = ingest.from_directory(ROOT / "transcripts", model="synthetic-template-assistant-v1")
    shipped = logs.load_jsonl(ROOT / "hpc_test_logs.jsonl")
    assert len(parsed) == len(shipped) and parsed.session_id.nunique() == 120
    m = parsed.merge(shipped, on=["session_id", "turn"], suffixes=("_p", "_s"))
    assert (m.role_p == m.role_s).all()
    assert (m.content_p.str.strip() == m.content_s.str.strip()).all()
    assert logs.validate(shipped) == []


def test_pii_screen_flags_exactly_the_planted_sessions():
    shipped = logs.load_jsonl(ROOT / "hpc_test_logs.jsonl")
    key = pd.read_csv(ROOT / "design_key.csv")
    screen = logs.screen_log(shipped)
    flagged = set(screen.loc[screen.total_hits > 0, "session_id"])
    planted = set(key.loc[key.identifiers_planted, "session_id"])
    assert flagged == planted


def test_features_extract_on_hpc_profile():
    ft.use_profile("hpc")
    try:
        f = ft.extract(logs.load_jsonl(ROOT / "hpc_test_logs.jsonl"), level="student")
    finally:
        ft.use_profile("general")
    assert len(f) == 120
    assert f.context_ratio.gt(0).sum() > 30      # fenced and bare code both counted
