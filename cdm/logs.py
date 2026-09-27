"""Interaction-log schema, loading, validation, and PII screening.

The log path starts here. One JSONL record per conversational turn; a *session*
is one student's continuous work on one task, identified by ``session_id``.

Nothing in this module interprets content — it only reads, checks, and screens.
Feature extraction lives in :mod:`cdm.features`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# --- schema -----------------------------------------------------------------

REQUIRED_FIELDS = ("session_id", "student", "role", "turn", "content")
OPTIONAL_FIELDS = ("discipline", "phase", "timestamp", "model", "rubric_version")
ROLES = ("user", "assistant", "system")

#: ``model`` should be a pinned identifier (e.g. ``claude-sonnet-4-5-20250929``),
#: not a family name. Model version is a known confound for interaction
#: behaviour, so it has to be recoverable per turn, not per study.


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Read a JSONL turn log into a DataFrame. Raises on malformed records."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            missing = [f for f in REQUIRED_FIELDS if f not in rec]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing field(s) {missing}")
            rows.append(rec)
    if not rows:
        raise ValueError(f"{path}: no records")
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of data problems. Empty list means the log is usable.

    These are the checks that catch a broken logging deployment before it
    quietly ruins a semester of collection.
    """
    problems: list[str] = []
    for f in REQUIRED_FIELDS:
        if f not in df.columns:
            problems.append(f"missing column: {f}")
    if problems:
        return problems

    bad_roles = sorted(set(df.role.unique()) - set(ROLES))
    if bad_roles:
        problems.append(f"unexpected role values: {bad_roles}")
    if df.content.isna().any():
        problems.append(f"{int(df.content.isna().sum())} turns with null content")

    for sid, g in df.groupby("session_id"):
        turns = g.turn.tolist()
        if len(set(turns)) != len(turns):
            problems.append(f"session {sid}: duplicate turn indices")
        if g.student.nunique() > 1:
            problems.append(f"session {sid}: more than one student")
        if not (g.sort_values("turn").role.iloc[0] in ("user", "system")):
            problems.append(f"session {sid}: does not open with a student turn")
        if (g.role == "user").sum() == 0:
            problems.append(f"session {sid}: no student turns")
    if "model" not in df.columns:
        problems.append("no `model` column — model version is not recoverable")
    return problems


def sessions(df: pd.DataFrame):
    """Yield ``(session_id, turns)`` with turns sorted, system turns dropped."""
    work = df[df.role != "system"]
    for sid, g in work.groupby("session_id", sort=True):
        yield sid, g.sort_values("turn").reset_index(drop=True)


# --- PII screening ----------------------------------------------------------
# Deliberately crude and high-recall. This is a screen to flag transcripts for
# human review before release, not a de-identification guarantee. Free-text
# prompts cannot be reliably de-identified by pattern matching, and this
# function must never be described as making a log releasable.

PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)"),
    "ssn_like": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "record_id": re.compile(r"\b(?:MRN|DOB|patient\s*(?:id|#)|student\s*id)\b[:# ]*\S*", re.I),
    "dob_like": re.compile(r"(?<!\d)(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}(?!\d)"),
    "url": re.compile(r"https?://\S+"),
}


def pii_hits(text: str) -> dict[str, int]:
    """Count apparent identifiers by type in one turn's text."""
    if not isinstance(text, str):
        return {}
    return {k: n for k, p in PII_PATTERNS.items() if (n := len(p.findall(text)))}


def screen_log(df: pd.DataFrame) -> pd.DataFrame:
    """Per-session PII screening report, worst sessions first.

    Use this to decide which transcripts need manual review. A zero count is
    not evidence a transcript is safe to publish.
    """
    rows = []
    for sid, turns in sessions(df):
        counts: dict[str, int] = {}
        for txt in turns.content:
            for k, n in pii_hits(txt).items():
                counts[k] = counts.get(k, 0) + n
        rows.append({
            "session_id": sid,
            "student": turns.student.iloc[0],
            "total_hits": sum(counts.values()),
            **counts,
        })
    out = pd.DataFrame(rows).fillna(0)
    return out.sort_values("total_hits", ascending=False).reset_index(drop=True)
