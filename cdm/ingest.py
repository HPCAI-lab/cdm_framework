"""Ingest real student transcripts into the canonical turn schema.

The scoring path consumes one JSON Lines record per conversational turn
(:mod:`cdm.logs`). Collected transcripts almost never arrive in that shape, so
this module converts the formats a course deployment actually produces.

Adapters here are best-effort readers of *other people's* formats. Run
``python -m cdm.ingest --dry-run`` first and read the report before trusting
any conversion: a silently mis-parsed role or a lost turn boundary corrupts
every feature downstream, and the failure is invisible in the scores.

Required of every output record: ``session_id``, ``student``, ``role``,
``turn``, ``content``. The pinned ``model`` identifier is required by the
measurement design (see docs/pilot_workflow.md); where an export does not
carry it, supply it with ``--model`` and record in the study log that it was
asserted rather than observed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

from cdm.logs import REQUIRED_FIELDS, screen_log, validate

CANONICAL = ("session_id", "student", "discipline", "phase", "turn",
             "role", "timestamp", "model", "content")

ROLE_ALIASES = {
    "user": "user", "human": "user", "student": "user", "you": "user", "me": "user",
    "prompt": "user", "q": "user",
    "assistant": "assistant", "ai": "assistant", "bot": "assistant", "model": "assistant",
    "chatgpt": "assistant", "claude": "assistant", "gpt": "assistant", "response": "assistant",
    "a": "assistant",
    "system": "system",
    # ChatGPT exports record browsing, code-interpreter and plugin output under
    # role "tool". It is neither the student nor the model's prose, so it is
    # kept as a system turn: recorded in the log, excluded from features.
    "tool": "system",
}


def _role(raw: str) -> str:
    key = str(raw).strip().strip(":").lower()
    if key in ROLE_ALIASES:
        return ROLE_ALIASES[key]
    raise ValueError(f"unrecognised role {raw!r}; extend ROLE_ALIASES")


def _rows_to_frame(rows: list[dict], model: str | None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in CANONICAL:
        if col not in df.columns:
            df[col] = None
    if model:
        df["model"] = df["model"].fillna(model)
    return df[list(CANONICAL)]


# --- adapters ---------------------------------------------------------------

def from_openai_export(path: str | Path, model: str | None = None) -> pd.DataFrame:
    """ChatGPT data export (``conversations.json``): a list of conversations,
    each with a ``mapping`` of message nodes."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    convs = data if isinstance(data, list) else [data]
    rows = []
    for conv in convs:
        sid = conv.get("conversation_id") or conv.get("id") or conv.get("title")
        mapping = conv.get("mapping", {})
        msgs = []
        for node in mapping.values():
            m = node.get("message")
            if not m:
                continue
            parts = (m.get("content") or {}).get("parts") or []
            text = "\n".join(p for p in parts if isinstance(p, str)).strip()
            if not text:
                continue
            msgs.append((m.get("create_time") or 0.0, m["author"]["role"], text,
                         m.get("metadata", {}).get("model_slug")))
        msgs.sort(key=lambda t: t[0])
        for i, (ts, role, text, slug) in enumerate(msgs):
            rows.append(dict(session_id=str(sid), student=None, turn=i,
                             role=_role(role), timestamp=ts, model=slug, content=text))
    return _rows_to_frame(rows, model)


def from_anthropic_export(path: str | Path, model: str | None = None) -> pd.DataFrame:
    """Claude data export: conversations with a flat ``chat_messages`` list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    convs = data if isinstance(data, list) else [data]
    rows = []
    for conv in convs:
        sid = conv.get("uuid") or conv.get("name")
        for i, m in enumerate(conv.get("chat_messages", [])):
            text = m.get("text")
            if not text and isinstance(m.get("content"), list):
                text = "\n".join(c.get("text", "") for c in m["content"]).strip()
            if not text:
                continue
            rows.append(dict(session_id=str(sid), student=None, turn=i,
                             role=_role(m.get("sender") or m.get("role")),
                             timestamp=m.get("created_at"), model=None, content=text))
    return _rows_to_frame(rows, model)


TURN_HEAD = re.compile(
    r"^\s*(?:\[|\*\*|##\s*)?\s*(user|human|student|you|me|prompt|assistant|ai|bot|model|"
    r"chatgpt|claude|gpt|response|system)\s*(?:\]|\*\*)?\s*:",
    re.I | re.M)


def from_text_transcript(path: str | Path, session_id: str | None = None,
                         model: str | None = None) -> pd.DataFrame:
    """A plain-text or Markdown transcript with speaker-labelled turns.

    Recognises ``User:``, ``**Student:**``, ``[Assistant]:``, ``## Claude:`` and
    similar at the start of a line. Text before the first label is dropped, and
    the drop is reported by :func:`ingest_report`.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    marks = list(TURN_HEAD.finditer(raw))
    rows = []
    sid = session_id or path.stem
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        text = raw[start:end].strip()
        if text:
            rows.append(dict(session_id=sid, student=None, turn=i,
                             role=_role(m.group(1)), timestamp=None,
                             model=None, content=text))
    return _rows_to_frame(rows, model)


def from_csv(path: str | Path, model: str | None = None, **colmap) -> pd.DataFrame:
    """A spreadsheet of turns. ``colmap`` renames source columns, e.g.
    ``from_csv('t.csv', session_id='convo', role='speaker', content='text')``."""
    df = pd.read_csv(path)
    if colmap:
        df = df.rename(columns={v: k for k, v in colmap.items()})
    df["role"] = df["role"].map(_role)
    if "turn" not in df.columns:
        df["turn"] = df.groupby("session_id").cumcount()
    return _rows_to_frame(df.to_dict("records"), model)


def from_directory(path: str | Path, pattern: str = "*.txt",
                   model: str | None = None) -> pd.DataFrame:
    """One transcript file per session. The file stem becomes the session id,
    and its leading digits, if any, become the student id."""
    frames = []
    for f in sorted(Path(path).glob(pattern)):
        df = from_text_transcript(f, session_id=f.stem, model=model)
        m = re.match(r"(\d+)", f.stem)
        if m:
            df["student"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no files matching {pattern} under {path}")
    return pd.concat(frames, ignore_index=True)


ADAPTERS = {
    "openai": from_openai_export,
    "anthropic": from_anthropic_export,
    "text": from_text_transcript,
    "csv": from_csv,
    "dir": from_directory,
}


# --- identity assignment ----------------------------------------------------

def assign_students(df: pd.DataFrame, roster: str | Path | None = None,
                    pseudonymise: bool = True) -> pd.DataFrame:
    """Attach a student id to every turn.

    ``roster`` is a two-column CSV mapping ``session_id`` to ``student``. With
    ``pseudonymise`` the roster's student values are replaced by sequential
    integers and the crosswalk is returned separately, so the scored corpus
    never carries a name. Keep that crosswalk outside the repository.
    """
    if roster is None:
        codes = {s: i + 1 for i, s in enumerate(sorted(df.session_id.unique()))}
        df = df.copy()
        df["student"] = df.session_id.map(codes)
        return df, pd.DataFrame({"session_id": list(codes), "student": list(codes.values())})

    r = pd.read_csv(roster)
    df = df.copy().drop(columns=["student"]).merge(r, on="session_id", how="left")
    missing = df[df.student.isna()].session_id.unique()
    if len(missing):
        raise ValueError(f"roster has no student for sessions: {list(missing)[:5]}")
    crosswalk = r.copy()
    if pseudonymise:
        codes = {s: i + 1 for i, s in enumerate(sorted(r.student.unique()))}
        crosswalk["pseudonym"] = crosswalk.student.map(codes)
        df["student"] = df.student.map(codes)
    return df, crosswalk


# --- reporting --------------------------------------------------------------

def ingest_report(df: pd.DataFrame) -> str:
    """Human-readable summary of what a conversion produced.

    Read this before scoring. The counts that matter most are turns per
    session and the student-to-assistant turn ratio: a parser that has merged
    turns or dropped a speaker shows up here and nowhere else.
    """
    lines = []
    lines.append(f"sessions            {df.session_id.nunique()}")
    lines.append(f"turns               {len(df)}")
    lines.append(f"students            {df.student.nunique(dropna=True)}")
    per = df.groupby("session_id").size()
    lines.append(f"turns per session   min {per.min()}  median {per.median():.0f}  max {per.max()}")
    roles = df.role.value_counts().to_dict()
    lines.append(f"roles               {roles}")
    u, a = roles.get("user", 0), roles.get("assistant", 0)
    ratio = (u / a) if a else float("inf")
    lines.append(f"student:assistant   {ratio:.2f}"
                 + ("   <-- inspect: expected near 1.0" if not 0.5 <= ratio <= 2.0 else ""))
    lines.append(f"model recoverable   {int(df.model.notna().sum())}/{len(df)} turns")
    empty = int((df.content.fillna('').str.strip() == '').sum())
    if empty:
        lines.append(f"empty content       {empty} turns  <-- dropped rows are silent data loss")
    problems = validate(df)
    lines.append("validation          " + ("clean" if not problems else f"{len(problems)} problem(s)"))
    for p in problems[:8]:
        lines.append(f"   - {p}")
    return "\n".join(lines)


def write_jsonl(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in df.to_dict("records"):
            fh.write(json.dumps({k: v for k, v in rec.items() if v is not None}) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert transcripts to the CDM turn schema.")
    ap.add_argument("--format", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--input", required=True)
    ap.add_argument("--pattern", default="*.txt", help="for --format dir")
    ap.add_argument("--model", help="pinned model identifier where the export lacks one")
    ap.add_argument("--roster", help="CSV mapping session_id to student")
    ap.add_argument("--keep-names", action="store_true",
                    help="do not pseudonymise roster student ids")
    ap.add_argument("--out", default="student_logs.jsonl")
    ap.add_argument("--crosswalk", default="crosswalk_KEEP_OUT_OF_REPO.csv")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    fn = ADAPTERS[a.format]
    df = fn(a.input, pattern=a.pattern, model=a.model) if a.format == "dir" \
        else fn(a.input, model=a.model)
    df, crosswalk = assign_students(df, a.roster, pseudonymise=not a.keep_names)

    print(ingest_report(df))
    scr = screen_log(df)
    flagged = int((scr.total_hits > 0).sum())
    print(f"\npii screen          {flagged}/{len(scr)} sessions contain apparent identifiers")
    if flagged:
        print("                    review these before any release; the screen does not")
        print("                    detect identifying narrative and cannot clear a corpus")

    if a.dry_run:
        print("\n[dry run] nothing written")
        return
    write_jsonl(df, a.out)
    crosswalk.to_csv(a.crosswalk, index=False)
    print(f"\nwrote {a.out} and {a.crosswalk}")
    print("keep the crosswalk outside the repository")


if __name__ == "__main__":
    main()
