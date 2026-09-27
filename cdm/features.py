"""Session-level feature extraction from interaction logs.

One feature per rubric item (25 total). Each feature is a *proxy* for its item,
chosen because it is computable from a transcript and auditable by a human
reading the same transcript. The mapping from feature to 1-5 item score lives in
:mod:`cdm.autoscore` and is a calibration question, not a measurement claim.

Backend: spaCy is used for lemmatisation and entity density when installed and a
model is loadable; otherwise a regex tokeniser is used. ``backend()`` reports
which, and the value is recorded in the output so runs are comparable.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from cdm.logs import pii_hits, sessions

DIMENSION_FEATURES = {
    "PromptEngineering": ["constraint_marker_density", "context_ratio", "refinement_ratio",
                          "specificity_index", "framing_clarity"],
    "ProblemDecomposition": ["subtask_count", "sequence_marker_density", "scope_qualifier_density",
                             "limitation_reference_count", "allocation_marker_count"],
    "OutputValidation": ["external_source_count", "error_assertion_count", "provenance_request_count",
                         "scrutiny_rate", "recheck_rate"],
    "EthicalIntegration": ["unprompted_ethics_rate", "privacy_handling", "bias_marker_count",
                           "stakeholder_marker_count", "attribution_marker_count"],
    "CollaborativeSensemaking": ["reconcile_count", "paraphrase_overlap", "why_how_rate",
                                 "disagreement_count", "shared_artifact_markers"],
}
FEATURES = [f for fs in DIMENSION_FEATURES.values() for f in fs]


# --- backend ----------------------------------------------------------------

@lru_cache(maxsize=1)
def _spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except Exception:
        return None


def backend() -> str:
    return "spacy:en_core_web_sm" if _spacy() is not None else "regex"


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
STOPWORDS = frozenset("""a an the and or but if then than that this these those of in on at to for with
from by as is are was were be been being do does did doing have has had having i you it its it's we they
he she them his her their our your my me not no yes so very just really can could would should will shall
may might must about into over under out up down more most some any all what which who whom how when where
why there here also too only even still much many few own same other another each both""".split())
VAGUE = frozenset("""stuff thing things something anything nice good better best bad great professional
appropriate proper fine ok okay interesting relevant stuff-like whatever etc improve improved awesome
quality general normal basic simple""".split())


def _tokens(text: str) -> list[str]:
    nlp = _spacy()
    if nlp is not None:
        return [t.text.lower() for t in nlp(text) if t.is_alpha]
    return [w.lower() for w in _WORD.findall(text)]


def _entity_rate(text: str) -> float:
    nlp = _spacy()
    if nlp is None:
        # regex fallback: capitalised mid-sentence tokens and numerals as a
        # coarse stand-in for named entities and figures
        toks = _WORD.findall(text)
        if not toks:
            return 0.0
        caps = sum(1 for t in toks[1:] if t[0].isupper())
        nums = len(re.findall(r"\d", text))
        return (caps + min(nums, len(toks))) / max(len(toks), 1)
    doc = nlp(text)
    return len(doc.ents) / max(len([t for t in doc if t.is_alpha]), 1)


# --- marker lexicons --------------------------------------------------------
# Every pattern here is a hypothesis about surface evidence for a construct.
# They are the first thing to revise when automated and human scores diverge.

from cdm import lexicons

#: Active marker lexicon. Profiles are defined in :mod:`cdm.lexicons`; select
#: one with ``lexicons.use("hpc")`` before extracting, and record which profile
#: a corpus was scored under, since features are not comparable across them.
M = lexicons.markers()


def use_profile(name: str) -> str:
    """Switch the active lexicon profile and rebind the marker table."""
    global M
    M = lexicons.use(name)
    return lexicons.active()


FENCE = r"```.*?```|\"{3}.*?\"{3}"

# Pasted code without fences. A line is code-like if it is a preprocessor
# directive, ends in ; { or }, or is dense in code punctuation. A run of two or
# more code-like lines counts, as does a single code-like line of 40+ chars
# (a pasted loop header, a kernel launch). Prose rarely meets any of these.
_PREPROC = re.compile(r"^\s*#\s*(?:include|pragma|define|ifn?def|if|endif|else)\b")
_CODE_END = re.compile(r"[;{}]\s*(?://.*)?$")
_CODE_PUNCT = set(";{}()[]=<>+*&|#_")


def _is_code_line(line: str) -> bool:
    s = line.strip()
    if len(s) < 3:
        return s in ("{", "}", "};")
    if _PREPROC.match(s) or _CODE_END.search(s):
        return True
    return len(s) >= 8 and sum(c in _CODE_PUNCT for c in s) / len(s) >= 0.12


def _bare_code_chars(text: str) -> int:
    """Characters of unfenced code in ``text`` (fenced blocks already removed)."""
    total, run = 0, []

    def flush():
        nonlocal total
        if len(run) >= 2 or (run and len(run[0].strip()) >= 40):
            total += sum(len(r) for r in run)
        run.clear()

    for line in text.splitlines():
        if _is_code_line(line):
            run.append(line)
        else:
            flush()
    flush()
    return total


def _count(rx: re.Pattern, text: str) -> int:
    return len(rx.findall(text)) if isinstance(text, str) else 0


def _jaccard(a: str, b: str) -> float:
    sa = {t for t in _tokens(a) if t not in STOPWORDS}
    sb = {t for t in _tokens(b) if t not in STOPWORDS}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _bigram_overlap(student: str, assistant: str) -> float:
    """Fraction of the student turn's bigrams also present in the AI turn.

    Near 0 means the student is not engaging with the output at all; near 1
    means reproduction. The rubric's top band is in between (item CS2).
    """
    st = [t for t in _tokens(student)]
    at = set(zip(_tokens(assistant), _tokens(assistant)[1:]))
    sb = list(zip(st, st[1:]))
    if not sb or not at:
        return 0.0
    return sum(1 for bg in sb if bg in at) / len(sb)


def session_features(turns: pd.DataFrame) -> dict[str, float]:
    """Compute the 25 features for one session."""
    stu = turns[turns.role == "user"].content.fillna("").tolist()
    ast = turns[turns.role == "assistant"].content.fillna("").tolist()
    n_stu = max(len(stu), 1)
    n_ast = max(len(ast), 1)
    all_stu = "\n".join(stu)
    toks = _tokens(all_stu)
    n_tok = max(len(toks), 1)

    # --- prompt engineering
    constraint = sum(_count(M["constraint"], t) for t in stu) / n_stu

    ctx_chars = 0
    for t in stu:
        blocks = re.findall(FENCE, t, re.S)
        ctx_chars += sum(len(b) for b in blocks)
        rest = re.sub(FENCE, "\n", t, flags=re.S)
        for para in re.split(r"\n\s*\n", rest):
            if len(para) > 300 and not blocks:
                ctx_chars += len(para)
            else:
                ctx_chars += _bare_code_chars(para)
    context_ratio = ctx_chars / max(len(all_stu), 1)

    refine = 0
    for i, t in enumerate(stu[1:], start=1):
        near_dup = any(_jaccard(t, prev) > 0.40 for prev in stu[:i])
        if near_dup or _count(M["revision"], t):
            refine += 1
    refinement_ratio = refine / n_stu

    content_toks = [t for t in toks if t not in STOPWORDS and len(t) > 3]
    vague = sum(1 for t in toks if t in VAGUE)
    specificity_index = (len(content_toks) / n_tok) - 2.0 * (vague / n_tok) + 0.5 * _entity_rate(all_stu)

    single_req = 0
    for t in stu:
        heads = _count(M["request_head"], t)
        if heads >= 1:
            single_req += 1 if heads == 1 else 0.5
        if _count(M["success_cond"], t):
            single_req += 0.5
    framing_clarity = min(single_req / n_stu, 1.5)

    # --- problem decomposition
    subtask_count = float(sum(_count(M["enumeration"], t) for t in stu))
    sequence_marker_density = sum(_count(M["sequence"], t) for t in stu) / n_stu
    scope_qualifier_density = sum(_count(M["scope"], t) for t in stu) / n_stu
    limitation_reference_count = float(sum(_count(M["limitation"], t) for t in stu))
    allocation_marker_count = float(sum(_count(M["allocation"], t) for t in stu))

    # --- output validation
    external_source_count = float(sum(_count(M["external_source"], t) for t in stu))
    error_assertion_count = float(sum(_count(M["error_assert"], t) for t in stu))
    provenance_request_count = float(sum(_count(M["provenance"], t) for t in stu))
    scrutiny = sum(_count(M["provenance"], t) + _count(M["hedge"], t) +
                   _count(M["error_assert"], t) + _count(M["external_source"], t) for t in stu)
    scrutiny_rate = scrutiny / n_ast

    # re-check: a student turn following an error assertion that confirms the fix
    rows = turns.sort_values("turn").reset_index(drop=True)
    err_turns = [i for i, r in rows.iterrows()
                 if r.role == "user" and _count(M["error_assert"], r.content or "")]
    rechecks = 0
    for i in err_turns:
        later = rows.iloc[i + 1:i + 5]
        if any(r.role == "user" and (_count(M["recheck"], r.content or "") or
                                     _count(M["hedge"], r.content or ""))
               for _, r in later.iterrows()):
            rechecks += 1
    recheck_rate = rechecks / len(err_turns) if err_turns else 0.0

    # --- ethical integration
    unprompted = 0
    for i, r in rows.iterrows():
        if r.role != "user" or not _count(M["ethics"], r.content or ""):
            continue
        prev_ai = rows.iloc[:i][rows.iloc[:i].role == "assistant"]
        primed = bool(len(prev_ai)) and bool(_count(M["ethics"], prev_ai.content.iloc[-1] or ""))
        if not primed:
            unprompted += 1
    unprompted_ethics_rate = unprompted / n_stu

    deid = sum(_count(M["privacy"], t) for t in stu)
    leaks = sum(sum(pii_hits(t).values()) for t in stu)
    privacy_handling = deid - 2.0 * leaks

    bias_marker_count = float(sum(_count(M["bias"], t) for t in stu))
    stakeholder_marker_count = float(sum(_count(M["stakeholder"], t) for t in stu))
    attribution_marker_count = float(sum(_count(M["attribution"], t) for t in stu))

    # --- collaborative sensemaking
    reconcile_count = float(sum(_count(M["reconcile"], t) for t in stu))
    overlaps = []
    for i, r in rows.iterrows():
        if r.role != "user" or i == 0:
            continue
        prev = rows.iloc[i - 1]
        if prev.role == "assistant" and len(_tokens(r.content or "")) >= 8:
            overlaps.append(_bigram_overlap(r.content or "", prev.content or ""))
    paraphrase_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    why_how_rate = sum(_count(M["why_how"], t) for t in stu) / n_stu
    disagreement_count = float(sum(_count(M["disagreement"], t) for t in stu))
    shared_artifact_markers = float(sum(_count(M["shared_artifact"], t) for t in stu))

    return dict(
        constraint_marker_density=constraint, context_ratio=context_ratio,
        refinement_ratio=refinement_ratio, specificity_index=specificity_index,
        framing_clarity=framing_clarity,
        subtask_count=subtask_count, sequence_marker_density=sequence_marker_density,
        scope_qualifier_density=scope_qualifier_density,
        limitation_reference_count=limitation_reference_count,
        allocation_marker_count=allocation_marker_count,
        external_source_count=external_source_count, error_assertion_count=error_assertion_count,
        provenance_request_count=provenance_request_count, scrutiny_rate=scrutiny_rate,
        recheck_rate=recheck_rate,
        unprompted_ethics_rate=unprompted_ethics_rate, privacy_handling=privacy_handling,
        bias_marker_count=bias_marker_count, stakeholder_marker_count=stakeholder_marker_count,
        attribution_marker_count=attribution_marker_count,
        reconcile_count=reconcile_count, paraphrase_overlap=paraphrase_overlap,
        why_how_rate=why_how_rate, disagreement_count=disagreement_count,
        shared_artifact_markers=shared_artifact_markers,
    )


def extract(df: pd.DataFrame, level: str = "student") -> pd.DataFrame:
    """Extract features from a turn log.

    ``level='session'`` returns one row per session; ``level='student'``
    averages sessions within student, which is the unit the pipeline scores.
    """
    rows = []
    for sid, turns in sessions(df):
        feats = session_features(turns)
        rows.append({
            "session_id": sid,
            "student": turns.student.iloc[0],
            "discipline": turns.discipline.iloc[0] if "discipline" in turns else None,
            "n_student_turns": int((turns.role == "user").sum()),
            "nlp_backend": backend(),
            "lexicon_profile": lexicons.active(),
            **feats,
        })
    out = pd.DataFrame(rows)
    if level == "session":
        return out
    if level != "student":
        raise ValueError("level must be 'session' or 'student'")
    agg = out.groupby("student", as_index=False).agg(
        {**{f: "mean" for f in FEATURES},
         "discipline": "first", "n_student_turns": "sum", "nlp_backend": "first",
         "lexicon_profile": "first"}
    )
    agg.insert(1, "n_sessions", out.groupby("student").size().values)
    return agg
