"""Synthetic interaction logs — SIMULATED, for exercising the log path only.

Every record carries ``provenance="SIMULATED"``. This generator exists so the
log -> feature -> score path can be run, tested, and inspected before any real
transcripts exist. It emits surface markers at rates driven by the same latent
dimension levels used in :mod:`cdm.model_core`.

> That construction makes agreement between automated scores and "human" scores
> on this data **circular**: both descend from the same latent draw. Nothing
> here is evidence that the feature set measures the constructs. It is evidence
> that the code runs and that the features respond to the behaviours they are
> supposed to respond to — which is a test, not a finding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cdm.model_core import DIMENSIONS, DISCIPLINES, MEANS, PHI, SDS

PROVENANCE = "SIMULATED"

TOPICS = {
    "Business": ("a market-entry brief for a regional grocery chain",
                 "a quarterly variance analysis", "a customer-churn summary"),
    "HealthSciences": ("a patient-education handout on hypertension",
                       "a medication-reconciliation checklist", "a shift-handoff protocol summary"),
    "Humanities": ("a close reading of a Dickinson poem", "a historiography note on Reconstruction",
                   "an argument about narrative unreliability"),
}
REQUESTS = ("Write", "Draft", "Summarise", "Outline", "Explain", "Compare")
CONSTRAINTS = ("in under 200 words", "as a table", "for a lay reader", "in three bullets",
               "without using jargon", "only cover the first section", "in a formal tone")
SEQ = ("First", "Then", "Next", "After that", "Before we draft")
SCOPE = ("just", "only the", "for now", "to start", "limited to")
LIMITS = ("I know you can't access our internal data, so",
          "your knowledge cut-off may matter here, so",
          "you might not know the current figures, so")
ALLOC = ("I'll do the analysis myself", "you draft the framing", "I'll handle the numbers on my own")
SOURCES = ("According to the textbook,", "I checked the guideline and", "My professor said",
           "I looked it up and")
ERRORS = ("That's wrong —", "This is incorrect:", "That citation doesn't exist.")
PROV = ("Cite your sources.", "Where did you get that?", "How do you know that?",
        "Show me your reasoning.")
HEDGE = ("Are you sure?", "Let me double-check that.", "I want to verify this.")
RECHECK = ("Now check whether the fix holds.", "Is that correct now?", "Did that fix it?")
ETHICS = ("I've de-identified the names first.", "Whose perspective is missing here?",
          "I should disclose the AI assistance.", "There's a privacy issue with the patient details.",
          "Who is affected if this is wrong?")
WHYHOW = ("Why does that follow?", "How does that mechanism work?", "Walk me through the reasoning.")
DISAGREE = ("I disagree —", "I'm not convinced.", "I still think my reading holds.")
RECONCILE = ("On the other hand, the article conflicts with that.",
             "That contradicts my notes.", "My teammate read it the other way.")
SHARED = ("our draft", "the team", "our report section")
SUCCESS = ("so that I can hand it in", "so I can brief the team", "the output I need is a one-pager")
FILLER = ("the framing depends on the audience", "several factors bear on this",
          "consider the underlying mechanism", "the evidence is mixed on this point",
          "one common approach is incremental refinement")


def _assistant_turn(rng, topic: str) -> str:
    bits = rng.choice(FILLER, size=rng.integers(2, 4), replace=False)
    return (f"Here is a draft on {topic}. " + " ".join(bits) +
            ". You may want to adjust the emphasis depending on your purpose.")


def _p(level: float) -> float:
    """Map a 1-5 latent level onto an emission probability."""
    return float(np.clip((level - 1.0) / 4.0, 0.02, 0.98))


def simulate_logs(n_per_group: int = 10, sessions_per_student: int = 3,
                  seed: int = 20260915, return_latents: bool = False):
    """Generate turn records for ``3 * n_per_group`` students.

    With ``return_latents=True`` also returns the latent dimension levels the
    markers were drawn from, so a paired "human coder" stream can be generated
    from the same ground truth. That pairing is what makes the demo circular —
    see the module docstring.
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(PHI)
    records: list[dict] = []
    latents: list[dict] = []
    sid = 0
    for disc in DISCIPLINES:
        mu, sd = np.array(MEANS[disc]), np.array(SDS[disc])
        for _ in range(n_per_group):
            sid += 1
            g = L @ rng.standard_normal(len(DIMENSIONS))
            F = np.clip(mu + sd * g, 1, 5)
            lvl = dict(zip(DIMENSIONS, F))
            latents.append(dict(student=sid, discipline=disc, **{d: float(F[i]) for i, d in enumerate(DIMENSIONS)}))
            pe, pd_, ov, ei, cs = (_p(lvl[d]) for d in DIMENSIONS)

            for s in range(sessions_per_student):
                session_id = f"S{sid:03d}-{s+1}"
                topic = TOPICS[disc][s % len(TOPICS[disc])]
                n_turns = int(rng.integers(3, 7))
                turn = 0
                last_ai = ""
                for k in range(n_turns):
                    parts = []
                    if k == 0 or rng.random() < 0.5:
                        parts.append(f"{rng.choice(REQUESTS)} {topic}")
                    if rng.random() < pe:
                        parts.append(", ".join(rng.choice(CONSTRAINTS, size=1 + (rng.random() < pe),
                                                          replace=False)))
                    if rng.random() < pe * 0.7 and k == 0:
                        parts.append("Here is my draft:\n\n" + ("draft paragraph " * 45))
                    if rng.random() < pe * 0.8:
                        parts.append(str(rng.choice(SUCCESS)))
                    if rng.random() < pe * 0.6 and k > 0:
                        parts.append(rng.choice(["Instead, make it shorter.", "That's not quite it — try again."]))
                    if rng.random() < pd_:
                        parts.append(f"{rng.choice(SEQ)} we need the criteria settled")
                    if rng.random() < pd_ * 0.8:
                        parts.append(f"Keep it {rng.choice(SCOPE)} opening section")
                    if rng.random() < pd_ * 0.5:
                        parts.append(str(rng.choice(LIMITS)) + " I'll supply the figures")
                    if rng.random() < pd_ * 0.4:
                        parts.append(str(rng.choice(ALLOC)))
                    if rng.random() < pd_ * 0.5:
                        parts.append("\n1. structure\n2. evidence\n3. wording")
                    if rng.random() < ov:
                        parts.append(str(rng.choice(PROV)))
                    if rng.random() < ov * 0.8:
                        parts.append(str(rng.choice(SOURCES)) + " it says otherwise")
                    if rng.random() < ov * 0.5:
                        parts.append(str(rng.choice(ERRORS)) + " the second claim is off")
                    if rng.random() < ov * 0.5:
                        parts.append(str(rng.choice(HEDGE)))
                    if rng.random() < ov * 0.4 and k > 1:
                        parts.append(str(rng.choice(RECHECK)))
                    if rng.random() < ei:
                        parts.append(str(rng.choice(ETHICS)))
                    if rng.random() < 0.06 * (1 - ei):     # low-ethics students leak identifiers
                        parts.append("Patient ID 44812, DOB 03/14/1991, jane.doe@example.com")
                    if rng.random() < cs:
                        parts.append(str(rng.choice(WHYHOW)))
                    if rng.random() < cs * 0.7:
                        parts.append(str(rng.choice(RECONCILE)))
                    if rng.random() < cs * 0.5:
                        parts.append(str(rng.choice(DISAGREE)) + " the framing misses the point")
                    if rng.random() < cs * 0.4:
                        parts.append(f"This goes into {rng.choice(SHARED)}")
                    # restatement behaviour: mid-range overlap is the target band
                    if last_ai and rng.random() < 0.7:
                        aw = last_ai.split()
                        if cs > 0.55:                      # reformulates in own words
                            take = aw[6:12]
                            parts.append("So the point is " + " ".join(take) + " in my own framing")
                        elif cs < 0.3:                     # reproduces verbatim
                            parts.append(" ".join(aw[:28]))
                    text = ". ".join(p for p in parts if p).strip()
                    if not text:
                        text = f"Continue with {topic}"
                    records.append(dict(session_id=session_id, student=sid, discipline=disc,
                                        phase=2, turn=turn, role="user", content=text,
                                        model="claude-sonnet-4-5-20250929",
                                        rubric_version="0.1-draft", provenance=PROVENANCE))
                    turn += 1
                    last_ai = _assistant_turn(rng, topic)
                    records.append(dict(session_id=session_id, student=sid, discipline=disc,
                                        phase=2, turn=turn, role="assistant", content=last_ai,
                                        model="claude-sonnet-4-5-20250929",
                                        rubric_version="0.1-draft", provenance=PROVENANCE))
                    turn += 1
    if return_latents:
        import pandas as pd
        return records, pd.DataFrame(latents)
    return records


def write_jsonl(records: list[dict], path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write a SIMULATED interaction log.")
    ap.add_argument("--n-per-group", type=int, default=10)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", default="cdm_simulated_logs.jsonl")
    a = ap.parse_args()
    recs = simulate_logs(a.n_per_group, a.sessions, a.seed)
    write_jsonl(recs, a.out)
    print(f"wrote {len(recs)} SIMULATED turns to {a.out}")
