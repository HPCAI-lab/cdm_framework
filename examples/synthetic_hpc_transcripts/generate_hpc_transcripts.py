"""Generate synthetic HPC/PDC student-AI transcripts for testing the CDM pipeline.

SYNTHETIC TEST DATA. Every transcript is assembled from templates. No student
wrote any of it, and it is not evidence about students or about the instrument.

Design
------
* 120 synthetic students, one session each, 20 per task across six tasks
  (MPI matmul, OpenMP race, CUDA tiling, MPI halo deadlock, OpenMP false
  sharing, scaling analysis).
* Each student gets a latent level per CDM dimension (correlated, mean 3).
  Each of the 25 items gets a target score 1-5 from its dimension's level plus
  item noise. OV4 and CS2 use ideal-point patterns: the top score is a middle
  band, and both extremes score low.
* The transcript is rendered so each item shows the behaviour its target score
  describes in docs/rubric.md. The targets are written to
  generator_item_labels.csv. They are the generator's intended scores, NOT human
  coding, and agreement against them is partly circular.

Usage: python generate_hpc_transcripts.py --out synthetic_hpc_transcripts --seed 20260924
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tasks import TASKS

DIMS = ["PromptEngineering", "ProblemDecomposition", "OutputValidation",
        "EthicalIntegration", "CollaborativeSensemaking"]
CODES = {"PromptEngineering": "PE", "ProblemDecomposition": "PD", "OutputValidation": "OV",
         "EthicalIntegration": "EI", "CollaborativeSensemaking": "CS"}
MODEL = "synthetic-template-assistant-v1"
HEADER = ("SYNTHETIC TEST TRANSCRIPT - generated from templates for pipeline testing. "
          "Not student data.")


def pick(rng, options):
    return options[int(rng.integers(len(options)))]


def code_block(rng, code, fenced):
    return f"```c\n{code}\n```" if fenced else code


class Session:
    """Builds one transcript as alternating student / assistant turns."""

    def __init__(self):
        self.turns: list[tuple[str, str]] = []

    def student(self, *parts):
        text = "\n\n".join(p for p in parts if p)
        if text.strip():
            self.turns.append(("user", text.strip()))
            return True
        return False

    def assistant(self, *parts):
        text = "\n\n".join(p for p in parts if p)
        self.turns.append(("assistant", text.strip()))


def draw_student(rng):
    """Latent dimension levels and item targets for one synthetic student."""
    corr = np.full((5, 5), 0.4) + np.eye(5) * 0.6
    z = rng.multivariate_normal(np.zeros(5), corr)
    latent = np.clip(3.0 + 1.15 * z, 1.0, 5.0)
    items = {}
    for d, lv in zip(DIMS, latent):
        for i in range(1, 6):
            items[f"{d}_i{i}"] = int(np.clip(np.round(lv + rng.normal(0, 0.7)), 1, 5))
    # ideal-point items
    ov = latent[2]
    if ov < 2.3:
        ov4_pat, ov4 = "accept_all", 1
    elif ov < 3.2:
        ov4_pat, ov4 = "uniform", 3
    elif ov < 3.8 or rng.random() < 0.55:
        ov4_pat, ov4 = "proportional", int(pick(rng, [4, 5, 5]))
    else:
        ov4_pat, ov4 = "over_scrutiny", int(pick(rng, [1, 2]))
    items["OutputValidation_i4"] = ov4
    cs = latent[4]
    if cs < 2.3:
        cs2_pat = pick(rng, ["verbatim", "loose"])
        cs2 = 1
    elif cs < 3.4:
        cs2_pat, cs2 = "close", 3
    else:
        cs2_pat, cs2 = "own_words", int(pick(rng, [4, 5, 5]))
    items["CollaborativeSensemaking_i2"] = cs2
    return latent, items, ov4_pat, cs2_pat


def lvl(items, code):
    """Item target by short code, e.g. 'PE1'."""
    dim = [d for d, c in CODES.items() if c == code[:2]][0]
    return items[f"{dim}_i{code[2]}"]


def render(t, items, rng, *, flaw, raise_policy, team, fenced, ov4_pat, cs2_pat):
    s = Session()
    L = lambda c: lvl(items, c)  # noqa: E731
    vague = L("PE4") <= 2
    symptom = t["vague_symptom"] if vague else t["symptom"]

    # ---------------- opening turn ----------------
    frame = {1: pick(rng, [f"{t['short'].lower()} broken", f"fix my {t['api'].lower()} code",
                           f"explain {t['api']} and fix this and also help with my report"]),
             2: f"can you fix my {t['short']}",
             3: f"Can you find the bug in my {t['short']}?",
             4: f"Can you find why {symptom} in my {t['short']}?",
             5: f"Can you find why {symptom}, {t['success']}?"}[L("PE5")]
    spec = {1: "idk its some parallel stuff, can u make it better",
            2: "its not working right, make it better",
            3: "I also want it optimized better.",
            4: f"I think it's a {t['api']} correctness problem, not a performance one.",
            5: (f"Specifically, {symptom}." if L("PE5") < 4 else f"I'm compiling with {t['config']}.")}[L("PE4")]
    has_code = L("PE2") >= 3
    context = []
    if has_code:
        context.append(code_block(rng, t["code"], fenced))
        if L("PE2") >= 4 and L("PE5") < 4:
            context.append(f"What happens: {symptom}.")
        if L("PE2") == 5:
            context.append(f"Build/run setup: {t['config']}.")
    elif L("PE2") == 1:
        context.append(pick(rng, ["its the one from lab", "you know the standard one from class"]))
    cons = {1: "", 2: f"in {t['lang']}", 3: f"Please {t['format_c']}.",
            4: f"Please {t['format_c']}, and it has to be correct.",
            5: f"Requirements: {t['goal']}; {t['format_c']}."}[L("PE1")]
    plan = []
    if L("PD1") == 1:
        plan.append(pick(rng, ["just do the whole thing", "can you just parallelize and fix all of it"]))
    elif L("PD1") in (3, 4):
        plan.append(f"I think there are two parts: {t['subparts'][0]} and {t['subparts'][1]}.")
    elif L("PD1") == 5:
        if rng.random() < 0.5:
            plan.append("I'd split it into: " + ", ".join(
                f"({i+1}) {p}" for i, p in enumerate(t["subparts"])) + ".")
        else:
            plan.append("My plan:\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(t["subparts"])))
    if L("PD2") in (3, 4):
        plan.append(f"First {t['first_part']}, then {t['later_part']}.")
    elif L("PD2") == 5:
        plan.append(f"{t['dependency']}.")
    if L("PD3") == 1:
        plan.append("write the whole assignment for me")
    elif L("PD3") in (3, 4):
        plan.append(f"Just look at {t['first_part']} in this function.")
    elif L("PD3") == 5:
        plan.append(f"For now just {t['first_part']}; we'll come back to {t['later_part']} later.")
    if L("PD4") == 5:
        plan.append(f"You can't run this on our cluster, so here are my measurements: {t['timings']}.")
    if L("PD5") == 1:
        plan.append("also write the analysis part of my report")
    elif L("PD5") == 5:
        plan.append(t["keep"][0].upper() + t["keep"][1:] + ".")
    policy_first = L("EI1") >= 4
    if policy_first:
        plan.append(t["policy"][0].upper() + t["policy"][1:]
                    + (" If not, just explain the problem and I'll write the fix myself." if L("EI1") == 5 else ""))
    privacy = ""
    if L("EI2") <= 2:
        privacy = t["privacy_leak"]
    elif L("EI2") >= 4:
        privacy = t["redact"] + (" because that's login and account information." if L("EI2") == 5 else "")
    if L("CS3") == 1:
        plan.append("just give me the fixed code")
    s.student(frame, spec, *context, cons, *plan, privacy)

    # ---------------- context request ----------------
    policy_note = ("(If this is for a course, check your course's AI-use policy and disclose "
                   "any code you use from here.)") if raise_policy else ""
    if not has_code:
        s.assistant("Could you paste the relevant code and describe exactly what goes wrong?")
        if L("PE2") == 2:
            s.student(code_block(rng, t["code"], fenced))
        else:
            s.student(pick(rng, ["idk just the normal one", "its the one we did in class"]))

    # ---------------- first answer: flawed or generic ----------------
    generic = (f"This kind of problem usually comes from how data is shared or split between "
               f"{'processes' if t['api'] == 'MPI' else 'threads'}. Check that each one has its own copies "
               "of anything it writes and that your loop bounds cover every element.")
    first = t["flaw"] if flaw else generic
    s.assistant(first, policy_note)

    # ---------------- reaction: OV2, PE3 ----------------
    ov2, pe3 = L("OV2"), L("PE3")
    react = []
    if ov2 == 2:
        react.append("hmm ok")
    elif ov2 in (3, 4):
        react.append("I'm not sure that's right, it still looks wrong to me.")
    elif ov2 == 5:
        react.append(("That won't work: " + t["flaw_rebuttal"] + ".") if flaw
                     else ("That doesn't explain it. " + t["error_specific"][0].upper()
                           + t["error_specific"][1:] + "."))
    switched = False
    if pe3 == 1:
        switched = rng.random() < 0.5
        react.append("ok different question, how do i install mpi on my laptop" if switched else "ok thanks")
    elif pe3 == 2:
        react.append("ok")
    elif pe3 == 3:
        react.append("Can you try again?")
    elif pe3 == 4:
        react.append(f"Can you try again and {t['format_c']}?")
    elif pe3 == 5:
        react.append((f"Not quite. Redo it: {t['format_c']}, and fix the actual cause instead of working around it.")
                     if flaw else
                     (f"That's too general. Try again, focusing on the case where {t['symptom']}, and {t['format_c']}."))
    if L("PD4") == 1:
        react.append(f"also can you run it on 16 {'ranks' if t['api'] == 'MPI' else 'threads'} and tell me the speedup")
    elif L("PD4") in (3, 4):
        react.append(f"Can you tell me how fast it will run on 16 {'ranks' if t['api'] == 'MPI' else 'threads'}?")
    s.student(*react)
    pushed = ov2 >= 3 or pe3 >= 3
    replies = []
    if pushed:
        if flaw:
            replies.append(t["flaw_ack"])
        replies += [t["diag"], code_block(rng, t["fix"], True)]
    elif switched:
        replies.append("On a laptop, the easiest route is your OS package manager: for example "
                       "'sudo apt install openmpi-bin libopenmpi-dev' on Ubuntu or 'brew install open-mpi' on macOS.")
    else:
        replies.append("Glad that helps. Let me know if anything else comes up.")
    if L("PD4") <= 4 and L("PD4") != 2:
        replies.append("I can't run code on your system, so I can't measure the speedup; if you run it and share the timings I can help interpret them.")
    s.assistant(*replies)
    if L("PD4") in (3, 4):
        s.student(f"ok, here are my timings then: {t['timings']}.")
        s.assistant("Thanks, that's useful. Those numbers show the scaling flattening out; we can look at why once the result is correct.")

    # ---------------- probing: OV3, CS3, OV4 over-scrutiny ----------------
    ov3, cs3 = L("OV3"), L("CS3")
    why = t["why"]
    n_why = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3}[cs3]
    probe = []
    if ov3 == 2:
        probe.append("source?")
    elif ov3 >= 3:
        probe.append(f"Can you explain {t['prov_q']}?")
    if n_why:
        probe.append(why[0][0][0].upper() + why[0][0][1:])
    if ov4_pat == "over_scrutiny":
        probe.append("Also, why did you choose those variable names, and are you sure the comments are worded "
                     "correctly? Please justify every line, including the formatting.")
    if s.student(*probe):
        rep = []
        if ov3 >= 3:
            rep.append(t["prov_a"])
            if ov3 == 5:
                rep.append(f"You could also use {t['dubious']}.")
        elif ov3 == 2:
            rep.append(t["prov_a"])
        if n_why:
            rep.append(why[0][1])
        if ov4_pat == "over_scrutiny":
            rep.append("The names and comments are conventional and don't affect correctness.")
        s.assistant(*rep)
        follow = []
        if ov3 == 5:
            follow.append(t["dubious_catch"][0].upper() + t["dubious_catch"][1:] + ".")
        elif ov3 == 4:
            follow.append("ok, I'll check that against the documentation.")
        if n_why >= 2:
            follow.append(why[1][0][0].upper() + why[1][0][1:])
        if s.student(*follow):
            rep = []
            if ov3 == 5:
                rep.append(t["dubious_ack"])
            if n_why >= 2:
                rep.append(why[1][1])
            s.assistant(*rep or ["Good idea."])
            if n_why == 3:
                s.student(why[2][0][0].upper() + why[2][0][1:])
                s.assistant(why[2][1])

    # ---------------- verification: OV1, OV5, OV4 ----------------
    ov1, ov5 = L("OV1"), L("OV5")
    ver = []
    if ov1 == 2:
        ver.append("Does this look right to you?")
    elif ov1 == 3:
        ver.append(f"I {t['spot']}.")
    elif ov1 == 4:
        ver.append(f"I compared it against {t['reference']} and it matches.")
    elif ov1 == 5:
        ver.append(f"I checked it against {t['reference']}: {t['ref_result']}.")
    if ov5 == 3:
        ver.append("I reran the case that was failing and read the output.")
    elif ov5 == 4:
        ver.append(f"I reran the failing case and a couple of others ({t['edge'].split(',')[0]}).")
    elif ov5 == 5:
        ver.append(f"After the fix I reran {t['edge']} to make sure it's still correct everywhere and "
                   "that the fix didn't introduce a new problem or slow it down.")
    if ov4_pat == "accept_all":
        ver.append(pick(rng, ["perfect, I'll use it as is", "great I'll just go with what you gave me"]))
    elif ov4_pat == "uniform":
        ver.append("I'm a little unsure about all of it, to be honest.")
    elif ov4_pat == "proportional":
        ver.append(f"I'm not worried about the naming or the comments, but I want to check "
                   f"the part about {t['subparts'][-1]} and any speedup numbers carefully myself.")
    s.student(*(ver or ["ok"]))
    s.assistant("That sounds right." if ver else "",
                f"One more thing: {t['model_claim']}",
                t["bias_claim"])

    # ---------------- performance claim: CS4, CS1, EI3 ----------------
    cs4, cs1, ei3 = L("CS4"), L("CS1"), L("EI3")
    perf = []
    if cs4 == 1:
        perf.append("oh ok, I'll go with that then")
    elif cs4 == 2:
        perf.append("ok")
    elif cs4 in (3, 4):
        perf.append("I'm not sure that's right, my numbers looked different. Can you check again?")
    if cs1 in (3, 4):
        perf.append(f"For reference, {t['profiler']}.")
    elif cs1 == 5:
        perf.append(t["reconcile"][0].upper() + t["reconcile"][1:] + ".")
    if cs4 == 5:
        perf.append("I disagree with that, and I tested it: I ran both versions and compared the timings, "
                    "and the measurement doesn't support the claim, so I'm keeping my explanation.")
    if ei3 in (3, 4):
        perf.append(t["bias_mid"])
    elif ei3 == 5:
        perf.append(t["bias_high"][0].upper() + t["bias_high"][1:] + ".")
    s.student(*(perf or ["ok"]))
    rep = []
    if cs4 >= 3 or cs1 >= 3:
        rep.append("Good point. The measurement is better evidence than my estimate, so go with what you measured.")
    if ei3 >= 3:
        rep.append("Fair, that claim depends on the setup; check it on your system before relying on it.")
    s.assistant(*rep or ["Great."])

    # ---------------- explanation and reflection: CS2, EI*, CS5 ----------------
    s.student(pick(rng, ["Can you summarize in two sentences what the bug was?",
                         "can you explain what was wrong in a couple sentences",
                         "Summarize the problem for me briefly."]))
    s.assistant(t["mech"])
    restate = {"verbatim": f"so for my report: {t['mech']}",
               "loose": f"so basically {t['loose_words']}",
               "close": f"So {t['close_words']}.",
               "own_words": f"So in my own words: {t['own_words']}."}[cs2_pat]
    refl = [restate]
    ei1, ei4, ei5, cs5 = L("EI1"), L("EI4"), L("EI5"), L("CS5")
    if raise_policy and ei1 in (2, 3) and not policy_first:
        refl.append("Good point about the course policy, I'll check it.")
    if ei4 in (3, 4):
        refl.append(t["stake_mid"][0].upper() + t["stake_mid"][1:] + ".")
    elif ei4 == 5:
        refl.append(t["stake_high"][0].upper() + t["stake_high"][1:] + ".")
    if ei5 in (3, 4):
        refl.append(t["attr_mid"][0].upper() + t["attr_mid"][1:] + ".")
    elif ei5 == 5:
        refl.append(t["attr_high"][0].upper() + t["attr_high"][1:] + ".")
    if team:
        if cs5 <= 2:
            refl.append("This is for a group project but I'm just doing my own piece.")
        elif cs5 in (3, 4):
            refl.append(t["team_mid"][0].upper() + t["team_mid"][1:] + ".")
        else:
            refl.append(t["team_high"][0].upper() + t["team_high"][1:] + ".")
    s.student(*refl)
    s.assistant("Sounds good. Good luck with the rest of it.")
    return s.turns


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="synthetic_hpc_transcripts")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--per-task", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    item_cols = [f"{d}_i{i}" for d in DIMS for i in range(1, 6)]
    jsonl, key_rows, label_rows = [], [], []
    sid = 0
    for task in TASKS:
        for _ in range(a.per_task):
            sid += 1
            latent, items, ov4_pat, cs2_pat = draw_student(rng)
            flaw = bool(rng.random() < 0.55)
            raise_policy = bool(rng.random() < 0.3)
            team = bool(rng.random() < 0.4)
            fenced = bool(rng.random() < 0.5)
            turns = render(TASKS[task], items, rng, flaw=flaw, raise_policy=raise_policy,
                           team=team, fenced=fenced, ov4_pat=ov4_pat, cs2_pat=cs2_pat)
            session = f"{sid:03d}_{task}"
            lines = [HEADER, ""]
            for i, (role, text) in enumerate(turns):
                lines.append(f"{'Student' if role == 'user' else 'Assistant'}: {text}")
                lines.append("")
                jsonl.append(dict(session_id=session, student=sid, discipline="HPC", phase="synthetic",
                                  turn=i, role=role, model=MODEL, content=text))
            (out / "transcripts" / f"{session}.txt").write_text("\n".join(lines), encoding="utf-8")
            pii = items["EthicalIntegration_i2"] <= 2
            key_rows.append(dict(student=sid, session_id=session, task=task, n_turns=len(turns),
                                 assistant_flawed_first_answer=flaw,
                                 assistant_raised_policy=raise_policy, team_session=team,
                                 code_fenced=fenced, identifiers_planted=pii,
                                 OV4_pattern=ov4_pat, CS2_pattern=cs2_pat,
                                 **{f"latent_{CODES[d]}": round(float(v), 3) for d, v in zip(DIMS, latent)}))
            lab = {"student": sid, "LABEL_SOURCE": "GENERATOR_TARGET_NOT_HUMAN_CODING"}
            for c in item_cols:
                lab[c] = items[c]
            if not team:
                lab["CollaborativeSensemaking_i5"] = "NA"
            label_rows.append(lab)
    with open(out / "hpc_test_logs.jsonl", "w", encoding="utf-8") as fh:
        for r in jsonl:
            fh.write(json.dumps(r) + "\n")
    for name, rows in (("design_key.csv", key_rows), ("generator_item_labels.csv", label_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {sid} transcripts, {len(jsonl)} turns to {out}/")


if __name__ == "__main__":
    main()
