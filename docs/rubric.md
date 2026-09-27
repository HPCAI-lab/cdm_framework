# CDM Scoring Rubric — 25 Items

The instrument: five dimensions, five items each, every item scored 1–5. This
document is the coder manual. It is also the specification the automated log
scorer implements — each item names the log feature that stands in for it in
`cdm/features.py`, so the human and machine paths score the same construct.

> [!IMPORTANT]
> These items are a **first draft for coder norming**, not a validated
> instrument. Anchors will move once three coders try to apply them to real
> transcripts and disagree. Expect to revise wording after the first norming
> round, and to revise it again after the second.

---

## How to score

**Unit of analysis.** One *session* — a single student's continuous work on one
task with the AI, from first prompt to last. A student contributes several
sessions; the student-level dimension score is the mean of their session scores.
Score every item on every session, even when evidence is thin (see level 1).

**The scale.** Anchors are given at 1, 3, and 5. Use 2 and 4 when the session
sits clearly between two anchors — 2 when it shows the level-1 pattern with one
isolated exception, 4 when it meets level 3 consistently and level 5 partially.
Do not use halves.

**Score what is in the transcript.** If a behavior is absent, it scores 1. A 1
means "no evidence in this session," not "the student is incapable." This is the
single most common coder drift: charitable inference from outside the transcript.

**Opportunity.** Some tasks give no occasion for an item — a factual lookup task
offers no stakeholder impact to consider. Mark the item `NA` rather than 1.
Items marked `NA` are dropped from that session's dimension mean, and a
dimension with three or more `NA` items is dropped for that session. Task design
should aim for at most one structural `NA` per dimension.

**Item ordering.** Within each dimension, items run from most to least demanding
(item 1 is the hardest to score highly on). This ordering is assumed by the
measurement model in `cdm/model_core.py`.

**HPC/PDC anchors.** Each item keeps its general anchors, which define the
construct, and adds HPC/PDC anchors that show what levels 1, 3 and 5 look like
on the pilot's task types: parallelizing a serial routine, debugging an MPI or
OpenMP program, and optimizing a CUDA kernel. Code HPC/PDC transcripts against
the HPC/PDC anchors; where one seems to conflict with the general anchor, the
general anchor governs and the conflict goes on the norming agenda. Like the
rest of this draft, these examples are expected to change after norming.

**Two items are non-monotone.** On OV4 and CS2, more of the behavior is not
better past a point; the anchors describe a middle band as the top score. Read
those two carefully — they are the items coders get backwards.

---

## Prompt Engineering

Eliciting useful output from an AI system.

**PE1 — Constraint specification** · `PromptEngineering_i1` · feature `constraint_marker_density`
Does the student state what the output must satisfy — format, length, audience, role, scope, exclusions?
- **1** — Bare topic or question with no stated requirements on the output.
- **3** — One or two constraints stated, usually format or length.
- **5** — Multiple constraints, including at least one about audience, purpose, or what to leave out; constraints are specific enough that a non-compliant answer would be identifiable.
- *HPC/PDC anchors:*
  - **1** — "Write matrix multiply in MPI."
  - **3** — Names the language and API ("in C with OpenMP") or a problem size, but nothing about correctness or performance.
  - **5** — States the requirements a wrong answer would violate: square N up to 4096, row-major doubles, result must match the serial version to 1e-9, at most 64 ranks, no external BLAS.

**PE2 — Context provision** · `PromptEngineering_i2` · feature `context_ratio`
Does the student supply the material the model needs, rather than assuming it?
- **1** — No context; references documents, data, or prior work the model cannot see ("fix my essay", "is this right?").
- **3** — Relevant material pasted or summarized, but partially — the model still has to guess at the task setting.
- **5** — Task-relevant material supplied deliberately, including the student's own goal or current draft state, with irrelevant material left out.
- *HPC/PDC anchors:*
  - **1** — "My CUDA kernel is slow, why?" with no kernel, launch configuration, or GPU named.
  - **3** — Pastes the kernel but not the launch configuration, problem size, compiler flags, or the error or profiler output.
  - **5** — Supplies the routine in question, the run configuration (grid/block sizes, ranks, threads), compiler and flags, hardware, and the actual error or profiler output; leaves unrelated files out.

**PE3 — Iterative refinement** · `PromptEngineering_i3` · feature `refinement_ratio`
When output falls short, does the student revise the prompt rather than abandon or accept it?
- **1** — No revision: either takes the first output as final, or drops the thread and starts an unrelated request.
- **3** — Re-asks when output is unsatisfactory, but the revision mostly repeats or rephrases without changing what was asked for.
- **5** — Revisions are diagnostic — the student changes the specific element that produced the shortfall (adds a constraint, narrows scope, supplies a missing input) and the change is traceable to the previous output.
- *HPC/PDC anchors:*
  - **1** — Accepts the first MPI version even though it deadlocks, or abandons the thread for an unrelated request.
  - **3** — "Still doesn't work, try again" with the same request.
  - **5** — After a race in generated OpenMP code, re-asks specifying that `sum` needs a reduction and the inner indices must be private — the change is traceable to the defect.

**PE4 — Lexical specificity** · `PromptEngineering_i4` · feature `specificity_index`
Does the student use precise, domain-appropriate terms rather than vague referents?
- **1** — Dominated by general placeholders ("stuff", "the thing", "make it better", "professional").
- **3** — Mixed: correct domain terms alongside vague qualifiers.
- **5** — Consistently precise, using the technical vocabulary of the task; qualifiers are operationalized ("under 200 words", "for a lay reader") rather than left to interpretation.
- *HPC/PDC anchors:*
  - **1** — "Make it faster", "parallel stuff", "fix the thing".
  - **3** — Correct terms (thread, rank, kernel) mixed with vague goals ("optimize it better").
  - **5** — Precise terms — false sharing, coalesced access, `MPI_Allreduce` vs `MPI_Reduce`, strong vs weak scaling — and operationalized goals ("at least 6× speedup on 8 threads at N = 2048").

**PE5 — Task framing** · `PromptEngineering_i5` · feature `framing_clarity`
Is the request itself identifiable — does the prompt state an action to perform?
- **1** — No identifiable request: a topic, a fragment, or several unrelated asks bundled into one turn.
- **3** — A clear single request, stated plainly.
- **5** — A clear single request with the success condition attached ("so that I can…", "the output I need is…").
- *HPC/PDC anchors:*
  - **1** — Pastes code with no request, or bundles "explain MPI, fix my code, and write my report" in one turn.
  - **3** — "Find the bug in this OpenMP loop."
  - **5** — "Find why this loop gives wrong results only for N > 64, so I can explain the race in my report."

---

## Problem Decomposition

Breaking a task into tractable sub-problems.

**PD1 — Sub-problem identification** · `ProblemDecomposition_i1` · feature `subtask_count`
Does the student break a composite task into parts rather than handing it over whole?
- **1** — The whole assignment is passed to the model in one request.
- **3** — The task is split into two or three parts, mostly along obvious lines.
- **5** — Parts are identified by what makes them separately hard, not just by order of appearance; the student names what each part requires.
- *HPC/PDC anchors:*
  - **1** — Pastes the whole assignment: "parallelize this program."
  - **3** — Splits along obvious lines: "first the serial version, then MPI."
  - **5** — Splits by what makes each part hard — distributing A's rows and broadcasting B, gathering results, the case where N is not divisible by the rank count, and the correctness check — and says what each needs.

**PD2 — Sequencing logic** · `ProblemDecomposition_i2` · feature `sequence_marker_density`
Are sub-tasks ordered with awareness of dependencies?
- **1** — No ordering; sub-requests arrive in arbitrary sequence, or later requests need outputs never obtained.
- **3** — Sensible order, mostly implicit.
- **5** — Order is stated and justified by dependency ("before we draft, I need the criteria settled, because…"); the student reorders when a dependency surfaces mid-session.
- *HPC/PDC anchors:*
  - **1** — Asks to tune a CUDA kernel before it produces correct results, or asks how to gather results before the data distribution exists.
  - **3** — Correctness before performance, but never stated.
  - **5** — "Before tuning tile size I need a correct baseline, otherwise I can't tell a speedup from a wrong answer"; reorders when a deadlock surfaces mid-session.

**PD3 — Scope control** · `ProblemDecomposition_i3` · feature `scope_qualifier_density`
Does the student bound the request to something the model can do well in one turn?
- **1** — Unbounded requests ("write my literature review").
- **3** — Requests are bounded, generally by length or count.
- **5** — Bounds are chosen to fit the model's working limits and the student's ability to check the result; the student explicitly defers parts to later turns.
- *HPC/PDC anchors:*
  - **1** — "Write my whole MPI + CUDA project."
  - **3** — Bounded to one file or function.
  - **5** — One kernel, loop, or communication step per turn, sized so the output can be compiled and tested before moving on; explicitly defers the rest ("we'll do the halo exchange later").

**PD4 — AI limitation awareness** · `ProblemDecomposition_i4` · feature `limitation_reference_count`
Does the student anticipate what the model cannot do and route around it?
- **1** — No sign of limits; asks for current data, private records, exact computation, or personal knowledge as if reliable.
- **3** — Limits acknowledged after being hit ("you don't have access to that, so…").
- **5** — Limits anticipated before they bite; the student pre-emptively supplies what the model lacks or reserves that part for a different tool.
- *HPC/PDC anchors:*
  - **1** — Asks the model for the actual runtime or speedup on the cluster, as if it could run the code.
  - **3** — Adjusts after hitting the limit: "you can't run it, so here is my timing output."
  - **5** — Supplies timings, core counts, GPU model and interconnect up front, and plans to verify performance claims by running them rather than asking.

**PD5 — Task-to-tool allocation** · `ProblemDecomposition_i5` · feature `allocation_marker_count`
Does the student decide which parts to delegate and which to keep?
- **1** — Everything delegated, or nothing (the AI used only as a search box).
- **3** — An implicit division of labor is visible in what does and does not get asked.
- **5** — The division is stated and reasoned ("I'll do the analysis myself since I need to defend it; you draft the framing").
- *HPC/PDC anchors:*
  - **1** — Delegates everything, including the scaling analysis in the report, or uses the model only to look up API syntax.
  - **3** — A division of labor is visible in what does and doesn't get asked.
  - **5** — "I'll write the decomposition and scaling analysis myself since I have to defend them; you check my `MPI_Scatterv` counts and displacements."

---

## Output Validation

Checking and correcting AI output.

**OV1 — Independent verification** · `OutputValidation_i1` · feature `external_source_count`
Does the student check claims against something outside the conversation?
- **1** — No verification of any claim.
- **3** — Spot-checks the most consequential claim, or asks the model to verify itself.
- **5** — Checks against an identifiable external source and says what it was; the check is aimed at the claims that matter most if wrong.
- *HPC/PDC anchors:*
  - **1** — Runs nothing; submits generated code on inspection.
  - **3** — Compiles and runs it once, or asks the model "is this correct?"
  - **5** — Checks against a named reference — the serial result, a known-answer test, BLAS/cuBLAS output, the MPI or OpenMP specification — aimed at what matters most if wrong: edge-case sizes and claimed speedups.

**OV2 — Error detection** · `OutputValidation_i2` · feature `error_assertion_count`
Does the student notice specific problems in the output?
- **1** — No errors noted, including where the transcript contains clear ones.
- **3** — Notices that something is wrong, stated generally ("this doesn't look right").
- **5** — Identifies the specific defect and what makes it wrong ("this citation is to a journal that stopped publishing in 2019").
- *HPC/PDC anchors:*
  - **1** — No errors noted, even where the generated code has an obvious race or mismatched send/receive.
  - **3** — "The output is wrong for big N" / "it hangs."
  - **5** — Names the defect and why: "`sum` is shared across threads, so the updates race"; "every rank posts a blocking send before its receive, so it deadlocks."

**OV3 — Provenance demand** · `OutputValidation_i3` · feature `provenance_request_count`
Does the student ask where output came from, and then look?
- **1** — Never asks for reasoning or sources.
- **3** — Asks for sources or reasoning.
- **5** — Asks, then inspects what comes back and reacts to its quality — notices when a citation is unverifiable or reasoning is circular.
- *HPC/PDC anchors:*
  - **1** — Never asks why a pragma, clause, or call was chosen.
  - **3** — Asks "why does this need `__syncthreads()`?" or for the relevant documentation.
  - **5** — Asks, then checks the answer — notices a cited MPI function that doesn't exist, or an OpenMP clause unsupported in the stated version, and says so.

**OV4 — Calibrated skepticism** · `OutputValidation_i4` · feature `scrutiny_rate` · **non-monotone**
Does the student treat output as provisional, in proportion to stakes?
- **1** — One of two extremes: accepts everything as fact, **or** rejects/re-asks everything indiscriminately.
- **3** — Uniform mild caution regardless of what the claim is.
- **5** — Scrutiny tracks consequence: heavy on load-bearing or checkable factual claims, light on phrasing and brainstorming. Both blanket trust and blanket distrust score low here.
- *HPC/PDC anchors:*
  - **1** — Accepts every generated kernel and speedup claim, **or** re-tests trivia and demands justification for formatting, comments, and variable names.
  - **3** — The same moderate caution everywhere.
  - **5** — Heavy scrutiny on synchronization, reductions, index arithmetic, and claimed speedups; light on boilerplate, naming, and comments.

**OV5 — Correction and re-check** · `OutputValidation_i5` · feature `recheck_rate`
After finding a problem, does the student confirm the fix?
- **1** — Errors found are not followed up, or the corrected output is accepted unexamined.
- **3** — Asks for a correction and reads it.
- **5** — Verifies the corrected output against the original defect, and checks whether the correction introduced new problems.
- *HPC/PDC anchors:*
  - **1** — After a fix for the race, accepts it without rerunning.
  - **3** — Reruns the failing case and reads the output.
  - **5** — Reruns the failing case and the ones that previously passed (other N, rank and thread counts), and checks the fix didn't create a new problem — e.g. the added critical section erased the speedup.

---

## Ethical Integration

Applying ethical judgment to AI use.

**EI1 — Spontaneous ethical flagging** · `EthicalIntegration_i1` · feature `unprompted_ethics_rate`
Does the student raise a concern the task and the model did not prompt?
- **1** — No ethical consideration appears, or it appears only after the model or the assignment raises it.
- **3** — Engages substantively once prompted.
- **5** — Raises a concern unprompted and pursues it — the concern changes what the student then asks for.
- *HPC/PDC anchors:*
  - **1** — None, or only once the model or assignment raises it.
  - **3** — Engages when the assignment asks for an AI-use statement.
  - **5** — Raises it unprompted — whether submitting a generated kernel is allowed under the course policy, or whether a large benchmark sweep is fair use of a shared allocation — and it changes what they ask for.

**EI2 — Privacy and data handling** · `EthicalIntegration_i2` · feature `privacy_handling`
How does the student treat identifiable or confidential material?
- **1** — Pastes identifiable third-party information (names with clinical, financial, or academic detail) without apparent consideration.
- **3** — Avoids obvious identifiers, or notes the issue in passing.
- **5** — De-identifies deliberately before sending, and says why; distinguishes what is safe to share from what is not.
- *HPC/PDC anchors:*
  - **1** — Pastes cluster credentials, SSH keys, allocation IDs, usernames or hostnames, another student's code, or unreleased research code without apparent thought.
  - **3** — Avoids obvious secrets, or mentions the issue in passing.
  - **5** — Redacts hostnames, usernames, tokens and proprietary code before sending and says why; distinguishes what can be shared (course starter code) from what cannot.

**EI3 — Bias recognition** · `EthicalIntegration_i3` · feature `bias_marker_count`
Does the student ask whose perspective the output carries?
- **1** — Output treated as neutral.
- **3** — Notes that a source or framing may be partial.
- **5** — Identifies a specific absence or skew and acts on it — asks for the missing perspective, or discounts the output accordingly.
- *HPC/PDC anchors:*
  - **1** — Treats the model's recommendation as neutral — the vendor, library, or single-benchmark performance claim it offers.
  - **3** — Notes a recommendation may be partial ("that advice is NVIDIA-specific").
  - **5** — Identifies the specific skew and acts on it — asks for a portable alternative (OpenMP offload, SYCL), or discounts a speedup measured on one input size or one machine.

**EI4 — Stakeholder impact** · `EthicalIntegration_i4` · feature `stakeholder_marker_count`
Does the student consider who is affected by the work downstream?
- **1** — No party beyond the student and the grader is considered.
- **3** — Affected parties named generically ("the patient", "customers").
- **5** — A specific consequence for a specific party is reasoned about, and it constrains the output the student accepts.
- *HPC/PDC anchors:*
  - **1** — No party beyond the student and the grader is considered.
  - **3** — Names affected parties generically ("other cluster users").
  - **5** — Reasons about a specific consequence for a specific party and it constrains what they run — won't run a 256-rank test on the login node, weighs a sweep's allocation or energy cost, or considers users of a shared library they are changing.

**EI5 — Attribution and disclosure** · `EthicalIntegration_i5` · feature `attribution_marker_count`
Does the student represent the AI's contribution honestly?
- **1** — No attention to what will be represented as the student's own work.
- **3** — Mentions citation or disclosure of AI use.
- **5** — Tracks which parts are AI-produced and states how they will be disclosed or rewritten; distinguishes drafting help from substantive authorship.
- *HPC/PDC anchors:*
  - **1** — No attention to what will be represented as the student's own work.
  - **3** — Mentions they will note AI use in the report.
  - **5** — Tracks which functions or kernels are AI-produced and states how each will be disclosed in the report or code comments, distinguishing debugging help from authored code.

---

## Collaborative Sensemaking

Building shared understanding with AI in the loop.

**CS1 — Perspective integration** · `CollaborativeSensemaking_i1` · feature `reconcile_count`
Does the student reconcile AI output with other sources or teammates?
- **1** — AI output used in isolation; conflicts with other sources go unremarked.
- **3** — Other sources mentioned alongside AI output.
- **5** — Conflicts are surfaced and resolved with a stated reason for preferring one account, or a synthesis that keeps what each contributes.
- *HPC/PDC anchors:*
  - **1** — Uses the model's explanation in isolation even when profiler output or documentation contradicts it.
  - **3** — Mentions profiler output or documentation alongside the model's account.
  - **5** — Surfaces the conflict (the model says compute-bound, `perf` or Nsight shows memory-bound) and resolves it with a stated reason.

**CS2 — Explanation quality** · `CollaborativeSensemaking_i2` · feature `paraphrase_overlap` · **non-monotone**
When the student restates AI output, is the restatement their own and accurate?
- **1** — One of two extremes: verbatim or near-verbatim reproduction with no reformulation, **or** restatement so loose it loses the content.
- **3** — Accurate restatement that stays close to the original wording.
- **5** — Reformulated in the student's own terms, accurately, and recast for the audience or purpose at hand. Copying scores low here even though it is faithful.
- *HPC/PDC anchors:*
  - **1** — Copies the model's explanation of the race into notes or the report verbatim, **or** restates it so loosely the mechanism is lost ("the threads mess it up").
  - **3** — Accurate, but close to the model's wording.
  - **5** — Own terms, accurate, recast for the purpose: "the barrier stops a fast warp from reading a half-loaded tile," written for the report's performance-analysis section.

**CS3 — Explanation seeking** · `CollaborativeSensemaking_i3` · feature `why_how_rate`
Does the student ask how and why, not only what?
- **1** — Only product requests ("give me", "write", "list").
- **3** — Occasional mechanism or reasoning questions.
- **5** — Sustained probing of reasoning, with follow-ups that build on the previous answer rather than restarting.
- *HPC/PDC anchors:*
  - **1** — Only "give me the code."
  - **3** — An occasional "why is this faster?"
  - **5** — Sustained probing that builds on each answer: why i-k-j loop order helps the cache, then how that interacts with tile size, then why speedup plateaus at 8 threads.

**CS4 — Disagreement handling** · `CollaborativeSensemaking_i4` · feature `disagreement_count`
When the AI conflicts with the student's understanding, what happens?
- **1** — Conflict ignored, or the student's view abandoned immediately on contradiction.
- **3** — States disagreement and re-asks.
- **5** — States the disagreement with grounds, tests it, and updates in whichever direction the evidence goes — including sticking with their own position when it holds up.
- *HPC/PDC anchors:*
  - **1** — Ignores the conflict, or drops a correct view as soon as the model contradicts it.
  - **3** — States the disagreement and re-asks.
  - **5** — States the disagreement with grounds, tests it by running both versions and comparing results or timings, and updates accordingly — keeping their own position when the test supports it.

**CS5 — Shared-artifact contribution** · `CollaborativeSensemaking_i5` · feature `shared_artifact_markers`
In team sessions, does the work build toward a joint product?
- **1** — Individual work in a shared setting; no reference to teammates' contributions.
- **3** — Own contribution framed as part of a larger whole.
- **5** — Explicitly builds on or reconciles teammates' parts; uses the AI to close gaps between them.
- *In single-student sessions this item is `NA`.*
- *HPC/PDC anchors:*
  - **1** — Works on their own module with no reference to teammates' parts.
  - **3** — Frames their kernel as one part of the team's code.
  - **5** — Uses the model to reconcile teammates' parts — matching data layouts between one member's MPI decomposition and another's CUDA kernel, or resolving who owns the halo exchange.

---

## Coder norming procedure

1. **Anchor pass.** Three coders independently score the same three transcripts,
   then meet. The output of the meeting is edits to this document, not a
   negotiated score.
2. **Second pass.** Three new transcripts, independent scoring, weighted κ
   computed per item. Any item below κ = 0.70 gets rewritten; items that
   survive two rounds are frozen.
3. **Holdout.** Reserve 15% of sessions, scored by all three coders throughout
   collection, to detect drift. The rest are single-coded once κ clears 0.80.
4. **Record everything.** Every version of this file is tagged, and every score
   carries the rubric version it was produced under. Scores from different
   rubric versions are not pooled.

## Crosswalk to the automated path

Every item above names a feature computed in `cdm/features.py`. The automated
scorer in `cdm/autoscore.py` converts each feature to a 1–5 item score using
declared cutpoints. Those cutpoints are **provisional guesses until calibrated
against human coding on the same sessions** — see `docs/pilot_workflow.md`. A
feature is a proxy for its item, never a definition of it: `why_how_rate` counts
question forms, while CS3 asks whether the probing built on what came before,
which no current feature detects. Where they diverge, the human score is the
criterion and the feature is the thing that needs fixing.
