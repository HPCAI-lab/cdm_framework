"""Marker lexicons, as swappable profiles.

Every pattern in a lexicon is a hypothesis about the surface evidence for a
construct, and that hypothesis is domain-bound. The ``general`` profile was
written against general academic coursework; its ethical-integration and
stakeholder patterns look for patients, clients and customers. None of those
appear in a transcript about debugging an MPI reduction, so on HPC coursework
those features go flat and their items carry no information whatever the
cutpoints are.

The ``hpc`` profile substitutes patterns that HPC and PDC coursework actually
produces: correctness against a serial baseline, race conditions and
deadlocks, profiler output, scaling behaviour, and the attribution language of
academic integrity rather than of patient privacy.

Which profile a corpus was scored under is part of the score's provenance. The
active profile name is recorded in the feature frame and carried into the
scored output.
"""
from __future__ import annotations

import re

_FLAGS = re.I | re.M


def _rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts), _FLAGS)


_BACKREF = r"(?<!\bfor )(?<!\bin )(?<!\bof )(?<!\bsee )(?<!\bstep )"


# --- shared across profiles -------------------------------------------------
# Structural markers: these describe how a request is phrased, not what it is
# about, and so do not vary by domain.

_SHARED = {
    "revision": _rx(r"\binstead\b", r"\bactually\b", r"\btry again\b", r"\bthat'?s not\b", r"\brewrite\b",
                    r"\bshorter\b", r"\blonger\b", r"\bmore specific\b", r"\bsame but\b", r"\bredo\b",
                    r"\blet'?s try\b", r"\bcloser but\b", r"\bnot quite\b"),
    "success_cond": _rx(r"\bso (?:that )?i can\b", r"\bso i can\b", r"\bthe (?:output|result) i need\b",
                        r"\bi(?:'|’)?m trying to\b", r"\bgoal is\b", r"\bin order to\b"),
    "request_head": _rx(r"^\s*(?:please\s+)?(?:write|draft|list|explain|summari[sz]e|compare|give|make|create|generate|outline|analy[sz]e|translate|rewrite|fix|check|find|suggest|help|implement|parallelis|parallelize|optimi[sz]e|debug|profile|port)\b",
                        r"\bcan you\b", r"\bcould you\b", r"\bwould you\b", r"\bi need you to\b"),
    # Line-start items ("1. ...", "- ...") and inline ones: "(1) ... (2) ...",
    # A later back-reference ("for (2) the docs say") is not a new item.
    # "(a) ... (b) ...", and "1) ..." after a colon, comma or semicolon. The
    # look-behinds keep code such as sleep(1), f(a) or A[i](2) from counting.
    "enumeration": _rx(r"^\s*\d+[.)]\s", r"^\s*[-*•]\s", r"\bfirst(?:ly)?\b.*\bsecond(?:ly)?\b",
                       _BACKREF + r"(?<![\w\])])\(\s*\d{1,2}\s*\)(?=\s)", _BACKREF + r"(?<![\w\])])\([a-h]\)(?=\s)",
                       r"(?<=[:;,]\s)\d{1,2}\)\s",
                       r"\btwo things\b", r"\bthree things\b", r"\bpart (?:one|two|1|2)\b"),
    "sequence": _rx(r"\bfirst\b", r"\bthen\b", r"\bnext\b", r"\bafter that\b", r"\bbefore (?:we|i|that)\b",
                    r"\bonce (?:we|i|that|you)\b", r"\bstep \d\b", r"\bfinally\b", r"\blater\b", r"\bsubsequently\b"),
    "scope": _rx(r"\bjust\b", r"\bonly the\b", r"\bfor now\b", r"\bto start\b", r"\blimit(?:ed)? to\b",
                 r"\bnarrow(?:ed)? (?:down|to)\b", r"\bone (?:section|part|piece|function|kernel|loop)\b",
                 r"\bset(?:ting)? aside\b", r"\bwe'?ll (?:get to|come back)\b"),
    "limitation": _rx(r"\byou (?:can'?t|cannot|don'?t have) (?:access|see|browse|know|run|compile)\b",
                      r"\byour (?:training|knowledge) (?:data|cut)", r"\bcut-?off\b", r"\bsince you can'?t\b",
                      r"\byou might (?:not know|make up|hallucinat)", r"\bhallucinat", r"\bno access to\b",
                      r"\bup-?to-?date\b", r"\bmake (?:something|it) up\b",
                      r"\byou can'?t (?:see|run) (?:my|the) (?:cluster|machine|output|hardware)\b"),
    "hedge": _rx(r"\bare you sure\b", r"\bis that right\b", r"\bdouble-?check\b", r"\bnot sure (?:that|this|about)\b",
                 r"\bseems? (?:off|odd|wrong)\b", r"\bverify\b", r"\bconfirm\b", r"\bcan i trust\b",
                 r"\bsceptical\b", r"\bskeptical\b"),
    "why_how": _rx(r"\bwhy\b", r"\bhow (?:does|do|would|did|is)\b", r"\bwhat makes\b", r"\bexplain (?:why|how)\b",
                   r"\bwhat'?s the (?:reasoning|mechanism|logic)\b", r"\bwalk me through\b"),
    "disagreement": _rx(r"\bi (?:disagree|don'?t think|doubt)\b", r"\bi'?m not convinced\b",
                        r"\bbut that (?:ignores|misses)\b", r"\bi still think\b", r"\bmy reading (?:is|was)\b",
                        r"\bpush back\b", r"\bi'?d argue\b", r"\bthat contradicts\b"),
    "allocation": _rx(r"\bi'?ll (?:do|handle|write|analy[sz]e|take|run|compile|test)\b",
                      r"\byou (?:do|handle|draft|take|write)\b", r"\bi'?ll (?:leave|keep) (?:that|this)\b",
                      r"\bmyself\b", r"\bon my own\b", r"\bi want to (?:do|write) (?:that|this|it)\b"),
    "shared_artifact": _rx(r"\bour (?:draft|report|code|repo|branch|section|analysis|project|submission)\b",
                           r"\bthe team\b", r"\bmy (?:teammate|group|partner|lab partner)",
                           r"\bwe'?(?:ll|re) (?:writing|adding|putting|submitting|pushing)\b",
                           r"\bgroup (?:deliverable|submission|report|assignment)\b"),
}

# --- general academic profile ----------------------------------------------

GENERAL = dict(_SHARED, **{
    "constraint": _rx(r"\bin (?:under|no more than|at most|exactly|about) \d+",
                      r"\b\d+\s*(?:words|sentences|bullets|pages|paragraphs|items)\b",
                      r"\bas (?:a|an) (?:table|list|bullet|outline|email|memo|abstract)\b",
                      r"\bformat(?:ted)? as\b",
                      r"\bfor (?:a|an) (?:lay|general|expert|undergraduate|clinical|executive) (?:reader|audience)\b",
                      r"\bdon'?t include\b", r"\bwithout (?:using|mentioning)\b", r"\bavoid\b", r"\btone\b",
                      r"\bact as\b", r"\byou are (?:a|an)\b", r"\bonly (?:use|include|cover)\b"),
    "external_source": _rx(r"\baccording to\b", r"\bi (?:looked|checked|read|found)\b",
                           r"\bthe (?:textbook|paper|article|syllabus|guideline|standard|manual|docs?)\b",
                           r"\bmy (?:notes|professor|instructor|lecture)\b", r"\bpubmed\b", r"\bgoogle scholar\b",
                           r"\bi verified\b", r"\bcross-?check", r"\bon the (?:cdc|who|nih|sec|irs) (?:site|website)\b"),
    "error_assert": _rx(r"\bthat'?s (?:wrong|incorrect|not right|false)\b",
                        r"\byou (?:got|have) (?:that|it|this) wrong\b",
                        r"\bthis is (?:wrong|inaccurate|incorrect)\b", r"\bdoesn'?t (?:exist|match)\b",
                        r"\bmade (?:that|this|it) up\b", r"\bmisquot", r"\bactually it'?s\b",
                        r"\bno,? (?:it|that|the)\b", r"\bcontradicts\b", r"\bthat citation\b"),
    "provenance": _rx(r"\bcite\b", r"\bcitation", r"\bsource(?:s)?\b", r"\bwhere did (?:you|that) (?:get|come)\b",
                      r"\bhow do you know\b", r"\bwhat'?s (?:your|the) (?:basis|evidence)\b", r"\breference(?:s)?\b",
                      r"\bshow (?:me )?your (?:work|reasoning)\b", r"\bis that (?:peer-?reviewed|documented)\b"),
    "recheck": _rx(r"\bnow (?:check|verify|confirm)\b", r"\bdid (?:that|you) fix\b",
                   r"\bis (?:it|that) (?:right|correct) now\b", r"\bre-?check\b", r"\bstill (?:wrong|off)\b",
                   r"\bbetter now\b", r"\bconfirm(?:ed)? (?:the|that) (?:fix|correction)\b"),
    "ethics": _rx(r"\bethic", r"\bconsent\b", r"\bprivacy\b", r"\bconfidential", r"\bbias(?:ed)?\b",
                  r"\bfair(?:ness)?\b", r"\bharm\b", r"\bplagiaris", r"\bacademic integrity\b", r"\bdisclos",
                  r"\bhipaa\b", r"\bferpa\b", r"\bde-?identif", r"\banonymi[sz]", r"\bstakeholder",
                  r"\bvulnerable\b", r"\bequit(?:y|able)\b"),
    "privacy": _rx(r"\bprivacy\b", r"\bconfidential", r"\bhipaa\b", r"\bferpa\b", r"\bde-?identif",
                   r"\banonymi[sz]", r"\bredact", r"\bi'?ve removed (?:the )?(?:names?|identifiers?)\b",
                   r"\bno (?:real )?names?\b", r"\bpseudonym", r"\bpersonal(?:ly)? identif"),
    "bias": _rx(r"\bbias(?:ed|es)?\b", r"\bstereotyp", r"\brepresentative\b", r"\bwhose perspective\b",
                r"\bleft out\b", r"\bmissing (?:voices?|perspectives?|groups?)\b", r"\bwestern\b", r"\bskew"),
    "stakeholder": _rx(r"\bpatient", r"\bclient", r"\bcustomer", r"\bparticipant", r"\bcommunity\b",
                       r"\bstakeholder", r"\bwho (?:is|would be) affected\b", r"\bimpact on\b",
                       r"\bfor the (?:reader|user|nurse|teacher)\b", r"\bend users?\b"),
    "attribution": _rx(r"\bcite (?:you|this|the ai|chatgpt|claude)\b", r"\bdisclos", r"\backnowledg",
                       r"\bmy own words\b", r"\bi'?ll rewrite\b", r"\bplagiaris", r"\bai-?(?:generated|assisted)\b",
                       r"\bhand (?:this|it) in\b", r"\battribut"),
    "reconcile": _rx(r"\bon the other hand\b", r"\bbut (?:the|my) (?:textbook|professor|article|notes|data)\b",
                     r"\bconflicts? with\b", r"\bcontradicts\b", r"\bwhereas\b", r"\bboth (?:say|claim|suggest)\b",
                     r"\breconcile\b", r"\bdisagrees? with\b", r"\bcombin(?:e|ing) (?:that|this|your|both)\b",
                     r"\bmy (?:teammate|group|partner)\b"),
})

# --- HPC / PDC profile ------------------------------------------------------
# Output validation in this setting means running the code: correctness against
# a serial baseline, a sanitizer or debugger, a profiler, a scaling study.
# Ethical integration means attribution and permitted-use rules on shared
# cluster resources, not patient privacy.

HPC = dict(_SHARED, **{
    "constraint": _rx(r"\bin (?:under|no more than|at most|exactly|about) \d+",
                      r"\b\d+\s*(?:lines|threads|ranks|processes|blocks|warps|iterations)\b",
                      r"\b(?:using|with|in) (?:mpi|openmp|cuda|openacc|pthreads|hip|sycl|kokkos)\b",
                      r"\bc\+\+\d*\b", r"\bfortran\b", r"\bcompile[sd]? with\b", r"\b-O[0-3]\b",
                      r"\bwithout (?:using|changing|modifying)\b", r"\bdon'?t (?:change|use|add)\b",
                      r"\bkeep the (?:signature|interface|api|loop structure)\b",
                      r"\bmust (?:be|remain) (?:thread-?safe|deterministic|reproducible)\b",
                      r"\bonly (?:use|include|modify|change)\b", r"\bact as\b", r"\byou are (?:a|an)\b"),
    "external_source": _rx(r"\baccording to\b", r"\bi (?:looked|checked|read|found|ran|tested|compiled|profiled)\b",
                           r"\bthe (?:docs?|documentation|man ?page|standard|spec|textbook|slides?)\b",
                           r"\b(?:mpi|openmp|cuda|nvidia|intel) (?:docs?|documentation|guide|manual|spec)\b",
                           r"\bmy (?:notes|professor|instructor|ta|lecture)\b",
                           r"\bserial (?:version|baseline|reference|implementation)\b",
                           r"\bagainst the (?:reference|baseline|known)\b", r"\bi verified\b", r"\bcross-?check",
                           r"\bstack ?overflow\b", r"\bvalgrind\b", r"\bnvprof\b", r"\bnsight\b",
                           r"\bgdb\b", r"\bperf\b", r"\bsanitizer\b", r"\bgprof\b"),
    "error_assert": _rx(r"\bthat'?s (?:wrong|incorrect|not right)\b", r"\bdoesn'?t (?:compile|build|link|run)\b",
                        r"\bsegfault", r"\bsegmentation fault\b", r"\bcore dump", r"\bdeadlock", r"\bhangs?\b",
                        r"\brace condition\b", r"\bwrong (?:answer|result|output|value)s?\b",
                        r"\bmismatch(?:es)?\b", r"\bdiverges?\b", r"\bnan\b", r"\boff by\b",
                        r"\bslower than (?:the )?serial\b", r"\bno speed-?up\b", r"\bthat api doesn'?t exist\b",
                        r"\bundefined (?:reference|behavior|behaviour)\b", r"\bcompiler error\b",
                        r"\bit returns? (?:the )?wrong\b", r"\bactually it'?s\b"),
    "provenance": _rx(r"\bwhy (?:does|do|is) (?:that|this|it)\b", r"\bhow do you know\b",
                      r"\bwhat'?s (?:your|the) (?:basis|reasoning)\b", r"\bshow (?:me )?your (?:work|reasoning)\b",
                      r"\bwhich (?:standard|version|spec)\b", r"\bis that in the (?:standard|spec|docs?)\b",
                      r"\bcite\b", r"\bsource(?:s)?\b", r"\breference(?:s)?\b",
                      r"\bwhere (?:in the docs?|does the standard)\b",
                      r"\bexplain (?:the|that) (?:pragma|directive|clause|flag)\b"),
    "recheck": _rx(r"\bnow (?:check|verify|confirm|run|test|profile)\b", r"\bdid (?:that|you) fix\b",
                   r"\bis (?:it|that) (?:right|correct) now\b", r"\bre-?(?:check|run|test|compile|profile)\b",
                   r"\bstill (?:wrong|off|slow|failing|hangs?)\b", r"\bbetter now\b", r"\bpasses now\b",
                   r"\bmatches (?:the )?(?:serial|reference|baseline)\b", r"\bspeed-?up is now\b"),
    "ethics": _rx(r"\bethic", r"\bacademic integrity\b", r"\bplagiaris", r"\bdisclos", r"\bcite (?:the )?ai\b",
                  r"\bhonor code\b", r"\ballowed to use\b", r"\bpermitted\b", r"\bcourse policy\b",
                  r"\blicen[cs]e\b", r"\bgpl\b", r"\bmy own work\b", r"\bsubmit(?:ting)? (?:this|it) as\b",
                  r"\bshared (?:cluster|allocation|node)\b", r"\bcompute (?:budget|allocation|quota)\b",
                  r"\benergy (?:cost|use|budget)\b", r"\bcarbon\b", r"\bfair (?:share|use)\b"),
    "privacy": _rx(r"\bprivacy\b", r"\bconfidential", r"\bproprietary\b", r"\bnda\b", r"\bunpublished\b",
                   r"\bcredential", r"\bapi key\b", r"\bpassword\b", r"\bssh key\b", r"\bi'?ve removed\b",
                   r"\bredact", r"\banonymi[sz]", r"\bcan'?t (?:share|paste) (?:the )?(?:real|full)\b",
                   r"\bsanitis(?:e|ed)\b", r"\bsanitiz(?:e|ed)\b"),
    "bias": _rx(r"\bbias(?:ed|es)?\b", r"\brepresentative\b", r"\bbenchmark (?:choice|selection)\b",
                r"\bcherry-?pick", r"\bonly (?:tested|measured) (?:on|with)\b", r"\bedge case",
                r"\bdoes (?:that|this) generali[sz]e\b", r"\bwhat about (?:other|larger|smaller)\b",
                r"\bproblem size\b", r"\bsingle (?:run|trial|measurement)\b", r"\bvariance (?:across|between) runs\b"),
    "stakeholder": _rx(r"\bother users?\b", r"\bshared (?:cluster|system|queue|node)\b", r"\bqueue\b",
                       r"\bwho (?:is|would be) affected\b", r"\bimpact on\b", r"\bdownstream\b",
                       r"\bwhoever (?:maintains|reads|runs)\b", r"\bmaintainer", r"\bthe grader\b",
                       r"\bfuture me\b", r"\bnext (?:person|student)\b", r"\bproduction\b", r"\bend users?\b"),
    "attribution": _rx(r"\bcite (?:you|this|the ai|chatgpt|claude|copilot)\b", r"\bdisclos", r"\backnowledg",
                       r"\bmy own (?:words|work|code)\b", r"\bi'?ll rewrite\b", r"\bplagiaris",
                       r"\bai-?(?:generated|assisted)\b", r"\bhand (?:this|it) in\b", r"\battribut",
                       r"\bcomment (?:saying|noting) (?:this|it) (?:was|came)\b", r"\bheader comment\b"),
    "reconcile": _rx(r"\bon the other hand\b", r"\bbut (?:the|my) (?:docs?|profiler|output|notes|data|serial|baseline)\b",
                     r"\bconflicts? with\b", r"\bcontradicts\b", r"\bwhereas\b", r"\bboth (?:say|claim|suggest)\b",
                     r"\breconcile\b", r"\bdisagrees? with\b", r"\bcombin(?:e|ing) (?:that|this|your|both)\b",
                     r"\bmy (?:teammate|group|partner|lab partner)\b",
                     r"\bthe profiler (?:says|shows|reports)\b", r"\bnvprof (?:says|shows)\b",
                     r"\bdoesn'?t match what (?:i|the)\b"),
})

PROFILES = {"general": GENERAL, "hpc": HPC}
DEFAULT_PROFILE = "general"

_active = DEFAULT_PROFILE


def use(profile: str) -> dict:
    """Select the active lexicon profile and return it."""
    global _active
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; have {sorted(PROFILES)}")
    _active = profile
    return PROFILES[profile]


def active() -> str:
    return _active


def markers() -> dict:
    return PROFILES[_active]


def coverage(profile: str = None) -> dict[str, int]:
    """Pattern count per marker family, for documenting a profile."""
    prof = PROFILES[profile or _active]
    return {k: len(v.pattern.split("|")) for k, v in sorted(prof.items())}
