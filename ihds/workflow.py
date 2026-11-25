"""Building blocks used by the LangGraph agents in ihds/agents.py.

Adapted from agentic-rag-system: the input guard, TF-IDF + cosine retrieval over a CSV knowledge
base, the token-overlap grounding score, and the reviewer checks. The orchestration itself (which
agent runs when, the revision loop) lives in ihds/agents.py.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import math

from .config import BORDERLINE, FEATURES, HIGH_MIN, KB_PATH, LOW_MAX, VALID_RANGES, ZERO_AS_MISSING

# Optional measurements: may be missing (imputed with the training median, and flagged).
OPTIONAL_FEATURES = ["BloodPressure", "SkinThickness", "Insulin"]

DISCLAIMER = "Advisory draft only - not a diagnosis. Clinician review and sign-off are required."

# ---------------------------------------------------------------- input guard
INJECTION_PATTERNS = [
    r"ignore (all|any|the|previous|prior) .*instructions",
    r"system prompt",
    r"you are now",
    r"jailbreak",
    r"disregard .*(rules|instructions)",
]
OUT_OF_SCOPE_PATTERNS = [
    r"\bdos(e|age|ing)\b",
    r"\bprescri(be|ption)\b",
    r"\bhow much insulin\b",
    r"\bmetformin\b",
    r"\bstop (taking|my) medication\b",
    r"\bdiagnose me\b",
]


@dataclass
class GuardResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    imputed: list[str] = field(default_factory=list)


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def normalize_record(record: dict) -> dict:
    """Raw-dataset convention: 0 in a ZERO_AS_MISSING column means 'not measured'."""
    out = dict(record)
    for f in ZERO_AS_MISSING:
        if f in out and not _is_missing(out[f]):
            try:
                if float(out[f]) == 0:
                    out[f] = None
            except (TypeError, ValueError):
                pass
    return out


def input_guard(record: dict, question: str = "") -> GuardResult:
    reasons, imputed = [], []
    for f in FEATURES:
        if f not in record or _is_missing(record[f]):
            if f in OPTIONAL_FEATURES:
                imputed.append(f)
            else:
                reasons.append(f"missing required value: {f}")
            continue
        try:
            v = float(record[f])
        except (TypeError, ValueError):
            reasons.append(f"non-numeric value: {f}")
            continue
        lo, hi = VALID_RANGES[f]
        if not lo <= v <= hi:
            reasons.append(f"{f}={v:g} outside plausible range [{lo}, {hi}]")
    q = (question or "").lower()
    if any(re.search(p, q) for p in INJECTION_PATTERNS):
        reasons.append("question looks like a prompt-injection attempt")
    if any(re.search(p, q) for p in OUT_OF_SCOPE_PATTERNS):
        reasons.append("question asks for treatment/medication or a diagnosis, which is out of scope")
    if len(q) > 500:
        reasons.append("question longer than 500 characters")
    return GuardResult(allowed=not reasons, reasons=reasons, imputed=imputed)


# ---------------------------------------------------------------- retrieval tool
class KnowledgeBase:
    """Evidence retrieval over data/clinical_knowledge.csv.

    IHDS_RETRIEVAL: "auto" (default: ChromaDB embeddings, keyword fallback), "chroma" (same, fallback still
    applies but is reported), or "keyword" (TF-IDF only, the original matcher). `last_method` records what
    actually served the latest query ("chroma" or "keyword"); see ihds/vector_store.py.
    """

    def __init__(self, path=KB_PATH, method: str | None = None):
        self.df = pd.read_csv(path)
        self.corpus = (self.df["topic"] + " " + self.df["fact"]).tolist()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.corpus)
        self.method = (method or os.environ.get("IHDS_RETRIEVAL", "auto")).lower()
        self._chroma = None
        self.chroma_error: str | None = None
        self.last_method = "keyword"

    def keyword_search(self, query: str, k: int = 3) -> list[dict]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).flatten()
        idx = scores.argsort()[::-1][:k]
        return [
            {
                "id": self.df.iloc[i]["id"],
                "fact": self.df.iloc[i]["fact"],
                "source": self.df.iloc[i]["source"],
                "score": round(float(scores[i]), 3),
            }
            for i in idx
            if scores[i] > 0
        ]

    def _chroma_index(self):
        if self._chroma is None and self.chroma_error is None:
            try:
                from .vector_store import ChromaIndex

                self._chroma = ChromaIndex(self.df)
            except Exception as exc:  # chromadb missing, model download failed, ...
                self.chroma_error = f"{type(exc).__name__}: {exc}"[:200]
        return self._chroma

    @property
    def embedding(self) -> str | None:
        return self._chroma.embedding if self._chroma is not None else None

    def retrieve(self, query: str, k: int = 3) -> tuple[list[dict], str]:
        """Returns (evidence, method) with method "chroma" or "keyword"."""
        if self.method != "keyword":
            idx = self._chroma_index()
            if idx is not None:
                try:
                    self.last_method = "chroma"
                    return idx.search(query, k), "chroma"
                except Exception as exc:
                    self.chroma_error = f"{type(exc).__name__}: {exc}"[:200]
                    self._chroma = None
        self.last_method = "keyword"
        return self.keyword_search(query, k), "keyword"

    def search(self, query: str, k: int = 3) -> list[dict]:
        return self.retrieve(query, k)[0]


# ---------------------------------------------------------------- helpers
QUERY_TERMS = {
    "Glucose": "glucose OGTT prediabetes impaired tolerance diabetes threshold",
    "BMI": "BMI overweight obesity risk factor",
    "Age": "age screening risk",
    "DiabetesPedigreeFunction": "family history pedigree genetic risk",
    "Pregnancies": "pregnancies gestational diabetes history",
    "BloodPressure": "blood pressure hypertension",
    "Insulin": "insulin resistance serum insulin",
    "SkinThickness": "BMI obesity risk factor",
}


def risk_band(p: float) -> str:
    if p < LOW_MAX:
        return "low"
    if p >= HIGH_MIN:
        return "high"
    return "moderate"


def build_query(top: list, question: str = "") -> str:
    """Query = topic terms of the two largest drivers (by |log-odds contribution|) + optional question."""
    terms = [QUERY_TERMS[c.feature] for c in top[:2]]
    return " ".join(terms + ([question] if question else []))


def _tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "with", "this", "that", "on", "be", "by", "as", "it", "not", "at"}
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in stop}


def grounding_score(answer: str, contexts: list[str]) -> float:
    """Share of answer content tokens found in the given sources (from agentic-rag-system)."""
    a = _tokens(answer)
    if not a:
        return 0.0
    c = set().union(*[_tokens(x) for x in contexts]) if contexts else set()
    return round(len(a & c) / len(a), 3)


# ---------------------------------------------------------------- drafting
def template_draft(prob: float, band: str, top: list, evidence: list[dict], imputed: list | None = None) -> str:
    drivers = ", ".join(
        f"{c.feature} = {c.value:g} ({'raises' if c.contribution > 0 else 'lowers'} risk)" for c in top[:3]
    )
    lines = [
        f"Estimated diabetes risk probability: {prob:.0%} ({band} band).",
        f"Main model drivers: {drivers}.",
    ]
    if imputed:
        lines.append(f"Not measured (imputed with training median): {', '.join(imputed)}.")
    for e in evidence:
        lines.append(f"[{e['id']}] {e['fact']}")
    if BORDERLINE[0] <= prob < BORDERLINE[1]:
        lines.append("This estimate is borderline; confirmatory testing may be more informative than the score.")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# ---------------------------------------------------------------- reviewer
FORBIDDEN = [r"\byou have diabetes\b", r"\bdiagnosed with\b", r"\bdiagnosis is\b", r"\bprescribe\b", r"\bmg of\b", r"\bunits of insulin\b"]


# Calibrated on 2026-11-25 for the sources below (docs/MLOPS.md): good gpt-4o-mini drafts of the default case
# score about 0.55-0.65, drafts with invented clinical claims 0.14-0.32. The earlier 0.2 belonged to the
# evidence-only score, where the same good drafts scored 0.18-0.22.
MIN_GROUNDING = 0.4


def unsupported_tokens(answer: str, contexts: list[str]) -> list[str]:
    """Answer content tokens that appear in none of the sources (what pulls the grounding score down)."""
    c = set().union(*[_tokens(x) for x in contexts]) if contexts else set()
    return sorted(_tokens(answer) - c)


def grounding_sources(evidence: list[dict], case_facts: list[str] | None = None) -> list[str]:
    """What the writer is given to write from: the retrieved evidence cards and the case data lines (model
    probability, drivers, population context, robustness check, imputed fields). Before 2026-11-25 only the
    evidence cards counted, so a draft that restated the case data it was told to state scored about 0.18-0.22
    against the old 0.2 threshold; the threshold is now MIN_GROUNDING = 0.4 (see docs/MLOPS.md)."""
    return [e["fact"] for e in evidence] + list(case_facts or [])


def _without_disclaimer(draft: str) -> str:
    """The mandatory disclaimer is checked on its own (has_disclaimer), so it does not count for grounding."""
    return re.sub(re.escape(DISCLAIMER), " ", draft, flags=re.IGNORECASE)


def review(draft: str, prob: float, evidence: list[dict], min_grounding: float = MIN_GROUNDING,
           case_facts: list[str] | None = None) -> dict:
    ids = {e["id"] for e in evidence}
    cited = set(re.findall(r"\[(K\d+)\]", draft))
    checks = {
        "has_disclaimer": DISCLAIMER.lower() in draft.lower(),
        "cites_retrieved_evidence": bool(cited) and cited <= ids,
        "states_model_probability": f"{prob:.0%}" in draft,
        "no_diagnostic_or_dosing_language": not any(re.search(p, draft.lower()) for p in FORBIDDEN),
    }
    sources = grounding_sources(evidence, case_facts)
    body = _without_disclaimer(draft)
    g = grounding_score(body, sources)
    checks["grounding_ok"] = g >= min_grounding
    unsupported = unsupported_tokens(body, sources)
    fixes = {
        "has_disclaimer": f"end with this sentence verbatim: {DISCLAIMER}",
        "cites_retrieved_evidence": "cite at least one evidence card, and only these: "
                                    + ", ".join(f"[{i}]" for i in sorted(ids)),
        "states_model_probability": f"state the model probability exactly as {prob:.0%}",
        "no_diagnostic_or_dosing_language": "remove diagnostic or dosing language",
        "grounding_ok": f"grounding {g} is below {min_grounding}; these words are not in the case data or the "
                        f"evidence: {', '.join(unsupported[:20])}. Keep to their wording and drop other claims",
    }
    return {"approved": all(checks.values()), "checks": checks, "grounding_score": g,
            "unsupported_terms": unsupported[:20],
            "feedback": [fixes[k] for k, ok in checks.items() if not ok]}


def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))
