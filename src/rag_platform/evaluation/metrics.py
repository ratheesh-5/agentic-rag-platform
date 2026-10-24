"""Evaluation metrics - plain, deterministic Python (no LLM involved).

Retrieval metrics (computed on ANSWERABLE questions only, at K = top_k):
- Precision@K : share of the K retrieved chunks that are relevant.        "how much noise?"
- Recall@K    : share of the question's evidence spans found in top K.    "did we find it all?"
- Hit rate@K  : 1 if at least one relevant chunk is in the top K, else 0.  "did we find anything?"
- MRR         : 1 / rank of the first relevant chunk (0 if none), averaged. "was it near the top?"
  Example: first relevant chunk at position 1 -> 1.0, at position 2 -> 0.5, at 4 -> 0.25.

Abstention metrics (ALL questions):
- abstention_accuracy       : answered when answerable + abstained when unanswerable, / all.
- false_abstention_rate     : answerable questions the system refused (too cautious).
- unanswerable_answered_rate: unanswerable questions the system answered anyway (hallucination).
Both error rates are reported because "refuse everything" would never hallucinate.

Answer quality (RAGAS, averaged over the questions the system answered; see ragas_eval.py):
faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness (0..1).
"""

import math
from dataclasses import dataclass, field, fields
from statistics import mean

from rag_platform.evaluation.golden import Span

# Names of the RAGAS scores stored per question (computed in ragas_eval.py). Defined here so this
# module stays plain Python and does not import the RAGAS library.
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


@dataclass
class QuestionResult:
    """Everything recorded for one golden question in one eval run (stored as JSON)."""

    id: str
    question: str
    answerable: bool  # gold label
    predicted_answerable: bool  # what the system did
    answer: str
    abstain_reason: str
    top_score: float | None  # cosine similarity of the best retrieved chunk
    retrieved: list[dict] = field(default_factory=list)  # doc, start, end, score, relevant
    retrieval: dict | None = None  # precision, recall, hit, rr (answerable questions only)
    ragas: dict | None = None  # RAGAS metric -> 0..1 or None (answered questions only)
    latency_s: float | None = None  # time to answer (includes rate-limiter waits)
    steps: list[str] = field(default_factory=list)  # agentic mode: graph trace

    @classmethod
    def from_dict(cls, d: dict) -> "QuestionResult":
        """Rebuild from stored JSON, ignoring keys this version doesn't know (older runs)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})  # **dict = named arguments


def overlaps(doc: str, start: int, end: int, span: Span) -> bool:
    """True if [start, end) in `doc` shares at least one character with the evidence span."""
    return doc == span.doc and start < span.end and span.start < end


def retrieval_scores(retrieved: list[tuple[str, int, int]], evidence: list[Span], k: int) -> dict:
    """Precision/recall/hit/reciprocal-rank for one question. `retrieved` is (doc, start, end)."""
    top = retrieved[:k]
    # relevant[i]: is the i-th retrieved chunk relevant?  (one bool per chunk)
    relevant = [any(overlaps(d, s, e, span) for span in evidence) for d, s, e in top]
    # covered[j]: is the j-th evidence span found by any chunk?  (one bool per span)
    covered = [any(overlaps(d, s, e, span) for d, s, e in top) for span in evidence]
    # Rank (1-based) of the first relevant chunk, or None. next(generator, default) returns the
    # first item, like LINQ .Select(...).FirstOrDefault().
    first = next((rank for rank, rel in enumerate(relevant, start=1) if rel), None)
    return {
        # sum() of bools counts the True values. Divide by what was actually returned (agentic
        # mode can return fewer than k graded passages); 0 when nothing was returned.
        "precision": sum(relevant) / len(top) if top else 0.0,
        "recall": sum(covered) / len(evidence),
        "hit": 1.0 if first else 0.0,
        "rr": 1.0 / first if first else 0.0,
    }


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _p95(values: list[float]) -> float | None:
    """95th percentile (nearest-rank): 95% of questions were answered at least this fast."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 4)


def summarize(results: list[QuestionResult]) -> dict:
    """Aggregate per-question results into the run's metrics."""
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    retrieval = [r.retrieval for r in answerable if r.retrieval]
    judged = [r.ragas for r in results if r.ragas]
    # A RAGAS metric can be None for some questions (no reference, or not scorable): skip those.
    ragas = {
        name: _avg([j[name] for j in judged if j.get(name) is not None]) for name in RAGAS_METRICS
    }
    return {
        "n_questions": len(results),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "precision_at_k": _avg([m["precision"] for m in retrieval]),
        "recall_at_k": _avg([m["recall"] for m in retrieval]),
        "hit_rate_at_k": _avg([m["hit"] for m in retrieval]),
        "mrr": _avg([m["rr"] for m in retrieval]),
        "abstention_accuracy": _avg(
            [float(r.predicted_answerable == r.answerable) for r in results]
        ),
        "false_abstention_rate": _avg([float(not r.predicted_answerable) for r in answerable]),
        "unanswerable_answered_rate": _avg([float(r.predicted_answerable) for r in unanswerable]),
        "n_judged": len(judged),
        **ragas,  # unpack the dict into this one (like spreading properties)
        "latency_avg_s": _avg([r.latency_s for r in results if r.latency_s is not None]),
        "latency_p95_s": _p95([r.latency_s for r in results if r.latency_s is not None]),
        "top_score_answerable": _avg([r.top_score for r in answerable if r.top_score is not None]),
        "top_score_unanswerable": _avg(
            [r.top_score for r in unanswerable if r.top_score is not None]
        ),
        **safe_threshold(answerable, unanswerable),
    }


def safe_threshold(answerable: list[QuestionResult], unanswerable: list[QuestionResult]) -> dict:
    """The highest similarity threshold that would NOT have refused any answerable question
    (rounded down, minus a 0.01 margin), and the share of unanswerable questions it would catch
    before any LLM call. A "safe" gate: it never costs a correct answer on this golden set."""
    tops = [r.top_score for r in answerable if r.top_score is not None]
    if not tops:
        return {"safe_threshold": None, "unanswerable_below_safe": None}
    t = max(0.0, math.floor(min(tops) * 100) / 100 - 0.01)
    caught = [float((r.top_score or 0.0) < t) for r in unanswerable]
    return {"safe_threshold": round(t, 2), "unanswerable_below_safe": _avg(caught)}


def calibrate_threshold(
    results: list[QuestionResult], current: float
) -> tuple[float, float | None]:
    """Find the similarity threshold that maximizes abstention accuracy on this run.

    Simulation: with threshold t, a question is refused if its top score < t, otherwise the
    system behaves as it did in this run. Only t >= `current` can be simulated (questions already
    refused by the current threshold never reached the LLM). Ties go to the lowest threshold,
    i.e. the least aggressive one. Returns (threshold, abstention accuracy at that threshold);
    with no results (a run that stopped before its first question) there is nothing to calibrate.
    """
    if not results:
        return current, None
    scores = sorted(
        {r.top_score for r in results if r.top_score is not None and r.top_score >= current}
    )
    best_t, best_acc = current, -1.0
    for t in [current, *scores]:
        acc = mean(
            float(((r.top_score or 0.0) >= t and r.predicted_answerable) == r.answerable)
            for r in results
        )
        if acc > best_acc:
            best_t, best_acc = t, acc
    return round(best_t, 4), round(best_acc, 4)
