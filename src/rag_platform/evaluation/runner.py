"""Run the golden set through a RAG pipeline and score every question.

For each question: run the answerer (fast or agentic - see answering.py) -> compute retrieval
metrics against the evidence spans -> optionally score the answer with RAGAS.
The caller stores the results and summary as an eval run, together with the config used.

The runner receives the real answerer that `rag ask` uses (not a copy), so the eval measures the
same code path users hit and a metric change always reflects a real behaviour change.

"Retrieved" = the passages the answer was generated from (Answer.context): all top-k chunks in
fast mode; only the passages the relevance grader kept in agentic mode.
"""

import time
from collections.abc import Callable

from rag_platform.evaluation.golden import GoldenItem
from rag_platform.evaluation.metrics import (
    QuestionResult,
    calibrate_threshold,
    overlaps,
    retrieval_scores,
    summarize,
)
from rag_platform.evaluation.ragas_eval import RagasEvaluator
from rag_platform.generation.answer import Answer
from rag_platform.retrieval.retriever import RetrievedChunk


def retrieval_only_answerer(
    retriever: Callable[[str], list[RetrievedChunk]], threshold: float
) -> Callable[[str], Answer]:
    """An "answerer" WITHOUT an LLM, for cheap experiments (`rag eval run --retrieval-only`,
    the optimizer). It retrieves, and "answers" iff the best similarity reaches the threshold.

    Retrieval metrics are exact; abstention metrics then measure the similarity gate alone.
    """

    def run(question: str) -> Answer:
        context = retriever(question)
        ok = bool(context) and max(c.score for c in context) >= threshold
        reason = "" if ok else "below similarity threshold"
        return Answer(question, ok, "", context=context, abstain_reason=reason)

    return run


def evaluate_question(
    item: GoldenItem,
    answerer: Callable[[str], Answer],
    k: int,
    ragas: RagasEvaluator | None,
) -> QuestionResult:
    started = time.perf_counter()  # high-resolution timer, like Stopwatch in .NET
    ans = answerer(item.question)
    latency = time.perf_counter() - started
    context = ans.context
    result = QuestionResult(
        id=item.id,
        question=item.question,
        answerable=item.answerable,
        predicted_answerable=ans.answerable,
        answer=ans.text,
        abstain_reason=ans.abstain_reason,
        top_score=max(c.score for c in context) if context else None,
        retrieved=[
            {
                "doc": c.doc_path,
                "start": c.start,
                "end": c.end,
                "score": round(c.score, 4),
                "relevant": any(overlaps(c.doc_path, c.start, c.end, s) for s in item.evidence),
            }
            for c in context
        ],
        latency_s=round(latency, 2),
        steps=ans.steps,
    )
    if item.answerable:
        spans = [(c.doc_path, c.start, c.end) for c in context]
        result.retrieval = retrieval_scores(spans, item.evidence, k)
    # RAGAS grades an answer, so it only runs when the system actually answered.
    if ragas is not None and ans.answerable:
        result.ragas = ragas.score(
            item.question, ans.text, [c.text for c in context], item.reference_answer
        )
    return result


def run_eval(
    golden: list[GoldenItem],
    answerer: Callable[[str], Answer],
    k: int,
    ragas: RagasEvaluator | None = None,  # None = retrieval/abstention metrics only (no LLM judge)
    # Callable[[QuestionResult], None] = a function taking a QuestionResult and returning nothing,
    # like Action<QuestionResult> in C#. The CLI passes one to print progress per question.
    on_result: Callable[[QuestionResult], None] | None = None,
) -> tuple[list[QuestionResult], str | None]:
    """Evaluate every question. Returns (results, error).

    If a question fails (typically an API quota error) the run STOPS and returns the results
    collected so far plus the error message, instead of losing them. The caller saves this
    partial run; `rag eval run --resume <id>` later evaluates only the missing questions.
    """
    results = []
    for item in golden:
        try:
            r = evaluate_question(item, answerer, k, ragas)
        except Exception as e:  # noqa: BLE001 - any failure: keep what we have and stop
            return results, f"{item.id}: {type(e).__name__}: {str(e)[:300]}"
        results.append(r)
        if on_result:
            on_result(r)
    return results, None


def run_metrics(
    results: list[QuestionResult], threshold: float, n_total: int, error: str | None
) -> dict:
    """Summary metrics stored with a run (shared by `rag eval run` and optimizer experiments)."""
    metrics = summarize(results)
    # Suggest a similarity threshold from this run's data (see metrics.calibrate_threshold).
    t, acc = calibrate_threshold(results, threshold)
    metrics["calibrated_threshold"] = t
    metrics["calibrated_abstention_accuracy"] = acc
    metrics["complete"] = len(results) == n_total
    metrics["error"] = error
    return metrics
