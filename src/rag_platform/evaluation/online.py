"""Online evaluation: quality measured from REAL chat usage, with no test data.

Two parts:
1. Scoring (needs the judge LLM): after an answer is shown, a sample of answers (default 1 in 5,
   ONLINE_EVAL_SAMPLE_RATE) is scored in the background with the two RAGAS metrics that need no
   reference answer: faithfulness (is every claim supported by the passages?) and answer
   relevancy (does it address the question?). About 3-4 judge calls per scored answer; the user
   never waits for it. Errors (e.g. quota) are stored, never raised.
2. Usage summary (plain code, free): unanswered questions, thumbs-down answers, low-scored
   answers and per-document numbers - the "suggestions" an admin acts on (add a missing
   document, fix a weak one).

The offline golden set (evaluation/golden.py) stays the release check for configuration
changes; this module watches what users actually ask. Like application monitoring next to
unit tests in .NET: both are needed, for different questions.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

LOW_SCORE = 0.7  # below this, an answer is listed for review
MAX_ITEMS = 20  # per suggestion list
MAX_RECENT = 200  # answers in the per-question list


def should_score(rate: float, rng: Callable[[], float] = random.random) -> bool:
    """Sample decision: True for about `rate` of the answers (0 = never, 1 = always)."""
    return rate > 0 and rng() < rate


def judge(svc, question: str, answer: str, contexts: list[str]) -> dict:
    """Score one answer now: {faithfulness, answer_relevancy, error, judge_calls}."""
    evaluator = svc.answer_evaluator()
    llm = getattr(evaluator, "llm", None)
    calls_before = getattr(llm, "calls", 0)
    try:
        scores = evaluator.score(question, answer, contexts, reference="")
        result = {
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 - a judge failure must not fail the caller
        result = {"faithfulness": None, "answer_relevancy": None, "error": str(e)[:300]}
    return {**result, "judge_calls": getattr(llm, "calls", 0) - calls_before}


def score_answer(svc, message_id: int, question: str, answer: str, contexts: list[str]) -> None:
    """Background task: score one chat answer and store the result (or the error)."""
    result = judge(svc, question, answer, contexts)
    store = svc.open_store()
    try:
        store.save_answer_score(message_id, **result)
    finally:
        store.close()


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _docs(details: dict) -> list[str]:
    return sorted({Path(s["document"]).name for s in details.get("sources", [])})


def usage_summary(rows: list[dict], sample_rate: float, include_admin: bool = False) -> dict:
    """Aggregate stored chat answers (Store.usage_rows) into totals and suggestion lists.
    Chats asked while signed in as admin (team testing) are left out unless include_admin."""
    admin_chats = sum(1 for r in rows if r["details"].get("by_admin"))
    if not include_admin:
        rows = [r for r in rows if not r["details"].get("by_admin")]
    answered = [r for r in rows if r["details"].get("answerable")]
    unanswered = [r for r in rows if not r["details"].get("answerable")]
    scored = [r for r in answered if r["scored_at"] and not r["score_error"]]
    faith = [r["faithfulness"] for r in scored if r["faithfulness"] is not None]
    relev = [r["answer_relevancy"] for r in scored if r["answer_relevancy"] is not None]

    # Unanswered questions, grouped (the same question asked twice counts twice), newest first.
    groups: dict[str, dict] = {}
    counts: Counter[str] = Counter()
    for r in sorted(unanswered, key=lambda r: r["id"], reverse=True):
        key = " ".join(r["question"].lower().split()).rstrip("?")
        counts[key] += 1
        groups.setdefault(
            key,
            {
                "question": r["question"],
                "understood_as": r["details"].get("standalone_question"),
                "reason": r["details"].get("abstain_reason", ""),
                "last_asked": r["created_at"],
            },
        )
    unanswered_list = [{**g, "count": counts[k]} for k, g in groups.items()][:MAX_ITEMS]

    def item(r: dict) -> dict:
        return {
            "message_id": r["id"],
            "question": r["question"],
            "answer": r["content"][:300],
            "documents": _docs(r["details"]),
            "faithfulness": r["faithfulness"],
            "answer_relevancy": r["answer_relevancy"],
            "created_at": r["created_at"],
        }

    newest = sorted(answered, key=lambda r: r["id"], reverse=True)
    # Thumbs-down on ANY answer, refusals included (a disliked refusal = a missing document).
    everything = sorted(rows, key=lambda r: r["id"], reverse=True)
    thumbs_down = [item(r) for r in everything if r["feedback"] == -1][:MAX_ITEMS]
    low = [
        item(r)
        for r in newest
        if any(v is not None and v < LOW_SCORE for v in (r["faithfulness"], r["answer_relevancy"]))
    ][:MAX_ITEMS]

    # Per document: how often it was cited, its answers' scores and thumbs-down.
    per_doc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in answered:
        for doc in _docs(r["details"]):
            d = per_doc[doc]
            d["answers"].append(1)
            d["down"].append(1 if r["feedback"] == -1 else 0)
            if r["faithfulness"] is not None:
                d["faith"].append(r["faithfulness"])
            if r["answer_relevancy"] is not None:
                d["relev"].append(r["answer_relevancy"])
    documents = sorted(
        (
            {
                "document": doc,
                "answers": len(d["answers"]),
                "thumbs_down": sum(d["down"]),
                "faithfulness": _avg(d["faith"]),
                "answer_relevancy": _avg(d["relev"]),
            }
            for doc, d in per_doc.items()
        ),
        key=lambda d: -d["answers"],
    )

    # Every answer, newest first: the per-question view (the UI filters and sorts it).
    recent = [
        {
            **item(r),
            "answerable": bool(r["details"].get("answerable")),
            "abstain_reason": r["details"].get("abstain_reason", ""),
            "feedback": r["feedback"],
            "by_admin": bool(r["details"].get("by_admin")),
        }
        for r in sorted(rows, key=lambda r: r["id"], reverse=True)[:MAX_RECENT]
    ]

    return {
        "sample_rate": sample_rate,
        "include_admin": include_admin,
        "admin_chats": admin_chats,
        "recent": recent,
        "totals": {
            "questions": len(rows),
            "answered": len(answered),
            "unanswered": len(unanswered),
            # Auto mode: questions the fast answer handed to the agent (details.route).
            "escalated": sum(
                1 for r in rows if r["details"].get("route", "").startswith("auto → agentic")
            ),
            "thumbs_up": sum(1 for r in rows if r["feedback"] == 1),
            "thumbs_down": sum(1 for r in rows if r["feedback"] == -1),
            "scored": len(scored),
            "faithfulness": _avg(faith),
            "answer_relevancy": _avg(relev),
        },
        "unanswered": unanswered_list,
        "thumbs_down": thumbs_down,
        "low_scores": low,
        "documents": documents,
    }
