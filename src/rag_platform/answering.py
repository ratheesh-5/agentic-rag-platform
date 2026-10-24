"""Choose the question-answering pipeline: fast, agentic or auto.

All pipelines are exposed as the same shape - a function `question -> Answer` - so callers
(`rag ask`, `rag eval`, the API and UI) don't care which one runs. .NET analogy: implementations
of one `IAnswerer` interface, picked by configuration (a strategy pattern).

- fast:    retrieve top-k (dense or hybrid, see retriever.py) -> generate. 1 LLM call.
- agentic: LangGraph workflow (plan, grade, rewrite, verify). ~4+ LLM calls. See
           agents/rag_graph/graph.py.
- auto:    fast first; agentic only when the fast answer struggled (an "escalation router").
           Simple questions keep the cost of fast mode, hard ones get the agent:

               question -> fast -> fine?                          -> done (1 call)
                                -> not answerable / no valid citation / only part answered
                                                                  -> agentic (~4 more calls)
                                -> best score below the threshold -> refuse, no escalation

           A threshold refusal is NOT escalated (owner decision, 2026-10-22): measured scores of
           off-topic and of unanswerable-but-related questions overlap with those of valid
           questions worded differently (0.60-0.68), and agentic mode would spend ~6 calls
           before refusing an off-topic question. A threshold that refuses valid questions is a
           calibration problem (quality check: safe threshold), not a routing one.
"""

from collections.abc import Callable
from dataclasses import replace

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag_platform.config import RagConfig
from rag_platform.generation.answer import BELOW_THRESHOLD, Answer, answer_question
from rag_platform.retrieval.bm25 import BM25Index
from rag_platform.retrieval.retriever import build_retriever
from rag_platform.store.db import Store
from rag_platform.store.vector_index import VectorIndex


def escalation_reason(ans: Answer) -> str:
    """Why auto mode should hand a fast answer to the agent ("" = keep the fast answer)."""
    if ans.answerable:
        return "" if ans.complete else "incomplete"  # answered only part of the question
    if ans.abstain_reason == BELOW_THRESHOLD:
        return ""  # nothing close enough: refuse cheaply (see the module docstring)
    return ans.abstain_reason  # the model said "not answerable", or no valid citation


def build_answerer(
    store: Store,
    index: VectorIndex,
    embedder: Embeddings,
    llm: BaseChatModel,
    cfg: RagConfig,
    bm25: BM25Index | None = None,  # prebuilt keyword index (API cache); built if None
) -> Callable[[str], Answer]:
    if cfg.mode != "fast":
        from rag_platform.agents.rag_graph.graph import agentic_answerer  # loads LangGraph

        agentic = agentic_answerer(store, index, embedder, llm, cfg, bm25)
        if cfg.mode == "agentic":
            return agentic

    retriever = build_retriever(store, index, embedder, cfg, bm25)  # dense or hybrid

    def fast(question: str) -> Answer:
        return answer_question(question, retriever(question), llm, cfg.similarity_threshold)

    if cfg.mode == "fast":
        return fast

    def auto(question: str) -> Answer:
        first = fast(question)
        reason = escalation_reason(first)
        if not reason:
            return replace(first, route="fast")
        route = f"auto → agentic ({reason})"
        second = agentic(question)
        if second.answerable or not first.answerable:
            return replace(second, route=route)
        # The agent refused but fast had a partial answer: never lose a good partial answer.
        return replace(first, steps=second.steps, route=route + ", kept the fast answer")

    return auto
