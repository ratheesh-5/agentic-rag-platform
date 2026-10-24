"""Workflow A - agentic RAG, built with LangGraph.

The fast pipeline always does: retrieve top-k -> generate. It fails when the question needs
several facts, when the wording doesn't match the documents, or when the model's answer drifts
from the passages. The agentic pipeline adds LLM decisions at those weak points:

    START -> plan -> retrieve -> grade --(relevant found)--------> generate -> verify -> END
                        ^          |                                  ^          |
                        |          +--(none, rewrites < max)-> rewrite           |
                        +---------------------------------------+    |          |
                                   +--(none, no rewrites left)--> abstain <------+
                                                             regenerate once ----+

- plan     (LLM)  Query Planner: splits the question into 1-3 search queries (e.g. a question
                  about two different documents gets one query per fact). The original
                  question is always searched as well.
- retrieve (tool) Runs every query through the normal dense retriever and merges the hits.
- grade    (LLM)  Relevance Grader: keeps only passages that help answer the question. Removes
                  noise that would otherwise distract the generator.
- rewrite  (LLM)  If nothing relevant was found, rephrase the search (max `max_rewrites`).
- generate (LLM)  The same cited-answer generator as fast mode, on the graded passages only -
                  without the similarity threshold: the grader already confirmed relevance.
                  If every citation was invalid, it retries once with feedback.
- verify   (LLM)  Answer Verifier: accept / regenerate once with feedback / abstain.
- abstain         Ends with an explicit "I don't know".

LangGraph basics (for a .NET developer): a graph is a state machine. The STATE is a typed
dictionary passed between NODES; each node is a plain function that receives the state and
returns only the fields it changed (LangGraph merges them in). EDGES decide which node runs
next; conditional edges call a small routing function. Think of it as a workflow engine
(like Durable Functions / Workflow Foundation) where some steps happen to be LLM calls.

Loop limits (rewrites, regenerations) are enforced in code by the routing functions, never left
to the model - an agent must always be guaranteed to stop.
"""

import operator
from collections.abc import Callable
from dataclasses import replace
from typing import Annotated, Literal, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag_platform.config import RagConfig
from rag_platform.generation.answer import (
    NO_VALID_CITATIONS,
    UNTRUSTED_PASSAGES,
    Answer,
    answer_question,
    valid_labels,
)
from rag_platform.retrieval.bm25 import BM25Index
from rag_platform.retrieval.retriever import RetrievedChunk, build_retriever
from rag_platform.store.db import Store
from rag_platform.store.vector_index import VectorIndex

# --- Prompts and structured outputs ------------------------------------------------------------

PLANNER_PROMPT = """You plan searches over an internal document collection.
Write 1 to 3 short search queries that together find everything needed to answer the question.
Use a single query unless the question asks about several distinct facts or entities.
Keep the important terms (names, codes, numbers) from the question."""

GRADER_PROMPT = (
    """You check which search results are useful for answering a question.
Return the numbers of the passages that contain information needed to answer it, even
partially. Return an empty list if none of them do."""
    + UNTRUSTED_PASSAGES
)

REWRITE_PROMPT = """A document search returned nothing relevant for the question below.
Write ONE new search query that uses different wording: synonyms, the likely terms a policy or
technical document would use, or a more specific phrase. Do not repeat a previous query."""

VERIFIER_PROMPT = (
    """You review an answer written from numbered passages.
- accept: every claim is supported by the cited passages and the question is answered as far
  as the passages allow (a partial answer that says what is not covered is fine).
- regenerate: the answer has an unsupported claim, a wrong citation, or leaves out information
  the passages do contain. Explain what to fix in feedback.
- abstain: the passages contain none of the facts the question asks for."""
    + UNTRUSTED_PASSAGES
)


class QueryPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3, description="Search queries")


class RelevanceGrade(BaseModel):
    relevant: list[int] = Field(description="Numbers of the useful passages")


class RewrittenQuery(BaseModel):
    query: str


class Verdict(BaseModel):
    verdict: Literal["accept", "regenerate", "abstain"]
    feedback: str = Field(default="", description="What to fix (for regenerate)")


def _passages(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))


# --- Graph state ---------------------------------------------------------------------------------


class RagState(TypedDict, total=False):
    """Data flowing through the graph. total=False: every key is optional (filled step by step).

    `Annotated[list[str], operator.add]` tells LangGraph to APPEND a node's returned `steps` to
    the existing list instead of replacing it (a "reducer"), so the trace accumulates.
    """

    question: str
    queries: list[str]  # current search queries (from the planner or the rewriter)
    candidates: list[RetrievedChunk]  # everything retrieved for the current queries
    context: list[RetrievedChunk]  # candidates the grader judged relevant
    rewrites: int
    regenerations: int
    feedback: str  # verifier feedback for the next generation
    next: str  # routing decision: "retry" (generate), "regenerate" (verify) or "done"
    answer: Answer
    steps: Annotated[list[str], operator.add]


# --- Graph construction --------------------------------------------------------------------------


def build_rag_graph(
    store: Store,
    index: VectorIndex,
    embedder: Embeddings,
    llm: BaseChatModel,
    cfg: RagConfig,
    bm25: BM25Index | None = None,  # prebuilt keyword index (API cache)
):
    """Compile the agentic RAG graph. Dependencies are captured by the node functions
    (closures), so the graph itself only carries the per-question state."""

    # Each node below is a nested function: it can use store/index/llm/cfg from this outer
    # function (a "closure"), like a C# lambda capturing variables.
    # Each node gets the current state and returns ONLY the fields it changed.

    retriever = build_retriever(store, index, embedder, cfg, bm25)  # dense or hybrid, per config

    def ask(system: str, human: str, schema: type[BaseModel]):
        """One LLM call that returns an instance of `schema` (structured output)."""
        return llm.with_structured_output(schema).invoke([("system", system), ("human", human)])

    def plan(state: RagState) -> dict:
        out: QueryPlan = ask(PLANNER_PROMPT, f"Question: {state['question']}", QueryPlan)
        # The user's own question is always searched too: the planner's keyword-style queries can
        # match worse (seen in testing: 0.655 for the planned query vs 0.690 for the question).
        # dict.fromkeys removes duplicates and keeps the order (like LINQ Distinct()).
        queries = list(dict.fromkeys([state["question"], *out.queries]))
        return {"queries": queries, "steps": [f"plan: {out.queries}"]}

    def retrieve_node(state: RagState) -> dict:
        # Several queries can find the same chunk: keep each chunk once, with its best score.
        best: dict[int, RetrievedChunk] = {}  # chunk_id -> hit with its best score
        for q in state["queries"]:
            for c in retriever(q):
                if c.chunk_id not in best or c.score > best[c.chunk_id].score:
                    best[c.chunk_id] = c
        # Highest score first. `lambda c: c.score` = inline function, like C# `c => c.Score`.
        hits = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return {"candidates": hits, "steps": [f"retrieve: {len(hits)} passages"]}

    def grade(state: RagState) -> dict:
        cands = state["candidates"]
        if not cands:
            return {"context": [], "steps": ["grade: nothing to grade"]}
        human = f"Question: {state['question']}\n\nPassages:\n\n{_passages(cands)}"
        out: RelevanceGrade = ask(GRADER_PROMPT, human, RelevanceGrade)
        # The LLM returns passage numbers (1-based). {...} builds a set (removes duplicates);
        # numbers outside 1..len(cands) are ignored - never trust the model's output blindly.
        keep = sorted({n for n in out.relevant if 1 <= n <= len(cands)})
        # n - 1: Python lists are 0-based. Candidates are sorted best-first, so keeping the first
        # top_k relevant ones caps the generator's context at top_k, as in fast mode (several
        # planned queries can return up to 3 x top_k candidates).
        context = [cands[n - 1] for n in keep][: cfg.top_k]
        return {"context": context, "steps": [f"grade: {len(context)}/{len(cands)} relevant"]}

    def rewrite(state: RagState) -> dict:
        human = f"Question: {state['question']}\nPrevious queries: {state['queries']}"
        out: RewrittenQuery = ask(REWRITE_PROMPT, human, RewrittenQuery)
        return {
            "queries": [out.query],
            # state.get("rewrites", 0) = value if present, else 0 (like TryGetValue + default)
            "rewrites": state.get("rewrites", 0) + 1,
            "steps": [f"rewrite: {out.query!r}"],
        }

    def generate(state: RagState) -> dict:
        # No similarity threshold here (0.0): the grader has already judged these passages
        # relevant, and the verifier checks the answer. The score threshold is a cheap filter
        # for fast mode; scores depend on the document collection (a table of names scored
        # 0.60 while holding the answer), so it would refuse passages the grader confirmed.
        ans = answer_question(
            state["question"],
            state["context"],
            llm,
            0.0,
            feedback=state.get("feedback", ""),
        )
        status = "answered" if ans.answerable else f"abstained ({ans.abstain_reason})"
        update: dict = {"answer": ans, "steps": [f"generate: {status}"], "next": "done"}
        # The model answered but cited labels that don't exist (e.g. a heading number): one
        # retry with feedback instead of refusing. Shares the regeneration budget with verify.
        if (
            ans.abstain_reason == NO_VALID_CITATIONS
            and state.get("regenerations", 0) < cfg.max_regenerations
        ):
            update["next"] = "retry"
            update["feedback"] = (
                "It cited passage labels that do not exist. Cite only "
                f"{valid_labels(len(state['context']))}; numbers in headings or section titles "
                "are not passage labels."
            )
            update["regenerations"] = state.get("regenerations", 0) + 1
        return update

    def verify(state: RagState) -> dict:
        ans = state["answer"]
        human = (
            f"Question: {state['question']}\n\nPassages:\n\n{_passages(ans.context)}"
            f"\n\nAnswer:\n{ans.text}"
        )
        out: Verdict = ask(VERIFIER_PROMPT, human, Verdict)
        update: dict = {"steps": [f"verify: {out.verdict}"], "next": "done"}
        if out.verdict == "regenerate" and state.get("regenerations", 0) < cfg.max_regenerations:
            update["next"] = "regenerate"
            update["feedback"] = out.feedback or "Answer only what the passages support."
            update["regenerations"] = state.get("regenerations", 0) + 1
        elif out.verdict != "accept":
            # "abstain", or a second "regenerate" after the budget is used up: refuse.
            # replace() copies a frozen dataclass with some fields changed (C# `record with {}`).
            update["answer"] = replace(
                ans,
                answerable=False,
                text="",
                citations=[],
                abstain_reason=f"verifier: {out.verdict}",
            )
        return update

    def abstain(state: RagState) -> dict:
        ans = Answer(
            state["question"],
            False,
            "",
            context=state.get("context", []),
            abstain_reason="no relevant passage found",
        )
        return {"answer": ans, "steps": ["abstain"]}

    # --- Routing functions: plain code decides the path, and enforces the loop limits. ---------

    def after_grade(state: RagState) -> str:
        if state["context"]:
            return "generate"
        return "rewrite" if state.get("rewrites", 0) < cfg.max_rewrites else "abstain"

    def after_generate(state: RagState) -> str:
        if state["next"] == "retry":
            return "generate"
        return "verify" if state["answer"].answerable else END

    def after_verify(state: RagState) -> str:
        return "generate" if state["next"] == "regenerate" else END

    g = StateGraph(RagState)
    for name, fn in [
        ("plan", plan),
        ("retrieve", retrieve_node),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("verify", verify),
        ("abstain", abstain),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "abstain"])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("generate", after_generate, ["generate", "verify", END])
    g.add_conditional_edges("verify", after_verify, ["generate", END])
    g.add_edge("abstain", END)
    return g.compile()


def agentic_answerer(
    store: Store,
    index: VectorIndex,
    embedder: Embeddings,
    llm: BaseChatModel,
    cfg: RagConfig,
    bm25: BM25Index | None = None,
) -> Callable[[str], Answer]:
    """Return a function question -> Answer that runs the agentic graph."""
    graph = build_rag_graph(store, index, embedder, llm, cfg, bm25)

    def run(question: str) -> Answer:
        final = graph.invoke({"question": question, "steps": []})
        return replace(final["answer"], steps=final["steps"])

    return run
