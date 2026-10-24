"""Tests for the agentic RAG graph (Workflow A) with a scripted fake LLM - no API calls.

Each test scripts what the "LLM" answers at each node (planner, grader, rewriter, generator,
verifier) and checks the path the graph takes, including that loop limits are enforced.
"""

from pathlib import Path

import pytest

from conftest import FakeLlm
from rag_platform.agents.rag_graph.graph import (
    QueryPlan,
    RelevanceGrade,
    RewrittenQuery,
    Verdict,
    agentic_answerer,
)
from rag_platform.answering import build_answerer
from rag_platform.config import RagConfig
from rag_platform.generation.answer import LlmAnswer
from rag_platform.ingestion.pipeline import ingest, open_index
from rag_platform.store.db import Store

CFG = RagConfig(mode="agentic", top_k=2, similarity_threshold=0.0)
GOOD = LlmAnswer(answerable=True, answer="Twenty days [1].")


@pytest.fixture
def deps(tmp_path: Path, fake_embedder):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "leave.md").write_text("Employees get twenty days annual leave.", encoding="utf-8")
    (corpus / "vpn.md").write_text("Connect to the VPN before using servers.", encoding="utf-8")
    store = Store(tmp_path / "rag.db")
    ingest(corpus, store, tmp_path / "f.index", fake_embedder, "fake", RagConfig())
    index = open_index(store, tmp_path / "f.index", "fake")
    yield store, index, fake_embedder
    store.close()


def _run(deps, replies: dict, question="how many days annual leave", cfg=CFG):
    store, index, embedder = deps
    llm = FakeLlm(replies)
    return agentic_answerer(store, index, embedder, llm, cfg)(question), llm


def test_happy_path_plan_grade_generate_accept(deps):
    ans, llm = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["annual leave days"]),
            RelevanceGrade: RelevanceGrade(relevant=[1]),
            LlmAnswer: GOOD,
            Verdict: Verdict(verdict="accept"),
        },
    )
    assert ans.answerable and ans.citations == [1]
    assert len(ans.context) == 1 and "twenty days" in ans.context[0].text  # graded context only
    assert [s.split(":")[0] for s in ans.steps] == [
        "plan", "retrieve", "grade", "generate", "verify"
    ]  # fmt: skip
    assert llm.schemas == [QueryPlan, RelevanceGrade, LlmAnswer, Verdict]


def test_grader_out_of_range_numbers_are_ignored(deps):
    ans, _ = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["annual leave"]),
            RelevanceGrade: RelevanceGrade(relevant=[1, 9]),  # 9 does not exist
            LlmAnswer: GOOD,
            Verdict: Verdict(verdict="accept"),
        },
    )
    assert len(ans.context) == 1


def test_no_relevant_passages_rewrites_twice_then_abstains(deps):
    ans, llm = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["ceo name"]),
            RelevanceGrade: RelevanceGrade(relevant=[]),
            RewrittenQuery: [RewrittenQuery(query="chief executive"), RewrittenQuery(query="ceo")],
        },
        question="who is the ceo",
    )
    assert not ans.answerable
    assert sum(s.startswith("rewrite") for s in ans.steps) == CFG.max_rewrites
    assert ans.steps[-1] == "abstain"
    assert LlmAnswer not in llm.schemas  # generator never called


def test_verifier_regenerate_once_with_feedback_then_accept(deps):
    ans, llm = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["annual leave"]),
            RelevanceGrade: RelevanceGrade(relevant=[1]),
            LlmAnswer: [LlmAnswer(answerable=True, answer="Thirty days [1]."), GOOD],
            Verdict: [
                Verdict(verdict="regenerate", feedback="Wrong number"),
                Verdict(verdict="accept"),
            ],
        },
    )
    assert ans.answerable and ans.text == "Twenty days [1]."
    second_generation_prompt = llm.prompts[llm.schemas.index(Verdict) + 1][1][1]
    assert "Wrong number" in second_generation_prompt


def test_verifier_regenerate_budget_is_enforced(deps):
    ans, llm = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["annual leave"]),
            RelevanceGrade: RelevanceGrade(relevant=[1]),
            LlmAnswer: GOOD,
            Verdict: Verdict(verdict="regenerate", feedback="still wrong"),  # always
        },
    )
    assert not ans.answerable and ans.abstain_reason == "verifier: regenerate"
    assert llm.schemas.count(LlmAnswer) == 1 + CFG.max_regenerations


def test_verifier_abstain(deps):
    ans, _ = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["annual leave"]),
            RelevanceGrade: RelevanceGrade(relevant=[1]),
            LlmAnswer: GOOD,
            Verdict: Verdict(verdict="abstain"),
        },
    )
    assert not ans.answerable and ans.text == ""


def test_planner_multiple_queries_are_merged_without_duplicates(deps):
    ans, llm = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["annual leave", "annual leave days", "vpn"]),
            RelevanceGrade: RelevanceGrade(relevant=[1, 2]),
            LlmAnswer: GOOD,
            Verdict: Verdict(verdict="accept"),
        },
    )
    assert "retrieve: 2 passages" in ans.steps  # 2 docs in the corpus, no duplicates
    assert len({c.chunk_id for c in ans.context}) == len(ans.context)


def test_graded_context_is_capped_at_top_k(deps):
    cfg = RagConfig(mode="agentic", top_k=1, similarity_threshold=0.0)
    ans, _ = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["annual leave", "vpn"]),  # 2 queries x top_k 1 = 2
            RelevanceGrade: RelevanceGrade(relevant=[1, 2]),  # grader keeps both
            LlmAnswer: GOOD,
            Verdict: Verdict(verdict="accept"),
        },
        cfg=cfg,
    )
    assert "retrieve: 2 passages" in ans.steps
    assert len(ans.context) == 1  # generator gets at most top_k passages


def test_build_answerer_fast_mode_makes_one_llm_call(deps):
    store, index, embedder = deps
    llm = FakeLlm(GOOD)
    ans = build_answerer(store, index, embedder, llm, RagConfig(top_k=2, similarity_threshold=0.0))(
        "how many days annual leave"
    )
    assert ans.answerable and ans.steps == [] and llm.schemas == [LlmAnswer]


def test_agents_that_read_passages_treat_them_as_untrusted_data():
    # Security plan B6: every prompt that shows document passages carries the same rule.
    from rag_platform.agents.rag_graph.graph import GRADER_PROMPT, VERIFIER_PROMPT
    from rag_platform.generation.answer import SYSTEM_PROMPT, UNTRUSTED_PASSAGES

    assert all(UNTRUSTED_PASSAGES in p for p in (SYSTEM_PROMPT, GRADER_PROMPT, VERIFIER_PROMPT))


def test_original_question_is_always_searched(deps, monkeypatch):
    # The planner's keyword queries can match worse than the user's own sentence, so the
    # question is searched too. monkeypatch swaps a method for this test only (like a mock).
    searched: list[str] = []
    embedder = deps[2]
    real = embedder.embed_query
    monkeypatch.setattr(embedder, "embed_query", lambda t: searched.append(t) or real(t))
    question = "how many days annual leave"
    _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["leave policy", question]),  # duplicate of the question
            RelevanceGrade: RelevanceGrade(relevant=[1]),
            LlmAnswer: GOOD,
            Verdict: Verdict(verdict="accept"),
        },
        question=question,
    )
    assert searched == [question, "leave policy"]  # question first, each query once


def test_graded_passages_are_not_refused_by_the_score_threshold(deps):
    # Agentic mode: once the grader confirms a passage, a low cosine score (e.g. a table of
    # names) must not turn the answer into "I don't know".
    strict = RagConfig(mode="agentic", top_k=2, similarity_threshold=0.99)
    ans, llm = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["annual leave days"]),
            RelevanceGrade: RelevanceGrade(relevant=[1]),
            LlmAnswer: GOOD,
            Verdict: Verdict(verdict="accept"),
        },
        cfg=strict,
    )
    assert max(c.score for c in ans.context) < 0.99  # below the threshold...
    assert ans.answerable and LlmAnswer in llm.schemas  # ...and still answered


def test_invalid_citations_get_one_retry_with_feedback(deps):
    # The model cites [P3] while only [P1] exists (a heading "3. ..." confused it): instead of
    # refusing, generate retries once and is told which labels are valid.
    bad = LlmAnswer(answerable=True, answer="Twenty days [P3].")
    ans, llm = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["annual leave"]),
            RelevanceGrade: RelevanceGrade(relevant=[1]),
            LlmAnswer: [bad, GOOD],
            Verdict: Verdict(verdict="accept"),
        },
    )
    assert ans.answerable and ans.text == "Twenty days [1]."
    retry_prompt = llm.prompts[llm.schemas.index(LlmAnswer) + 1][1][1]
    assert "Cite only [P1]" in retry_prompt
    assert [s.split(":")[0] for s in ans.steps][-3:] == ["generate", "generate", "verify"]


def test_invalid_citations_retry_budget_is_enforced(deps):
    bad = LlmAnswer(answerable=True, answer="Twenty days [P3].")
    ans, llm = _run(
        deps,
        {
            QueryPlan: QueryPlan(queries=["annual leave"]),
            RelevanceGrade: RelevanceGrade(relevant=[1]),
            LlmAnswer: bad,  # always wrong
        },
    )
    assert not ans.answerable and ans.abstain_reason == "answer had no valid citations"
    assert llm.schemas.count(LlmAnswer) == 1 + CFG.max_regenerations
    assert Verdict not in llm.schemas  # refused answers are never verified


# --- auto mode: fast first, agentic only when the fast answer struggled -------------------------

AUTO = RagConfig(mode="auto", top_k=2, similarity_threshold=0.0)
AGENT_OK = {
    QueryPlan: QueryPlan(queries=["annual leave"]),
    RelevanceGrade: RelevanceGrade(relevant=[1]),
    Verdict: Verdict(verdict="accept"),
}


def _auto(deps, llm, cfg=AUTO, question="how many days annual leave"):
    store, index, embedder = deps
    return build_answerer(store, index, embedder, llm, cfg)(question)


def test_auto_keeps_a_good_fast_answer_at_one_call(deps):
    llm = FakeLlm(GOOD)
    ans = _auto(deps, llm)
    assert ans.answerable and ans.route == "fast"
    assert llm.schemas == [LlmAnswer]  # no planner, grader or verifier


def test_auto_escalates_a_partial_answer(deps):
    # The fast answer covered only part of the question (complete=False): the agent tries,
    # with one search per part.
    partial = LlmAnswer(answerable=True, answer="Twenty days [1]; not covered: carry-over.",
                        complete=False)  # fmt: skip
    llm = FakeLlm({**AGENT_OK, LlmAnswer: [partial, GOOD]})
    ans = _auto(deps, llm)
    assert ans.route == "auto → agentic (incomplete)"
    assert ans.text == "Twenty days [1]." and ans.steps  # the agent's answer and trace
    assert llm.schemas == [LlmAnswer, QueryPlan, RelevanceGrade, LlmAnswer, Verdict]


def test_auto_escalates_when_the_model_says_not_answerable(deps):
    llm = FakeLlm({**AGENT_OK, LlmAnswer: [LlmAnswer(answerable=False, answer=""), GOOD]})
    ans = _auto(deps, llm)
    assert ans.answerable and ans.route == "auto → agentic (model judged the context insufficient)"


def test_auto_does_not_escalate_a_threshold_refusal(deps):
    # Nothing close enough (off-topic): refuse cheaply - no answer call, no agent (owner decision).
    llm = FakeLlm(GOOD)
    ans = _auto(deps, llm, cfg=RagConfig(mode="auto", top_k=2, similarity_threshold=0.99))
    assert not ans.answerable and ans.route == "fast"
    assert llm.schemas == []


def test_auto_keeps_the_fast_partial_answer_when_the_agent_refuses(deps):
    partial = LlmAnswer(answerable=True, answer="Twenty days [1].", complete=False)
    agent_refuses = {**AGENT_OK, Verdict: Verdict(verdict="abstain")}
    llm = FakeLlm({**agent_refuses, LlmAnswer: [partial, GOOD]})
    ans = _auto(deps, llm)
    assert ans.answerable and ans.text == "Twenty days [1]."  # never lose a good partial answer
    assert ans.route == "auto → agentic (incomplete), kept the fast answer"
    assert ans.steps  # the agent's trace stays visible for developers


def test_escalation_reasons():
    from rag_platform.answering import escalation_reason
    from rag_platform.generation.answer import BELOW_THRESHOLD, NO_VALID_CITATIONS, Answer

    assert escalation_reason(Answer("q", True, "x [1]")) == ""
    assert escalation_reason(Answer("q", True, "x [1]", complete=False)) == "incomplete"
    assert escalation_reason(Answer("q", False, "", abstain_reason=BELOW_THRESHOLD)) == ""
    assert escalation_reason(Answer("q", False, "", abstain_reason=NO_VALID_CITATIONS)) == (
        NO_VALID_CITATIONS
    )
