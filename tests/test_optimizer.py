"""Tests for the Phase 4 optimizer: the experiment runner (tool) and the multi-agent graph.

The graph test scripts every LLM decision (diagnosis, tool calls, report) with FakeLlm and the
experiments run for real on a tiny corpus with the fake embedder - no API calls. It checks the
whole flow, including the human-approval interrupt, budgets and argument validation.
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from conftest import FakeLlm
from rag_platform.agents.optimizer_graph.experiments import ExperimentRunner, objective
from rag_platform.agents.optimizer_graph.graph import (
    Diagnosis,
    NextAction,
    ReportText,
    build_optimizer_graph,
)
from rag_platform.config import RagConfig, Settings, load_rag_config
from rag_platform.evaluation.runner import retrieval_only_answerer
from rag_platform.ingestion.pipeline import ingest
from rag_platform.retrieval.retriever import RetrievedChunk
from rag_platform.store.db import Store

# One fact sentence per document (an evidence quote must match exactly once) plus unique
# filler, so each document splits into several chunks.
FACTS = {
    "codes.md": "Error code LUM-2001 means a gateway timeout after 30 seconds.",
    "leave.md": "Employees get twenty days of annual leave per year.",
    "vpn.md": "Always connect to the VPN before using production servers.",
}
TEXT = {
    name: fact + " " + " ".join(f"Filler line {i} of {name}." for i in range(8))
    for name, fact in FACTS.items()
}


@pytest.fixture
def setup(tmp_path: Path, fake_embedder):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in TEXT.items():
        (corpus / name).write_text(text, encoding="utf-8")
    s = Settings(_env_file=None, gemini_api_key="x", index_dir=tmp_path / "idx")
    main = Store(s.db_path)
    ingest(corpus, main, s.faiss_path, fake_embedder, s.embedding_model, RagConfig())
    docs = {Path(p).name: p for p in main.document_paths()}
    main.close()
    rows = [
        ("q1", "what does LUM-2001 mean", True, "codes.md", "LUM-2001 means a gateway timeout"),
        ("q2", "how many days of annual leave", True, "leave.md", "twenty days of annual leave"),
        # Misleading wording (matches codes.md filler) -> evidence not ranked first -> a failure,
        # so the optimizer always has something to improve.
        ("q3", "filler line of codes md", True, "leave.md", "Employees get twenty days"),
        ("u1", "who is the ceo", False, None, None),
    ]
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(
            json.dumps({
                "id": qid, "question": q, "answerable": ok, "reference_answer": "",
                "evidence": [{"doc": docs[doc], "quote": quote}] if ok else [],
            })
            for qid, q, ok, doc, quote in rows
        ),
        encoding="utf-8",
    )  # fmt: skip
    base = RagConfig(chunk_size=120, chunk_overlap=20, top_k=2)
    # Re-index the main store with the base chunking so the main index matches `base`.
    main = Store(s.db_path)
    ingest(corpus, main, s.faiss_path, fake_embedder, s.embedding_model, base)
    main.close()
    runner = ExperimentRunner(s, fake_embedder, golden, base)
    yield runner, tmp_path
    runner.close()


def test_retrieval_only_answerer_uses_similarity_gate():
    chunk = RetrievedChunk(1, "d", 0, 1, "t", 0.6)
    assert retrieval_only_answerer(lambda q: [chunk], 0.5)("q").answerable
    assert not retrieval_only_answerer(lambda q: [chunk], 0.7)("q").answerable
    assert not retrieval_only_answerer(lambda q: [], 0.0)("q").answerable


def test_experiment_runner_reuses_main_index_and_builds_experiment_index(setup):
    runner, tmp = setup
    base = runner.run(runner.base, "base")
    assert base.changes == {} and base.metrics["recall_at_k"] is not None
    assert runner.run(runner.base, "again") is base  # identical config is never re-run
    other = runner.run(RagConfig(**{**runner.base.model_dump(), "chunk_size": 200}), "c200")
    assert other.changes == {"chunk_size": 200}
    assert (tmp / "idx" / "experiments" / "chunk-200-20" / "rag.db").exists()
    assert [r["label"] for r in runner.main.list_eval_runs()] == ["base", "c200"]


def test_objective_prefers_recall_then_mrr():
    a = {"recall_at_k": 1.0, "mrr": 0.5, "unanswerable_below_safe": 0.0}
    b = {"recall_at_k": 0.9, "mrr": 1.0, "unanswerable_below_safe": 1.0}
    c = {"recall_at_k": 1.0, "mrr": 0.8, "unanswerable_below_safe": 0.0}
    assert max([a, b, c], key=objective) is c


def test_apply_writes_config_and_reindexes_on_new_chunking(setup):
    runner, tmp = setup
    cfg = RagConfig(**{**runner.base.model_dump(), "chunk_size": 200, "top_k": 3})
    runner.run(cfg, "warm cache")  # experiment index fills the shared embedding cache
    assert runner.apply(cfg, tmp / "rag_config.json") is True
    assert load_rag_config(tmp / "rag_config.json") == cfg
    assert runner.main.chunkers() == {"200/20"}


def _tool_call(call_id: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "RunExperiment", "args": args, "id": call_id}])


def _graph(runner, tmp, llm, **kw):
    return build_optimizer_graph(runner, llm, tmp / "rag_config.json", **kw)


SCRIPT = {
    Diagnosis: Diagnosis(findings=["codes ranked low"], suggestions=["try hybrid"]),
    NextAction: NextAction(action="report", reason="good enough"),
    ReportText: ReportText(summary="Use hybrid", rationale="higher MRR", risks="None"),
}


def test_full_flow_with_approval_writes_config(setup):
    runner, tmp = setup
    llm = FakeLlm(
        SCRIPT,
        tool_replies=[
            _tool_call("c1", reason="codes", retrieval="hybrid"),
            _tool_call("c2", reason="bad", chunk_size=100, chunk_overlap=200),  # invalid
            _tool_call("c3", reason="more", top_k=3),
        ],
    )
    graph = _graph(runner, tmp, llm, experiments_per_round=2, max_rounds=2)
    thread = {"configurable": {"thread_id": "t1"}}
    out = graph.invoke({"steps": []}, thread)

    report = out["__interrupt__"][0].value["report"]  # paused for human approval
    assert [e["label"] for e in report["experiments"]] == ["opt-baseline", "opt-r1-e1", "opt-r1-e2"]
    assert report["changes"]  # something to approve (at least the calibrated threshold)
    # Tool results the agent received (other prompts are (role, text) tuples, skipped).
    tool_results = [m for p in llm.prompts for m in p if getattr(m, "type", None) == "tool"]
    assert any("Invalid arguments" in m.content for m in tool_results)

    final = graph.invoke(Command(resume=True), thread)
    assert final["decision"] == "approved"
    applied = load_rag_config(tmp / "rag_config.json")
    assert applied.model_dump() == report["recommended_config"]


def test_reject_leaves_config_untouched(setup):
    runner, tmp = setup
    llm = FakeLlm(SCRIPT, tool_replies=[_tool_call("c1", reason="x", retrieval="hybrid")])
    graph = _graph(runner, tmp, llm, experiments_per_round=1, max_rounds=1)
    thread = {"configurable": {"thread_id": "t2"}}
    graph.invoke({"steps": []}, thread)
    final = graph.invoke(Command(resume=False), thread)
    assert final["decision"] == "rejected"
    assert not (tmp / "rag_config.json").exists()


def test_repeated_experiment_is_not_listed_twice(setup):
    runner, tmp = setup
    same = [_tool_call(f"c{i}", reason="x", retrieval="hybrid") for i in range(2)]
    llm = FakeLlm(SCRIPT, tool_replies=same)
    graph = _graph(runner, tmp, llm, experiments_per_round=3, max_rounds=1)
    out = graph.invoke({"steps": []}, {"configurable": {"thread_id": "t7"}})
    labels = [e["label"] for e in out["__interrupt__"][0].value["report"]["experiments"]]
    assert labels == ["opt-baseline", "opt-r1-e1"]
    tool_results = [m for p in llm.prompts for m in p if getattr(m, "type", None) == "tool"]
    assert tool_results[1].content.startswith("Already run")


def test_budget_is_enforced_in_code(setup):
    runner, tmp = setup
    calls = [_tool_call(f"c{i}", reason="x", top_k=i) for i in range(1, 6)]
    llm = FakeLlm(SCRIPT, tool_replies=calls)
    graph = _graph(runner, tmp, llm, experiments_per_round=2, max_rounds=1)
    out = graph.invoke({"steps": []}, {"configurable": {"thread_id": "t3"}})
    report = out["__interrupt__"][0].value["report"]
    assert len(report["experiments"]) == 1 + 2  # baseline + budget, never more


def test_failing_experiment_is_reported_to_the_agent_not_fatal(setup, monkeypatch):
    runner, tmp = setup
    real_run = runner.run

    def flaky(cfg, label):  # baseline works, every experiment hits a "quota" error
        if label != "opt-baseline":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return real_run(cfg, label)

    monkeypatch.setattr(runner, "run", flaky)  # pytest helper: replace a method for this test
    llm = FakeLlm(SCRIPT, tool_replies=[_tool_call("c1", reason="x", retrieval="hybrid")])
    out = _graph(runner, tmp, llm, experiments_per_round=1, max_rounds=1).invoke(
        {"steps": []}, {"configurable": {"thread_id": "t6"}}
    )
    tool_results = [m for p in llm.prompts for m in p if getattr(m, "type", None) == "tool"]
    assert "Experiment failed" in tool_results[0].content
    assert "__interrupt__" in out  # the run went on to the report


def test_embedding_throttle_splits_batches_and_waits():
    from rag_platform.llm import RateLimitedEmbeddings

    class Inner:
        def __init__(self):
            self.batches = []

        def embed_documents(self, texts):
            self.batches.append(len(texts))
            return [[0.0]] * len(texts)

        def embed_query(self, text):
            return [0.0]

    clock = {"t": 0.0}
    waits = []

    def sleep(seconds):  # fake clock: "waiting" just moves time forward
        waits.append(seconds)
        clock["t"] += seconds

    inner = Inner()
    emb = RateLimitedEmbeddings(inner, 10, clock=lambda: clock["t"], sleep=sleep)
    assert len(emb.embed_documents([f"t{i}" for i in range(25)])) == 25
    assert inner.batches == [10, 10, 5]  # never more than the per-minute budget at once
    assert len(waits) == 2 and all(w > 59 for w in waits)


def test_finish_without_llm_when_nothing_fails(setup, tmp_path):
    runner, tmp = setup
    golden = tmp_path / "easy.jsonl"
    lines = [
        line for line in runner.golden_path.read_text("utf-8").splitlines() if '"q3"' not in line
    ]
    golden.write_text("\n".join(lines), encoding="utf-8")
    runner.golden_path = golden  # only well-ranked questions left
    llm = FakeLlm(SCRIPT)
    out = _graph(runner, tmp, llm).invoke({"steps": []}, {"configurable": {"thread_id": "t5"}})
    assert out["steps"][-1] == "supervisor: finish"
    assert llm.prompts == []  # no LLM call at all: the decision was made in code


def test_supervisor_cannot_pick_an_unallowed_action(setup):
    runner, tmp = setup
    script = {**SCRIPT, NextAction: NextAction(action="delete everything", reason="?")}
    llm = FakeLlm(script, tool_replies=[_tool_call("c1", reason="x", retrieval="hybrid")])
    graph = _graph(runner, tmp, llm, experiments_per_round=1, max_rounds=2)
    out = graph.invoke({"steps": []}, {"configurable": {"thread_id": "t4"}})
    # Invalid choice falls back to the first allowed action ("diagnose", then "report").
    assert "__interrupt__" in out or out.get("decision") == "no-change"
