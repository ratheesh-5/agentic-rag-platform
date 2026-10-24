"""Tests for the evaluation package: golden-set loading, metrics, threshold calibration, the
RAGAS adapters, the end-to-end runner and eval-run storage. Fakes only - no API calls.

RAGAS's own metric logic is not re-tested here (that's the library's job); we test our
adapters and how we call and aggregate the metrics."""

import asyncio
import json
from pathlib import Path

import pytest
from ragas.metrics.result import MetricResult

from conftest import FakeLlm
from rag_platform.answering import build_answerer
from rag_platform.config import RagConfig
from rag_platform.evaluation.golden import GoldenItem, Span, find_span, load_golden
from rag_platform.evaluation.metrics import (
    RAGAS_METRICS,
    QuestionResult,
    calibrate_threshold,
    retrieval_scores,
    summarize,
)
from rag_platform.evaluation.ragas_eval import (
    LangChainRagasEmbeddings,
    LangChainRagasLLM,
    RagasEvaluator,
)
from rag_platform.evaluation.runner import run_eval
from rag_platform.generation.answer import Answer, LlmAnswer
from rag_platform.ingestion.pipeline import ingest, open_index
from rag_platform.store.db import Store


def test_find_span_ignores_whitespace_differences():
    text = "Header\nThe team default is 100\n   characters of overlap."
    span = find_span("d", text, "default is 100 characters of")
    assert text[span.start : span.end] == "default is 100\n   characters of"


def test_find_span_rejects_missing_and_ambiguous_quotes():
    with pytest.raises(ValueError, match="not found"):
        find_span("d", "abc", "xyz")
    with pytest.raises(ValueError, match="2 times"):
        find_span("d", "rule. rule.", "rule.")


def test_load_golden_resolves_evidence(tmp_path: Path):
    path = tmp_path / "golden.jsonl"
    rows = [
        {"id": "a", "question": "q?", "answerable": True, "reference_answer": "r",
         "evidence": [{"doc": "d.md", "quote": "world"}]},
        {"id": "b", "question": "u?", "answerable": False, "evidence": []},
    ]  # fmt: skip
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    items = load_golden(path, {"d.md": "hello world\n"})
    assert items[0].evidence == [Span("d.md", 6, 11)]
    assert not items[1].answerable


def test_retrieval_scores():
    evidence = [Span("d", 100, 120), Span("e", 0, 10)]
    retrieved = [("d", 0, 50), ("d", 90, 110), ("x", 0, 10)]  # only the 2nd overlaps
    m = retrieval_scores(retrieved, evidence, k=3)
    assert m == {"precision": pytest.approx(1 / 3), "recall": 0.5, "hit": 1.0, "rr": 0.5}
    assert retrieval_scores([("x", 0, 1)], evidence, k=1)["rr"] == 0.0


def _result(qid, answerable, predicted, top_score, hit=1.0) -> QuestionResult:
    r = QuestionResult(qid, "q", answerable, predicted, "", "", top_score)
    if answerable:
        r.retrieval = {"precision": hit, "recall": hit, "hit": hit, "rr": hit}
    return r


def test_summarize_abstention_metrics():
    results = [
        _result("a1", True, True, 0.8),
        _result("a2", True, False, 0.7, hit=0.0),  # wrongly abstained
        _result("u1", False, False, 0.5),
        _result("u2", False, True, 0.6),  # answered an unanswerable question
    ]
    m = summarize(results)
    assert m["abstention_accuracy"] == 0.5
    assert m["false_abstention_rate"] == 0.5
    assert m["unanswerable_answered_rate"] == 0.5
    assert m["hit_rate_at_k"] == 0.5
    assert m["faithfulness"] is None  # nothing judged


def test_run_eval_stops_on_error_and_keeps_partial_results():
    def answerer(question):
        if question == "boom":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return Answer(question, False, "", abstain_reason="test")

    golden = [GoldenItem(i, q, False, "", []) for i, q in [("a", "ok"), ("b", "boom"), ("c", "x")]]
    results, error = run_eval(golden, answerer, k=5)
    assert [r.id for r in results] == ["a"]  # "c" is not attempted after the failure
    assert error.startswith("b: RuntimeError") and "RESOURCE_EXHAUSTED" in error


def test_question_result_from_dict_ignores_unknown_keys():
    r = QuestionResult.from_dict(
        {"id": "q", "question": "?", "answerable": True, "predicted_answerable": True,
         "answer": "a", "abstain_reason": "", "top_score": 0.7, "judge": {"old": 1}}
    )  # fmt: skip
    assert r.id == "q" and r.latency_s is None


def test_calibrate_threshold_finds_separating_score():
    results = [
        _result("a1", True, True, 0.80),
        _result("a2", True, True, 0.75),
        _result("u1", False, True, 0.60),  # answered, but a threshold of 0.75 would stop it
        _result("u2", False, False, 0.55),
    ]
    assert calibrate_threshold(results, current=0.0) == (0.75, 1.0)
    assert calibrate_threshold([], current=0.35) == (0.35, None)  # run stopped before any Q
    assert summarize([])["abstention_accuracy"] is None


def test_ragas_llm_adapter_delegates_to_langchain_and_counts_calls():
    llm = FakeLlm(LlmAnswer(answerable=True, answer="x"))
    adapter = LangChainRagasLLM(llm)
    assert adapter.generate("prompt", LlmAnswer).answer == "x"
    assert asyncio.run(adapter.agenerate("prompt 2", LlmAnswer)).answer == "x"
    assert adapter.calls == 2 and llm.prompts == ["prompt", "prompt 2"]


def test_ragas_embeddings_adapter_uses_query_embedding(fake_embedder):
    emb = LangChainRagasEmbeddings(fake_embedder)
    assert emb.embed_text("annual leave") == fake_embedder.embed_query("annual leave")


class _FakeMetric:
    def __init__(self, value):
        self.value, self.calls = value, 0

    def score(self, **kwargs):
        self.calls += 1
        return MetricResult(value=self.value)


def test_ragas_evaluator_skips_reference_metrics_without_reference(fake_embedder):
    ev = RagasEvaluator(FakeLlm(), fake_embedder)
    for name in RAGAS_METRICS:
        setattr(ev, name, _FakeMetric(0.8))
    ev.faithfulness = _FakeMetric(float("nan"))  # RAGAS returns NaN when it can't score

    with_ref = ev.score("q", "a", ["ctx"], reference="ref")
    assert with_ref["faithfulness"] is None and with_ref["answer_correctness"] == 0.8
    no_ref = ev.score("q", "a", ["ctx"], reference="")
    assert no_ref["answer_relevancy"] == 0.8
    assert no_ref["context_precision"] is None and ev.context_precision.calls == 1


class _FakeRagas:
    def score(self, question, answer, contexts, reference):
        return {name: 1.0 for name in RAGAS_METRICS}


def test_run_eval_end_to_end(tmp_path: Path, fake_embedder):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "leave.md").write_text("Employees get twenty days annual leave.", encoding="utf-8")
    (corpus / "vpn.md").write_text("Connect to the VPN before using servers.", encoding="utf-8")
    store = Store(tmp_path / "rag.db")
    ingest(corpus, store, tmp_path / "f.index", fake_embedder, "fake", RagConfig())
    doc = next(p for p in store.document_paths() if p.endswith("leave.md"))
    golden = [
        GoldenItem("q1", "how many days annual leave", True, "twenty days",
                   [find_span(doc, store.document_texts()[doc], "twenty days")]),
        GoldenItem("u1", "who is the ceo", False, "", []),
    ]  # fmt: skip
    llm = FakeLlm(LlmAnswer(answerable=True, answer="Twenty days [1]."))
    index = open_index(store, tmp_path / "f.index", "fake")
    cfg = RagConfig(top_k=2, similarity_threshold=0.0)
    answerer = build_answerer(store, index, fake_embedder, llm, cfg)
    results, error = run_eval(golden, answerer, cfg.top_k, ragas=_FakeRagas())
    assert error is None

    q1, u1 = results
    assert q1.retrieval["hit"] == 1.0 and q1.retrieved[0]["relevant"]
    assert q1.ragas["answer_correctness"] == 1.0
    assert summarize(results)["faithfulness"] == 1.0
    assert q1.latency_s is not None and summarize(results)["latency_p95_s"] is not None
    assert u1.retrieval is None and u1.predicted_answerable  # fake LLM always answers
    run_id = store.save_eval_run("t", cfg.model_dump(), summarize(results), [])
    assert store.get_eval_run(run_id)["metrics"]["unanswerable_answered_rate"] == 1.0
    assert [r["id"] for r in store.list_eval_runs()] == [run_id]
    store.close()
