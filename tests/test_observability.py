"""Tests for observability (Phase 7): the usage log callback, embedding logging, the usage
report and the LangSmith switch. No API calls: LangChain results are faked."""

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from conftest import FakeLlm
from rag_platform.api.app import app
from rag_platform.api.deps import Services, get_services
from rag_platform.config import Settings
from rag_platform.llm import RateLimitedEmbeddings
from rag_platform.observability import (
    UsageLogger,
    configure_tracing,
    purpose_of,
    tracing_info,
    usage_report,
)
from rag_platform.store.db import Store


def _result(tokens_in: int, tokens_out: int) -> LLMResult:
    msg = AIMessage(
        "ok",
        usage_metadata={
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def _rows(db: Path) -> list[dict]:
    store = Store(db)
    try:
        return store.llm_calls_since("2000-01-01")
    finally:
        store.close()


def test_purpose_comes_from_metadata_then_langgraph_node_then_default():
    assert purpose_of({"purpose": "answer", "langgraph_node": "generate"}, "chat") == "answer"
    assert purpose_of({"langgraph_node": "grade"}, "chat") == "agent: grade"
    assert purpose_of(None, "judge") == "judge"


def test_usage_logger_records_tokens_latency_and_errors(tmp_path: Path):
    db = tmp_path / "rag.db"
    log = UsageLogger(db, "model-x", default_purpose="chat")
    ok, bad = uuid.uuid4(), uuid.uuid4()
    log.on_chat_model_start({}, [], run_id=ok, metadata={"purpose": "rewrite"})
    log.on_llm_end(_result(120, 30), run_id=ok)
    log.on_chat_model_start({}, [], run_id=bad, metadata={"langgraph_node": "verify"})
    log.on_llm_error(RuntimeError("429 RESOURCE_EXHAUSTED"), run_id=bad)
    first, second = _rows(db)
    assert (first["purpose"], first["input_tokens"], first["output_tokens"]) == ("rewrite", 120, 30)
    assert first["run_id"] == str(ok) and first["error"] is None and first["latency_ms"] >= 0
    assert second["purpose"] == "agent: verify" and "RESOURCE_EXHAUSTED" in second["error"]


def test_logging_failures_never_break_the_call(tmp_path: Path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")  # a FILE where the db's folder should be: writing must fail
    log = UsageLogger(blocker / "rag.db", "m")
    run = uuid.uuid4()
    log.on_chat_model_start({}, [], run_id=run)
    log.on_llm_end(_result(1, 1), run_id=run)  # no exception


class _Inner:
    def embed_documents(self, texts):
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        return [1.0]


def test_embedding_requests_are_logged_with_their_text_count(tmp_path: Path):
    db = tmp_path / "rag.db"
    emb = RateLimitedEmbeddings(_Inner(), 90, sleep=lambda s: None, usage_db=db, model="emb")
    emb.embed_documents(["a", "b", "c"])
    emb.embed_query("q")
    assert [(r["kind"], r["texts"]) for r in _rows(db)] == [("embed", 3), ("embed", 1)]
    RateLimitedEmbeddings(_Inner(), 90).embed_query("q")  # no db: nothing logged, no error


def _row(day, purpose, model="m", kind="chat", tin=100, tout=10, ms=1000, error=None):
    return {
        "created_at": f"{day}T10:00:00+00:00", "model": model, "purpose": purpose,
        "kind": kind, "input_tokens": tin if kind == "chat" else None,
        "output_tokens": tout if kind == "chat" else None, "texts": 5 if kind == "embed" else None,
        "latency_ms": ms, "error": error, "run_id": "r",
    }  # fmt: skip


def test_usage_report_totals_breakdowns_days_and_cost():
    rows = [
        _row("2026-10-01", "answer", ms=900),  # outside a 7-day window ending 10-12
        _row("2026-10-10", "answer", ms=1000),
        _row("2026-10-12", "answer", ms=3000),
        _row("2026-10-12", "judge", tin=1000, tout=100, ms=2000),
        _row("2026-10-12", "answer", ms=500, error="Timeout"),
        _row("2026-10-12", "embed", kind="embed", ms=200),
    ]
    prices = {"m": {"input_per_million": 1.0, "output_per_million": 2.0}}
    rep = usage_report(rows, prices, days=7, today=datetime(2026, 10, 12, tzinfo=UTC))
    t = rep["totals"]
    assert t["calls"] == 4 and t["embed_texts"] == 5 and t["errors"] == 1
    assert t["input_tokens"] == 1300 and t["output_tokens"] == 130
    assert t["avg_latency_ms"] == 2000  # failed calls excluded from latency
    assert t["cost"] == pytest.approx((1300 * 1.0 + 130 * 2.0) / 1e6)
    assert [p["purpose"] for p in rep["by_purpose"]][:2] == ["answer", "judge"]
    assert len(rep["per_day"]) == 7 and rep["per_day"][-1]["calls"] == 3
    calls = rep["calls"]  # every call in the window, newest first (embeddings included)
    assert len(calls) == 5 and max(c["latency_ms"] for c in calls) == 3000
    assert {c["kind"] for c in calls} == {"chat", "embed"}
    assert rep["recent_errors"][0]["error"] == "Timeout"
    embed_only = next(p for p in rep["by_purpose"] if p["purpose"] == "embed")
    assert embed_only["avg_latency_ms"] == 200  # embedding-only groups show their own latency


def test_no_prices_means_no_cost():
    rep = usage_report([_row("2026-10-12", "answer")], {}, 1, datetime(2026, 10, 12, tzinfo=UTC))
    assert rep["totals"]["cost"] is None


def test_tracing_is_off_without_key_and_exported_with_it(monkeypatch):
    for k in [k for k in os.environ if k.startswith("LANGSMITH_")]:
        monkeypatch.delenv(k)
    off = Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="")
    assert configure_tracing(off) is False and "LANGSMITH_TRACING" not in os.environ
    on = Settings(
        _env_file=None,
        langsmith_tracing=True,
        langsmith_api_key="lsv2_test",
        langsmith_project="p1",
        langsmith_sampling_rate=0.5,
    )
    # monkeypatch.setenv registers the keys, so they are removed again after this test.
    for k in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT",
              "LANGSMITH_TRACING_SAMPLING_RATE"):  # fmt: skip
        monkeypatch.setenv(k, "")
        monkeypatch.delenv(k)
    assert configure_tracing(on) is True
    assert os.environ["LANGSMITH_PROJECT"] == "p1"
    assert os.environ["LANGSMITH_TRACING_SAMPLING_RATE"] == "0.5"
    info = tracing_info(on)
    assert info["enabled"] and info["url"] == "https://smith.langchain.com"
    assert "lsv2_test" not in str(info)  # the key never leaves the server


def test_usage_endpoint_is_admin_only_and_reports(tmp_path: Path, fake_embedder):
    settings = Settings(
        _env_file=None, gemini_api_key="x", api_keys="admin", index_dir=tmp_path / "idx"
    )
    svc = Services(settings, fake_embedder, FakeLlm(), prices_path=tmp_path / "none.json")
    store = svc.open_store()
    store.add_llm_call(
        model="m", purpose="answer", kind="chat", input_tokens=10, output_tokens=2, texts=None,
        latency_ms=1500, error=None, run_id="r1",
    )  # fmt: skip
    store.close()
    app.dependency_overrides[get_services] = lambda: svc
    try:
        client = TestClient(app)
        body = client.get("/usage/summary?days=3", headers={"X-API-Key": "admin"}).json()
        assert body["totals"]["calls"] == 1 and body["tracing"]["enabled"] is False
        guest = {"X-Guest-Id": "3f1c2a9e-7b4d-4c1a-9e2f-0a1b2c3d4e5f"}
        assert client.get("/usage/summary", headers=guest).status_code == 401
        assert client.get("/usage/trace/r1", headers={"X-API-Key": "admin"}).status_code == 404
    finally:
        app.dependency_overrides.clear()
