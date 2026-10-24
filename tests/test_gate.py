"""Tests for the CI quality gate (evaluation/gate.py): snapshot export, offline run, release
gates, regression vs the baseline and the "missing vectors" path. Fake embedder, no API."""

import json
from pathlib import Path

import pytest

from rag_platform.config import RagConfig
from rag_platform.evaluation.gate import (
    MissingVectors,
    OfflineEmbeddings,
    export_snapshot,
    run_gate,
)
from rag_platform.ingestion.pipeline import ingest
from rag_platform.retrieval.query_cache import CachedQueryEmbedder
from rag_platform.store.db import Store

FACTS = {
    "leave.md": "Employees get twenty days of annual leave per year.",
    "vpn.md": "Connect to the VPN before using internal servers.",
}


@pytest.fixture
def setup(tmp_path: Path, fake_embedder, monkeypatch):
    """A tiny corpus + golden set + a "local" index whose cache becomes the snapshot."""
    monkeypatch.chdir(tmp_path)  # the gate works with repo-relative paths, like in CI
    corpus = Path("data/sample_corpus")
    corpus.mkdir(parents=True)
    for name, text in FACTS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    golden = Path("golden.jsonl")
    rows = [
        {"id": "q1", "question": "How many days of annual leave?", "answerable": True,
         "reference_answer": "Twenty.",
         "evidence": [{"doc": "data/sample_corpus/leave.md", "quote": "twenty days"}]},
        {"id": "u1", "question": "Who is the CEO?", "answerable": False,
         "reference_answer": "", "evidence": []},
    ]  # fmt: skip
    golden.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    gates = Path("gates.json")
    gates.write_text(
        json.dumps(
            [
                {
                    "metric": "recall_at_k",
                    "label": "Recall@5",
                    "op": ">=",
                    "threshold": 0.85,
                    "blocking": True,
                    "needs_answers": False,
                }
            ]  # fmt: skip
        ),
        encoding="utf-8",
    )
    # "Local" index: ingest + embed the questions once, so the cache holds every vector.
    local = Store(Path("local.db"))
    ingest(corpus, local, Path("local.index"), fake_embedder, "fake", RagConfig())
    queries = CachedQueryEmbedder(fake_embedder, local, "fake")
    for r in rows:
        queries.embed_query(r["question"])
    local.close()
    return {"golden": golden, "gates": gates, "corpus": corpus}


def _gate(setup, embedder, snapshot, baseline=Path("none.json")):
    return run_gate(
        RagConfig(similarity_threshold=0.0),
        embedder,
        "fake",
        snapshot=snapshot,
        baseline=baseline,
        corpus=setup["corpus"],
        golden=setup["golden"],
        gates_path=setup["gates"],
    )


def test_exported_snapshot_lets_the_gate_run_without_any_api(setup):
    snap = Path("snapshot.db")
    n = export_snapshot(Path("local.db"), snap, "fake", golden=setup["golden"])
    assert n == 4  # 2 chunks + 2 questions
    result = _gate(setup, OfflineEmbeddings(), snap)  # would raise on ANY embedding call
    assert result.passed and result.metrics["recall_at_k"] == 1.0
    assert [c.status for c in result.checks] == ["pass"]


def test_a_drop_below_the_baseline_fails_the_gate(setup):
    snap = Path("snapshot.db")
    export_snapshot(Path("local.db"), snap, "fake", golden=setup["golden"])
    baseline = Path("baseline.json")
    baseline.write_text(json.dumps({"recall_at_k": 1.0, "mrr": 1.5}), encoding="utf-8")
    result = _gate(setup, OfflineEmbeddings(), snap, baseline)
    assert not result.passed and result.drops and result.drops[0].startswith("mrr")


def test_missing_vectors_are_reported_not_failed(setup):
    with pytest.raises(MissingVectors):
        _gate(setup, OfflineEmbeddings(), Path("no-snapshot.db"))
