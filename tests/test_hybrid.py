"""Tests for hybrid retrieval (BM25 + vector, fused with RRF), query caching and the active
config file. Fake embedder only - no API calls."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from rag_platform.config import RagConfig, load_rag_config, save_rag_config
from rag_platform.ingestion.pipeline import ingest, open_index
from rag_platform.retrieval.bm25 import BM25Index, tokenize
from rag_platform.retrieval.query_cache import CachedQueryEmbedder
from rag_platform.retrieval.retriever import build_retriever, hybrid_retrieve, rrf_fuse
from rag_platform.store.db import Store
from rag_platform.store.vector_index import VectorIndex, normalize


def test_tokenize_keeps_codes_together():
    assert tokenize("Error LUM-2001 in gemini-3.5, top_k!") == [
        "error", "lum-2001", "in", "gemini-3.5", "top_k"
    ]  # fmt: skip


def test_bm25_ranks_exact_code_first():
    idx = BM25Index([1, 2, 3], ["general timeout policy", "LUM-2001 gateway timeout", "other"])
    hits = idx.search("what is LUM-2001", 3)
    assert hits[0][0] == 2
    assert all(cid != 3 for cid, _ in hits)  # no shared word -> not returned


def test_rrf_rewards_agreement():
    # 7 is 2nd in both lists; 1 and 9 are 1st in one list only -> 7 wins.
    assert rrf_fuse([[1, 7, 3], [9, 7, 4]])[0] == 7


def test_vector_index_returns_stored_vectors():
    vecs = normalize(np.eye(3))
    idx = VectorIndex()
    idx.add([10, 11, 12], vecs)
    idx.remove([11])
    got = idx.vectors([10, 11, 12])
    assert set(got) == {10, 12} and np.allclose(got[12], vecs[2])


@pytest.fixture
def indexed(tmp_path: Path, fake_embedder):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "codes.md").write_text("Error code LUM-2001 means gateway timeout.", encoding="utf-8")
    (corpus / "leave.md").write_text("Employees get twenty days annual leave.", encoding="utf-8")
    (corpus / "vpn.md").write_text("Connect to the VPN before using servers.", encoding="utf-8")
    store = Store(tmp_path / "rag.db")
    ingest(corpus, store, tmp_path / "f.index", fake_embedder, "fake", RagConfig())
    yield store, open_index(store, tmp_path / "f.index", "fake")
    store.close()


def test_hybrid_returns_cosine_scores_and_finds_code(indexed, fake_embedder):
    store, index = indexed
    hits = hybrid_retrieve(
        "LUM-2001", store, index, fake_embedder, BM25Index.from_store(store), 2, 3
    )
    assert hits[0].text.startswith("Error code LUM-2001")
    q = normalize(np.array([fake_embedder.embed_query("LUM-2001")]))[0]
    for h in hits:  # score is the real cosine similarity, also for BM25-only hits
        vec = index.vectors([h.chunk_id])[h.chunk_id]
        assert h.score == pytest.approx(float(vec @ q), abs=1e-5)


def test_build_retriever_respects_mode_and_top_k(indexed, fake_embedder):
    store, index = indexed
    dense = build_retriever(store, index, fake_embedder, RagConfig(top_k=1))
    hybrid = build_retriever(store, index, fake_embedder, RagConfig(top_k=2, retrieval="hybrid"))
    assert len(dense("annual leave")) == 1
    assert len(hybrid("annual leave")) == 2


def test_query_cache_embeds_each_question_once(tmp_path: Path, fake_embedder):
    store = Store(tmp_path / "rag.db")
    cached = CachedQueryEmbedder(fake_embedder, store, "fake")
    first = cached.embed_query("how many days?")
    second = cached.embed_query("how many days?")
    assert cached.api_calls == 1 and np.allclose(first, second)
    store.close()


def test_config_file_round_trip_and_overrides(tmp_path: Path):
    path = tmp_path / "rag_config.json"
    assert load_rag_config(path) == RagConfig()  # missing file -> code defaults
    save_rag_config(RagConfig(top_k=7, retrieval="hybrid"), path)
    cfg = load_rag_config(path, top_k=None, mode="agentic")  # None = keep the file's value
    assert (cfg.top_k, cfg.retrieval, cfg.mode) == (7, "hybrid", "agentic")
    with pytest.raises(ValidationError):
        load_rag_config(path, retrieval="sparse")
