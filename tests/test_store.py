"""Tests for incremental indexing (SQLite + FAISS + embedding cache), using FakeEmbedder.

`tmp_path` is a built-in pytest fixture: a fresh temporary directory per test. The `env`
fixture below uses `yield`: code before it is setup, code after it is teardown (like
IDisposable / xUnit's IAsyncLifetime).
"""

from pathlib import Path

import numpy as np
import pytest

from rag_platform.config import RagConfig
from rag_platform.ingestion.pipeline import delete, ingest, open_index
from rag_platform.store.db import Store
from rag_platform.store.vector_index import VectorIndex, normalize

CFG = RagConfig(chunk_size=120, chunk_overlap=20)
MODEL = "fake"


@pytest.fixture
def env(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("Alpha policy. " * 30, encoding="utf-8")
    (corpus / "b.txt").write_text("Bravo travel rules. " * 30, encoding="utf-8")
    store = Store(tmp_path / "idx" / "rag.db")
    yield corpus, store, tmp_path / "idx" / "faiss.index"
    store.close()


def _in_sync(store: Store, index_path: Path) -> bool:
    return VectorIndex.load(index_path).ids == set(store.chunk_ids())


def test_first_ingest_indexes_everything(env, fake_embedder):
    corpus, store, index_path = env
    st = ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    assert (st.added, st.updated, st.unchanged) == (2, 0, 0)
    assert store.counts() == (2, st.chunks_added)
    assert _in_sync(store, index_path)


def test_reingest_unchanged_skips_and_embeds_nothing(env, fake_embedder):
    corpus, store, index_path = env
    ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    fake_embedder.calls.clear()
    st = ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    assert (st.added, st.updated, st.unchanged) == (0, 0, 2)
    assert fake_embedder.calls == []


def test_changed_doc_replaces_chunks_and_reuses_cache(env, fake_embedder):
    corpus, store, index_path = env
    ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    (corpus / "a.md").write_text(
        "Alpha policy. " * 30 + "\n\nNew closing section.", encoding="utf-8"
    )
    fake_embedder.calls.clear()
    st = ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    assert (st.updated, st.unchanged) == (1, 1)
    assert st.cache_hits > 0  # identical leading chunks are not re-embedded
    assert len(fake_embedder.calls) < st.chunks_added
    assert _in_sync(store, index_path)


def test_chunking_config_change_rechunks(env, fake_embedder):
    corpus, store, index_path = env
    ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    st = ingest(corpus, store, index_path, fake_embedder, MODEL, RagConfig(chunk_size=200))
    assert st.updated == 2
    assert _in_sync(store, index_path)


def test_missing_index_is_rebuilt_from_sqlite(env, fake_embedder):
    corpus, store, index_path = env
    ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    index_path.unlink()
    assert open_index(store, index_path, MODEL).ids == set(store.chunk_ids())


def test_removed_file_is_deleted_on_directory_ingest(env, fake_embedder):
    corpus, store, index_path = env
    ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    (corpus / "b.txt").unlink()
    st = ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    assert (st.deleted, st.unchanged) == (1, 1)
    assert st.chunks_removed > 0
    assert store.counts()[0] == 1
    assert _in_sync(store, index_path)


def test_single_file_ingest_does_not_delete_siblings(env, fake_embedder):
    corpus, store, index_path = env
    ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    st = ingest(corpus / "a.md", store, index_path, fake_embedder, MODEL, CFG)
    assert (st.deleted, st.unchanged) == (0, 1)
    assert store.counts()[0] == 2


def test_delete_document(env, fake_embedder):
    corpus, store, index_path = env
    ingest(corpus, store, index_path, fake_embedder, MODEL, CFG)
    assert delete(corpus / "a.md", store, index_path, MODEL) > 0
    assert delete(corpus / "a.md", store, index_path, MODEL) is None
    assert store.counts()[0] == 1
    assert _in_sync(store, index_path)


def test_vector_index_search_and_remove():
    vecs = normalize(np.eye(4))
    idx = VectorIndex()
    idx.add([10, 11, 12, 13], vecs)
    assert idx.search(vecs[2], k=1) == [(12, pytest.approx(1.0))]
    idx.remove([12])
    assert 12 not in [i for i, _ in idx.search(vecs[2], k=4)]
