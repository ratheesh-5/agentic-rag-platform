"""Indexing pipeline: load -> chunk -> embed -> store (SQLite) -> index (FAISS), incrementally.

Why incremental? Embedding calls cost money and time. Re-running `rag ingest` should only do
work for what actually changed:

  document unchanged (same content hash AND same chunking config)  -> skip entirely
  document new or changed                                          -> replace all its chunks
  chunk text already embedded before (embedding cache hit)         -> no API call
  file deleted from the ingested folder                            -> remove from the index

Order matters for safety: we call the embedding API BEFORE writing to SQLite, so an API failure
leaves the database untouched. FAISS is saved in a `finally` block so it always matches what
SQLite committed; if it still drifts (e.g. a crash), `open_index` rebuilds it from SQLite.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from rag_platform.config import DEFAULT_COLLECTION_ID, RagConfig
from rag_platform.ingestion.chunking import Chunk, chunk_text, text_hash
from rag_platform.ingestion.loaders import load_corpus, load_document
from rag_platform.store.db import Store
from rag_platform.store.vector_index import VectorIndex, normalize


# Not frozen: counters are incremented while ingesting (a mutable DTO).
@dataclass
class IngestStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0  # docs whose source file no longer exists under the ingested directory
    chunks_added: int = 0
    chunks_removed: int = 0
    embedded: int = 0  # texts sent to the embedding API
    cache_hits: int = 0


def doc_key(path: Path) -> str:
    """Stable document id: path relative to the working directory when possible."""
    p = path.resolve()
    try:
        return p.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return p.as_posix()


def open_index(
    store: Store, path: Path, model: str, collection_id: int = DEFAULT_COLLECTION_ID
) -> VectorIndex:
    """Load one collection's FAISS index; rebuild it from SQLite if missing or out of sync.

    Rebuilding is cheap: all vectors are in the embedding cache, so no API calls are needed.
    Each collection has its own index file (Settings.collection_index_path) holding only that
    collection's passages, so a search can never return another collection's text.
    """
    index = VectorIndex.load(path)
    if index.ids != set(store.chunk_ids(collection_id=collection_id)):
        index = VectorIndex()
        ids, vecs = store.chunk_vectors(model, collection_id)
        if vecs is not None:
            index.add(ids, vecs)
    return index


def embed_chunks(
    chunks: list[Chunk], store: Store, embedder: Embeddings, model: str, stats: IngestStats
) -> np.ndarray:
    """Return one normalized vector per chunk, calling the embedding API only for cache misses."""
    hashes = [c.text_hash for c in chunks]
    vectors = store.get_vectors(hashes, model)
    # Dict comprehension keyed by hash also de-duplicates identical chunk texts.
    missing = {c.text_hash: c.text for c in chunks if c.text_hash not in vectors}
    stats.cache_hits += len(set(hashes)) - len(missing)
    if missing:
        # One batched API call for all missing texts; normalize so dot product == cosine.
        new = normalize(np.array(embedder.embed_documents(list(missing.values()))))
        fresh = dict(zip(missing.keys(), new, strict=True))
        store.put_vectors(fresh, model)
        vectors |= fresh  # merge dicts (like adding all entries of one Dictionary to another)
        stats.embedded += len(missing)
    return np.stack([vectors[h] for h in hashes])  # matrix in the same order as `chunks`


def ingest(
    root: Path | list[Path],
    store: Store,
    index_path: Path,
    embedder: Embeddings,
    model: str,
    cfg: RagConfig,
    collection_id: int = DEFAULT_COLLECTION_ID,
) -> IngestStats:
    """Incrementally index a file, a directory or a list of files: skip unchanged docs, replace
    changed ones and, for a directory, delete docs whose files are gone from it.

    A list of files (used to re-index known documents with another chunking config, e.g. for
    optimizer experiments) never prunes anything. Everything happens inside one collection:
    `index_path` must be that collection's index file."""
    stats = IngestStats()
    # Stored per document: changing chunk size/overlap must re-chunk even unchanged files,
    # otherwise the index would silently mix chunks made with different settings.
    chunker = f"{cfg.chunk_size}/{cfg.chunk_overlap}"
    index = open_index(store, index_path, model, collection_id)
    seen: set[str] = set()
    docs = load_corpus(root) if isinstance(root, Path) else [load_document(p) for p in root]
    try:
        for doc in docs:
            key = doc_key(doc.path)
            seen.add(key)
            content_hash = text_hash(doc.text)
            existing = store.get_document(key)
            if (
                existing
                and existing["content_hash"] == content_hash
                and existing["chunker"] == chunker
            ):
                stats.unchanged += 1
                continue
            chunks = chunk_text(doc.text, cfg)
            vecs = embed_chunks(chunks, store, embedder, model, stats)  # before DB write
            old_ids, new_ids = store.replace_document(
                key, doc.doc_type, content_hash, chunker, doc.text, chunks, collection_id
            )
            index.remove(old_ids)
            index.add(new_ids, vecs)
            stats.updated += bool(existing)  # bool counts as 0/1 in Python
            stats.added += not existing
            stats.chunks_removed += len(old_ids)
            stats.chunks_added += len(new_ids)
        # Prune: only for a directory, and only documents that live under that directory.
        if isinstance(root, Path) and root.is_dir():
            prefix = "" if doc_key(root) == "." else doc_key(root) + "/"
            for key in store.document_paths(collection_id):
                if key.startswith(prefix) and key not in seen:
                    removed = store.delete_document(key) or []
                    index.remove(removed)
                    stats.deleted += 1
                    stats.chunks_removed += len(removed)
    finally:
        index.save(index_path)  # keep FAISS in step with whatever SQLite committed
    return stats


def delete(
    path: Path,
    store: Store,
    index_path: Path,
    model: str,
    collection_id: int = DEFAULT_COLLECTION_ID,
) -> int | None:
    """Remove one document from SQLite and its collection's FAISS index. Returns chunks removed,
    None if not indexed."""
    index = open_index(store, index_path, model, collection_id)
    removed = store.delete_document(doc_key(path))
    if removed is None:
        return None
    index.remove(removed)
    index.save(index_path)
    return len(removed)
