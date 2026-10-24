"""Retrieval: given a question, find the most relevant chunks (the "R" in RAG).

Dense retrieval = semantic search:
  1. Embed the question with the SAME embedding model used for the chunks (vectors from
     different models live in different "spaces" and can't be compared).
  2. Ask FAISS for the top_k chunk vectors with the highest cosine similarity.
  3. Load those chunks' text and source from SQLite.

Hybrid retrieval (RagConfig.retrieval = "hybrid") adds BM25 keyword search (bm25.py) for exact
tokens that dense search can miss (codes, names, table rows). Both retrievers return a ranked
candidate list; Reciprocal Rank Fusion (RRF) merges them by RANK, not score, because cosine
similarities and BM25 scores are on different scales:

    rrf_score(chunk) = sum over retrievers of 1 / (60 + rank of chunk in that retriever)

A chunk ranked high by either retriever (or decently by both) ends up near the top. 60 is the
conventional constant that keeps the very first ranks from dominating.

Every returned chunk keeps its COSINE score (also BM25-only hits), so the abstention threshold
means the same thing in both modes. In hybrid mode the list is ordered by RRF, not by score.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from langchain_core.embeddings import Embeddings

from rag_platform.config import RagConfig
from rag_platform.retrieval.bm25 import BM25Index
from rag_platform.store.db import Store
from rag_platform.store.vector_index import VectorIndex, normalize

RRF_K = 60


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    doc_path: str
    start: int  # char offsets into the document's normalized text
    end: int
    text: str
    score: float  # cosine similarity to the query


def embed_query(embedder: Embeddings, query: str) -> np.ndarray:
    # embed_query (not embed_documents): Gemini uses a query-specific task type for questions.
    # normalize() expects a 2-D batch, so wrap in [...] and take row [0] back out.
    return normalize(np.array([embedder.embed_query(query)]))[0]


def hydrate(store: Store, hits: list[tuple[int, float]]) -> list[RetrievedChunk]:
    """(chunk_id, score) pairs -> RetrievedChunks with text and source, in the given order."""
    rows = store.get_chunks([cid for cid, _ in hits])
    return [
        RetrievedChunk(
            chunk_id=cid,
            doc_path=rows[cid]["path"],
            start=rows[cid]["start_char"],
            end=rows[cid]["end_char"],
            text=rows[cid]["text"],
            score=score,
        )
        for cid, score in hits
        if cid in rows  # defensive: skip a hit whose chunk is no longer in SQLite
    ]


def retrieve(
    query: str, store: Store, index: VectorIndex, embedder: Embeddings, top_k: int
) -> list[RetrievedChunk]:
    """Dense retrieval: embed the query, search FAISS, hydrate chunks from SQLite. Best first."""
    return hydrate(store, index.search(embed_query(embedder, query), top_k))


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Merge several best-first id lists with Reciprocal Rank Fusion; returns ids best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def hybrid_retrieve(
    query: str,
    store: Store,
    index: VectorIndex,
    embedder: Embeddings,
    bm25: BM25Index,
    top_k: int,
    candidates: int,
) -> list[RetrievedChunk]:
    """Dense + BM25 candidates, fused with RRF; top_k chunks in fused order, cosine scores."""
    qvec = embed_query(embedder, query)
    dense = index.search(qvec, candidates)
    lexical = bm25.search(query, candidates)
    fused = rrf_fuse([[cid for cid, _ in dense], [cid for cid, _ in lexical]])[:top_k]
    cosine = dict(dense)
    missing = [cid for cid in fused if cid not in cosine]  # found by BM25 only
    for cid, vec in index.vectors(missing).items():
        cosine[cid] = float(vec @ qvec)  # dot product of normalized vectors = cosine
    return hydrate(store, [(cid, cosine.get(cid, 0.0)) for cid in fused])


def build_retriever(
    store: Store,
    index: VectorIndex,
    embedder: Embeddings,
    cfg: RagConfig,
    bm25: BM25Index | None = None,
) -> Callable[[str], list[RetrievedChunk]]:
    """Return a `query -> chunks` function for the configured retrieval mode.
    `bm25`: a prebuilt keyword index (the API keeps one in memory); built here if not given."""
    if cfg.retrieval == "hybrid":
        bm25 = bm25 or BM25Index.from_store(store)  # built once, reused for every query
        return lambda q: hybrid_retrieve(
            q, store, index, embedder, bm25, cfg.top_k, cfg.hybrid_candidates
        )
    return lambda q: retrieve(q, store, index, embedder, cfg.top_k)
