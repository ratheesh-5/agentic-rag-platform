"""Cache for QUESTION embeddings, so repeated evaluations don't re-embed the same questions.

The optimizer runs many experiments over the same 50 golden questions. Without a cache every
experiment would embed all 50 questions again (50 API calls each). Question vectors only depend
on the question text and the embedding model, so they are stored in SQLite's embedding_cache
under a separate key ("<model>#query") - query and document embeddings use different task
types, so they must never be mixed.

This is a decorator around any LangChain `Embeddings` (.NET analogy: a caching decorator around
an interface). Document embedding passes straight through; the ingest pipeline has its own cache.
"""

import numpy as np
from langchain_core.embeddings import Embeddings

from rag_platform.ingestion.chunking import text_hash
from rag_platform.store.db import Store
from rag_platform.store.vector_index import normalize


class CachedQueryEmbedder(Embeddings):
    def __init__(self, embedder: Embeddings, store: Store, model: str):
        self.embedder = embedder
        self.store = store
        self.key = f"{model}#query"
        self.api_calls = 0  # cache misses = real embedding calls

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        h = text_hash(text)
        cached = self.store.get_vectors([h], self.key)
        if h in cached:
            return cached[h].tolist()
        vec = normalize(np.array([self.embedder.embed_query(text)]))[0]
        self.store.put_vectors({h: vec}, self.key)
        self.api_calls += 1
        return vec.tolist()
