"""BM25 keyword search over all chunks - the "lexical" half of hybrid retrieval.

Vector (dense) search matches MEANING but can miss exact tokens: error codes like "LUM-2001",
names, or one row of a table. BM25 scores a chunk by how often the query's words occur in it,
weighted by how rare each word is across all chunks (rare words like "LUM-2001" count a lot,
common words like "the" almost nothing), with a correction for chunk length.

The index is built in memory from the chunks in SQLite. For a corpus of a few hundred chunks
that takes milliseconds, so it is rebuilt per process instead of being stored.
"""

import re

from rank_bm25 import BM25Okapi

from rag_platform.config import DEFAULT_COLLECTION_ID
from rag_platform.store.db import Store

# Words, keeping hyphenated/dotted codes together: "LUM-2001", "gemini-3.5", "top_k".
TOKEN = re.compile(r"\w+(?:[-.]\w+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunk_ids: list[int], texts: list[str]):
        self.chunk_ids = chunk_ids
        # BM25Okapi needs a non-empty corpus; an empty index simply returns no hits.
        self.bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    @classmethod
    def from_store(cls, store: Store, collection_id: int = DEFAULT_COLLECTION_ID) -> "BM25Index":
        """Keyword index over one collection's passages (the same isolation as its FAISS file)."""
        rows = store.chunk_texts(collection_id)
        return cls([r[0] for r in rows], [r[1] for r in rows])

    def search(self, query: str, n: int) -> list[tuple[int, float]]:
        """Top-n (chunk_id, bm25 score), best first. Chunks sharing no query word are skipped."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(self.chunk_ids[i], float(scores[i])) for i in best if scores[i] > 0]
