"""Similarity threshold per collection, set automatically (plan section I, phase 1).

Why: the threshold decides "is anything in these documents about this question?" before any
chat call (below it: "I couldn't find this", free and safe). Scores depend on the documents -
a short resume scores lower than a long handbook - so one global value (0.68, measured on the
demo collection) refused answerable resume questions (found in testing, 2026-10-24: 0.64-0.675).
Users can't be expected to pick such a number, so the system measures it.

How (no chat calls): ask the collection 12 clearly unrelated questions ("What is the capital of
France?") and see how high unrelated questions score; the threshold goes just above the highest
of them. Measured on the owner's data: AI handbook 0.61, HR policies 0.58, Resume 0.58 - every
real question scored above its collection's value. The probe vectors are embedded once and
cached (CachedQueryEmbedder), so re-measuring after each upload or delete costs nothing.

Which value wins (`effective_threshold`):
1. the demo collection keeps its MEASURED value from rag_config.json (0.68, from a test set with
   tricky near-topic questions - stricter than the probes can see);
2. an admin's manual override;
3. the automatic value;
4. the global value, until the first measurement.
Limit: probes catch clearly off-topic questions. A near-topic question the documents don't answer
passes the threshold; the model then still says "not in the documents" (one chat call, safe).
"""

from langchain_core.embeddings import Embeddings

from rag_platform.config import DEFAULT_COLLECTION_ID
from rag_platform.retrieval.query_cache import CachedQueryEmbedder
from rag_platform.store.db import Store
from rag_platform.store.vector_index import VectorIndex

PROBES = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the football world cup?",
    "What is the weather like tomorrow?",
    "How tall is Mount Everest?",
    "Recommend a good movie to watch",
    "How do I change a car tyre?",
    "What is the price of gold today?",
    "Tell me a joke",
    "How many legs does a spider have?",
    "What is photosynthesis?",
    "Translate hello into Spanish",
]
MARGIN = 0.01  # the threshold sits just above the best-scoring unrelated question


def auto_threshold(
    store: Store, index: VectorIndex, embedder: Embeddings, model: str
) -> float | None:
    """Highest score any probe question reaches in this index + MARGIN; None if it's empty."""
    if not index.ids:
        return None
    queries = CachedQueryEmbedder(embedder, store, model)
    best = 0.0
    for question in PROBES:
        hits = index.search(queries.embed_query(question), 1)  # cached vectors are normalized
        if hits:
            best = max(best, hits[0][1])
    return round(best + MARGIN, 2)


def effective_threshold(collection: dict, configured: float) -> tuple[float, str]:
    """(threshold, where it comes from) for a collection row; `configured` = rag_config.json."""
    if collection["id"] == DEFAULT_COLLECTION_ID:
        return configured, "measured"
    if collection.get("threshold") is not None:
        return collection["threshold"], "manual"
    if collection.get("auto_threshold") is not None:
        return collection["auto_threshold"], "automatic"
    return configured, "default"
