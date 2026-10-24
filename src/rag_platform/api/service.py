"""Shared API logic: run the answer pipeline and map provider failures to HTTP errors.

Used by both POST /ask (single question) and POST /chat (conversation), so the pipeline call and
the error handling exist exactly once.

SearchCache (C1, production-readiness plan): the FAISS vector index and the BM25 keyword index
stay in memory between requests instead of being loaded from disk / rebuilt from every
passage's text on each question. Before each use a cheap stamp - number of chunks, highest chunk
id (ids are never reused) and the FAISS file's modification time - is compared; any upload,
delete or `rag ingest` (also from the host while Docker runs) changes it and the next request
rebuilds. .NET analogy: IMemoryCache with a change token. One entry per collection (each has
its own index file), so a question only ever searches its own collection.
"""

import logging
import threading
from dataclasses import dataclass, field

from fastapi import HTTPException

from rag_platform.answering import build_answerer
from rag_platform.api.deps import Caller, Services, can_see
from rag_platform.api.schemas import Source
from rag_platform.config import DEFAULT_COLLECTION_ID, RagConfig, load_rag_config
from rag_platform.generation.analyse import analyse_question
from rag_platform.generation.answer import Answer
from rag_platform.ingestion.pipeline import open_index
from rag_platform.llm import provider_problem
from rag_platform.retrieval.bm25 import BM25Index
from rag_platform.retrieval.calibration import auto_threshold, effective_threshold
from rag_platform.retrieval.retriever import embed_query
from rag_platform.store.db import Store
from rag_platform.store.vector_index import VectorIndex

log = logging.getLogger("rag_platform.api")


@dataclass
class _Entry:
    """One collection's cached indexes and the stamp they were built for."""

    stamp: tuple
    index: VectorIndex
    bm25: BM25Index | None = None


@dataclass
class SearchCache:
    """In-memory FAISS + BM25 indexes for the API process, one entry per collection, rebuilt
    only when that collection's documents change."""

    _entries: dict[int, _Entry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    builds: int = 0  # how often an index was (re)built - for tests and diagnostics

    def get(
        self,
        store: Store,
        svc: Services,
        need_bm25: bool,
        collection_id: int = DEFAULT_COLLECTION_ID,
    ) -> tuple[VectorIndex, BM25Index | None]:
        path = svc.settings.collection_index_path(collection_id)
        mtime = path.stat().st_mtime_ns if path.exists() else 0
        stamp = (store.chunks_version(collection_id), mtime)
        with self._lock:  # requests run in a thread pool: one rebuild at a time
            entry = self._entries.get(collection_id)
            if entry is None or entry.stamp != stamp:
                index = open_index(store, path, svc.settings.embedding_model, collection_id)
                entry = self._entries[collection_id] = _Entry(stamp, index)  # BM25: lazily
                self.builds += 1
            if need_bm25 and entry.bm25 is None:
                entry.bm25 = BM25Index.from_store(store, collection_id)
            return entry.index, entry.bm25

    def forget(self, collection_id: int) -> None:
        """Drop a deleted collection's indexes."""
        with self._lock:
            self._entries.pop(collection_id, None)


def provider_error(e: Exception) -> HTTPException:
    """Map a model-provider failure to an HTTP error a client can act on.

    Quota/rate limits -> 429 Too Many Requests ("try again later", with Retry-After); a slow or
    overloaded provider (timeouts, 503/504, DEADLINE_EXCEEDED) -> 503 with a plain-language
    message; anything else -> 503. The provider's raw message is not echoed to clients.
    """
    problem = provider_problem(e)  # with fallback on, every provider had this problem
    if problem == "quota":
        return HTTPException(
            429,
            "Model quota exceeded (free tier limits). Try again later or switch the answer model "
            "in Admin › Settings.",
            headers={"Retry-After": "60"},
        )
    if problem == "slow":
        return HTTPException(
            503,
            "The AI service is slow or overloaded right now. Please try again in a minute.",
            headers={"Retry-After": "60"},
        )
    return HTTPException(503, f"Model provider error: {type(e).__name__}")


def recalibrate(svc: Services, store: Store, collection_id: int) -> None:
    """Measure a collection's similarity threshold again (after an upload or delete). No chat
    calls; the probe questions' vectors are cached. The demo collection keeps its measured value."""
    if collection_id == DEFAULT_COLLECTION_ID:
        return
    index, _ = svc.search.get(store, svc, False, collection_id)
    value = auto_threshold(store, index, svc.embedder, svc.settings.embedding_model)
    store.set_auto_threshold(collection_id, value)


def threshold_for(svc: Services, store: Store, collection: dict) -> tuple[float, str]:
    """The collection's effective threshold and its source (see retrieval/calibration.py);
    measures it first if it has documents but no value yet (collections from before thresholds)."""
    if (
        collection["id"] != DEFAULT_COLLECTION_ID
        and collection.get("auto_threshold") is None
        and store.chunk_ids(collection_id=collection["id"])
    ):
        recalibrate(svc, store, collection["id"])
        collection = store.get_collection(collection["id"]) or collection
    configured = load_rag_config(svc.config_path).similarity_threshold
    return effective_threshold(collection, configured)


def answer(
    svc: Services, cfg: RagConfig, question: str, collection_id: int = DEFAULT_COLLECTION_ID
) -> Answer:
    """Run the configured pipeline (fast, agentic or auto) for one question, searching only the
    given collection with that collection's similarity threshold."""
    store = svc.open_store()
    try:
        collection = store.get_collection(collection_id)
        if collection is not None and collection_id != DEFAULT_COLLECTION_ID:
            threshold, _ = threshold_for(svc, store, collection)
            cfg = cfg.model_copy(update={"similarity_threshold": threshold})  # frozen: a copy
        need_bm25 = cfg.retrieval == "hybrid"
        index, bm25 = svc.search.get(store, svc, need_bm25, collection_id)
        answerer = build_answerer(store, index, svc.embedder, svc.llm, cfg, bm25)
        try:
            ans = answerer(question)
        except Exception as e:  # noqa: BLE001 - provider/quota errors -> 429/503, not a crash
            error = provider_error(e)
            log.warning(
                "model provider error",
                extra={"error": type(e).__name__, "status": error.status_code},
            )
            raise error from e
        # The outcome without any text: enough to spot refusals and escalations in the log.
        log.info(
            "answer",
            extra={
                "route": ans.route or cfg.mode,
                "answerable": ans.answerable,
                "abstain_reason": ans.abstain_reason or None,
                "citations": len(ans.citations),
                "passages": len(ans.context),
            },
        )
        return ans
    finally:
        store.close()


def analyse(svc: Services, question: str, collection_id: int = DEFAULT_COLLECTION_ID) -> Answer:
    """Analyse mode: answer from every document of the collection (map-reduce, see
    generation/analyse.py), with the collection's similarity threshold."""
    store = svc.open_store()
    try:
        collection = store.get_collection(collection_id)
        threshold = load_rag_config(svc.config_path).similarity_threshold
        if collection is not None:
            threshold, _ = threshold_for(svc, store, collection)
        index, _ = svc.search.get(store, svc, False, collection_id)
        try:
            ans = analyse_question(question, store, index, svc.embedder, svc.llm, threshold)
        except Exception as e:  # noqa: BLE001 - provider/quota errors -> 429/503, not a crash
            error = provider_error(e)
            log.warning(
                "model provider error",
                extra={"error": type(e).__name__, "status": error.status_code},
            )
            raise error from e
        log.info(
            "answer",
            extra={
                "route": ans.route,
                "answerable": ans.answerable,
                "abstain_reason": ans.abstain_reason or None,
                "citations": len(ans.citations),
                "passages": len(ans.context),
            },
        )
        return ans
    finally:
        store.close()


def collections_with_matches(
    svc: Services, question: str, collection_id: int, caller: Caller | None = None
) -> list[dict]:
    """Other collections THE CALLER MAY SEE with a passage scoring at least their threshold.

    Used after "I couldn't find this" (no passage of the chat's collection reached the
    threshold): the chat then offers "Ask in <collection>" instead of a dead end. One embedding
    call and a vector search per collection (milliseconds) - no chat call, and only the
    collections' names are returned, never their text - and someone else's private collection
    is never even checked.
    """
    store = svc.open_store()
    try:
        others = [
            c
            for c in store.list_collections()
            if c["id"] != collection_id and c["passages"] and can_see(caller, c)
        ]
        if not others:
            return []
        vec = embed_query(svc.embedder, question)
        found = []
        for c in others:
            index, _ = svc.search.get(store, svc, False, c["id"])
            hits = index.search(vec, 1)
            if hits and hits[0][1] >= threshold_for(svc, store, c)[0]:
                found.append({"id": c["id"], "name": c["name"]})
        return found
    finally:
        store.close()


def sources_of(ans: Answer) -> list[Source]:
    """The passages the answer cites, numbered as in the answer text."""
    return [
        Source(
            number=n,
            document=c.doc_path,
            start=c.start,
            end=c.end,
            score=round(c.score, 4),
            text=c.text,
        )
        for n, c in zip(ans.citations, ans.cited_chunks, strict=True)
    ]
