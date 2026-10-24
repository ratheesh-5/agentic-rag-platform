"""Experiments - the optimizer's tool. Plain, deterministic Python: no LLM here.

An experiment = "evaluate this RagConfig on the golden set, retrieval-only":
1. Pick the index: the main index if its chunking matches the config, otherwise an experiment
   index under data/index/experiments/chunk-<size>-<overlap>/ (built once, then reused).
   Experiment indexes share the embedding cache with the main database in both directions, so
   identical chunk texts are never embedded twice, and applying a winning chunk size later
   re-indexes the main index without any embedding calls.
2. Run the golden set through retrieval only (no chat calls; question vectors are cached).
3. Save it as a normal eval run (so `rag eval list/compare` show experiments too) and return a
   compact summary for the agents.

`objective()` defines "better" in code, not by an LLM: Recall@K first (did the evidence reach
the LLM?), then MRR (was it near the top?), then how many unanswerable questions the safe
similarity threshold blocks. The same inputs always give the same winner.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag_platform.config import DEFAULT_COLLECTION_ID, RagConfig, Settings, save_rag_config
from rag_platform.evaluation.golden import load_golden
from rag_platform.evaluation.runner import retrieval_only_answerer, run_eval, run_metrics
from rag_platform.ingestion.chunking import text_hash
from rag_platform.ingestion.pipeline import ingest, open_index
from rag_platform.retrieval.query_cache import CachedQueryEmbedder
from rag_platform.retrieval.retriever import build_retriever
from rag_platform.store.db import Store

# Metrics the agents see (the full metrics are in the saved eval run).
KEY_METRICS = [
    "recall_at_k",
    "mrr",
    "precision_at_k",
    "hit_rate_at_k",
    "safe_threshold",
    "unanswerable_below_safe",
    "top_score_answerable",
    "top_score_unanswerable",
]


@dataclass
class Experiment:
    label: str
    changes: dict  # settings that differ from the base config
    config: dict  # full RagConfig
    metrics: dict  # KEY_METRICS only
    run_id: int
    failures: list[dict] = field(default_factory=list)  # questions retrieval got wrong-ish


def objective(metrics: dict) -> tuple[float, float, float]:
    """Sort key for "better" (higher tuple wins): recall, then MRR, then unanswerables blocked."""
    return (
        metrics.get("recall_at_k") or 0.0,
        metrics.get("mrr") or 0.0,
        metrics.get("unanswerable_below_safe") or 0.0,
    )


def _failures(results) -> list[dict]:
    """Answerable questions whose evidence was missed or not ranked first, for the diagnosis."""
    out = []
    for r in results:
        if r.answerable and r.retrieval and r.retrieval["rr"] < 1.0:
            rank = round(1 / r.retrieval["rr"]) if r.retrieval["rr"] else None
            top_doc = r.retrieved[0]["doc"].rsplit("/", 1)[-1] if r.retrieved else None
            out.append(
                {"id": r.id, "question": r.question, "evidence_rank": rank, "top_doc": top_doc}
            )
    return out


class ExperimentRunner:
    def __init__(
        self,
        settings: Settings,
        embedder: Embeddings,
        golden_path: Path,
        base: RagConfig,
        limit: int = 0,
        log: Callable[[str], None] = lambda _msg: None,
    ):
        self.s = settings
        self.embedder = embedder
        self.golden_path = golden_path
        self.base = base
        self.limit = limit
        self.log = log
        self.main = Store(settings.db_path)
        self._done: dict[str, Experiment] = {}  # config JSON -> result: never run twice

    def close(self) -> None:
        self.main.close()

    def changes(self, cfg: RagConfig) -> dict:
        base = self.base.model_dump()
        return {k: v for k, v in cfg.model_dump().items() if base[k] != v}

    def _chunker(self, cfg: RagConfig) -> str:
        return f"{cfg.chunk_size}/{cfg.chunk_overlap}"

    def _index_for(self, cfg: RagConfig) -> tuple[Store, Path]:
        """Main index if the chunking matches, else a (cached) experiment index. Experiments
        use the demo collection only: the golden set belongs to it."""
        if self.main.chunkers(DEFAULT_COLLECTION_ID) == {self._chunker(cfg)}:
            return self.main, self.s.faiss_path
        folder = self.s.index_dir / "experiments" / f"chunk-{cfg.chunk_size}-{cfg.chunk_overlap}"
        store = Store(folder / "rag.db")
        store.copy_embeddings_from(self.s.db_path)  # reuse every vector we already paid for
        docs = [Path(p) for p in self.main.document_paths(DEFAULT_COLLECTION_ID)]
        stats = ingest(
            docs, store, folder / "faiss.index", self.embedder, self.s.embedding_model, cfg
        )
        if stats.embedded:
            self.log(f"  indexed chunk {self._chunker(cfg)}: {stats.embedded} new embeddings")
        self.main.copy_embeddings_from(folder / "rag.db")  # share new vectors back
        return store, folder / "faiss.index"

    def run(self, cfg: RagConfig, label: str, experiment: bool = True) -> Experiment:
        """experiment=False: a plain "quality check" (admin UI), not an optimizer try."""
        key = cfg.model_dump_json()
        if key in self._done:
            return self._done[key]
        store, index_path = self._index_for(cfg)
        try:
            index = open_index(store, index_path, self.s.embedding_model)
            queries = CachedQueryEmbedder(self.embedder, self.main, self.s.embedding_model)
            retriever = build_retriever(store, index, queries, cfg)
            items = load_golden(self.golden_path, store.document_texts())
            if self.limit:
                items = items[: self.limit]
            answerer = retrieval_only_answerer(retriever, cfg.similarity_threshold)
            results, error = run_eval(items, answerer, cfg.top_k)
            if error:
                raise RuntimeError(f"experiment {label} failed: {error}")
            metrics = run_metrics(results, cfg.similarity_threshold, len(items), None)
            config = {
                "rag": cfg.model_dump(),
                "retrieval_only": True,
                "experiment": experiment,
                "chat_model": None,
                "embedding_model": self.s.embedding_model,
                "judge_model": None,
                "golden": self.golden_path.as_posix(),
                "golden_hash": text_hash(self.golden_path.read_text(encoding="utf-8")),
                "n_questions": len(items),
            }
            run_id = self.main.save_eval_run(label, config, metrics, [asdict(r) for r in results])
        finally:
            if store is not self.main:
                store.close()
        exp = Experiment(
            label=label,
            changes=self.changes(cfg),
            config=cfg.model_dump(),
            metrics={k: metrics.get(k) for k in KEY_METRICS},
            run_id=run_id,
            failures=_failures(results),
        )
        self._done[key] = exp
        return exp

    def apply(self, cfg: RagConfig, config_path: Path) -> bool:
        """Write the approved config; re-index the main index if the chunking changed.
        Returns True if a re-index happened (it reuses cached vectors: no embedding calls)."""
        reindex = self.main.chunkers(DEFAULT_COLLECTION_ID) != {self._chunker(cfg)}
        save_rag_config(cfg, config_path)
        if reindex:  # the demo collection; other collections keep their chunking (phase 4)
            docs = [Path(p) for p in self.main.document_paths(DEFAULT_COLLECTION_ID)]
            ingest(docs, self.main, self.s.faiss_path, self.embedder, self.s.embedding_model, cfg)
        return reindex
