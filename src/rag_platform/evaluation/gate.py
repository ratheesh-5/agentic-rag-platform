"""CI evaluation gate (Phase 7): fail the build when retrieval quality drops.

`rag gate` builds a fresh index of the sample corpus in a temp folder, runs the retrieval-only
evaluation of the golden set with the ACTIVE rag_config.json, and fails (exit code 1) if:
- a release gate is missed (data/eval/release_gates.json: Recall@5 >= 0.85, MRR >= 0.70), or
- a metric dropped more than 0.02 below the recorded baseline (data/eval/gate_baseline.json),
  the playbook's "no blocking metric may drop by more than 2 points" rule.

No API key needed: vectors come from a committed snapshot of the embedding cache
(data/eval/embedding_snapshot.db - the sample-corpus chunks and the golden questions). Only a
change that creates NEW texts (e.g. another chunk size) needs embeddings: with GEMINI_API_KEY
set they are fetched, without it the gate is skipped with a warning (never a false failure).
`rag gate --export` refreshes the snapshot from the local index; `--update-baseline` records
the current metrics as the new baseline (both reviewed in a pull request like any change).

Only the deterministic retrieval metrics are gated: answer metrics (faithfulness, correctness)
need many LLM judge calls, which CI should not spend on every push.
"""

import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag_platform.config import RagConfig
from rag_platform.evaluation.gates import GateCheck, check_gates, load_gates, overall
from rag_platform.evaluation.golden import load_golden
from rag_platform.evaluation.runner import retrieval_only_answerer, run_eval, run_metrics
from rag_platform.ingestion.chunking import text_hash
from rag_platform.ingestion.pipeline import ingest, open_index
from rag_platform.retrieval.query_cache import CachedQueryEmbedder
from rag_platform.retrieval.retriever import build_retriever
from rag_platform.store.db import Store

SNAPSHOT = Path("data/eval/embedding_snapshot.db")
BASELINE = Path("data/eval/gate_baseline.json")
CORPUS = Path("data/sample_corpus")
GOLDEN = Path("data/eval/golden.jsonl")
MAX_DROP = 0.02  # playbook: no blocking metric may drop more than 2 points
GATED = ("recall_at_k", "mrr")


class MissingVectors(RuntimeError):
    """A text has no vector in the snapshot and no API key is available to embed it."""


class OfflineEmbeddings(Embeddings):
    """Embeddings that never call an API: every request means the snapshot is incomplete."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise MissingVectors(f"{len(texts)} chunk text(s) not in the snapshot")

    def embed_query(self, text: str) -> list[float]:
        raise MissingVectors(f"question not in the snapshot: {text[:60]!r}")


@dataclass
class GateResult:
    passed: bool
    metrics: dict
    checks: list[GateCheck]
    drops: list[str]  # metrics that fell more than MAX_DROP below the baseline


def run_gate(
    cfg: RagConfig,
    embedder: Embeddings,
    model: str,
    snapshot: Path = SNAPSHOT,
    baseline: Path = BASELINE,
    corpus: Path = CORPUS,
    golden: Path = GOLDEN,
    gates_path: Path | None = None,
) -> GateResult:
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "rag.db")
        try:
            if snapshot.exists():
                store.copy_embeddings_from(snapshot)  # every vector we already paid for
            ingest(corpus, store, Path(tmp) / "faiss.index", embedder, model, cfg)
            index = open_index(store, Path(tmp) / "faiss.index", model)
            queries = CachedQueryEmbedder(embedder, store, model)
            retriever = build_retriever(store, index, queries, cfg)
            items = load_golden(golden, store.document_texts())
            results, error = run_eval(items, retrieval_only_answerer(retriever, 0.0), cfg.top_k)
        finally:
            store.close()
    if error:
        if "MissingVectors" in error:
            raise MissingVectors(error)
        raise RuntimeError(error)
    metrics = run_metrics(results, cfg.similarity_threshold, len(items), None)
    checks = check_gates(metrics, True, load_gates(gates_path) if gates_path else load_gates())
    drops = []
    if baseline.exists():
        base = json.loads(baseline.read_text("utf-8"))
        for key in GATED:
            if key in base and metrics.get(key) is not None:
                if metrics[key] < base[key] - MAX_DROP:
                    drops.append(
                        f"{key} {metrics[key]:.3f} < baseline {base[key]:.3f} - {MAX_DROP}"
                    )
    return GateResult(overall(checks) != "fail" and not drops, metrics, checks, drops)


def export_snapshot(src_db: Path, snapshot: Path, model: str, golden: Path = GOLDEN) -> int:
    """Copy only what the gate needs from the local cache: vectors of the sample-corpus chunks
    (current chunking) and of the golden questions. Returns the number of vectors written."""
    questions = [json.loads(line)["question"] for line in golden.read_text("utf-8").splitlines()]
    query_hashes = [text_hash(q) for q in questions]
    snapshot.unlink(missing_ok=True)
    out = Store(snapshot)  # same schema, so copy_embeddings_from() can read it later
    out.close()
    con = sqlite3.connect(snapshot)
    try:
        con.execute("ATTACH DATABASE ? AS src", (str(src_db),))
        with con:
            con.execute(
                "INSERT INTO embedding_cache SELECT e.* FROM src.embedding_cache e"
                " WHERE e.model = ? AND e.text_hash IN (SELECT c.text_hash FROM src.chunks c"
                " JOIN src.documents d ON d.id = c.doc_id WHERE d.path LIKE ?)",
                (model, f"{CORPUS.as_posix()}/%"),
            )
            con.executemany(
                "INSERT OR IGNORE INTO embedding_cache SELECT * FROM src.embedding_cache"
                " WHERE model = ? AND text_hash = ?",
                [(f"{model}#query", h) for h in query_hashes],
            )
        count = con.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
    finally:
        con.close()
    con = sqlite3.connect(snapshot)
    con.execute("VACUUM")  # smallest possible file for git
    con.close()
    return count
