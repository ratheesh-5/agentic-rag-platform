"""SQLite store - the source of truth for everything we index.

Tables:
- collections      named groups of documents ("AI Engineering Handbook", "HR policies"); a chat
                   searches exactly one collection (plan section I). Collection 1 is the demo
                   collection: evaluation, optimizer and CI gate work on it. `owner` is empty for
                   shared collections (managed by admins, searchable by everyone) and the owner
                   hash of a signed-in user for PRIVATE ones (phase 2: only that user sees them).
                   Names are unique per owner (two users may both have "My resume").
- documents        one row per source file: normalized text + content hash + chunking config,
                   and the collection it belongs to.
- chunks           the pieces of each document, with char offsets. A chunk's `id` is also its
                   id in the FAISS vector index, which is how a search hit is mapped back to text.
- embedding_cache  vector per (chunk-text hash, model). Embedding calls cost money and time;
                   identical text is never embedded twice, even across documents or re-ingests.
- eval_runs        one row per evaluation run: the exact config used, summary metrics and
                   per-question results (JSON), so any two runs can be compared later.
                   `hidden` = left out of the admin UI's default list (old test runs).
- answer_scores    online evaluation: faithfulness / relevance of sampled chat answers.
- llm_calls        usage log: one row per model call (model, purpose, tokens, latency, error)
                   for the Admin › Usage page; written by observability.UsageLogger.
- recommendations  optimizer recommendations (report JSON) awaiting or after a human decision:
                   pending -> applied | rejected | superseded (a newer one replaced it).
- conversations    chat conversations (Phase 6), each owned by one caller (stored as a hash)
                   and bound to the collection it started in.
- messages         the turns of a conversation; assistant messages keep their sources and the
                   user's feedback (thumbs up/down) for later evaluation.

FAISS (see vector_index.py) only holds vectors and can be rebuilt from this database at any
time, so if the two ever disagree, SQLite wins.

Python notes for .NET developers:
- `with self.conn:` opens a transaction that commits on success and rolls back on an exception
  (like `using var tx = conn.BeginTransaction(); ... tx.Commit();`).
- `?` placeholders are parameterized queries (never string-format values into SQL).
- `sqlite3.Row` lets you read columns by name: `row["path"]`.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_platform.config import DEFAULT_COLLECTION_ID
from rag_platform.ingestion.chunking import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    doc_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunker TEXT NOT NULL,          -- chunking config used, e.g. "800/100"
    text TEXT NOT NULL,             -- normalized text; chunk offsets refer to this
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused; also the FAISS id
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,           -- float32, L2-normalized
    PRIMARY KEY (text_hash, model)
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    label TEXT NOT NULL,
    config TEXT NOT NULL,           -- JSON: RagConfig + model names + golden set used
    metrics TEXT NOT NULL,          -- JSON: summary metrics
    results TEXT NOT NULL,          -- JSON: per-question results
    hidden INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS answer_scores (
    message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    faithfulness REAL,
    answer_relevancy REAL,
    judge_calls INTEGER NOT NULL,
    error TEXT,                     -- e.g. quota error: no scores, kept so it is visible
    scored_at TEXT NOT NULL,
    judge TEXT NOT NULL DEFAULT ''  -- the judge model (scores of different judges differ)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    purpose TEXT NOT NULL,          -- answer, rewrite, judge, agent: <node>, embed, ...
    kind TEXT NOT NULL,             -- chat | embed
    input_tokens INTEGER,
    output_tokens INTEGER,
    texts INTEGER,                  -- embeddings: number of texts in the request
    latency_ms INTEGER NOT NULL,
    error TEXT,
    run_id TEXT                     -- LangChain run id = LangSmith run id (trace link)
);
CREATE INDEX IF NOT EXISTS ix_llm_calls_created ON llm_calls(created_at);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    report TEXT NOT NULL,           -- JSON: the optimizer's report (changes, metrics, summary)
    status TEXT NOT NULL,           -- pending | applied | rejected | superseded
    decided_at TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner TEXT NOT NULL,            -- hash of the API key that created it (never the key)
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_conversations_owner ON conversations(owner, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,             -- "user" or "assistant"
    content TEXT NOT NULL,
    details TEXT NOT NULL,          -- JSON: sources, answerable, standalone question, steps...
    feedback INTEGER,               -- assistant messages: 1 = helpful, -1 = not helpful
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_conversation ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS collections (    -- groups of documents; a chat searches one of them
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                 -- unique per owner (index ix_collections_owner_name)
    description TEXT NOT NULL,
    owner TEXT NOT NULL DEFAULT '',     -- '' = shared (admin-managed); else a user's owner hash
    created_at TEXT NOT NULL,
    threshold REAL,                     -- similarity threshold: admin's manual override
    auto_threshold REAL                 -- ... measured automatically (retrieval/calibration.py)
);
CREATE TABLE IF NOT EXISTS users (      -- people who signed in with Google (plan B1)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    google_sub TEXT NOT NULL UNIQUE,    -- Google's stable user id ("sub" claim)
    email TEXT NOT NULL,
    name TEXT NOT NULL,
    picture TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_login_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (   -- our own sign-in sessions, revocable on sign-out
    token_hash TEXT PRIMARY KEY,        -- SHA-256 of the token (the token itself is never stored)
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
"""


def _in_collection(table: str, collection_id: int | None) -> tuple[str, tuple]:
    """SQL filter " WHERE <table>.collection_id = ?" and its argument; nothing for None (all).
    Every search-related read goes through this, so one collection never sees another's text."""
    if collection_id is None:
        return "", ()
    return f" WHERE {table}.collection_id = ?", (collection_id,)


class Store:
    """SQLite source of truth: documents, chunks and the embedding cache."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")  # SQLite needs this for ON DELETE CASCADE
        self.conn.executescript(SCHEMA)  # idempotent: CREATE ... IF NOT EXISTS
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that databases created by an older version lack (like an EF migration,
        but by hand: CREATE TABLE IF NOT EXISTS never changes an existing table)."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(eval_runs)")}
        if "hidden" not in cols:
            with self.conn:
                self.conn.execute(
                    "ALTER TABLE eval_runs ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0"
                )
        # Collections (2026-10-24): the demo collection exists in every database, and documents
        # and conversations from before collections belong to it (DEFAULT 1 fills old rows).
        # No foreign key on the new columns (SQLite can't add one with a default): deleting a
        # collection removes its documents and conversations in code (delete_collection).
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO collections (id, name, description, created_at)"
                " VALUES (?, 'AI Engineering Handbook',"
                " 'The demo knowledge base: an AI engineering team handbook.', ?)",
                (DEFAULT_COLLECTION_ID, datetime.now(UTC).isoformat(timespec="seconds")),
            )
            for table in ("documents", "conversations"):
                cols = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
                if "collection_id" not in cols:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN collection_id INTEGER NOT NULL"
                        f" DEFAULT {DEFAULT_COLLECTION_ID}"
                    )
            # Similarity threshold per collection (retrieval/calibration.py): an admin's override
            # and the automatically measured value; NULL = not set / not measured yet.
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(collections)")}
            for col in ("threshold", "auto_threshold"):
                if col not in cols:
                    self.conn.execute(f"ALTER TABLE collections ADD COLUMN {col} REAL")
            # Model choice in Settings (plan G7): each score records its judge model.
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(answer_scores)")}
            if "judge" not in cols:
                self.conn.execute(
                    "ALTER TABLE answer_scores ADD COLUMN judge TEXT NOT NULL DEFAULT ''"
                )
            # Phase 2: names were unique for everyone; with private collections they are unique
            # per owner. SQLite can't drop a column constraint, so an old table is rebuilt once
            # (same ids; nothing references collections by foreign key).
            sql = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'collections'"
            ).fetchone()[0]
            if "name TEXT NOT NULL UNIQUE" in sql:
                self.conn.execute(
                    "CREATE TABLE collections_new (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    " name TEXT NOT NULL, description TEXT NOT NULL,"
                    " owner TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,"
                    " threshold REAL, auto_threshold REAL)"
                )
                self.conn.execute(
                    "INSERT INTO collections_new (id, name, description, owner, created_at,"
                    " threshold, auto_threshold) SELECT id, name, description, owner, created_at,"
                    " threshold, auto_threshold FROM collections"
                )
                self.conn.execute("DROP TABLE collections")
                self.conn.execute("ALTER TABLE collections_new RENAME TO collections")
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_collections_owner_name"
                " ON collections(owner, name)"
            )

    def close(self) -> None:
        self.conn.close()

    def get_document(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()

    def replace_document(
        self,
        path: str,
        doc_type: str,
        content_hash: str,
        chunker: str,
        text: str,
        chunks: list[Chunk],
        collection_id: int = DEFAULT_COLLECTION_ID,
    ) -> tuple[list[int], list[int]]:
        """Insert or replace a document and its chunks atomically. Returns (old_ids, new_ids).

        A changed document gets ALL its chunks replaced (simpler and safer than diffing). The
        caller uses old_ids/new_ids to apply the same change to the FAISS index; unchanged
        chunk texts still skip the embedding API thanks to the cache.
        """
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            row = self.get_document(path)
            old_ids: list[int] = []
            if row:
                doc_id = row["id"]
                old_ids = self.chunk_ids(doc_id)
                self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                self.conn.execute(
                    "UPDATE documents SET doc_type = ?, content_hash = ?, chunker = ?, text = ?,"
                    " ingested_at = ?, collection_id = ? WHERE id = ?",
                    (doc_type, content_hash, chunker, text, now, collection_id, doc_id),
                )
            else:
                doc_id = self.conn.execute(
                    "INSERT INTO documents (path, doc_type, content_hash, chunker, text,"
                    " ingested_at, collection_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (path, doc_type, content_hash, chunker, text, now, collection_id),
                ).lastrowid
            # List comprehension: insert each chunk, collect its new id (like .Select().ToList()).
            new_ids = [
                self.conn.execute(
                    "INSERT INTO chunks (doc_id, idx, start_char, end_char, text, text_hash)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, c.index, c.start, c.end, c.text, c.text_hash),
                ).lastrowid
                for c in chunks
            ]
        return old_ids, new_ids

    def delete_document(self, path: str) -> list[int] | None:
        """Delete a document and its chunks. Returns the removed chunk ids, None if not found."""
        with self.conn:
            row = self.get_document(path)
            if row is None:
                return None
            ids = self.chunk_ids(row["id"])
            self.conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))  # cascades
        return ids

    def list_documents(self, collection_id: int | None = None) -> list[dict]:
        """Documents with their chunk counts (no text), for listings such as the API.
        `collection_id` limits it to one collection (None = all)."""
        where, args = _in_collection("d", collection_id)
        rows = self.conn.execute(
            "SELECT d.id, d.path, d.doc_type, d.chunker, d.ingested_at, d.collection_id,"
            " COUNT(c.id) AS chunks FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id"
            f"{where} GROUP BY d.id ORDER BY d.path",
            args,
        ).fetchall()
        return [dict(r) for r in rows]  # sqlite3.Row -> plain dict

    def get_document_by_id(self, doc_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

    def document_paths(self, collection_id: int | None = None) -> list[str]:
        where, args = _in_collection("documents", collection_id)
        sql = f"SELECT path FROM documents{where} ORDER BY path"
        return [r[0] for r in self.conn.execute(sql, args)]

    def document_texts(self) -> dict[str, str]:
        """Normalized text per document path (golden-set evidence quotes are resolved in it)."""
        return {r[0]: r[1] for r in self.conn.execute("SELECT path, text FROM documents")}

    def chunkers(self, collection_id: int | None = None) -> set[str]:
        """Distinct chunking configs in the index, e.g. {"800/100"}."""
        where, args = _in_collection("documents", collection_id)
        sql = f"SELECT DISTINCT chunker FROM documents{where}"
        return {r[0] for r in self.conn.execute(sql, args)}

    def chunks_version(self, collection_id: int | None = None) -> tuple[int, int]:
        """(number of chunks, highest chunk id): changes on every add, replace or delete, because
        chunk ids are never reused. A cheap check (~0.02 ms) for the API's in-memory index cache."""
        where, args = _in_collection("d", collection_id)
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(c.id), 0) FROM chunks c"
            f" JOIN documents d ON d.id = c.doc_id{where}",
            args,
        ).fetchone()
        return row[0], row[1]

    def chunk_ids(self, doc_id: int | None = None, collection_id: int | None = None) -> list[int]:
        """Chunk ids of one document, of one collection, or (no argument) of all documents."""
        if doc_id is not None:
            rows = self.conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ? ORDER BY id", (doc_id,)
            )
        else:
            where, args = _in_collection("d", collection_id)
            rows = self.conn.execute(
                f"SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id{where}"
                " ORDER BY c.id",
                args,
            )
        return [r[0] for r in rows]

    def chunk_texts(self, collection_id: int | None = None) -> list[tuple[int, str]]:
        """(chunk id, text) of one collection (None = all), for the BM25 keyword index."""
        where, args = _in_collection("d", collection_id)
        rows = self.conn.execute(
            f"SELECT c.id, c.text FROM chunks c JOIN documents d ON d.id = c.doc_id{where}"
            " ORDER BY c.id",
            args,
        )
        return [(r[0], r[1]) for r in rows]

    def get_chunks(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        """Chunks by id, with their document path. Used to turn FAISS hits back into text."""
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))  # "?,?,?" - one placeholder per id for IN (...)
        rows = self.conn.execute(
            "SELECT c.*, d.path FROM chunks c JOIN documents d ON d.id = c.doc_id"
            f" WHERE c.id IN ({marks})",
            ids,
        )
        return {r["id"]: r for r in rows}  # dict comprehension, like .ToDictionary(r => r.Id)

    def counts(self, collection_id: int | None = None) -> tuple[int, int]:
        where, args = _in_collection("documents", collection_id)
        docs = self.conn.execute(f"SELECT COUNT(*) FROM documents{where}", args).fetchone()[0]
        return docs, len(self.chunk_ids(collection_id=collection_id))

    # --- Embedding cache -------------------------------------------------------------------
    # Vectors are stored as raw float32 bytes (BLOB). numpy converts both ways:
    # array.tobytes() -> bytes, np.frombuffer(bytes, float32) -> array. A 768-dim vector is
    # 768 * 4 = 3 KB.

    def get_vectors(self, hashes: list[str], model: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for h in set(hashes):
            row = self.conn.execute(
                "SELECT vector FROM embedding_cache WHERE text_hash = ? AND model = ?", (h, model)
            ).fetchone()
            if row:
                out[h] = np.frombuffer(row[0], dtype=np.float32)
        return out

    def put_vectors(self, vectors: dict[str, np.ndarray], model: str) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (text_hash, model, vector)"
                " VALUES (?, ?, ?)",
                [(h, model, v.astype(np.float32).tobytes()) for h, v in vectors.items()],
            )

    # --- Eval runs -----------------------------------------------------------------------------
    # Stored as JSON text: flexible (new metrics need no schema change) and easy to read back.

    def save_eval_run(self, label: str, config: dict, metrics: dict, results: list[dict]) -> int:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            return self.conn.execute(
                "INSERT INTO eval_runs (created_at, label, config, metrics, results)"
                " VALUES (?, ?, ?, ?, ?)",
                (now, label, json.dumps(config), json.dumps(metrics), json.dumps(results)),
            ).lastrowid

    def get_eval_run(self, run_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "label": row["label"],
            "config": json.loads(row["config"]),
            "metrics": json.loads(row["metrics"]),
            "results": json.loads(row["results"]),
        }

    def list_eval_runs(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, created_at, label, config, metrics, hidden FROM eval_runs ORDER BY id"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "label": r["label"],
                "config": json.loads(r["config"]),
                "metrics": json.loads(r["metrics"]),
                "hidden": bool(r["hidden"]),
            }
            for r in rows
        ]

    def set_eval_run_hidden(self, run_id: int, hidden: bool) -> bool:
        with self.conn:
            cur = self.conn.execute(
                "UPDATE eval_runs SET hidden = ? WHERE id = ?", (int(hidden), run_id)
            )
        return cur.rowcount > 0

    # --- Online evaluation (scores of real chat answers) ---------------------------------------

    def save_answer_score(
        self,
        message_id: int,
        faithfulness: float | None,
        answer_relevancy: float | None,
        judge_calls: int,
        error: str | None,
        judge: str = "",
    ) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO answer_scores (message_id, faithfulness,"
                " answer_relevancy, judge_calls, error, scored_at, judge)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, faithfulness, answer_relevancy, judge_calls, error, now, judge),
            )

    def usage_rows(self) -> list[dict]:
        """Every assistant message with the user question before it and its score (if any),
        across ALL owners but only in SHARED collections: for the admin's usage summary
        (questions and answers in private collections are the user's own business). The
        question is the latest user message before the answer in the same conversation (ids
        are global, so a plain id - 1 could belong to another conversation)."""
        rows = self.conn.execute(
            """
            SELECT a.id, a.content, a.details, a.feedback, a.created_at,
                   (SELECT u.content FROM messages u
                     WHERE u.conversation_id = a.conversation_id AND u.role = 'user'
                       AND u.id < a.id ORDER BY u.id DESC LIMIT 1) AS question,
                   s.faithfulness, s.answer_relevancy, s.error AS score_error, s.scored_at
            FROM messages a LEFT JOIN answer_scores s ON s.message_id = a.id
            JOIN conversations v ON v.id = a.conversation_id
            JOIN collections k ON k.id = v.collection_id
            WHERE a.role = 'assistant' AND k.owner = '' ORDER BY a.id
            """
        ).fetchall()
        return [{**dict(r), "details": json.loads(r["details"])} for r in rows]

    # --- Usage log (observability) -----------------------------------------------------------

    def add_llm_call(self, **row) -> None:
        """Insert one usage row; `row` keys are llm_calls column names."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        cols = ["created_at", *row]
        marks = ", ".join("?" * len(cols))
        with self.conn:
            self.conn.execute(
                f"INSERT INTO llm_calls ({', '.join(cols)}) VALUES ({marks})",
                (now, *row.values()),
            )

    def llm_calls_since(self, since_iso: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM llm_calls WHERE created_at >= ? ORDER BY id", (since_iso,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Optimizer recommendations (human approval from the CLI or the admin UI) ---------------

    def save_recommendation(self, report: dict) -> int:
        """Store a new pending recommendation; older pending ones are superseded by it."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                "UPDATE recommendations SET status = 'superseded', decided_at = ?"
                " WHERE status = 'pending'",
                (now,),
            )
            return self.conn.execute(
                "INSERT INTO recommendations (created_at, report, status) VALUES (?, ?, 'pending')",
                (now, json.dumps(report)),
            ).lastrowid

    def latest_recommendation(self) -> dict | None:
        row = self.conn.execute("SELECT * FROM recommendations ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return {**dict(row), "report": json.loads(row["report"])}

    def decide_recommendation(self, rec_id: int, status: str) -> bool:
        """pending -> applied / rejected. False if it doesn't exist or was already decided."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            cur = self.conn.execute(
                "UPDATE recommendations SET status = ?, decided_at = ?"
                " WHERE id = ? AND status = 'pending'",
                (status, now, rec_id),
            )
        return cur.rowcount > 0

    # --- Conversations (chat memory) ---------------------------------------------------------
    # Every query filters by `owner`, so one API key can never read or change another key's
    # conversations (a missing row and a foreign row look the same: None / False).

    def create_conversation(
        self, owner: str, title: str, collection_id: int = DEFAULT_COLLECTION_ID
    ) -> int:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            return self.conn.execute(
                "INSERT INTO conversations (owner, title, created_at, updated_at, collection_id)"
                " VALUES (?, ?, ?, ?, ?)",
                (owner, title, now, now, collection_id),
            ).lastrowid

    def get_conversation(self, conversation_id: int, owner: str) -> dict | None:
        row = self.conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, c.collection_id,"
            " k.name AS collection FROM conversations c"
            " JOIN collections k ON k.id = c.collection_id"
            " WHERE c.id = ? AND c.owner = ?",
            (conversation_id, owner),
        ).fetchone()
        return dict(row) if row else None

    def list_conversations(self, owner: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, c.collection_id,"
            " k.name AS collection, COUNT(m.id) AS messages"
            " FROM conversations c JOIN messages m ON m.conversation_id = c.id"
            " JOIN collections k ON k.id = c.collection_id"
            # JOIN (not LEFT JOIN): conversations without messages are left out - older
            # versions created one before answering, so a failed first question left it empty.
            " WHERE c.owner = ? GROUP BY c.id ORDER BY c.updated_at DESC, c.id DESC",
            (owner,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conversation_id: int, owner: str) -> bool:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM conversations WHERE id = ? AND owner = ?", (conversation_id, owner)
            )
        return cur.rowcount > 0  # its messages go too (ON DELETE CASCADE)

    def delete_conversations(self, owner: str) -> int:
        """Delete all of one owner's conversations ("delete my data"); returns how many."""
        with self.conn:
            cur = self.conn.execute("DELETE FROM conversations WHERE owner = ?", (owner,))
        return cur.rowcount

    # --- Collections (plan section I, phase 1) ------------------------------------------------

    def create_collection(self, name: str, description: str = "", owner: str = "") -> int:
        """A new collection: shared (owner '') or private to `owner` (a user's owner hash).
        sqlite3.IntegrityError if this owner already has a collection with that name."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            return self.conn.execute(
                "INSERT INTO collections (name, description, owner, created_at)"
                " VALUES (?, ?, ?, ?)",
                (name, description, owner, now),
            ).lastrowid

    def get_collection(self, collection_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM collections WHERE id = ?", (collection_id,)
        ).fetchone()
        return dict(row) if row else None

    def collection_by_name(self, name: str, owner: str = "") -> dict | None:
        """A collection by name for one owner ('' = shared; the CLI manages shared ones)."""
        row = self.conn.execute(
            "SELECT * FROM collections WHERE name = ? AND owner = ?", (name, owner)
        ).fetchone()
        return dict(row) if row else None

    def list_collections(self) -> list[dict]:
        """Every collection with its number of documents and passages, demo collection first."""
        rows = self.conn.execute(
            "SELECT k.id, k.name, k.description, k.owner, k.created_at, k.threshold,"
            " k.auto_threshold, COUNT(DISTINCT d.id) AS documents, COUNT(c.id) AS passages"
            " FROM collections k LEFT JOIN documents d ON d.collection_id = k.id"
            " LEFT JOIN chunks c ON c.doc_id = d.id"
            " GROUP BY k.id ORDER BY k.id"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_collection(
        self, collection_id: int, name: str, description: str, threshold: float | None = None
    ) -> bool:
        """Rename / describe; `threshold` = manual override of the similarity threshold
        (None = automatic)."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE collections SET name = ?, description = ?, threshold = ? WHERE id = ?",
                (name, description, threshold, collection_id),
            )
        return cur.rowcount > 0

    def owner_usage(self, owner: str) -> dict:
        """How much one user stores in private collections: for the per-user limits."""
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT k.id), COUNT(DISTINCT d.id), COUNT(c.id) FROM collections k"
            " LEFT JOIN documents d ON d.collection_id = k.id"
            " LEFT JOIN chunks c ON c.doc_id = d.id WHERE k.owner = ?",
            (owner,),
        ).fetchone()
        return {"collections": row[0], "documents": row[1], "passages": row[2]}

    def private_totals(self) -> dict:
        """Totals over ALL private collections - all an admin may know about them (no names,
        no documents, no text)."""
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT k.id), COUNT(DISTINCT k.owner), COUNT(DISTINCT d.id),"
            " COUNT(c.id) FROM collections k LEFT JOIN documents d ON d.collection_id = k.id"
            " LEFT JOIN chunks c ON c.doc_id = d.id WHERE k.owner <> ''"
        ).fetchone()
        return {"collections": row[0], "users": row[1], "documents": row[2], "passages": row[3]}

    def collection_ids_of(self, owner: str) -> list[int]:
        rows = self.conn.execute("SELECT id FROM collections WHERE owner = ?", (owner,))
        return [r[0] for r in rows]

    def delete_user(self, user_id: int, owner: str) -> None:
        """Delete an account: its conversations (messages, scores cascade) and the user row
        (sessions cascade). Private collections are deleted first by the caller (files and
        search indexes live outside the database)."""
        with self.conn:
            self.conn.execute("DELETE FROM conversations WHERE owner = ?", (owner,))
            self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def set_auto_threshold(self, collection_id: int, value: float | None) -> None:
        """Store the measured threshold (None = measure again before the next answer)."""
        with self.conn:
            self.conn.execute(
                "UPDATE collections SET auto_threshold = ? WHERE id = ?", (value, collection_id)
            )

    def delete_collection(self, collection_id: int) -> list[str] | None:
        """Delete a collection with its documents (passages cascade) and its conversations
        (messages cascade), in one transaction. Returns the deleted documents' paths (the caller
        removes uploaded files and the collection's search index), None if not found."""
        with self.conn:
            if self.get_collection(collection_id) is None:
                return None
            paths = self.document_paths(collection_id)
            self.conn.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
            self.conn.execute("DELETE FROM conversations WHERE collection_id = ?", (collection_id,))
            self.conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
        return paths

    # --- Users and sessions (Sign in with Google, plan B1) --------------------------------------

    def upsert_user(self, google_sub: str, email: str, name: str, picture: str) -> int:
        """Create the user on first sign-in, refresh name/email/picture later; returns the id."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                "INSERT INTO users (google_sub, email, name, picture, created_at, last_login_at)"
                " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(google_sub) DO UPDATE SET"
                " email = excluded.email, name = excluded.name, picture = excluded.picture,"
                " last_login_at = excluded.last_login_at",
                (google_sub, email, name, picture, now, now),
            )
        return self.conn.execute(
            "SELECT id FROM users WHERE google_sub = ?", (google_sub,)
        ).fetchone()[0]

    def create_session(self, token_hash: str, user_id: int, expires_at: str) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (token_hash, user_id, now, expires_at),
            )

    def session_user(self, token_hash: str) -> dict | None:
        """The user of a session that has not expired yet, else None."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        row = self.conn.execute(
            "SELECT u.id, u.email, u.name, u.picture FROM sessions s"
            " JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires_at > ?",
            (token_hash, now),
        ).fetchone()
        return dict(row) if row else None

    def delete_session(self, token_hash: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

    def purge_older_than(self, chat_cutoff: str | None, usage_cutoff: str | None) -> dict:
        """Retention (plan B10): delete conversations last active before `chat_cutoff` - their
        messages, feedback and online scores go too (ON DELETE CASCADE) - and usage-log rows
        created before `usage_cutoff`. Cutoffs are ISO-8601 UTC times like the stored ones, so
        text comparison is time comparison. None = keep that data."""
        counts = {"conversations": 0, "llm_calls": 0}
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:  # one transaction
            # Expired sign-in sessions are useless; removing them keeps the table small.
            self.conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            if chat_cutoff:
                counts["conversations"] = self.conn.execute(
                    "DELETE FROM conversations WHERE updated_at < ?", (chat_cutoff,)
                ).rowcount
            if usage_cutoff:
                counts["llm_calls"] = self.conn.execute(
                    "DELETE FROM llm_calls WHERE created_at < ?", (usage_cutoff,)
                ).rowcount
        return counts

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        details: dict,
        created_at: str | None = None,  # ISO time; default now (questions pass "when asked")
    ) -> int:
        now = created_at or datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            message_id = self.conn.execute(
                "INSERT INTO messages (conversation_id, role, content, details, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, json.dumps(details), now),
            ).lastrowid
            self.conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )
        return message_id

    def get_messages(self, conversation_id: int, last: int | None = None) -> list[dict]:
        """Messages oldest first; `last` = only the most recent N (for chat memory)."""
        rows = self.conn.execute(
            "SELECT m.id, m.role, m.content, m.details, m.feedback, m.created_at,"
            " s.faithfulness, s.answer_relevancy"  # online-eval score, if this answer was sampled
            " FROM messages m LEFT JOIN answer_scores s ON s.message_id = m.id"
            " WHERE m.conversation_id = ? ORDER BY m.id DESC LIMIT ?",
            (conversation_id, last if last is not None else -1),  # LIMIT -1 = no limit
        ).fetchall()
        out = [{**dict(r), "details": json.loads(r["details"])} for r in rows]
        return out[::-1]  # [::-1] reverses the list: back to oldest first

    def set_feedback(self, message_id: int, owner: str, rating: int | None) -> bool:
        """Thumbs up (1) / down (-1) / cleared (None) on an assistant message the owner can see."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE messages SET feedback = ? WHERE id = ? AND role = 'assistant'"
                " AND conversation_id IN (SELECT id FROM conversations WHERE owner = ?)",
                (rating, message_id, owner),
            )
        return cur.rowcount > 0

    def copy_embeddings_from(self, other_db: Path) -> int:
        """Copy another database's embedding cache into this one (existing rows are kept).

        Optimizer experiments index the same documents with other chunk sizes in separate
        databases; sharing the cache both ways means identical chunk texts and questions are
        never embedded twice. ATTACH lets one SQLite connection query a second database file.
        """
        before = self.conn.total_changes
        with self.conn:
            self.conn.execute("ATTACH DATABASE ? AS other", (str(other_db),))
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT OR IGNORE INTO embedding_cache SELECT * FROM other.embedding_cache"
                )
        finally:
            self.conn.execute("DETACH DATABASE other")
        return self.conn.total_changes - before

    def chunk_vectors(
        self, model: str, collection_id: int | None = None
    ) -> tuple[list[int], np.ndarray | None]:
        """Chunk ids (of one collection, or all) with their cached vectors; used to rebuild a
        FAISS index."""
        where, args = _in_collection("d", collection_id)
        rows = self.conn.execute(
            "SELECT c.id, e.vector FROM chunks c JOIN embedding_cache e"
            " ON e.text_hash = c.text_hash AND e.model = ?"
            f" JOIN documents d ON d.id = c.doc_id{where} ORDER BY c.id",
            (model, *args),
        ).fetchall()
        if not rows:
            return [], None
        # np.stack turns a list of 1-D vectors into a 2-D matrix: one row per chunk.
        return [r[0] for r in rows], np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
