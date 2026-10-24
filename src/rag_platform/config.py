"""Configuration for the whole platform.

Two kinds of settings, kept separate on purpose:

- `Settings`  - environment-level values (API key, model names, file locations). Read from
  environment variables / the `.env` file. .NET analogy: `IConfiguration` + `appsettings.json`
  / user secrets, bound to an options class.
- `RagConfig` - the tunable knobs of the RAG pipeline (chunk size, top-k, thresholds). These are
  what evaluation measures and the optimizer (Phase 4) changes, so every eval run stores the
  exact RagConfig it used. Keeping them in one immutable object makes runs reproducible.
"""

import os
from pathlib import Path
from typing import Literal  # Literal["a", "b"] = a string restricted to these values (~ enum)

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# The demo collection (plan section I): every database has it; documents from before collections
# live in it, and evaluation, the optimizer and the CI gate work on it. Its search index keeps the
# original file name (Settings.faiss_path).
DEFAULT_COLLECTION_ID = 1


class Settings(BaseSettings):
    """Environment-level settings (secrets, model names). Loaded from .env.

    pydantic-settings maps each field to an environment variable of the same name
    (case-insensitive): `gemini_api_key` <- GEMINI_API_KEY. Values are type-validated, like
    binding `IOptions<T>` with data annotations.
    """

    # Which env file to read: ".env" by default. Set RAG_ENV_FILE=.env.alt to use another profile
    # (e.g. a second API key from a different Google Cloud project, which has its own quota).
    model_config = SettingsConfigDict(
        env_file=os.environ.get("RAG_ENV_FILE", ".env"), extra="ignore"
    )

    gemini_api_key: str = ""
    chat_model: str = "gemini-3.5-flash-lite"  # LLM that writes answers
    embedding_model: str = "gemini-embedding-001"  # model that turns text into vectors
    judge_model: str = ""  # LLM RAGAS uses to grade answers; empty = same as chat_model
    # Client-side cap on chat calls per model (Gemini free tier: 15/min). Raise on a paid tier.
    chat_requests_per_minute: int = 14
    # Calls that may go out back to back after an idle moment (a chat question needs 2: rewrite
    # + answer). The steady rate is lowered by the same amount, so a minute never exceeds
    # chat_requests_per_minute. 1 = no burst (every call waited ~4.3 s on the free tier).
    chat_burst: int = 2
    # A slow / overloaded provider must not hang a question for minutes: each model request
    # gives up after this many seconds and is retried at most this many times (worst case ~1
    # minute per call). Library defaults were no timeout + 6 retries (91-126 s seen live).
    chat_timeout_seconds: float = 20
    chat_max_retries: int = 2
    # Client-side cap on texts embedded per minute (free tier: 100/min, each text counts).
    embed_requests_per_minute: int = 90
    # Second chat provider (plan G2/G5): Groq, used as the main model or as the automatic
    # fallback when Gemini is out of quota or overloaded (chosen in Admin › Settings, stored in
    # rag_config.json). Empty key = Groq is off. Only CHAT moves; embeddings stay on Gemini.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    # Client-side cap. Groq's free tier allows 1,000 requests/day but only ~8,000 TOKENS per
    # minute per model, and one answer prompt is ~2,000 tokens - so about 4 calls a minute.
    groq_requests_per_minute: int = 5
    index_dir: Path = Path("data/index")  # SQLite db + FAISS index
    # REST API (Phase 5): comma-separated ADMIN keys accepted in the X-API-Key header (documents,
    # evaluation, settings, /ask). Empty = admin endpoints refuse every request (503).
    api_keys: str = ""
    # Guests (no key) may chat; each guest browser/IP gets this many questions per hour.
    # 0 = guests disabled (chat then needs an admin key too).
    guest_questions_per_hour: int = 20
    # Sign in with Google (security plan B1). The OAuth client id of THIS app (Google Cloud
    # Console > Credentials; public, not a secret). Empty = Google sign-in off: the UI falls back
    # to the admin-key dialog. Users sign in with their own Google account; admins are the
    # signed-in users whose email is in ADMIN_EMAILS (comma-separated). API keys keep working
    # for scripts and the CLI.
    google_client_id: str = ""
    admin_emails: str = ""
    session_days: int = 7  # how long a sign-in lasts (our own session, not Google's 1-h token)
    user_questions_per_hour: int = 50  # per signed-in user (guests: GUEST_QUESTIONS_PER_HOUR)
    # Private collections (plan section I, phase 2): what one signed-in user may store. Protects
    # the shared embedding quota and the disk; uploads beyond a limit are refused BEFORE any
    # embedding call.
    user_max_collections: int = 5
    user_max_documents: int = 20
    user_max_passages: int = 500
    # Online evaluation: share of chat answers scored in the background for faithfulness and
    # relevance (0.2 = 1 in 5; 0 = off). ~3-4 judge calls per scored answer.
    online_eval_sample_rate: float = 0.2
    # Data retention (plan B10): conversations without activity for this many days are deleted
    # (with their messages, feedback and scores), and usage-log rows older than this. Checked
    # when the API starts and then daily. 0 = keep forever.
    chat_retention_days: int = 90
    usage_retention_days: int = 90
    # LangSmith tracing (Phase 7, optional): copied to the environment by
    # observability.configure_tracing so LangChain sends traces. Off unless both are set.
    # The API traces; CLI batch runs only with --trace-langsmith.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-platform"
    langsmith_sampling_rate: float = 1.0  # share of requests traced (free tier: monthly limit)
    langsmith_endpoint: str = ""  # only for the EU region (https://eu.api.smith.langchain.com)
    upload_dir: Path = Path("data/uploads")  # documents uploaded through the API
    max_upload_mb: int = 10
    # Browser apps allowed to call the API (CORS), comma-separated. Default: the Angular dev
    # server. Never "*": with credentials-like headers that would let any site call the API.
    cors_origins: str = "http://localhost:4200"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def admin_email_list(self) -> list[str]:
        return [e.strip().lower() for e in self.admin_emails.split(",") if e.strip()]

    # @property = a read-only computed property, like `public Path DbPath => IndexDir / "rag.db";`
    @property
    def db_path(self) -> Path:
        return self.index_dir / "rag.db"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.index"

    def collection_index_path(self, collection_id: int) -> Path:
        """Each collection has its own FAISS file, so a search can only ever see that collection
        (isolation by construction); the demo collection keeps data/index/faiss.index."""
        if collection_id == DEFAULT_COLLECTION_ID:
            return self.faiss_path
        return self.index_dir / "collections" / f"{collection_id}.faiss"


class RagConfig(BaseModel, frozen=True):
    """All pipeline tunables. Stored with every eval run.

    `frozen=True` makes instances immutable (like a C# `record` with init-only properties);
    create a variant with `RagConfig(chunk_size=500)` instead of mutating one.
    """

    # Chunking: documents are split into pieces of at most `chunk_size` characters. Each piece
    # repeats the last `chunk_overlap` characters of the previous one, so a fact sitting on a
    # boundary is still complete in at least one chunk.
    chunk_size: int = 800
    chunk_overlap: int = 100
    # Retrieval: how many of the most similar chunks are given to the LLM as context.
    top_k: int = 5
    # "dense" = vector similarity only. "hybrid" = vector + BM25 keyword search merged with
    # Reciprocal Rank Fusion; helps with exact terms (codes, names, table rows) (Phase 4).
    retrieval: Literal["dense", "hybrid"] = "dense"
    hybrid_candidates: int = 20  # hybrid: candidates taken from EACH retriever before fusion
    # Abstention: if even the best chunk's cosine similarity is below this, answer "I don't
    # know" without calling the LLM. Must be calibrated per embedding model (Phase 2).
    similarity_threshold: float = 0.35
    # Pipeline: "fast" = retrieve -> generate (1 LLM call). "agentic" = LangGraph workflow with
    # query planning, relevance grading, query rewrites and answer verification (Phase 3).
    # "auto" = fast first; agentic only when the fast answer struggled (see answering.py).
    mode: Literal["fast", "agentic", "auto"] = "fast"
    max_rewrites: int = 2  # agentic: query rewrites when no relevant passage is found
    max_regenerations: int = 1  # agentic: answer regenerations requested by the verifier
    # Answer model (plan G2/G3), chosen in Admin › Settings: the provider that writes answers
    # first, and whether the other one takes over when it is out of quota, overloaded or too
    # slow (see llm.FallbackChat). Embeddings always stay on Gemini.
    chat_provider: Literal["gemini", "groq"] = "gemini"
    chat_fallback: bool = True


# Settings the admin switches directly in Admin › Settings. Optimizer recommendations measure
# search only: they ignore these fields and applying one keeps them.
ADMIN_CHOICES = ("mode", "chat_provider", "chat_fallback")


# The ACTIVE pipeline config, versioned in git (like appsettings.json). Every command starts from
# it; the optimizer (Phase 4) rewrites it after human approval. Missing file = code defaults.
RAG_CONFIG_FILE = Path("rag_config.json")


# What the chatbot covers (name, description, topics with example questions), shown to users.
# Domain-specific content lives here, not in the UI. Missing file = generic defaults.
ASSISTANT_FILE = Path("assistant.json")


def load_rag_config(path: Path = RAG_CONFIG_FILE, **overrides) -> RagConfig:
    """Active config from `rag_config.json`, with optional overrides (None values are ignored).

    Building a new RagConfig (instead of model_copy) re-runs validation on the overrides.
    """
    base = RagConfig.model_validate_json(path.read_text("utf-8")) if path.exists() else RagConfig()
    changes = {k: v for k, v in overrides.items() if v is not None}
    return RagConfig(**{**base.model_dump(), **changes})


def save_rag_config(cfg: RagConfig, path: Path = RAG_CONFIG_FILE) -> None:
    path.write_text(cfg.model_dump_json(indent=2) + "\n", encoding="utf-8")
