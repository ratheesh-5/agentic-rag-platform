"""Dependencies for the API: shared services, who is calling (user, admin, guest), limits.

FastAPI's dependency injection: an endpoint declares `svc: Services = Depends(get_services)`
and FastAPI calls get_services() and passes the result in. .NET analogy: constructor injection
with services registered in Program.cs; tests replace a registration via
`app.dependency_overrides[get_services] = ...` (like WebApplicationFactory.ConfigureServices).

Services are created once per process (functools.lru_cache = a lazy singleton). Database
connections are NOT shared: SQLite connections may only be used by the thread that created
them, and FastAPI runs normal `def` endpoints in a thread pool, so each request opens its own
Store (cheap: a local file).

Writes (upload, delete) take a process-wide lock so two requests never update SQLite and the
FAISS file at the same time; reads (ask, list) run in parallel.

Callers (see current_caller):
- Signed-in user (Sign in with Google, plan B1; api/auth.py): sends our session token as
  "Authorization: Bearer ...". Chats on every device; USER_QUESTIONS_PER_HOUR. An admin when the
  email is in ADMIN_EMAILS.
- Admin by key: sends one of API_KEYS in X-API-Key (scripts, CLI tools, the UI's fallback when
  Google sign-in is not configured). Can use every endpoint.
- Guest: no sign-in, only X-Guest-Id - a random id the browser generated once. Guests can chat
  (their own conversations only) and are limited to GUEST_QUESTIONS_PER_HOUR questions per IP.
"""

import hashlib
import random
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, Header, HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag_platform.config import ASSISTANT_FILE, RAG_CONFIG_FILE, Settings
from rag_platform.evaluation.gates import GATES_FILE
from rag_platform.observability import PRICES_FILE
from rag_platform.store.db import Store

write_lock = threading.Lock()


class RateLimiter:
    """Sliding window: at most `limit` hits per `window` seconds for each key (here: guest IP).
    In memory, so it resets when the server restarts - enough to protect the model quota."""

    def __init__(self, window: float = 3600.0):
        self.window = window
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()  # endpoints run in a thread pool

    def hit(self, key: str, limit: int, cost: int = 1) -> float:
        """Record `cost` hits (an analysis counts as several questions); returns 0 if allowed,
        else the seconds until enough earlier hits have expired."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()  # forget hits older than the window
            if len(hits) + cost > limit:
                if cost > limit:
                    return self.window
                # the oldest hits that must expire first; the last of them decides the wait
                return self.window - (now - hits[len(hits) + cost - limit - 1])
            hits.extend([now] * cost)
            return 0.0


@dataclass
class Services:
    settings: Settings
    embedder: Embeddings
    llm: BaseChatModel
    config_path: Path = RAG_CONFIG_FILE  # rag_config.json with the active settings
    assistant_path: Path = ASSISTANT_FILE  # assistant.json: topics + example questions for users
    golden_path: Path = Path("data/eval/golden.jsonl")  # default test set for quality checks
    gates_path: Path = GATES_FILE  # release gate thresholds
    prices_path: Path = PRICES_FILE  # optional model prices for the usage report
    rng: Callable[[], float] = random.random  # online-eval sampling (tests make it fixed)
    evaluator: object | None = None  # online-eval judge (RagasEvaluator), created on first use
    # In-memory FAISS + BM25 cache (api/service.py SearchCache); created lazily to avoid an
    # import cycle (service imports deps).
    _search: object | None = None

    @property
    def search(self):
        if self._search is None:
            from rag_platform.api.service import SearchCache

            self._search = SearchCache()
        return self._search

    def answer_evaluator(self):
        """The RAGAS judge for online evaluation, built once (tests assign a fake)."""
        if self.evaluator is None:
            from rag_platform.evaluation.ragas_eval import RagasEvaluator  # loads RAGAS lazily
            from rag_platform.llm import get_judge_model

            self.evaluator = RagasEvaluator(get_judge_model(self.settings), self.embedder)
        return self.evaluator

    # field(default_factory=...) = a new limiter per Services object (tests get a fresh one).
    guest_limiter: RateLimiter = field(default_factory=RateLimiter)

    def open_store(self) -> Store:
        return Store(self.settings.db_path)


@lru_cache  # first call builds the services; later calls return the same object
def get_services() -> Services:
    from rag_platform.llm import get_chat_model, get_embedder

    s = Settings()
    if not s.gemini_api_key:
        raise HTTPException(503, "Server not configured: GEMINI_API_KEY is missing")
    from rag_platform.observability import configure_tracing

    configure_tracing(s)  # the live API is traced when LangSmith is configured (before any model)
    return Services(settings=s, embedder=get_embedder(s), llm=get_chat_model(s))


# Declare the headers in the OpenAPI docs (/docs shows an "Authorize" button for them).
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer = HTTPBearer(auto_error=False)  # "Authorization: Bearer <session token>"


def _is_admin_key(key: str | None, svc: Services) -> bool:
    """True for a valid admin key, False for no key; 401 for a wrong key, 503 if none set."""
    if not key:
        return False
    keys = svc.settings.api_key_list
    if not keys:
        raise HTTPException(503, "API keys are not configured (set API_KEYS)")
    # compare_digest takes the same time whether the first or last character differs, so an
    # attacker can't guess a key character by character from response times.
    if not any(secrets.compare_digest(key, k) for k in keys):
        raise HTTPException(
            401, "Invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"}
        )
    return True


def session_user(token: str, svc: Services) -> dict:
    """The signed-in user of a session token; 401 if it is unknown, expired or signed out."""
    store = svc.open_store()
    try:
        user = store.session_user(token_hash(token))
    finally:
        store.close()
    if user is None:
        raise HTTPException(
            401, "Your sign-in has expired. Please sign in again.", headers=_BEARER_CHALLENGE
        )
    return user


# A guest id is a random UUID made by the browser (crypto.randomUUID); anything else is refused.
_GUEST_ID = re.compile(r"^[A-Za-z0-9-]{16,64}$")
_BEARER_CHALLENGE = {"WWW-Authenticate": "Bearer"}


@dataclass
class Caller:
    owner: str  # who owns the conversations (a hash; the key/guest id itself is never stored)
    is_admin: bool
    user: dict | None = None  # the signed-in user (id, email, name, picture), if any

    @property
    def limit_key(self) -> str | None:
        """Question-limit bucket: per user when signed in, per IP for guests (None = no limit)."""
        if self.is_admin:
            return None
        return f"user:{self.user['id']}" if self.user else ""


def current_caller(
    key: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
    guest_id: str | None = Header(None, alias="X-Guest-Id"),
    svc: Services = Depends(get_services),
) -> Caller:
    """Who is calling, strongest proof first:
    1. a session token (signed in with Google) -> that user; admin if the email is in ADMIN_EMAILS;
    2. a valid X-API-Key -> admin (scripts, CLI tools, the UI's fallback when Google is off);
    3. an X-Guest-Id -> a guest.
    Conversations belong to the user when signed in, else to the guest id (the browser's
    identity, also for an admin using a key in the UI), else to the key.
    """
    if bearer is not None:
        user = session_user(bearer.credentials, svc)
        is_admin = user["email"].lower() in svc.settings.admin_email_list
        return Caller(user_owner(user["id"]), is_admin, user)
    is_admin = _is_admin_key(key, svc)
    if guest_id is not None:
        if not _GUEST_ID.match(guest_id):
            raise HTTPException(422, "Invalid X-Guest-Id")
        if not is_admin and svc.settings.guest_questions_per_hour <= 0:
            raise HTTPException(401, "Guest access is disabled; please sign in")
        return Caller(guest_owner(guest_id), is_admin)
    if is_admin:
        return Caller(_hash(key or ""), True)
    raise HTTPException(401, "Sign in, or send an API key (X-API-Key) or a guest id (X-Guest-Id)")


def require_admin(caller: Caller = Depends(current_caller), svc: Services = Depends(get_services)):
    """Admin endpoints: an admin user (signed in, email in ADMIN_EMAILS) or a valid API key."""
    s = svc.settings
    if not s.api_key_list and not s.admin_email_list:
        raise HTTPException(503, "No admins configured (set ADMIN_EMAILS or API_KEYS)")
    if not caller.is_admin:
        if caller.user:  # signed in, but not an admin: authenticated, not allowed
            raise HTTPException(403, "Admins only")
        raise HTTPException(
            401, "Invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"}
        )


def current_owner(caller: Caller = Depends(current_caller)) -> str:
    return caller.owner


def optional_caller(
    key: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
    guest_id: str | None = Header(None, alias="X-Guest-Id"),
    svc: Services = Depends(get_services),
) -> Caller | None:
    """Like current_caller, but None for a request without any credential: for open endpoints
    (/collections, /assistant) that show private collections only to their owner."""
    if key is None and bearer is None and guest_id is None:
        return None
    return current_caller(key, bearer, guest_id, svc)


# --- Collections: who may see / manage one (plan section I, phase 2) -------------------------
# Every endpoint that takes a collection asks these, so the rule exists exactly once:
#   shared (owner '')   -> everyone may search it; admins manage it;
#   private (owner set) -> only its owner, a signed-in user, may search AND manage it - not
#                          other users, not guests, not admins (they only see totals).


def private_owner(caller: Caller | None) -> str | None:
    """The owner value a signed-in user's private collections carry (None: not signed in)."""
    return user_owner(caller.user["id"]) if caller and caller.user else None


def can_see(caller: Caller | None, collection: dict) -> bool:
    return collection["owner"] == "" or collection["owner"] == private_owner(caller)


def can_manage(caller: Caller | None, collection: dict) -> bool:
    if collection["owner"] == "":
        return bool(caller and caller.is_admin)
    return collection["owner"] == private_owner(caller)


def visible_collection(store: Store, caller: Caller | None, collection_id: int) -> dict:
    """The collection if the caller may search it; 404 otherwise - also for someone else's
    private collection, so its existence isn't revealed."""
    collection = store.get_collection(collection_id)
    if collection is None or not can_see(caller, collection):
        raise HTTPException(404, f"No collection with id {collection_id}")
    return collection


def manageable_collection(store: Store, caller: Caller | None, collection_id: int) -> dict:
    """The collection if the caller may change it (documents, name, delete); 404 if they can't
    even see it, 403 if they can see but not change it (a shared one, for a non-admin)."""
    collection = visible_collection(store, caller, collection_id)
    if not can_manage(caller, collection):
        raise HTTPException(403, "Only admins can change shared collections")
    return collection


def limit_questions(
    request: Request,
    caller: Caller = Depends(current_caller),
    svc: Services = Depends(get_services),
) -> None:
    """Guests may ask GUEST_QUESTIONS_PER_HOUR questions per IP (a new guest id doesn't reset
    it), signed-in users USER_QUESTIONS_PER_HOUR each; admins are not limited. Protects the
    model quota on a public chat page."""
    bucket = caller.limit_key
    if bucket is None:
        return
    if bucket:
        limit = svc.settings.user_questions_per_hour
    else:
        bucket = "ip:" + (request.client.host if request.client else "unknown")
        limit = svc.settings.guest_questions_per_hour
    wait = svc.guest_limiter.hit(bucket, limit)
    if wait:
        raise HTTPException(
            429,
            "You have reached the hourly question limit. Please try again later.",
            headers={"Retry-After": str(int(wait) + 1)},
        )


def user_owner(user_id: int) -> str:
    """Conversation owner of a signed-in user (the same on every device)."""
    return _hash(f"user:{user_id}")


def guest_owner(guest_id: str) -> str:
    """Conversation owner of a guest browser."""
    return _hash("guest:" + guest_id)


def token_hash(token: str) -> str:
    """Sessions are stored by the SHA-256 of their token: a copy of the database alone can't be
    used to sign in (like storing password hashes, but a random 256-bit token needs no salt)."""
    return hashlib.sha256(token.encode()).hexdigest()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]
