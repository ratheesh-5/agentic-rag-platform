"""REST API (FastAPI) over the same building blocks as the CLI.

Run it:  uv run rag serve            (or: uv run uvicorn rag_platform.api.app:app --reload)
Docs:    http://127.0.0.1:8000/docs  (interactive Swagger UI generated from schemas.py)

Endpoints (admin = X-API-Key header with one of API_KEYS; guest = X-Guest-Id header):
    GET    /health              liveness check: the process is alive (open)
    GET    /ready               readiness check: config, database and search index work (open)
    GET    /me                  the caller's role: admin or guest (guest or admin)
    GET    /assistant           what the chatbot covers: topics + example questions (open)
    POST   /ask                 one independent question, cited answer (no memory)
    POST   /chat                question within a conversation (memory; see api/chat.py)
    GET    /conversations       the caller's conversations; GET/DELETE /conversations/{id}
    PUT    /messages/{id}/feedback   thumbs up/down on an answer
                                (chat endpoints: guest or admin; everything below: admin)
    /collections                groups of documents, shared or private (api/collections.py)
    DELETE /me                  delete my account: private collections, chats, sign-in (user)
    GET    /documents           documents of the collections the caller may manage
                                (admins: shared ones; every signed-in user: their private ones)
    POST   /documents           upload + validate + index one file (multipart/form-data)
    DELETE /documents/{id}      remove a document from the index (and its uploaded file)
    GET    /config              the active rag_config.json
    PUT    /config/mode         switch the answer mode: fast / agentic / auto (admin)
    GET    /config/providers    chat providers and whether each is set up (admin)
    PUT    /config/model        answer model: Gemini or Groq first, fallback on/off (admin)
    /config/models...           choose answer / backup / judge models, test, quality check
                                (admin, see api/models.py)
    /eval/..., /optimizer/...   quality checks, runs, release gates, optimizer approval
                                (see api/evaluation.py)
    /usage/...                  usage log: calls, tokens, latency, cost; LangSmith links

Endpoints are plain `def` functions: the RAG code is synchronous, so FastAPI runs them in a
thread pool and slow LLM calls don't block other requests. Long jobs (full eval runs with
answers + RAGAS, the optimizer itself) stay in the CLI.

FastAPI (like ASP.NET Core minimal APIs): a decorator binds a function to a route; parameters
come from the path, query, body (pydantic model) or `Depends(...)`; the return value is
serialized through `response_model`.

Request log (plan D1): a middleware gives every request an id (the caller's X-Request-ID if it
looks safe, else a new one), returns it in the X-Request-ID response header and writes one JSON
line per request (method, path, status, ms, which credential was sent - never its value, never
the question). See observability.py for the log format.
"""

import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from rag_platform import llm
from rag_platform.api import retention, service
from rag_platform.api.auth import router as auth_router
from rag_platform.api.chat import router as chat_router
from rag_platform.api.collections import remove_collection
from rag_platform.api.collections import router as collections_router
from rag_platform.api.deps import (
    Caller,
    Services,
    can_manage,
    current_caller,
    get_services,
    manageable_collection,
    optional_caller,
    private_owner,
    require_admin,
    visible_collection,
    write_lock,
)
from rag_platform.api.evaluation import router as evaluation_router
from rag_platform.api.models import router as models_router
from rag_platform.api.schemas import (
    AskRequest,
    AskResponse,
    AssistantProfile,
    ChatProvider,
    DocumentInfo,
    Me,
    ModeChange,
    ModelChange,
    Topic,
    UploadResponse,
    UserInfo,
)
from rag_platform.api.uploads import UploadError, check_content, safe_filename, suspicious_lines
from rag_platform.api.usage import router as usage_router
from rag_platform.config import (
    DEFAULT_COLLECTION_ID,
    RagConfig,
    Settings,
    load_rag_config,
    save_rag_config,
)
from rag_platform.ingestion.chunking import chunk_text
from rag_platform.ingestion.loaders import load_document
from rag_platform.ingestion.pipeline import delete, doc_key, ingest
from rag_platform.observability import request_id


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup / shutdown (.NET: IHostedService.StartAsync / StopAsync). Code before `yield` runs
    when the server starts, code after it when it stops. Tests' TestClient(app) skips it."""
    cleanup = asyncio.create_task(retention.run_daily(Settings()))  # plan B10, once a day
    yield
    cleanup.cancel()


app = FastAPI(
    title="Agentic RAG Platform",
    description="Cited answers from your documents, with abstention and evaluation.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: browsers block a web page from calling an API on another origin (here the Angular dev
# server on :4200 calling this API on :8000) unless the API says that origin is allowed. Only
# the configured origins, methods and headers are allowed (.NET: app.UseCors with a policy).
app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings().cors_origin_list,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # PUT: answer feedback
    allow_headers=["Authorization", "X-API-Key", "X-Guest-Id", "Content-Type"],
    expose_headers=["X-Request-ID"],  # lets the browser read the id (e.g. for a bug report)
)

log = logging.getLogger("rag_platform.api")
SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9-]{8,64}")  # a caller's id is reused only if harmless
QUIET_PATHS = {"/health", "/ready"}  # Docker probes every 30 s: logged only when they fail


@app.middleware("http")  # runs around every request (.NET: app.Use(async (ctx, next) => ...))
async def request_log(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", "")
    if not SAFE_REQUEST_ID.fullmatch(rid):
        rid = uuid.uuid4().hex[:12]
    # Not reset afterwards on purpose: each request runs in its own context, and uvicorn's log of
    # an unhandled error (written after this function) then still carries the id.
    request_id.set(rid)
    started = time.perf_counter()
    fields = {
        "method": request.method,
        "path": request.url.path,  # no query string
        "auth": _credential(request),
    }
    try:
        response = await call_next(request)
    except Exception:
        # uvicorn logs the traceback right after this (with the same request_id).
        log.error("request", extra=fields | {"status": 500, "ms": _ms(started)})
        raise
    response.headers["X-Request-ID"] = rid
    if request.url.path not in QUIET_PATHS or response.status_code >= 400:
        level = logging.WARNING if response.status_code >= 500 else logging.INFO
        log.log(
            level, "request", extra=fields | {"status": response.status_code, "ms": _ms(started)}
        )
    return response


def _credential(request: Request) -> str:
    """Which credential the caller sent (not whether it was valid - the status shows that)."""
    if "authorization" in request.headers:
        return "session"
    if "x-api-key" in request.headers:
        return "api-key"
    return "guest-id" if "x-guest-id" in request.headers else "none"


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


# Every route on this router requires a valid admin key (one declaration instead of per route).
api = APIRouter(dependencies=[Depends(require_admin)])


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness: answers as long as the process runs (a restart is the only fix if it doesn't)."""
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
def ready(response: Response, svc: Services = Depends(get_services)) -> dict:
    """Readiness (plan D2): can this instance answer questions right now?

    Liveness vs readiness - the split Kubernetes and cloud container platforms use (.NET:
    MapHealthChecks with tagged checks): /health only proves the process runs; /ready also
    checks what a question needs. Docker's healthcheck calls it (docker-compose.yml), so
    `docker ps` shows "unhealthy" instead of "Up" when, e.g., the database is broken.

    No model call (it would spend quota every 30 s): a missing GEMINI_API_KEY already fails in
    get_services with 503. Failures name only the error type, as the endpoint is open.
    The search-index check loads FAISS into the in-memory cache, so it also warms it up.
    """
    checks = {"model_key": "ok"}  # get_services() succeeded, so a key is configured
    try:
        load_rag_config(svc.config_path)
        checks["config"] = "ok"
    except Exception as e:  # noqa: BLE001 - report every failure, never crash the probe
        checks["config"] = f"failed: {type(e).__name__}"
    try:
        store = svc.open_store()
        try:
            passages, _ = store.chunks_version()
            checks["database"] = "ok"
            svc.search.get(store, svc, need_bm25=False)
            checks["search_index"] = f"ok ({passages} passages)"
        finally:
            store.close()
    except Exception as e:  # noqa: BLE001
        failed = f"failed: {type(e).__name__}"
        if "database" in checks:  # the database answered; loading the index failed
            checks["search_index"] = failed
        else:
            checks["database"], checks["search_index"] = failed, "not checked"
    ok = all(v.startswith("ok") for v in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ok else "not ready", "checks": checks}


@app.get("/assistant", response_model=AssistantProfile, tags=["system"])
def assistant(
    collection_id: int = Query(DEFAULT_COLLECTION_ID, description="Topics of this collection"),
    caller: Caller | None = Depends(optional_caller),
    svc: Services = Depends(get_services),
) -> AssistantProfile:
    """Public on purpose (like /health): it only describes what the documents cover.

    Built from assistant.json AND the indexed documents (no LLM call):
    - hand-written topics whose documents are not indexed (deleted) are left out, so every
      example shown can be answered;
    - every indexed document no topic mentions (e.g. a new upload) gets a generic topic of its
      own, with an overview question the pipeline answers as a cited summary.
    """
    path = svc.assistant_path
    profile = (
        AssistantProfile.model_validate_json(path.read_text("utf-8"))
        if path.exists()
        else AssistantProfile()
    )
    store = svc.open_store()
    try:
        visible_collection(store, caller, collection_id)  # someone else's private one: 404
        indexed = sorted({Path(d["path"]).name for d in store.list_documents(collection_id)})
    finally:
        store.close()
    topics = [t for t in profile.topics if not t.documents or set(indexed) & set(t.documents)]
    covered = {doc for t in topics for doc in t.documents}
    topics += [_document_topic(name) for name in indexed if name not in covered]
    profile.topics = topics
    profile.chat_retention_days = svc.settings.chat_retention_days  # privacy note in the chat
    return profile


def _document_topic(file_name: str) -> Topic:
    """Generic topic for a document assistant.json doesn't describe: "team_notes.md" ->
    "Team notes" with the example "What does Team notes cover?"."""
    title = Path(file_name).stem.replace("_", " ").replace("-", " ").strip()
    title = title[:1].upper() + title[1:]
    return Topic(
        title=title,
        description=f"Uploaded document: {file_name}",
        examples=[f"What does {title} cover?"],
        documents=[file_name],
        generated=True,
    )


@app.get("/me", response_model=Me, tags=["system"])
def me(caller: Caller = Depends(current_caller)) -> Me:
    """Who is calling: the UI shows the Admin pages only for role "admin"."""
    user = UserInfo(**caller.user) if caller.user else None
    role = "admin" if caller.is_admin else ("user" if user else "guest")
    return Me(role=role, user=user)


@app.delete("/me", status_code=status.HTTP_204_NO_CONTENT, tags=["system"])
def delete_my_account(
    caller: Caller = Depends(current_caller), svc: Services = Depends(get_services)
) -> None:
    """ "Delete my account" (privacy, plan B10 + section I): the signed-in user's private
    collections (documents, search indexes, files), their chats and the account with its
    sessions. Shared collections are untouched. Signing in again later starts from zero."""
    owner = private_owner(caller)
    if owner is None:
        raise HTTPException(403, "Only a signed-in user can delete an account")
    with write_lock:
        store = svc.open_store()
        try:
            for collection_id in store.collection_ids_of(owner):
                remove_collection(svc, store, collection_id)
            store.delete_user(caller.user["id"], owner)
        finally:
            store.close()
    log.info("account deleted", extra={"user_id": caller.user["id"]})


@api.post("/ask", response_model=AskResponse, tags=["answers"])
def ask(
    req: AskRequest,
    caller: Caller = Depends(current_caller),
    svc: Services = Depends(get_services),
) -> AskResponse:
    """One independent question (no conversation memory); see POST /chat for conversations."""
    store = svc.open_store()
    try:
        visible_collection(store, caller, req.collection_id)  # admins: shared ones only
    finally:
        store.close()
    cfg = load_rag_config(svc.config_path, mode=req.mode, retrieval=req.retrieval, top_k=req.top_k)
    ans = service.answer(svc, cfg, req.question, req.collection_id)
    return AskResponse(
        question=req.question,
        answerable=ans.answerable,
        answer=ans.text,
        sources=service.sources_of(ans),
        abstain_reason=ans.abstain_reason,
        mode=cfg.mode,
        steps=ans.steps,
    )


@app.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
def list_documents(
    collection_id: int | None = Query(None, description="Only this collection (default: all)"),
    caller: Caller = Depends(current_caller),
    svc: Services = Depends(get_services),
) -> list[dict]:
    """Documents of the collections the caller may manage - admins: the shared ones; every
    signed-in user: their own private ones. Admins never list users' private documents."""
    if not caller.is_admin and caller.user is None:
        raise HTTPException(403, "Sign in with Google to manage your own documents")
    store = svc.open_store()
    try:
        if collection_id is not None:
            manageable_collection(store, caller, collection_id)
            return _with_warnings(store, store.list_documents(collection_id))
        mine = {c["id"] for c in store.list_collections() if can_manage(caller, c)}
        docs = [d for d in store.list_documents() if d["collection_id"] in mine]
        return _with_warnings(store, docs)
    finally:
        store.close()


def _with_warnings(store, docs: list[dict]) -> list[dict]:
    """Add the prompt-injection scan (B6) of each document's stored text. Scanned on every
    listing (milliseconds for a few documents), so documents added with the CLI are covered and
    a fixed or replaced document loses its warning right away."""
    texts = store.document_texts()
    return [d | {"warnings": suspicious_lines(texts.get(d["path"], ""))} for d in docs]


@app.post(
    "/documents",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
def upload_document(
    file: UploadFile = File(..., description="A .md, .txt, .pdf or .docx file"),
    collection_id: int = Form(DEFAULT_COLLECTION_ID, description="Collection to add it to"),
    caller: Caller = Depends(current_caller),
    svc: Services = Depends(get_services),
) -> UploadResponse:
    """Upload into a collection the caller may manage: a shared one (admins) or the caller's
    own private one (any signed-in user, within the per-user limits)."""
    s = svc.settings
    store = svc.open_store()
    try:
        collection = manageable_collection(store, caller, collection_id)
    finally:
        store.close()
    max_bytes = s.max_upload_mb * 1024 * 1024
    try:
        name = safe_filename(file.filename or "")
        data = file.file.read(max_bytes + 1)  # never read more than limit + 1 into memory
        check_content(name, data, max_bytes)
    except UploadError as e:
        raise HTTPException(e.status, e.message) from None

    folder = _upload_folder(s, collection_id)
    path = folder / name
    with write_lock:  # one writer at a time for SQLite + FAISS
        folder.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        previous = path.read_bytes() if existed else None
        path.write_bytes(data)
        try:
            doc = load_document(path)  # the parser must be able to read it, before we index it
        except Exception as e:  # noqa: BLE001 - corrupt file despite a valid signature
            _restore(path, previous)
            raise HTTPException(422, f"Could not read the document: {type(e).__name__}") from e
        store = svc.open_store()
        try:
            cfg: RagConfig = load_rag_config(svc.config_path)
            if collection["owner"]:  # private: the user's limits, checked BEFORE embedding
                refused = _over_user_limits(store, s, collection["owner"], path, doc.text, cfg)
                if refused:
                    _restore(path, previous)
                    raise HTTPException(403, refused)
            try:
                index_path = s.collection_index_path(collection_id)
                st = ingest(
                    path, store, index_path, svc.embedder, s.embedding_model, cfg, collection_id
                )
            except Exception as e:  # noqa: BLE001 - e.g. embedding quota
                _restore(path, previous)
                raise service.provider_error(e) from e
            service.recalibrate(svc, store, collection_id)  # its threshold, measured again
            key = doc_key(path)
            info = next(
                d for d in _with_warnings(store, store.list_documents()) if d["path"] == key
            )
        finally:
            store.close()
    if info["warnings"]:  # visible in the API log too (D1), without the document's text
        log.warning("suspicious document", extra={"path": key, "lines": len(info["warnings"])})
    state = "unchanged" if st.unchanged else ("updated" if st.updated else "added")
    return UploadResponse(
        document=DocumentInfo(**info),
        status=state,
        embeddings_new=st.embedded,
        embeddings_cached=st.cache_hits,
    )


def _over_user_limits(store, s: Settings, owner: str, path: Path, text: str, cfg) -> str:
    """Why this upload would exceed the user's limits ('' = it fits). Chunking is free, so the
    new passage count is known before any embedding call; replacing a document frees its old
    passages first."""
    usage = store.owner_usage(owner)
    existing = store.get_document(doc_key(path))
    old_passages = len(store.chunk_ids(existing["id"])) if existing else 0
    if not existing and usage["documents"] >= s.user_max_documents:
        return f"You can store at most {s.user_max_documents} documents in your collections"
    passages = usage["passages"] - old_passages + len(chunk_text(text, cfg))
    if passages > s.user_max_passages:
        return (
            f"This document would bring your collections to {passages} passages; the limit is "
            f"{s.user_max_passages}. Remove a document or upload a shorter one."
        )
    return ""


def _upload_folder(s: Settings, collection_id: int) -> Path:
    """Uploads of the demo collection stay in data/uploads/ (as before collections); others go to
    data/uploads/<collection id>/, so the same file name can exist in two collections."""
    if collection_id == DEFAULT_COLLECTION_ID:
        return s.upload_dir
    return s.upload_dir / str(collection_id)


def _restore(path: Path, previous: bytes | None) -> None:
    """Undo a failed upload: put the old file back, or remove the new one."""
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(previous)


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"])
def delete_document(
    doc_id: int, caller: Caller = Depends(current_caller), svc: Services = Depends(get_services)
) -> None:
    s = svc.settings
    with write_lock:
        store = svc.open_store()
        try:
            row = store.get_document_by_id(doc_id)
            if row is None:
                raise HTTPException(404, f"No document with id {doc_id}")
            collection_id = row["collection_id"]
            manageable_collection(store, caller, collection_id)  # 404 / 403 like an upload
            index_path = s.collection_index_path(collection_id)
            delete(Path(row["path"]), store, index_path, s.embedding_model, collection_id)
            service.recalibrate(svc, store, collection_id)
        finally:
            store.close()
        # Files uploaded through the API are owned by the API: remove them too, so a later
        # `rag ingest data/uploads` doesn't bring the document back. Other sources are untouched.
        file = Path(row["path"])
        if file.resolve().parent == _upload_folder(s, collection_id).resolve():
            file.unlink(missing_ok=True)


@api.get("/config", response_model=RagConfig, tags=["system"])
def get_config(svc: Services = Depends(get_services)) -> RagConfig:
    return load_rag_config(svc.config_path)


@api.put("/config/mode", response_model=RagConfig, tags=["system"])
def set_mode(body: ModeChange, svc: Services = Depends(get_services)) -> RagConfig:
    """Switch the answer mode directly (owner decision, 2026-10-22). Unlike chunking, search or
    the threshold - changed only through a measured, approved optimizer recommendation - the
    mode is a cost / quality choice the admin makes; it applies from the next question."""
    with write_lock:  # one writer of rag_config.json at a time (also used by "apply")
        cfg = load_rag_config(svc.config_path, mode=body.mode)
        save_rag_config(cfg, svc.config_path)
    return cfg


@api.get("/config/providers", response_model=list[ChatProvider], tags=["system"])
def chat_providers(svc: Services = Depends(get_services)) -> list[ChatProvider]:
    """The chat providers the server knows and whether each has its key (never the key)."""
    configured = llm.chat_providers(svc.settings)
    return [
        ChatProvider(name="gemini", model=svc.settings.chat_model, configured=True),
        ChatProvider(name="groq", model=svc.settings.groq_model, configured="groq" in configured),
    ]


@api.put("/config/model", response_model=RagConfig, tags=["system"])
def set_model(body: ModelChange, svc: Services = Depends(get_services)) -> RagConfig:
    """Answer model (plan G3): which provider writes answers first and whether the other one
    takes over when it is out of quota or overloaded. Applies from the next question."""
    if body.chat_provider not in llm.chat_providers(svc.settings):
        raise HTTPException(422, f"{body.chat_provider} is not set up on the server (API key)")
    with write_lock:
        cfg = load_rag_config(
            svc.config_path, chat_provider=body.chat_provider, chat_fallback=body.chat_fallback
        )
        save_rag_config(cfg, svc.config_path)
    return cfg


app.include_router(api)
app.include_router(auth_router)  # /auth/... sign in with Google (api/auth.py)
app.include_router(chat_router)  # /chat, /conversations, /messages (api/chat.py)
app.include_router(collections_router)  # /collections (api/collections.py)
app.include_router(evaluation_router)  # /eval/..., /optimizer/... (api/evaluation.py)
app.include_router(usage_router)  # /usage/... (api/usage.py)
app.include_router(models_router)  # /config/models... (api/models.py)
