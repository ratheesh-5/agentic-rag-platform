"""Factory functions for the AI models (the only place that knows we use Gemini and Groq).

The rest of the code depends on LangChain's abstract types, not on Gemini classes:
- `BaseChatModel` - any chat LLM: `.invoke(messages)` returns the model's reply.
- `Embeddings`    - any embedding model: `.embed_documents(texts)` / `.embed_query(text)`
                    return vectors (lists of floats).
.NET analogy: these are interfaces (`IChatClient`, `IEmbeddingGenerator` in
Microsoft.Extensions.AI) and these functions are the composition root. Swapping providers means
changing only this file; tests pass fake implementations of the same interfaces.

Rate limiting: hosted models enforce requests-per-minute quotas (the Gemini free tier allows
15/min per chat model). Instead of failing with RESOURCE_EXHAUSTED mid-run, every chat model
gets a client-side token-bucket limiter that makes calls wait for a free slot. The limiter is
shared per model name, because the provider's quota is per model, not per client object.

Usage log (Phase 7): every chat model gets a UsageLogger callback (one SQLite row per call) and
embeddings are logged by RateLimitedEmbeddings; see observability.py. LangSmith tracing is
switched on by the API (deps.get_services) and by CLI commands given --trace-langsmith.

Two chat providers (plan G2/G5): Gemini, and Groq when GROQ_API_KEY is set. FallbackChat puts
them in the order chosen in Admin › Settings and moves on to the next one when a provider is out
of quota, overloaded or too slow. Embeddings always come from Gemini: vectors of different
models can't be compared, so switching would mean re-indexing every document.

Imports are inside the functions so that commands which never call a model (e.g. `rag inspect`)
don't pay the cost of loading the Google SDK.
"""

import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from rag_platform.config import RAG_CONFIG_FILE, Settings, load_rag_config
from rag_platform.observability import UsageLogger, log_embedding

log = logging.getLogger(__name__)

_limiters: dict[tuple, InMemoryRateLimiter] = {}  # one shared limiter per chat model name


def _rate_limiter(model: str, requests_per_minute: int, burst: int = 1) -> InMemoryRateLimiter:
    """Token bucket (.NET: TokenBucketRateLimiter): tokens refill at a steady rate and are saved
    up to `burst` while idle, so a question's 2 calls go out without waiting. Refill = limit -
    burst per minute, so even a full burst plus a minute of refills stays within the limit."""
    key = (model, requests_per_minute, burst)
    if key not in _limiters:
        _limiters[key] = InMemoryRateLimiter(
            requests_per_second=max(1, requests_per_minute - burst) / 60,
            check_every_n_seconds=0.1,
            max_bucket_size=burst,
        )
        # The limiter starts its refill clock at its first acquire(); start it now, so the
        # time until the first question already fills the bucket (else that question waits).
        _limiters[key].acquire(blocking=False)
    return _limiters[key]


def _chat(s: Settings, model: str, purpose: str) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=s.gemini_api_key,
        rate_limiter=_rate_limiter(model, s.chat_requests_per_minute, s.chat_burst),
        timeout=s.chat_timeout_seconds,  # per attempt; library default = wait forever
        max_retries=s.chat_max_retries,  # library default 6 retries with growing waits
        # Usage log: purpose = run metadata if the caller set one, else this default.
        callbacks=[UsageLogger(s.db_path, model, default_purpose=purpose)],
    )


def _groq(s: Settings, purpose: str, model: str = "") -> BaseChatModel:
    from langchain_groq import ChatGroq

    class JsonSchemaGroq(ChatGroq):
        """Structured output through JSON schema instead of the default tool call: tested on
        golden questions (2026-10-24), gpt-oss answered correctly but dropped the [P1] citation
        markers with tool calls, and kept them with JSON schema."""

        def with_structured_output(self, schema, *, method="json_schema", **kwargs):
            return super().with_structured_output(schema, method=method, **kwargs)

    model = model or s.groq_model
    return JsonSchemaGroq(
        model=model,
        api_key=s.groq_api_key,
        rate_limiter=_rate_limiter(model, s.groq_requests_per_minute, s.chat_burst),
        timeout=s.chat_timeout_seconds,
        max_retries=s.chat_max_retries,
        callbacks=[UsageLogger(s.db_path, model, default_purpose=purpose)],
    )


def provider_problem(e: Exception) -> str | None:
    """ "quota" (rate limit / quota exceeded) or "slow" (timeout, overloaded, 503/504) - problems
    of one provider that another one may not have; None for anything else (a bad request, a bug).
    Used for the fallback below and for the API's error messages (api/service.provider_error)."""
    text = f"{type(e).__name__} {e}"
    if "RateLimit" in text or "RESOURCE_EXHAUSTED" in text or " 429" in text:
        return "quota"
    slow = ("Timeout", "DEADLINE_EXCEEDED", "UNAVAILABLE", "overloaded", " 503", " 504")
    if any(x in text for x in slow):
        return "slow"
    return None


class FallbackChat:
    """The app's chat model: Gemini and/or Groq, in the order chosen in Admin › Settings.

    It offers what this app calls on a chat model (invoke, with_structured_output, bind_tools),
    so the rest of the code doesn't know there are two providers (.NET: a decorator implementing
    the same interface). The order is read from rag_config.json on every call: a change in
    Settings applies from the next question, no restart. When a provider fails with a quota,
    overload or timeout error, the next one gets the same request; any other error (a bad
    request, a bug) is raised at once - switching provider would only hide it.
    """

    def __init__(
        self,
        models: dict[str, BaseChatModel],
        config_path: Path = RAG_CONFIG_FILE,
        factory: Callable[[str, str], BaseChatModel] | None = None,
    ):
        self.models = models  # provider name -> its .env default model; only configured ones
        self.config_path = config_path
        # (provider, model name) -> a new model: for the models chosen in Admin › Settings.
        self.factory = factory
        self._named: dict[tuple[str, str], BaseChatModel] = {}  # built once per name
        self._lock = threading.Lock()  # requests run in a thread pool

    def order(self, cfg=None) -> list[str]:
        """Providers to try, first choice first (only the first one if fallback is off)."""
        cfg = cfg or load_rag_config(self.config_path)
        first = cfg.chat_provider if cfg.chat_provider in self.models else next(iter(self.models))
        rest = [name for name in self.models if name != first]
        return [first, *rest] if cfg.chat_fallback else [first]

    def named(self, provider: str, model: str = "") -> BaseChatModel:
        """A provider's model by name ('' = its .env default) - also used by the Settings
        "Test" button and quality check, which must call exactly that model, without fallback."""
        if not model or self.factory is None:
            return self.models[provider]
        with self._lock:
            if (provider, model) not in self._named:
                self._named[(provider, model)] = self.factory(provider, model)
            return self._named[(provider, model)]

    def attempts(self, cfg) -> list[tuple[str, str]]:
        """(provider, model) pairs to try in order: each provider's model, then its backup
        model (quota and overload are per model, so another model of the same provider often
        still works). Fallback off = exactly one model."""
        pairs = []
        for name in self.order(cfg):
            pairs.append((name, getattr(cfg, f"{name}_model", "")))
            backup = getattr(cfg, f"{name}_backup_model", "")
            if backup and self.factory is not None:
                pairs.append((name, backup))
        return pairs if cfg.chat_fallback else pairs[:1]

    def run(self, call: Callable[[BaseChatModel], object]):
        cfg = load_rag_config(self.config_path)  # read once per call: order + chosen models
        pairs = self.attempts(cfg)
        for i, (name, model) in enumerate(pairs):
            try:
                return call(self.named(name, model))
            except Exception as e:
                problem = provider_problem(e)
                if problem is None or i + 1 == len(pairs):
                    raise
                log.warning(  # the model calls themselves are in the usage log (llm_calls)
                    "chat model fallback",
                    extra={"provider": name, "model": model or "default", "problem": problem},
                )

    def invoke(self, messages, config=None, **kwargs):
        return self.run(lambda m: m.invoke(messages, config, **kwargs))

    def with_structured_output(self, schema, **kwargs) -> "_Bound":
        return _Bound(self, lambda m: m.with_structured_output(schema, **kwargs))

    def bind_tools(self, tools, **kwargs) -> "_Bound":
        return _Bound(self, lambda m: m.bind_tools(tools, **kwargs))


class _Bound:
    """with_structured_output / bind_tools of FallbackChat: the same wrapping, applied to
    whichever provider ends up answering."""

    def __init__(self, chat: FallbackChat, wrap: Callable[[BaseChatModel], object]):
        self.chat, self.wrap = chat, wrap

    def invoke(self, messages, config=None, **kwargs):
        return self.chat.run(lambda m: self.wrap(m).invoke(messages, config, **kwargs))


def chat_providers(s: Settings) -> dict[str, str]:
    """Configured chat providers -> their model names (Gemini always; Groq when keyed)."""
    providers = {"gemini": s.chat_model}
    if s.groq_api_key:
        providers["groq"] = s.groq_model
    return providers


def get_chat_model(s: Settings, config_path: Path = RAG_CONFIG_FILE) -> FallbackChat:
    models = {"gemini": _chat(s, s.chat_model, "chat")}
    if s.groq_api_key:
        models["groq"] = _groq(s, "chat")

    def factory(provider: str, model: str) -> BaseChatModel:
        return _chat(s, model, "chat") if provider == "gemini" else _groq(s, "chat", model)

    return FallbackChat(models, config_path, factory)


# Text-chat models only: the providers also list image, speech, transcription, robotics and
# guard models that can't write a cited answer.
_GEMINI_CHAT = re.compile(r"^gemini-\d+(\.\d+)?-(flash|pro)(-lite)?(-preview)?$")
_NOT_CHAT = ("whisper", "guard", "orpheus", "tts", "safeguard")
_model_lists: dict[str, tuple[float, list[str]]] = {}  # provider -> (fetched at, names)


def list_chat_models(s: Settings, provider: str) -> list[str]:
    """The chat models a provider offers today (cached 10 minutes; [] if the list can't be read
    - Settings then shows only the current model). Listing models costs no quota."""
    cached = _model_lists.get(provider)
    if cached and time.monotonic() - cached[0] < 600:
        return cached[1]
    try:
        if provider == "gemini":
            from google import genai

            # Keep the client in a variable while reading: the list is fetched page by page, and
            # a client nobody refers to any more is closed before the pages are read.
            client = genai.Client(api_key=s.gemini_api_key)
            names = [
                m.name.removeprefix("models/")
                for m in client.models.list()
                if "generateContent" in (getattr(m, "supported_actions", None) or [])
            ]
            names = [n for n in names if _GEMINI_CHAT.match(n)]
        else:
            import groq

            client = groq.Groq(api_key=s.groq_api_key)
            names = [
                m.id
                for m in client.models.list().data
                if not any(x in m.id for x in _NOT_CHAT)
                and (getattr(m, "context_window", 0) or 0) >= 32000
            ]
    except Exception as e:  # noqa: BLE001 - the list is a convenience, never an error
        log.warning(
            "model list unavailable", extra={"provider": provider, "error": type(e).__name__}
        )
        return []
    names = sorted(set(names))
    _model_lists[provider] = (time.monotonic(), names)
    return names


def get_judge_model(s: Settings, model: str = "") -> BaseChatModel:
    """LLM used as evaluation judge (by RAGAS, see evaluation/ragas_eval.py).

    Judges are usually run at temperature 0 for repeatability, but gemini-3.x models use fixed
    sampling defaults and ignore (and warn about) a temperature setting, so none is passed.
    Judge runs may therefore vary slightly between runs.
    """
    return _chat(s, model or s.judge_model or s.chat_model, "judge")


class RateLimitedEmbeddings(Embeddings):
    """Keeps embedding calls under a texts-per-minute budget (decorator around an Embeddings).

    The Gemini free tier allows 100 embedding requests per minute and EVERY TEXT in a batch
    counts as one request. Indexing 150 chunks in one batch would fail, so large batches are
    split and each part waits until it fits in the rolling 60-second window.
    """

    def __init__(
        self,
        inner: Embeddings,
        texts_per_minute: int,
        clock=time.monotonic,
        sleep=time.sleep,
        usage_db=None,  # SQLite db for the usage log (None = don't log, e.g. in tests)
        model: str = "",
    ):
        self.inner = inner
        self.usage_db, self.model = usage_db, model
        self.limit = texts_per_minute
        self.clock, self.sleep = clock, sleep  # injectable, so tests don't really wait
        self.sent: deque[float] = deque()  # send time of each text in the last 60 s
        # RAGAS embeds from worker threads: the check-and-reserve below must be atomic, or two
        # threads could both see room and exceed the limit (like `lock (...) { }` in C#).
        self.lock = threading.Lock()

    def _wait_for(self, n: int) -> None:
        while True:
            with self.lock:
                now = self.clock()
                while self.sent and now - self.sent[0] >= 60:
                    self.sent.popleft()  # older than a minute: no longer counts
                if len(self.sent) + n <= self.limit:
                    self.sent.extend([now] * n)
                    return
                wait = 60 - (now - self.sent[0]) + 0.1  # until the oldest text drops out
            self.sleep(wait)  # sleep outside the lock so other threads aren't blocked

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.limit):
            part = texts[i : i + self.limit]
            self._wait_for(len(part))
            started = time.perf_counter()
            out.extend(self.inner.embed_documents(part))
            log_embedding(self.usage_db, self.model, len(part), started)
        return out

    def embed_query(self, text: str) -> list[float]:
        self._wait_for(1)
        started = time.perf_counter()
        vector = self.inner.embed_query(text)
        log_embedding(self.usage_db, self.model, 1, started)
        return vector


def get_embedder(s: Settings) -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # Gemini embeddings are "asymmetric": embed_documents() uses a document task type and
    # embed_query() a query task type. Always use the matching method for each side.
    inner = GoogleGenerativeAIEmbeddings(model=s.embedding_model, google_api_key=s.gemini_api_key)
    return RateLimitedEmbeddings(
        inner, s.embed_requests_per_minute, usage_db=s.db_path, model=s.embedding_model
    )
