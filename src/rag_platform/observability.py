"""Observability: LangSmith tracing (hosted) + a local usage log (SQLite) of every model call.

Two complementary views (Phase 7):
- LangSmith (optional): every request becomes a TRACE - a tree of steps with prompts, outputs,
  tokens, latency and errors - viewed in the LangSmith web app. LangChain sends traces by itself
  once LANGSMITH_TRACING / LANGSMITH_API_KEY are environment variables; `configure_tracing`
  copies them from .env (pydantic-settings reads .env into Settings, not into os.environ).
  The live API is traced; CLI batch runs (eval, optimize) only with --trace-langsmith, so the
  free tier's monthly trace allowance isn't used up by batch jobs.
- Usage log (always on, local, no limits): one row per model call - model, purpose, tokens,
  latency, error - written by `UsageLogger`, a LangChain callback handler attached to every chat
  model. The Admin › Usage page aggregates it (`usage_report`); cost is computed only from
  prices the owner enters in model_prices.json (no real prices are hard-coded).

- Application log (plan D1, API only): one JSON line per event on stdout - per HTTP request
  (method, path, status, duration), per model call (purpose, tokens, latency, LangSmith run id)
  and per answer (route, answerable, citations). Every line of one request carries the same
  `request_id`, so `docker compose logs api | grep <id>` shows that request's whole story, and
  the run id leads on to its LangSmith trace. Question and answer TEXT is never logged (privacy:
  logs get copied to more places than the database); the database keeps the conversation.

Purpose labels come from LangChain run metadata: callers pass config={"metadata": {"purpose":
"answer"}}; LangGraph adds "langgraph_node" (plan, grade, verify, supervisor...) to every call
it makes, so agent steps are labelled automatically. Evaluation code does not depend on this
log (the CI gate never reads it).

.NET analogy: LangSmith ~ Application Insights distributed tracing; UsageLogger ~ a
DelegatingHandler that records every outgoing call's telemetry; the application log ~ ILogger
with Serilog's JSON output and a correlation-id middleware.
"""

import json
import logging
import os
import time
from collections import defaultdict
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

PRICES_FILE = Path("model_prices.json")
MAX_CALLS = 500  # calls returned to the Usage page per request
LANGSMITH_US = "https://smith.langchain.com"

log = logging.getLogger("rag_platform.model")


# --- Application log (JSON lines) ------------------------------------------------------------

# The current request's id. A ContextVar is like .NET's AsyncLocal<T>: each request (async task,
# and the worker thread FastAPI runs a sync endpoint in) sees its own value, so log lines written
# deep inside the RAG code still know which request they belong to.
request_id: ContextVar[str] = ContextVar("request_id", default="")

# Attributes every LogRecord has; anything else was passed via `extra={...}` and is output.
# ("color_message": uvicorn's copy of its message with terminal colour codes.)
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "color_message",
}
_default_record = logging.getLogRecordFactory()


def _record_with_request_id(*args, **kwargs) -> logging.LogRecord:
    """Stamp the current request id on every log record when it is created (in the request's
    context), so it is right even if the record is formatted later or elsewhere."""
    record = _default_record(*args, **kwargs)
    record.request_id = request_id.get()
    return record


logging.setLogRecordFactory(_record_with_request_id)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, msg, request_id, then the `extra` fields.

    Usage: log.info("answer", extra={"route": "fast", "citations": 2}). JSON lines are what log
    tools (Docker, Azure Monitor, Grafana Loki) parse and filter without regexes.
    """

    def format(self, record: logging.LogRecord) -> str:
        line = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = {k: v for k, v in vars(record).items() if k not in _STANDARD}
        if rid := extra.pop("request_id", ""):  # := assigns and tests in one step
            line["request_id"] = rid
        line |= extra
        if record.exc_info:
            line["traceback"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str, ensure_ascii=False)


# Logging setup for `rag serve`, passed to uvicorn (which applies it in every worker process,
# also with --reload). Our own loggers and uvicorn's log INFO; third-party libraries only
# WARNING - e.g. HTTP client libraries log every request URL at INFO.
# Like appsettings.json "Logging" with a default level and per-namespace overrides.
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"json": {"()": JsonFormatter}},
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {"handlers": ["stdout"], "level": "WARNING"},
    "loggers": {
        "rag_platform": {"level": "INFO"},
        "uvicorn": {"level": "INFO"},
        "uvicorn.access": {"level": "WARNING"},  # our request log replaces its access lines
    },
}

# Without a configured handler (CLI, tests) Python would print warnings from our loggers to
# stderr; a NullHandler keeps the CLI output as it is. `rag serve` routes them to stdout.
logging.getLogger("rag_platform").addHandler(logging.NullHandler())


# --- LangSmith -------------------------------------------------------------------------------


def configure_tracing(s) -> bool:
    """Export the LANGSMITH_* settings to the process environment so LangChain traces.

    Must run before the first model call: langsmith caches these variables on first read.
    Values already set in the real environment win (setdefault). Returns True if tracing is on.
    """
    if not (s.langsmith_tracing and s.langsmith_api_key):
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", s.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", s.langsmith_project)
    # 1.0 = every request; e.g. 0.5 = half of them (keeps volume inside the free allowance).
    os.environ.setdefault("LANGSMITH_TRACING_SAMPLING_RATE", str(s.langsmith_sampling_rate))
    if s.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", s.langsmith_endpoint)
    return True


def tracing_info(s) -> dict:
    """For the Usage page: is tracing on, which project, where to look (never the key)."""
    enabled = bool(s.langsmith_tracing and s.langsmith_api_key)
    web = LANGSMITH_US
    if s.langsmith_endpoint and "eu." in s.langsmith_endpoint:
        web = "https://eu.smith.langchain.com"
    return {
        "enabled": enabled,
        "project": s.langsmith_project,
        "sampling_rate": s.langsmith_sampling_rate,
        "url": web if enabled else None,
    }


def trace_url(run_id: str, project: str) -> str:
    """Link to one call's trace in LangSmith (one lookup request to LangSmith, no LLM call)."""
    from langsmith import Client  # imported lazily: only needed when someone clicks a link

    client = Client()
    return client.get_run_url(run=client.read_run(run_id), project_name=project)


# --- Usage log ---------------------------------------------------------------------------------


def purpose_of(metadata: dict | None, default: str) -> str:
    """Explicit purpose first, then the LangGraph node (agent step), then the model's default."""
    metadata = metadata or {}
    if metadata.get("purpose"):
        return metadata["purpose"]
    if metadata.get("langgraph_node"):
        return f"agent: {metadata['langgraph_node']}"
    return default


def _usage(response: Any) -> tuple[int | None, int | None]:
    """(input_tokens, output_tokens) from a LangChain LLMResult, if the provider reported them."""
    try:
        meta = response.generations[0][0].message.usage_metadata or {}
        return meta.get("input_tokens"), meta.get("output_tokens")
    except (AttributeError, IndexError, TypeError):
        return None, None


class UsageLogger(BaseCallbackHandler):
    """Records one row per chat-model call in SQLite (model, purpose, tokens, latency, error).

    A LangChain callback handler: LangChain calls on_chat_model_start before the request and
    on_llm_end / on_llm_error after it, with the same run_id. Logging must never break a user
    request, so every failure here is swallowed.
    """

    def __init__(self, db_path: Path, model: str, default_purpose: str = "chat"):
        self.db_path, self.model, self.default_purpose = db_path, model, default_purpose
        self._started: dict[Any, tuple[float, str]] = {}  # run_id -> (start time, purpose)

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self._started[run_id] = (time.perf_counter(), purpose_of(metadata, self.default_purpose))

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs):
        self._started[run_id] = (time.perf_counter(), purpose_of(metadata, self.default_purpose))

    def on_llm_end(self, response, *, run_id, **kwargs):
        tokens_in, tokens_out = _usage(response)
        self._write(run_id, tokens_in, tokens_out, None)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._write(run_id, None, None, f"{type(error).__name__}: {str(error)[:200]}")

    def _write(self, run_id, tokens_in, tokens_out, error) -> None:
        started, purpose = self._started.pop(run_id, (time.perf_counter(), self.default_purpose))
        _record(
            self.db_path,
            model=self.model,
            purpose=purpose,
            kind="chat",
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            texts=None,
            latency_ms=round((time.perf_counter() - started) * 1000),
            error=error,
            run_id=str(run_id),
        )


def log_embedding(db_path: Path | None, model: str, texts: int, started: float) -> None:
    """One row per embedding request (Gemini reports no token counts for embeddings)."""
    if db_path is not None:
        _record(
            db_path,
            model=model,
            purpose="embed",
            kind="embed",
            input_tokens=None,
            output_tokens=None,
            texts=texts,
            latency_ms=round((time.perf_counter() - started) * 1000),
            error=None,
            run_id=None,
        )


def _record(db_path: Path, **row) -> None:
    # The application log gets the same facts (never prompt or answer text), tied to the request.
    fields = {k: v for k, v in row.items() if v is not None}
    log.log(logging.WARNING if row["error"] else logging.INFO, "model call", extra=fields)
    try:
        from rag_platform.store.db import Store  # local import: store imports nothing from here

        store = Store(db_path)
        try:
            store.add_llm_call(**row)
        finally:
            store.close()
    except Exception:  # noqa: BLE001 - telemetry must never fail the request
        pass


# --- Usage report ------------------------------------------------------------------------------


def load_prices(path: Path = PRICES_FILE) -> dict:
    """{model: {"input_per_million": float, "output_per_million": float}} entered by the owner."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text("utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}  # "_comment" keys ignored


def _cost(row: dict, prices: dict) -> float | None:
    p = prices.get(row["model"])
    if not p or row["kind"] != "chat":
        return None
    tin, tout = row["input_tokens"] or 0, row["output_tokens"] or 0
    return tin / 1e6 * p.get("input_per_million", 0) + tout / 1e6 * p.get("output_per_million", 0)


def _p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def _summary(group: list[dict]) -> dict:
    chats = [r for r in group if r["kind"] == "chat"]
    costs = [r["cost"] for r in group if r["cost"] is not None]
    # Latency of chat calls; a group with only embeddings (e.g. the embedding model) uses those.
    timed = chats or [r for r in group if r["kind"] == "embed"]
    latencies = [r["latency_ms"] for r in timed if r["error"] is None]
    return {
        "calls": len(chats),
        "embed_texts": sum(r["texts"] or 0 for r in group if r["kind"] == "embed"),
        "input_tokens": sum(r["input_tokens"] or 0 for r in chats),
        "output_tokens": sum(r["output_tokens"] or 0 for r in chats),
        "errors": sum(1 for r in group if r["error"]),
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "p95_latency_ms": _p95(latencies),
        "cost": round(sum(costs), 6) if costs else None,
    }


def usage_report(rows: list[dict], prices: dict, days: int, today: datetime | None = None) -> dict:
    """Aggregate llm_calls rows of the last `days` days for the Usage page."""
    today_date = (today or datetime.now(UTC)).date()
    since = today_date - timedelta(days=days - 1)
    rows = [dict(r) for r in rows if datetime.fromisoformat(r["created_at"]).date() >= since]
    for r in rows:
        r["cost"] = _cost(r, prices)

    groups: dict[str, dict[str, list]] = {
        "purpose": defaultdict(list),
        "model": defaultdict(list),
        "day": defaultdict(list),
    }
    for r in rows:
        groups["purpose"][r["purpose"]].append(r)
        groups["model"][r["model"]].append(r)
        groups["day"][r["created_at"][:10]].append(r)

    per_day = []
    for i in range(days):  # every day in the window, also empty ones (for the chart)
        d = (since + timedelta(days=i)).isoformat()
        per_day.append({"date": d, **_summary(groups["day"].get(d, []))})

    def ranked(kind: str) -> list[dict]:
        items = [{kind: k, **_summary(v)} for k, v in groups[kind].items()]
        return sorted(items, key=lambda x: (-x["calls"], -x["embed_texts"]))

    keep = ("id", "created_at", "model", "purpose", "kind", "latency_ms", "input_tokens",
            "output_tokens", "texts", "error", "run_id", "cost")  # fmt: skip
    return {
        "days": days,
        "totals": _summary(rows),
        "priced_models": sorted(prices),
        "by_purpose": ranked("purpose"),
        "by_model": ranked("model"),
        "per_day": per_day,
        # Every call of the window, newest first (capped): the UI filters, sorts and searches it.
        "calls": [
            {k: r.get(k) for k in keep}
            for r in sorted(rows, key=lambda r: (r["created_at"], r.get("id") or 0), reverse=True)[
                :MAX_CALLS
            ]
        ],
        "recent_errors": [
            {k: r[k] for k in ("created_at", "model", "purpose", "error")}
            for r in sorted(rows, key=lambda r: r["created_at"], reverse=True)
            if r["error"]
        ][:10],
    }
