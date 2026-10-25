"""Admin › Settings › Models (plan G7): choose the answer and judge models without editing .env.

    GET  /config/models          current / default / backup model per slot, the model lists,
                                 the embedding model (read-only) and the latest quality checks
    PUT  /config/models          use the chosen models (saved in rag_config.json)
    POST /config/models/test     one tiny call to a model: does it answer at all?
    POST /config/models/check    quality check of an answer model, in the background

Why a check before use: a model can answer "hello" but still break the RAG contract (JSON
output, [P1] citations, saying "I don't know"). The check answers a few demo questions with
exactly that model (no fallback) and needs CHECK_MINIMUM passes; only then can it write answers
or serve as a backup. A backup model is another model of the same provider, tried when the main
one is out of quota or overloaded (Gemini's limits are per model; see llm.FallbackChat).
The judge needs no check, but its scores aren't comparable with another judge's, so every score
records its judge (answer_scores.judge) and the UI warns before switching.

The embedding model is only shown: changing it means re-indexing every document and
re-exporting the CI snapshot, so it stays a deliberate .env change.

Cost: listing models is free; Test = 1 call; a check = up to 6 answer calls.
"""

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from rag_platform import llm
from rag_platform.answering import build_answerer
from rag_platform.api import service
from rag_platform.api.deps import Services, get_services, require_admin, write_lock
from rag_platform.api.schemas import (
    ModelCheck,
    ModelCheckItem,
    ModelRequest,
    ModelsChange,
    ModelsInfo,
    ModelSlot,
    ModelTestResult,
)
from rag_platform.config import DEFAULT_COLLECTION_ID, RagConfig, load_rag_config, save_rag_config
from rag_platform.evaluation.golden import load_golden

router = APIRouter(dependencies=[Depends(require_admin)], tags=["system"])

# Five answerable demo questions from different documents + one the documents can't answer.
CHECK_IDS = ("q06", "q11", "q16", "q26", "q36", "u01")
CHECK_MINIMUM = 5
_check_lock = threading.Lock()  # one model check at a time (they share the provider quota)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _checks_path(svc: Services) -> Path:
    return svc.settings.index_dir / "model_checks.json"


def _load_checks(svc: Services) -> dict[str, ModelCheck]:
    """Latest check per "kind:model" (a JSON file: survives restarts, no table needed)."""
    path = _checks_path(svc)
    if not path.exists():
        return {}
    return {k: ModelCheck(**v) for k, v in json.loads(path.read_text("utf-8")).items()}


def _save_check(svc: Services, check: ModelCheck) -> None:
    with write_lock:
        checks = _load_checks(svc)
        checks[f"{check.kind}:{check.model}"] = check
        data = {k: c.model_dump() for k, c in checks.items()}
        _checks_path(svc).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _defaults(svc: Services) -> dict[str, str]:
    s = svc.settings
    return {"gemini": s.chat_model, "groq": s.groq_model, "judge": s.judge_model or s.chat_model}


def _model(svc: Services, kind: str, name: str):
    """Exactly this model, without fallback (tests: the fake LLM / patched judge factory)."""
    if kind == "judge":
        return llm.get_judge_model(svc.settings, name)
    named = getattr(svc.llm, "named", None)  # FallbackChat in production
    return named(kind, name) if named else svc.llm


@router.get("/config/models", response_model=ModelsInfo)
def models_info(svc: Services = Depends(get_services)) -> ModelsInfo:
    """What Settings › Models shows. Model lists are cached 10 minutes; no model calls."""
    cfg = load_rag_config(svc.config_path)
    configured = llm.chat_providers(svc.settings)
    defaults = _defaults(svc)
    slots = []
    for kind in ("gemini", "groq", "judge"):
        provider = "gemini" if kind == "judge" else kind  # the judge stays on Gemini
        ok = provider in configured
        slots.append(
            ModelSlot(
                kind=kind,
                configured=ok,
                current=getattr(cfg, f"{kind}_model") or defaults[kind],
                default=defaults[kind],
                backup=getattr(cfg, f"{kind}_backup_model", ""),
                options=llm.list_chat_models(svc.settings, provider) if ok else [],
            )
        )
    return ModelsInfo(
        slots=slots,
        embedding=svc.settings.embedding_model,
        checks=list(_load_checks(svc).values()),
        check_minimum=CHECK_MINIMUM,
    )


@router.put("/config/models", response_model=RagConfig)
def set_models(body: ModelsChange, svc: Services = Depends(get_services)) -> RagConfig:
    """Use the chosen models from the next question on. A new answer or backup model must have
    passed its quality check (409 otherwise); the .env default and the models in use are always
    allowed."""
    cfg = load_rag_config(svc.config_path)
    configured = llm.chat_providers(svc.settings)
    defaults = _defaults(svc)
    checks = _load_checks(svc)

    def checked(kind: str, name: str) -> None:
        check = checks.get(f"{kind}:{name}")
        if check is None or not check.ok:
            raise HTTPException(409, f"Run a passing quality check for {name} first")

    changes = {}
    for kind in ("gemini", "groq", "judge"):
        name = getattr(body, f"{kind}_model").strip()
        main = name or defaults[kind]
        if name == defaults[kind]:
            name = ""  # stored as "use the .env default", so a later .env change still applies
        if name and kind == "groq" and "groq" not in configured:
            raise HTTPException(422, "Groq is not set up on the server (API key)")
        if name and name != getattr(cfg, f"{kind}_model") and kind != "judge":
            checked(kind, name)
        changes[f"{kind}_model"] = name
        if kind == "judge":
            continue
        backup = getattr(body, f"{kind}_backup_model").strip()
        if backup and backup == main:
            raise HTTPException(422, "The backup model must differ from the main model")
        if backup and kind not in configured:
            raise HTTPException(422, f"{kind} is not set up on the server (API key)")
        if backup and backup != getattr(cfg, f"{kind}_backup_model") and backup != defaults[kind]:
            checked(kind, backup)
        changes[f"{kind}_backup_model"] = backup
    with write_lock:
        cfg = load_rag_config(svc.config_path).model_copy(update=changes)
        save_rag_config(cfg, svc.config_path)
    return cfg


@router.post("/config/models/test", response_model=ModelTestResult)
def test_model(body: ModelRequest, svc: Services = Depends(get_services)) -> ModelTestResult:
    """One tiny call: is the model there, and does our key get an answer from it now?"""
    if body.kind == "groq" and "groq" not in llm.chat_providers(svc.settings):
        raise HTTPException(422, "Groq is not set up on the server (API key)")
    started = time.perf_counter()
    try:
        reply = _model(svc, body.kind, body.model).invoke("Reply with the single word: OK")
        ok, message = True, str(getattr(reply, "content", reply)).strip()[:100]
    except Exception as e:  # noqa: BLE001 - a failing model is a result here, not an error
        problem = llm.provider_problem(e)
        ok = False
        message = {
            "quota": "Out of quota for this model right now.",
            "slow": "The model is slow or overloaded right now.",
        }.get(problem, f"{type(e).__name__}: {str(e)[:200]}")
    return ModelTestResult(ok=ok, message=message, seconds=round(time.perf_counter() - started, 1))


@router.post(
    "/config/models/check", response_model=ModelCheck, status_code=status.HTTP_202_ACCEPTED
)
def start_check(
    body: ModelRequest, background: BackgroundTasks, svc: Services = Depends(get_services)
) -> ModelCheck:
    """Start the quality check of an answer model; the page polls GET /config/models."""
    if body.kind == "judge":
        raise HTTPException(422, "The judge model needs no quality check")
    if body.kind not in llm.chat_providers(svc.settings):
        raise HTTPException(422, f"{body.kind} is not set up on the server (API key)")
    if not _check_lock.acquire(blocking=False):
        raise HTTPException(409, "A model check is already running")
    check = ModelCheck(kind=body.kind, model=body.model, state="running", started_at=_now())
    _save_check(svc, check)
    background.add_task(_run_check, svc, check)  # runs after the response (releases the lock)
    return check


def _run_check(svc: Services, check: ModelCheck) -> None:
    """Answer the check questions in fast mode on the demo collection with exactly this model.
    Pass = an answerable question is answered citing its evidence document; the unanswerable
    one is refused."""
    store = svc.open_store()
    try:
        items = [i for i in load_golden(svc.golden_path, store.document_texts())]
        items = [i for i in items if i.id in CHECK_IDS]
        cfg = load_rag_config(svc.config_path).model_copy(update={"mode": "fast"})
        index, bm25 = svc.search.get(store, svc, cfg.retrieval == "hybrid", DEFAULT_COLLECTION_ID)
        answer = build_answerer(
            store, index, svc.embedder, _model(svc, check.kind, check.model), cfg, bm25
        )
        check.total = len(items)
        for item in items:
            ans = answer(item.question)  # a provider error stops the check (quota, overload)
            cited = {s.document for s in service.sources_of(ans)}
            if item.answerable:
                ok = ans.answerable and bool(cited & {e.doc for e in item.evidence})
                note = (
                    "cited the right document"
                    if ok
                    else ("answered without the right source" if ans.answerable else "refused")
                )
            else:
                ok = not ans.answerable
                note = "refused correctly" if ok else "answered a question the documents can't"
            check.items.append(ModelCheckItem(id=item.id, question=item.question, ok=ok, note=note))
            check.passed += ok
        check.ok = check.passed >= CHECK_MINIMUM
        check.state = "done"
    except Exception as e:  # noqa: BLE001 - shown on the page
        check.state = "failed"
        check.error = service.provider_error(e).detail
    finally:
        store.close()
        check.finished_at = _now()
        _save_check(svc, check)
        _check_lock.release()
