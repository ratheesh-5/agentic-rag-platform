"""Admin endpoints for quality: evaluation runs, release gates and optimizer recommendations.

    GET  /eval/runs                       runs with kind, gate status and "uses the active config"
    GET  /eval/runs/{id}                  one run in full, with its gate checks
    POST /eval/runs                       run a quality check now (optional uploaded test set)
    PUT  /eval/runs/{id}/hidden           hide / show a run in the default list
    GET  /eval/gates                      the release gate thresholds
    POST /optimizer/runs                  start the optimizer in the background (optional test set)
    GET  /optimizer/run                   progress / outcome of the optimizer started from the UI
    GET  /optimizer/recommendation        the latest optimizer recommendation (or null)
    POST /optimizer/recommendation/{id}/apply | /reject     the human decision
    GET  /quality/usage                   live quality from real chats + suggestions
    POST /quality/test                    answer ONE test question and score it now (not saved)

A "quality check" is the retrieval-only evaluation the optimizer uses: no chat calls, and the
question vectors are cached, so it takes seconds. Full evaluations (answers + RAGAS, hundreds of
LLM calls) stay in the CLI (`rag eval run`).

Applying a recommendation is the human-approval step of the optimizer (Workflow B), moved from
the terminal into the UI. It refuses if the active config changed since the recommendation was
made, because its numbers were measured against the old one.

The optimizer itself can also be started from the UI (same as `rag optimize --decision later`):
it takes minutes (experiments may re-chunk and embed), longer than a web request should wait, so
it runs as a background task and the page polls GET /optimizer/run for progress. Its result is a
saved recommendation - nothing changes until an admin applies it.
"""

import json
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from rag_platform.agents.optimizer_graph.experiments import ExperimentRunner
from rag_platform.api import service
from rag_platform.api.deps import (
    Caller,
    Services,
    current_caller,
    get_services,
    require_admin,
    visible_collection,
    write_lock,
)
from rag_platform.api.schemas import (
    EvalRunSummary,
    HiddenFlag,
    OptimizerRun,
    QualityTest,
    QualityTestResult,
    Recommendation,
    TestPassage,
)
from rag_platform.config import ADMIN_CHOICES, RagConfig, load_rag_config
from rag_platform.evaluation.gates import GateCheck, check_gates, load_gates, overall
from rag_platform.evaluation.golden import load_golden
from rag_platform.evaluation.online import judge, usage_summary

router = APIRouter(dependencies=[Depends(require_admin)], tags=["evaluation"])

# One quality job at a time - a quality check or an optimizer run (a second click gets 409
# instead of a duplicate run; both read the index and write eval runs).
_check_lock = threading.Lock()
BUSY = "A quality check or optimizer run is already running"
MAX_GOLDEN_BYTES = 1024 * 1024
# Optimizer budget for runs started from the UI (the CLI defaults: 3 experiments x 2 rounds).
EXPERIMENTS_PER_ROUND = 3
ROUNDS = 2
# The optimizer started from the UI. In memory (one API process): a restart forgets it, but the
# recommendation it produced is saved in SQLite.
_optimizer = OptimizerRun()


def _kind(config: dict) -> str:
    if config.get("experiment"):
        return "experiment"  # an optimizer try
    return "check" if config.get("retrieval_only") else "full"


def _summary(run: dict, active: dict, svc: Services) -> EvalRunSummary:
    cfg = run.get("config", {})
    checks = check_gates(
        run["metrics"], bool(cfg.get("retrieval_only")), load_gates(svc.gates_path)
    )
    return EvalRunSummary(
        id=run["id"],
        created_at=run["created_at"],
        label=run["label"],
        metrics=run["metrics"],
        kind=_kind(cfg),
        hidden=run.get("hidden", False),
        current=cfg.get("rag") == active,
        gates=overall(checks),
        test_set=Path(cfg.get("golden", "")).name,
    )


@router.get("/eval/runs", response_model=list[EvalRunSummary])
def list_eval_runs(svc: Services = Depends(get_services)) -> list[EvalRunSummary]:
    active = load_rag_config(svc.config_path).model_dump()
    store = svc.open_store()
    try:
        return [_summary(r, active, svc) for r in store.list_eval_runs()]
    finally:
        store.close()


@router.get("/eval/runs/{run_id}")
def get_eval_run(run_id: int, svc: Services = Depends(get_services)) -> dict:
    store = svc.open_store()
    try:
        run = store.get_eval_run(run_id)
    finally:
        store.close()
    if run is None:
        raise HTTPException(404, f"No eval run with id {run_id}")
    retrieval_only = bool(run["config"].get("retrieval_only"))
    gates = check_gates(run["metrics"], retrieval_only, load_gates(svc.gates_path))
    return {**run, "kind": _kind(run["config"]), "gates": [g.model_dump() for g in gates]}


@router.get("/eval/gates", response_model=list[GateCheck])
def gates(svc: Services = Depends(get_services)) -> list[GateCheck]:
    """The thresholds only (value/status empty), for the UI's legend."""
    return check_gates({}, False, load_gates(svc.gates_path))


@router.post("/eval/runs", response_model=EvalRunSummary, status_code=status.HTTP_201_CREATED)
def run_quality_check(
    golden: UploadFile | None = File(None, description="Optional test set (.jsonl golden format)"),
    label: str = Form("quality-check", max_length=60),
    svc: Services = Depends(get_services),
) -> EvalRunSummary:
    """Retrieval-only evaluation of the ACTIVE config on the default or an uploaded test set."""
    golden_path = svc.golden_path
    if golden is not None:
        golden_path = _save_test_set(golden, svc)
    else:
        _check_documents(golden_path, svc)  # e.g. a document was deleted after the set was made
    if not _check_lock.acquire(blocking=False):
        raise HTTPException(409, BUSY)
    try:
        cfg = load_rag_config(svc.config_path)
        runner = ExperimentRunner(svc.settings, svc.embedder, golden_path, cfg)
        try:
            run_id = runner.run(cfg, label, experiment=False).run_id
        except ValueError as e:  # test set no longer matches the documents (e.g. text changed)
            raise HTTPException(409, f"The test set does not match the documents: {e}") from None
        except Exception as e:  # noqa: BLE001 - e.g. embedding quota for new test questions
            raise service.provider_error(e) from e
        finally:
            runner.close()
    finally:
        _check_lock.release()
    store = svc.open_store()
    try:
        run = next(r for r in store.list_eval_runs() if r["id"] == run_id)
    finally:
        store.close()
    return _summary(run, cfg.model_dump(), svc)


def _check_documents(golden_path: Path, svc: Services) -> None:
    """The test set's questions quote indexed documents. If one was deleted, say which, instead
    of failing deep inside the run (that showed up as a misleading "provider error")."""
    store = svc.open_store()
    try:
        indexed = set(store.document_texts())
    finally:
        store.close()
    needed = set()
    for line in golden_path.read_text("utf-8").splitlines():
        if line.strip():
            needed |= {e["doc"] for e in json.loads(line).get("evidence", [])}
    missing = sorted(Path(d).name for d in needed - indexed)
    if missing:
        raise HTTPException(
            409,
            f"The test set {golden_path.name} has questions about documents that are no longer "
            f"indexed: {', '.join(missing)}. Upload them again, or run the check with your own "
            "test set for the current documents.",
        )


def _save_test_set(file: UploadFile, svc: Services) -> Path:
    """Validate an uploaded test set BEFORE running anything: every evidence quote must be
    found exactly once in an indexed document (same rule as the default golden set)."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", (file.filename or "").replace("\\", "/").split("/")[-1])
    if not name.lower().endswith(".jsonl"):
        raise HTTPException(415, "The test set must be a .jsonl file (one question per line)")
    data = file.file.read(MAX_GOLDEN_BYTES + 1)
    if len(data) > MAX_GOLDEN_BYTES:
        raise HTTPException(413, "Test set too large (max 1 MB)")
    folder = svc.settings.upload_dir / "test_sets"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    tmp = folder / (name + ".tmp")
    tmp.write_bytes(data)
    store = svc.open_store()
    try:
        items = load_golden(tmp, store.document_texts())
        if not items:
            raise ValueError("the file contains no questions")
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as e:
        tmp.unlink(missing_ok=True)
        message = str(e).replace(str(tmp), name)  # show the file name, not the temp path
        raise HTTPException(422, f"Invalid test set: {message}") from None
    finally:
        store.close()
    tmp.replace(path)  # only a valid file replaces an earlier one with the same name
    return path


@router.put("/eval/runs/{run_id}/hidden", status_code=status.HTTP_204_NO_CONTENT)
def set_hidden(run_id: int, body: HiddenFlag, svc: Services = Depends(get_services)) -> None:
    store = svc.open_store()
    try:
        if not store.set_eval_run_hidden(run_id, body.hidden):
            raise HTTPException(404, f"No eval run with id {run_id}")
    finally:
        store.close()


# --- Optimizer recommendation: human approval in the UI ----------------------------------------


@router.get("/optimizer/recommendation", response_model=Recommendation | None, tags=["optimizer"])
def latest_recommendation(svc: Services = Depends(get_services)) -> Recommendation | None:
    store = svc.open_store()
    try:
        rec = store.latest_recommendation()
    finally:
        store.close()
    if rec is None:
        return None
    base = rec["report"].get("base_config")
    stale = base is not None and _tuned(base) != _tuned(
        load_rag_config(svc.config_path).model_dump()
    )
    return Recommendation(**rec, stale=stale)


def _tuned(config: dict) -> dict:
    """The settings a recommendation is measured on - everything except the admin's direct
    choices (answer mode and model, ADMIN_CHOICES), which the optimizer never changes."""
    return {k: v for k, v in config.items() if k not in ADMIN_CHOICES}


def _pending(store, rec_id: int) -> dict:
    rec = store.latest_recommendation()
    if rec is None or rec["id"] != rec_id:
        raise HTTPException(404, f"No current recommendation with id {rec_id}")
    if rec["status"] != "pending":
        raise HTTPException(409, f"Recommendation {rec_id} is already {rec['status']}")
    return rec


@router.post(
    "/optimizer/recommendation/{rec_id}/apply",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["optimizer"],
)
def apply_recommendation(rec_id: int, svc: Services = Depends(get_services)) -> None:
    with write_lock:  # may re-index: one writer at a time
        store = svc.open_store()
        try:
            rec = _pending(store, rec_id)
            current = load_rag_config(svc.config_path)
            base = rec["report"].get("base_config")
            if base is not None and _tuned(base) != _tuned(current.model_dump()):
                raise HTTPException(
                    409,
                    "The configuration changed after this recommendation was made; "
                    "run `rag optimize` again",
                )
            # Keep the admin's current choices (answer mode and model): the optimizer tunes
            # search settings only, and the admin may have switched them since.
            choices = {k: getattr(current, k) for k in ADMIN_CHOICES}
            cfg = RagConfig(**{**rec["report"]["recommended_config"], **choices})
            runner = ExperimentRunner(svc.settings, svc.embedder, svc.golden_path, current)
            try:
                runner.apply(cfg, svc.config_path)  # writes rag_config.json; re-chunks if needed
            except Exception as e:  # noqa: BLE001 - provider errors -> 429/503
                raise service.provider_error(e) from e
            finally:
                runner.close()
            store.decide_recommendation(rec_id, "applied")
        finally:
            store.close()


@router.post(
    "/optimizer/recommendation/{rec_id}/reject",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["optimizer"],
)
def reject_recommendation(rec_id: int, svc: Services = Depends(get_services)) -> None:
    store = svc.open_store()
    try:
        _pending(store, rec_id)
        store.decide_recommendation(rec_id, "rejected")
    finally:
        store.close()


# --- Optimizer started from the UI (background task) -------------------------------------------


@router.post(
    "/optimizer/runs",
    response_model=OptimizerRun,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["optimizer"],
)
def start_optimizer(
    background: BackgroundTasks,
    golden: UploadFile | None = File(None, description="Optional test set (.jsonl golden format)"),
    svc: Services = Depends(get_services),
) -> OptimizerRun:
    """Start the multi-agent optimizer (= `rag optimize --decision later`) in the background.
    Cost: a few chat calls (diagnosis, choosing experiments, report) plus embeddings when an
    experiment tries a new chunk size; the experiments themselves make no chat calls."""
    global _optimizer  # module-level state (like a static field in C#)
    golden_path = svc.golden_path
    if golden is not None:
        golden_path = _save_test_set(golden, svc)  # validated before anything starts
    else:
        _check_documents(golden_path, svc)
    if not _check_lock.acquire(blocking=False):
        raise HTTPException(409, BUSY)
    _optimizer = OptimizerRun(state="running", test_set=golden_path.name, started_at=_now())
    # BackgroundTasks run after the response is sent (like queuing work to a hosted service in
    # ASP.NET); the task releases the lock when it ends.
    background.add_task(_run_optimizer, svc, golden_path, _optimizer)
    return _optimizer


@router.get("/optimizer/run", response_model=OptimizerRun, tags=["optimizer"])
def optimizer_run() -> OptimizerRun:
    """Progress and outcome of the optimizer started from the UI (idle if none since restart)."""
    return _optimizer


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _run_optimizer(svc: Services, golden_path: Path, job: OptimizerRun) -> None:
    """The same steps as the CLI: stream the graph, collect its progress lines, and save the
    report as a recommendation when the graph pauses for human approval."""
    from rag_platform.agents.optimizer_graph.graph import build_optimizer_graph

    try:
        cfg = load_rag_config(svc.config_path)
        runner = ExperimentRunner(
            svc.settings, svc.embedder, golden_path, cfg, log=lambda m: job.steps.append(m.strip())
        )
        try:
            graph = build_optimizer_graph(
                runner, svc.llm, svc.config_path, EXPERIMENTS_PER_ROUND, ROUNDS
            )
            thread = {"configurable": {"thread_id": "optimize"}}
            job.outcome = "nothing-to-improve"  # unless a step below says otherwise
            for update in graph.stream({"steps": []}, thread, stream_mode="updates"):
                for node, change in update.items():
                    if node == "__interrupt__":  # paused for approval: save it for Settings
                        report = {**change[0].value["report"], "test_set": golden_path.name}
                        job.recommendation_id = runner.main.save_recommendation(report)
                        job.outcome = "recommendation"
                        continue
                    job.steps.extend((change or {}).get("steps", []))
                    if (change or {}).get("decision") == "no-change":
                        job.outcome = "no-change"  # the current settings won
        finally:
            runner.close()
        job.state = "done"
    except Exception as e:  # noqa: BLE001 - e.g. quota or an overloaded model: shown on the page
        job.state = "failed"
        # A document changed during the run, so a quote no longer matches. (Checked by message:
        # pydantic's ValidationError - a bad model reply - is also a ValueError.)
        if isinstance(e, ValueError) and "evidence quote" in str(e):
            job.error = f"The test set does not match the documents: {e}"
        else:
            job.error = service.provider_error(e).detail
    finally:
        job.finished_at = _now()
        _check_lock.release()


# --- Live quality from real usage (no test data) ---------------------------------------------


@router.get("/quality/usage", tags=["quality"])
def quality_usage(include_admin: bool = False, svc: Services = Depends(get_services)) -> dict:
    """Totals, unanswered questions, thumbs-down and low-scored answers, every answer with its
    scores, per-document numbers. Plain aggregation of stored chats: no LLM calls.
    Admin chats (team testing) are excluded unless ?include_admin=true."""
    store = svc.open_store()
    try:
        rows = store.usage_rows()
    finally:
        store.close()
    return usage_summary(rows, svc.settings.online_eval_sample_rate, include_admin)


@router.post("/quality/test", response_model=QualityTestResult, tags=["quality"])
def quality_test(
    req: QualityTest,
    svc: Services = Depends(get_services),
    caller: Caller = Depends(current_caller),
) -> QualityTestResult:
    """Admin playground: run the live pipeline on one question and score the answer right away
    (always, not sampled). Nothing is saved, so tests never distort the Live quality numbers.
    It searches the chosen collection with that collection's similarity threshold, exactly like a
    new chat there (someone else's private collection: 404, as everywhere).
    Cost: 1 answer call (0 if the similarity threshold refuses) + ~4 judge calls."""
    store = svc.open_store()
    try:
        visible_collection(store, caller, req.collection_id)
    finally:
        store.close()
    started = time.perf_counter()
    ans = service.answer(svc, load_rag_config(svc.config_path), req.question, req.collection_id)
    sources = service.sources_of(ans)
    cited = {(s.document, s.start) for s in sources}
    passages = [
        TestPassage(
            document=c.doc_path,
            score=round(c.score, 4),
            cited=(c.doc_path, c.start) in cited,
            text=c.text,
        )
        for c in ans.context
    ]
    scores = {"faithfulness": None, "answer_relevancy": None, "error": None, "judge_calls": 0}
    scores["judge"] = svc.judge_name()
    if ans.answerable:  # a refusal has no claims to judge
        scores = judge(svc, req.question, ans.text, [c.text for c in ans.context])
    return QualityTestResult(
        question=req.question,
        answerable=ans.answerable,
        answer=ans.text,
        abstain_reason=ans.abstain_reason,
        sources=sources,
        passages=passages,
        faithfulness=scores["faithfulness"],
        answer_relevancy=scores["answer_relevancy"],
        judge_calls=scores["judge_calls"],
        judge_model=scores["judge"],
        score_error=scores["error"],
        seconds=round(time.perf_counter() - started, 1),
    )
