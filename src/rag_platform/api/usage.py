"""Admin endpoints for the usage log (Phase 7 observability).

    GET /usage/summary?days=14     calls, tokens, latency, errors, cost per purpose/model/day
    GET /usage/trace/{run_id}      link to that call's trace in LangSmith (if tracing is on)

The data comes from the local llm_calls table (observability.UsageLogger), so the page works
without LangSmith. Cost appears only for models the owner priced in model_prices.json.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from rag_platform.api.deps import Services, get_services, require_admin
from rag_platform.observability import load_prices, trace_url, tracing_info, usage_report

router = APIRouter(dependencies=[Depends(require_admin)], tags=["usage"])


@router.get("/usage/summary")
def usage_summary(
    days: int = Query(14, ge=1, le=90), svc: Services = Depends(get_services)
) -> dict:
    since = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()
    store = svc.open_store()
    try:
        rows = store.llm_calls_since(since)
    finally:
        store.close()
    report = usage_report(rows, load_prices(svc.prices_path), days)
    return {**report, "tracing": tracing_info(svc.settings)}


@router.get("/usage/trace/{run_id}")
def usage_trace(run_id: str, svc: Services = Depends(get_services)) -> dict:
    """One lookup at LangSmith (no LLM call): the web link to this call's trace."""
    info = tracing_info(svc.settings)
    if not info["enabled"]:
        raise HTTPException(404, "LangSmith tracing is not configured")
    try:
        return {"url": trace_url(run_id, info["project"])}
    except Exception as e:  # noqa: BLE001 - e.g. not traced (sampling) or not ingested yet
        raise HTTPException(404, f"Trace not found in LangSmith: {type(e).__name__}") from e
