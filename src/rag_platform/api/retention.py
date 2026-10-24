"""Data retention (plan B10): personal data is not kept forever.

Chat questions can contain personal details, and the usage log records who used the models when.
Privacy rules (GDPR, India's DPDP Act) expect data to be kept only as long as it is needed, and
less stored data means less to leak. So, when the API starts and then once a day:
- conversations without activity for CHAT_RETENTION_DAYS (default 90) are deleted, together with
  their messages, feedback and online-evaluation scores;
- usage-log rows (`llm_calls`) older than USAGE_RETENTION_DAYS (default 90) are deleted.
0 keeps that data forever. Eval runs, optimizer recommendations and documents are not personal
data (test questions, settings, the knowledge base) and stay. Users can also delete all their
conversations at once (DELETE /conversations).

The daily loop runs inside the API process (started by app.py's lifespan) - .NET analogy: a
BackgroundService / IHostedService with a PeriodicTimer. No extra container or cron job needed.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from rag_platform.config import Settings
from rag_platform.store.db import Store

log = logging.getLogger("rag_platform.retention")
INTERVAL_SECONDS = 24 * 60 * 60  # once a day


def _cutoff(now: datetime, days: int) -> str | None:
    """ISO time `days` before now (same format as the stored times), or None for "keep"."""
    return (now - timedelta(days=days)).isoformat(timespec="seconds") if days > 0 else None


def purge(settings: Settings, now: datetime | None = None) -> dict:
    """Delete data older than the retention periods; returns how many rows went."""
    now = now or datetime.now(UTC)
    store = Store(settings.db_path)
    try:
        counts = store.purge_older_than(
            _cutoff(now, settings.chat_retention_days), _cutoff(now, settings.usage_retention_days)
        )
    finally:
        store.close()
    log.info(
        "retention purge",
        extra=counts
        | {
            "chat_retention_days": settings.chat_retention_days,
            "usage_retention_days": settings.usage_retention_days,
        },
    )
    return counts


async def run_daily(settings: Settings) -> None:
    """Purge now, then every 24 hours, until the API stops (the task is cancelled)."""
    while True:
        try:
            # The SQLite work is synchronous: run it in a worker thread so the server keeps
            # answering requests meanwhile (like Task.Run in .NET).
            await asyncio.to_thread(purge, settings)
        except Exception:  # noqa: BLE001 - a failed purge is logged and retried tomorrow
            log.exception("retention purge failed")
        await asyncio.sleep(INTERVAL_SECONDS)
