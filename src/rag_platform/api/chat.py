"""Chat endpoints: conversations with memory, stored in SQLite.

    POST   /chat                       ask within a conversation (new one if no id given)
    GET    /conversations              the caller's conversations, most recent first
    DELETE /conversations              delete all of the caller's conversations
    GET    /conversations/{id}         one conversation with all its messages
    DELETE /conversations/{id}         delete it (and its messages)
    PUT    /messages/{id}/feedback     thumbs up (1) / down (-1) / clear (null) on an answer

How memory works for one POST /chat:
1. Load the last few turns of the conversation.
2. Condense: rewrite a follow-up into a standalone question (generation/condense.py; skipped for
   the first message, so it costs no extra call). If the rewrite call fails (provider slow or
   overloaded), the message is answered as typed instead of failing the turn.
3. Answer the standalone question with the normal pipeline (fast or agentic, citations,
   abstention) - nothing else in the RAG code needs to know about conversations.
4. Save both messages; the assistant message keeps its sources and the standalone question.
5. Online evaluation: for a sample of answers (ONLINE_EVAL_SAMPLE_RATE), score faithfulness and
   relevance in the background AFTER the response is sent (evaluation/online.py).

Analyse mode (`analyse: true`, signed-in users and admins): the question is answered from every
document of the conversation's collection (generation/analyse.py); it counts as ANALYSE_COST
questions toward the hourly limit.

Guests (X-Guest-Id, no key) and admins (X-API-Key) can chat; see deps.current_caller. Every
conversation belongs to the guest id or key that created it: anyone else gets 404 for it, exactly
as if it did not exist. Guests are limited per hour (deps.limit_questions) and cannot
override the answer mode or search method - those are backend settings (rag_config.json).
"""

import re
import time
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from rag_platform.api import service
from rag_platform.api.deps import (
    Caller,
    Services,
    current_caller,
    current_owner,
    get_services,
    limit_questions,
    visible_collection,
)
from rag_platform.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Conversation,
    ConversationSummary,
    Feedback,
)
from rag_platform.config import load_rag_config
from rag_platform.evaluation import online
from rag_platform.generation.answer import BELOW_THRESHOLD
from rag_platform.generation.condense import MEMORY_TURNS, condense_question

router = APIRouter(tags=["chat"])

# One analysis (generation/analyse.py) makes up to 11 AI calls instead of 1, so it counts as this
# many questions toward the hourly limit (the limit dependency already counted one).
ANALYSE_COST = 5


def _same_question(a: str, b: str) -> bool:
    """True if two questions differ only in case, spacing or punctuation ("what is RAG" vs
    "What is RAG?"), so the rewrite is not worth showing."""

    def norm(s: str) -> str:
        return re.sub(r"[\W_]+", " ", s).strip().lower()  # \W = any non-word character

    return norm(a) == norm(b)


def _message(row: dict) -> ChatMessage:
    """Stored message (details JSON) -> API model."""
    return ChatMessage(
        id=row["id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        feedback=row["feedback"],
        faithfulness=row.get("faithfulness"),
        answer_relevancy=row.get("answer_relevancy"),
        **row["details"],  # answerable, sources, standalone_question, ... (assistant only)
    )


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(limit_questions)])
def chat(
    req: ChatRequest,
    background: BackgroundTasks,  # FastAPI injects it; tasks run after the response is sent
    svc: Services = Depends(get_services),
    caller: Caller = Depends(current_caller),
) -> ChatResponse:
    if (req.mode or req.retrieval) and not caller.is_admin:
        raise HTTPException(403, "Only admins can override the answer mode or search method")
    if req.analyse:
        if caller.limit_key == "":  # a guest (no sign-in, no admin key)
            raise HTTPException(403, "Sign in to analyse all documents")
        if caller.limit_key:  # signed-in user: 4 more hits on top of the one already counted
            limit = svc.settings.user_questions_per_hour
            wait = svc.guest_limiter.hit(caller.limit_key, limit, ANALYSE_COST - 1)
            if wait:
                raise HTTPException(
                    429,
                    f"An analysis counts as {ANALYSE_COST} questions and would pass your hourly "
                    "limit. Ask a normal question or try again later.",
                    headers={"Retry-After": str(int(wait) + 1)},
                )
    # The question is stored with the time it was ASKED (messages are saved after the answer).
    asked_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.perf_counter()
    owner = caller.owner
    cfg = load_rag_config(svc.config_path, mode=req.mode, retrieval=req.retrieval)
    store = svc.open_store()
    try:
        if req.conversation_id is None:
            # A new conversation is created only AFTER a successful answer (below): a failed
            # first question (provider overloaded) must not leave an empty chat in the sidebar.
            visible_collection(store, caller, req.collection_id)  # private: owner only
            conversation_id = None
            collection_id = req.collection_id
            title = req.message.strip()[:60]  # first question names the conversation
            history: list[dict] = []
        else:
            conv = store.get_conversation(req.conversation_id, owner)
            if conv is None:
                raise HTTPException(404, f"No conversation with id {req.conversation_id}")
            # A conversation keeps its collection: follow-ups never search other documents.
            conversation_id, title = conv["id"], conv["title"]
            collection_id = conv["collection_id"]
            history = store.get_messages(conversation_id, last=2 * MEMORY_TURNS)
    finally:
        store.close()

    try:
        standalone = condense_question(history, req.message, svc.llm)
    except Exception:  # noqa: BLE001 - graceful degradation: the rewrite is optional
        # Provider slow/overloaded/quota: answer the message as typed instead of failing the
        # whole turn. A complete question still gets its answer; a real follow-up ("and its
        # overlap?") may be refused as not covered - never guessed. If the answer call fails
        # too, service.answer raises the usual 429/503.
        standalone = req.message
    if req.analyse:
        ans = service.analyse(svc, standalone, collection_id)
    else:
        ans = service.answer(svc, cfg, standalone, collection_id)
    found_in: list[dict] = []
    if ans.abstain_reason == BELOW_THRESHOLD:  # nothing relevant here: maybe in another one?
        try:
            found_in = service.collections_with_matches(svc, standalone, collection_id, caller)
        except Exception:  # noqa: BLE001 - only a hint: never fail the turn because of it
            found_in = []

    details = {
        "answerable": ans.answerable,
        "seconds": round(time.perf_counter() - started, 1),  # answer time, shown in the chat
        "sources": [s.model_dump() for s in service.sources_of(ans)],
        "standalone_question": None if _same_question(standalone, req.message) else standalone,
        "abstain_reason": ans.abstain_reason,
        "steps": ans.steps,
        "route": ans.route,  # auto mode: answered by fast, or escalated to agentic (and why)
        "found_in": found_in,  # other collections with matching passages (after a refusal)
        # Asked while signed in as admin: left out of Live quality by default (team testing).
        "by_admin": caller.is_admin,
    }
    store = svc.open_store()
    try:
        if conversation_id is None:
            conversation_id = store.create_conversation(owner, title, collection_id)
        user_id = store.add_message(conversation_id, "user", req.message, {}, created_at=asked_at)
        bot_id = store.add_message(conversation_id, "assistant", ans.text, details)
        saved = {m["id"]: m for m in store.get_messages(conversation_id, last=2)}
    finally:
        store.close()
    if ans.answerable and online.should_score(svc.settings.online_eval_sample_rate, svc.rng):
        contexts = [c.text for c in ans.context]  # everything the generator read
        background.add_task(online.score_answer, svc, bot_id, standalone, ans.text, contexts)
    return ChatResponse(
        conversation_id=conversation_id,
        collection_id=collection_id,
        title=title,
        user_message=_message(saved[user_id]),
        assistant_message=_message(saved[bot_id]),
    )


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(
    svc: Services = Depends(get_services), owner: str = Depends(current_owner)
) -> list[dict]:
    store = svc.open_store()
    try:
        return store.list_conversations(owner)
    finally:
        store.close()


@router.delete("/conversations", status_code=status.HTTP_204_NO_CONTENT)
def delete_all_conversations(
    svc: Services = Depends(get_services), owner: str = Depends(current_owner)
) -> None:
    """Delete my data (plan B10): every conversation of this caller, with all its messages."""
    store = svc.open_store()
    try:
        store.delete_conversations(owner)
    finally:
        store.close()


@router.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(
    conversation_id: int,
    svc: Services = Depends(get_services),
    owner: str = Depends(current_owner),
) -> Conversation:
    store = svc.open_store()
    try:
        conv = store.get_conversation(conversation_id, owner)
        if conv is None:
            raise HTTPException(404, f"No conversation with id {conversation_id}")
        messages = [_message(m) for m in store.get_messages(conversation_id)]
    finally:
        store.close()
    return Conversation(**conv, messages=messages)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    svc: Services = Depends(get_services),
    owner: str = Depends(current_owner),
) -> None:
    store = svc.open_store()
    try:
        if not store.delete_conversation(conversation_id, owner):
            raise HTTPException(404, f"No conversation with id {conversation_id}")
    finally:
        store.close()


@router.put("/messages/{message_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
def set_feedback(
    message_id: int,
    body: Feedback,
    svc: Services = Depends(get_services),
    owner: str = Depends(current_owner),
) -> None:
    store = svc.open_store()
    try:
        if not store.set_feedback(message_id, owner, body.rating):
            raise HTTPException(404, f"No answer with id {message_id}")
    finally:
        store.close()
