"""Tests for conversation memory: question condensing and the chat API. Fakes only - no API calls.

FakeLlm answers per requested schema: StandaloneQuestion (the follow-up rewriter) and LlmAnswer
(the answer generator), so a test can see exactly which LLM calls happened and with what prompt.
"""

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLlm
from rag_platform.api import retention
from rag_platform.api.app import app
from rag_platform.api.deps import Services, get_services
from rag_platform.config import RagConfig, Settings, save_rag_config
from rag_platform.generation.answer import LlmAnswer
from rag_platform.generation.condense import StandaloneQuestion, condense_question

KEY = {"X-API-Key": "key-a"}
OTHER = {"X-API-Key": "key-b"}
DOC = b"The team default chunk size is 800 characters. The default chunk overlap is 100 characters."


# --- condense_question -------------------------------------------------------------------------


def test_first_message_is_not_rewritten_and_costs_no_call():
    llm = FakeLlm()
    assert condense_question([], "What is the chunk size?", llm) == "What is the chunk size?"
    assert llm.prompts == []


def test_follow_up_is_rewritten_with_recent_turns():
    llm = FakeLlm({StandaloneQuestion: StandaloneQuestion(question="What is the chunk overlap?")})
    history = [
        {"role": "user", "content": "What is the chunk size?"},
        {"role": "assistant", "content": "800 characters [1]."},
    ]
    assert condense_question(history, "and its overlap?", llm) == "What is the chunk overlap?"
    prompt = llm.prompts[0][1][1]
    assert (
        "user: What is the chunk size?" in prompt and "Latest message: and its overlap?" in prompt
    )


def test_empty_rewrite_falls_back_to_the_original_question():
    llm = FakeLlm({StandaloneQuestion: StandaloneQuestion(question="  ")})
    history = [{"role": "user", "content": "x"}]
    assert condense_question(history, "and its overlap?", llm) == "and its overlap?"


# --- chat API -----------------------------------------------------------------------------------


@pytest.fixture
def chat(tmp_path: Path, fake_embedder):
    settings = Settings(
        _env_file=None,
        online_eval_sample_rate=0,  # no background judge calls in tests
        gemini_api_key="x",
        api_keys="key-a,key-b",
        index_dir=tmp_path / "idx",
        upload_dir=tmp_path / "uploads",
    )
    save_rag_config(RagConfig(similarity_threshold=0.0), tmp_path / "rag_config.json")
    llm = FakeLlm(
        {
            StandaloneQuestion: StandaloneQuestion(question="What is the default chunk overlap?"),
            LlmAnswer: LlmAnswer(answerable=True, answer="100 characters [1]."),
        }
    )
    svc = Services(settings, fake_embedder, llm, config_path=tmp_path / "rag_config.json")
    app.dependency_overrides[get_services] = lambda: svc
    client = TestClient(app)
    r = client.post("/documents", headers=KEY, files={"file": ("defaults.md", DOC)})
    assert r.status_code == 201
    yield client, llm
    app.dependency_overrides.clear()


def _send(client, message: str, conversation_id: int | None = None, headers=KEY):
    body = {"message": message, **({"conversation_id": conversation_id} if conversation_id else {})}
    return client.post("/chat", headers=headers, json=body)


def test_conversation_with_follow_up_uses_memory(chat):
    client, llm = chat
    first = _send(client, "What is the default chunk size?").json()
    cid = first["conversation_id"]
    assert first["title"] == "What is the default chunk size?"
    assert first["assistant_message"]["standalone_question"] is None  # first message: as-is
    assert StandaloneQuestion not in llm.schemas  # ...and no rewrite call

    second = _send(client, "and its overlap?", cid).json()
    assert second["conversation_id"] == cid
    bot = second["assistant_message"]
    assert bot["standalone_question"] == "What is the default chunk overlap?"
    assert bot["seconds"] is not None and bot["seconds"] >= 0  # answer time for the chat
    assert second["user_message"]["created_at"] <= bot["created_at"]  # asked before answered
    assert bot["answerable"] and bot["sources"][0]["document"].endswith("defaults.md")
    # The answer generator saw the rewritten question, not "and its overlap?".
    answer_prompt = llm.prompts[-1][1][1]
    assert "Question: What is the default chunk overlap?" in answer_prompt

    conv = client.get(f"/conversations/{cid}", headers=KEY).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conv["messages"][2]["content"] == "and its overlap?"


def test_conversations_are_private_to_their_api_key(chat):
    client, _ = chat
    cid = _send(client, "What is the default chunk size?").json()["conversation_id"]
    assert client.get(f"/conversations/{cid}", headers=OTHER).status_code == 404
    assert client.get("/conversations", headers=OTHER).json() == []
    assert _send(client, "follow-up", cid, headers=OTHER).status_code == 404
    assert client.delete(f"/conversations/{cid}", headers=OTHER).status_code == 404
    assert len(client.get("/conversations", headers=KEY).json()) == 1


def test_list_newest_first_and_delete(chat):
    client, _ = chat
    a = _send(client, "First conversation question").json()["conversation_id"]
    b = _send(client, "Second conversation question").json()["conversation_id"]
    listed = client.get("/conversations", headers=KEY).json()
    assert {c["id"] for c in listed} == {a, b} and all(c["messages"] == 2 for c in listed)
    assert client.delete(f"/conversations/{a}", headers=KEY).status_code == 204
    assert [c["id"] for c in client.get("/conversations", headers=KEY).json()] == [b]
    assert client.get(f"/conversations/{a}", headers=KEY).status_code == 404


def test_feedback_on_answers_only(chat):
    client, _ = chat
    res = _send(client, "What is the default chunk size?").json()
    bot_id, user_id = res["assistant_message"]["id"], res["user_message"]["id"]
    cid = res["conversation_id"]

    assert (
        client.put(f"/messages/{bot_id}/feedback", headers=KEY, json={"rating": -1}).status_code
        == 204
    )
    msgs = client.get(f"/conversations/{cid}", headers=KEY).json()["messages"]
    assert msgs[1]["feedback"] == -1
    assert (
        client.put(f"/messages/{bot_id}/feedback", headers=KEY, json={"rating": None}).status_code
        == 204
    )
    # Not on user messages, not on someone else's answer, only 1 / -1 / null.
    assert (
        client.put(f"/messages/{user_id}/feedback", headers=KEY, json={"rating": 1}).status_code
        == 404
    )
    assert (
        client.put(f"/messages/{bot_id}/feedback", headers=OTHER, json={"rating": 1}).status_code
        == 404
    )
    assert (
        client.put(f"/messages/{bot_id}/feedback", headers=KEY, json={"rating": 5}).status_code
        == 422
    )


def test_unknown_conversation_and_quota_errors(chat):
    client, llm = chat
    assert _send(client, "hello there", 999).status_code == 404
    cid = _send(client, "What is the default chunk size?").json()["conversation_id"]

    class Quota(FakeLlm):
        def with_structured_output(self, schema):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    app.dependency_overrides[get_services]().llm = Quota()
    r = _send(client, "and its overlap?", cid)  # rewrite AND answer calls hit the quota
    assert r.status_code == 429
    # Nothing half-saved: the failed turn left the conversation unchanged.
    assert len(client.get(f"/conversations/{cid}", headers=KEY).json()["messages"]) == 2


def test_a_failed_first_question_leaves_no_empty_conversation(chat):
    # Seen live: the provider was overloaded, and every failed first question left an empty
    # chat in the sidebar (created before answering). It's created only after an answer now.
    client, _ = chat

    class Overloaded(FakeLlm):
        def with_structured_output(self, schema):
            raise RuntimeError("503 UNAVAILABLE high demand")

    app.dependency_overrides[get_services]().llm = Overloaded()
    assert _send(client, "What is the default chunk size?").status_code == 503
    assert client.get("/conversations", headers=KEY).json() == []


def test_older_empty_conversations_are_hidden(chat):
    from rag_platform.api.deps import _hash

    client, _ = chat
    svc = app.dependency_overrides[get_services]()
    store = svc.open_store()
    try:  # what older versions left behind: a conversation without messages
        # Admin without a guest id: the owner is the hash of the key (api/deps.py).
        store.create_conversation(_hash(KEY["X-API-Key"]), "What is the GPU budget?")
    finally:
        store.close()
    assert client.get("/conversations", headers=KEY).json() == []


def test_failed_rewrite_answers_the_message_as_typed(chat):
    client, llm = chat
    cid = _send(client, "What is the default chunk size?").json()["conversation_id"]

    class RewriteDown(FakeLlm):  # only the follow-up rewriter is overloaded
        def with_structured_output(self, schema):
            if schema is StandaloneQuestion:
                raise RuntimeError("503 UNAVAILABLE high demand")
            return super().with_structured_output(schema)

    down = RewriteDown({LlmAnswer: LlmAnswer(answerable=True, answer="100 characters [1].")})
    app.dependency_overrides[get_services]().llm = down
    r = _send(client, "What is the default chunk overlap?", cid)
    assert r.status_code == 200
    bot = r.json()["assistant_message"]
    assert bot["answerable"] and bot["standalone_question"] is None  # answered as typed
    assert "Question: What is the default chunk overlap?" in down.prompts[-1][1][1]


# --- guests (no key) vs admins -------------------------------------------------------------------

GUEST = {"X-Guest-Id": "3f1c2a9e-7b4d-4c1a-9e2f-0a1b2c3d4e5f"}
GUEST2 = {"X-Guest-Id": "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"}


def test_me_reports_the_role(chat):
    client, _ = chat
    assert client.get("/me", headers=GUEST).json() == {"role": "guest", "user": None}
    assert client.get("/me", headers=KEY).json() == {"role": "admin", "user": None}
    assert client.get("/me", headers={**GUEST, **KEY}).json() == {"role": "admin", "user": None}
    assert client.get("/me").status_code == 401  # neither
    assert client.get("/me", headers={"X-API-Key": "wrong", **GUEST}).status_code == 401
    assert client.get("/me", headers={"X-Guest-Id": "x"}).status_code == 422  # not a random id


def test_guests_chat_privately_but_cannot_use_admin_endpoints(chat):
    client, _ = chat
    res = _send(client, "What is the default chunk size?", headers=GUEST)
    assert res.status_code == 200 and res.json()["assistant_message"]["answerable"]
    cid = res.json()["conversation_id"]
    assert [c["id"] for c in client.get("/conversations", headers=GUEST).json()] == [cid]
    assert client.get(f"/conversations/{cid}", headers=GUEST2).status_code == 404
    assert client.get("/conversations", headers=KEY).json() == []  # admin key's own list

    for method, path in [("GET", "/config"), ("GET", "/eval/runs")]:
        assert client.request(method, path, headers=GUEST).status_code == 401
    # Documents: signed-in users manage their own private ones; a guest has none (sign in).
    assert client.get("/documents", headers=GUEST).status_code == 403
    assert client.post("/ask", headers=GUEST, json={"question": "chunk size?"}).status_code == 401
    # Mode/search are backend settings: a guest can't switch to the costlier agentic mode.
    r = client.post("/chat", headers=GUEST, json={"message": "chunk size?", "mode": "agentic"})
    assert r.status_code == 403


def test_admin_in_the_ui_keeps_the_browser_s_conversations(chat):
    # The UI always sends its guest id; signing in as admin must not hide the guest's history.
    client, _ = chat
    cid = _send(client, "What is the default chunk size?", headers=GUEST).json()["conversation_id"]
    listed = client.get("/conversations", headers={**GUEST, **KEY}).json()
    assert [c["id"] for c in listed] == [cid]


def test_guest_hourly_limit_admins_unlimited(chat):
    client, _ = chat
    svc = app.dependency_overrides[get_services]()
    svc.settings.guest_questions_per_hour = 2
    assert _send(client, "What is the default chunk size?", headers=GUEST).status_code == 200
    assert _send(client, "What is the default chunk size?", headers=GUEST).status_code == 200
    blocked = _send(client, "What is the default chunk size?", headers=GUEST2)  # same IP
    assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) > 0
    assert _send(client, "What is the default chunk size?", headers=KEY).status_code == 200

    svc.settings.guest_questions_per_hour = 0  # guests disabled
    assert client.get("/conversations", headers=GUEST).status_code == 401


def test_trivial_rewrite_is_not_stored(chat):
    # The rewriter only fixed case/punctuation: nothing worth showing as "Understood as".
    client, llm = chat
    cid = _send(client, "What is the default chunk size?").json()["conversation_id"]
    llm.reply[StandaloneQuestion] = StandaloneQuestion(question="What is the default overlap?")
    bot = _send(client, "what is the default overlap", cid).json()["assistant_message"]
    assert bot["standalone_question"] is None


# --- online evaluation: sampled answers scored in the background -------------------------------


class _FakeJudge:
    """Stands in for RagasEvaluator: records what it was asked, returns fixed scores."""

    def __init__(self, scores=None, error=None):
        self.scores, self.error, self.seen = scores or {}, error, []

    def score(self, question, answer, contexts, reference):
        self.seen.append((question, answer, contexts, reference))
        if self.error:
            raise self.error
        return self.scores


def _online(chat, judge, rate=1.0):
    svc = app.dependency_overrides[get_services]()
    svc.settings.online_eval_sample_rate = rate
    svc.evaluator = judge
    return svc


def test_sampled_answers_are_scored_after_the_response(chat):
    client, _ = chat
    judge = _FakeJudge({"faithfulness": 0.9, "answer_relevancy": 0.4})
    _online(chat, judge)
    res = _send(client, "What is the default chunk size?").json()
    # TestClient runs background tasks right after the response, so the score exists now.
    question, answer, contexts, reference = judge.seen[0]
    assert question == "What is the default chunk size?" and answer == "100 characters [1]."
    assert contexts and reference == ""  # reference-free metrics only
    usage = client.get("/quality/usage?include_admin=true", headers=KEY).json()
    assert usage["totals"]["scored"] == 1 and usage["totals"]["faithfulness"] == 0.9
    assert usage["low_scores"][0]["message_id"] == res["assistant_message"]["id"]  # 0.4 < 0.7
    assert usage["documents"][0]["document"] == "defaults.md"


def test_not_sampled_or_unanswered_answers_are_not_scored(chat):
    client, _ = chat
    judge = _FakeJudge({"faithfulness": 1.0})
    svc = _online(chat, judge, rate=0.2)
    svc.rng = lambda: 0.5  # above the 0.2 rate: not in the sample
    _send(client, "What is the default chunk size?")
    assert judge.seen == []
    svc.settings.online_eval_sample_rate = 0  # off
    svc.rng = lambda: 0.0
    _send(client, "What is the default chunk size?")
    assert judge.seen == []


def test_judge_errors_are_recorded_not_raised(chat):
    client, _ = chat
    _online(chat, _FakeJudge(error=RuntimeError("429 RESOURCE_EXHAUSTED")))
    assert _send(client, "What is the default chunk size?").status_code == 200  # user unaffected
    usage = client.get("/quality/usage?include_admin=true", headers=KEY).json()
    assert usage["totals"]["answered"] == 1 and usage["totals"]["scored"] == 0


def test_usage_lists_unanswered_questions_and_thumbs_down(chat):
    client, llm = chat
    llm.reply[LlmAnswer] = LlmAnswer(answerable=False, answer="")
    _send(client, "What is the CEO salary?", headers=GUEST)
    _send(client, "what is the ceo salary", headers=GUEST2)  # same question, other user
    llm.reply[LlmAnswer] = LlmAnswer(answerable=True, answer="100 characters [1].")
    bot = _send(client, "What is the default chunk size?").json()["assistant_message"]
    client.put(f"/messages/{bot['id']}/feedback", headers=KEY, json={"rating": -1})

    usage = client.get("/quality/usage?include_admin=true", headers=KEY).json()
    assert usage["totals"] == {
        "questions": 3, "answered": 1, "unanswered": 2, "escalated": 0, "thumbs_up": 0,
        "thumbs_down": 1, "scored": 0, "faithfulness": None, "answer_relevancy": None,
    }  # fmt: skip
    assert len(usage["unanswered"]) == 1 and usage["unanswered"][0]["count"] == 2
    refusal = client.get("/conversations", headers=GUEST).json()[0]["id"]
    refusal_msg = client.get(f"/conversations/{refusal}", headers=GUEST).json()["messages"][1]
    client.put(f"/messages/{refusal_msg['id']}/feedback", headers=GUEST, json={"rating": -1})
    disliked = client.get("/quality/usage", headers=KEY).json()["thumbs_down"]
    assert [d["question"] for d in disliked] == ["What is the CEO salary?"]  # refusal included
    assert usage["thumbs_down"][0]["question"] == "What is the default chunk size?"
    assert client.get("/quality/usage", headers=GUEST).status_code == 401  # admins only


def test_admin_chats_are_excluded_by_default_and_listed_per_answer(chat):
    client, _ = chat
    _send(client, "What is the default chunk size?")  # admin key: team testing
    _send(client, "What is the default chunk size?", headers=GUEST)  # a real user
    usage = client.get("/quality/usage", headers=KEY).json()
    assert usage["totals"]["questions"] == 1 and usage["admin_chats"] == 1
    assert [a["by_admin"] for a in usage["recent"]] == [False]
    both = client.get("/quality/usage?include_admin=true", headers=KEY).json()
    assert both["totals"]["questions"] == 2 and len(both["recent"]) == 2


def test_scores_appear_on_the_message_after_scoring(chat):
    client, _ = chat
    _online(chat, _FakeJudge({"faithfulness": 0.8, "answer_relevancy": 0.9}))
    cid = _send(client, "What is the default chunk size?", headers=GUEST).json()["conversation_id"]
    bot = client.get(f"/conversations/{cid}", headers=GUEST).json()["messages"][1]
    assert bot["faithfulness"] == 0.8 and bot["answer_relevancy"] == 0.9


def test_quality_test_scores_now_and_saves_nothing(chat):
    client, _ = chat
    judge = _FakeJudge({"faithfulness": 1.0, "answer_relevancy": 0.7})
    _online(chat, judge, rate=0)  # sampling off: a test is ALWAYS scored anyway
    r = client.post("/quality/test", headers=KEY, json={"question": "default chunk size?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answerable"] and body["answer"] == "100 characters [1]."
    assert body["faithfulness"] == 1.0 and body["answer_relevancy"] == 0.7
    assert body["passages"] and any(p["cited"] for p in body["passages"])
    assert len(judge.seen) == 1
    usage = client.get("/quality/usage?include_admin=true", headers=KEY).json()
    assert usage["totals"]["questions"] == 0  # not saved as a conversation
    assert client.post("/quality/test", headers=GUEST, json={"question": "x?"}).status_code == 401


def test_quality_test_does_not_judge_a_refusal(chat):
    client, llm = chat
    judge = _FakeJudge({"faithfulness": 1.0})
    _online(chat, judge)
    llm.reply[LlmAnswer] = LlmAnswer(answerable=False, answer="")
    body = client.post("/quality/test", headers=KEY, json={"question": "CEO salary?"}).json()
    assert body["answerable"] is False and body["faithfulness"] is None and judge.seen == []


# --- data retention (plan B10) ---------------------------------------------------------------


def test_delete_all_conversations_removes_only_the_callers(chat):
    client, _ = chat
    _send(client, "What is the default chunk size?")
    _send(client, "What is the default overlap?")
    _send(client, "What is the default chunk size?", headers=OTHER)
    assert client.delete("/conversations", headers=KEY).status_code == 204
    assert client.get("/conversations", headers=KEY).json() == []
    assert len(client.get("/conversations", headers=OTHER).json()) == 1  # others keep theirs


def test_retention_deletes_old_chats_and_usage_rows_only(chat):
    client, _ = chat
    svc = app.dependency_overrides[get_services]()
    _send(client, "What is the default chunk size?")
    store = svc.open_store()
    store.add_llm_call(
        model="m", purpose="answer", kind="chat", input_tokens=1, output_tokens=1, texts=None,
        latency_ms=5, error=None, run_id="r1",
    )  # fmt: skip
    store.close()
    now = datetime.now(UTC)

    kept = retention.purge(svc.settings, now=now + timedelta(days=89))  # still within 90 days
    assert kept == {"conversations": 0, "llm_calls": 0}
    forever = svc.settings.model_copy(update={"chat_retention_days": 0, "usage_retention_days": 0})
    assert retention.purge(forever, now=now + timedelta(days=999)) == kept  # 0 = keep forever

    gone = retention.purge(svc.settings, now=now + timedelta(days=91))
    assert gone == {"conversations": 1, "llm_calls": 1}
    assert client.get("/conversations", headers=KEY).json() == []
    assert client.get("/assistant").json()["chat_retention_days"] == 90  # shown in the chat


def test_the_api_starts_the_daily_cleanup(chat, monkeypatch):
    calls = []
    monkeypatch.setattr(retention, "purge", lambda settings: calls.append(settings))
    with TestClient(app):  # `with` runs the lifespan (startup + shutdown)
        deadline = time.monotonic() + 3
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
    assert len(calls) == 1  # purged once at startup; the next run is a day later
