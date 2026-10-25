"""Tests for Admin › Settings › Models (plan G7): models chosen by name, a backup model of the
same provider, the model lists, Test, the quality check before use and the judge per score.

Fake models only (conftest.FakeLlm and a tiny echo model); the providers' model lists are
monkeypatched. No API calls.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLlm
from rag_platform import llm
from rag_platform.api import models as models_api
from rag_platform.api.app import app
from rag_platform.api.deps import Services, get_services
from rag_platform.config import RagConfig, Settings, load_rag_config, save_rag_config
from rag_platform.evaluation.online import judge
from rag_platform.generation.answer import LlmAnswer
from rag_platform.llm import FallbackChat
from rag_platform.store.db import Store

KEY = {"X-API-Key": "test-key"}
ANSWER = LlmAnswer(answerable=True, answer="Twenty days [P1].")
REFUSE = LlmAnswer(answerable=False, answer="")


class QuotaError(Exception):
    def __init__(self):
        super().__init__("429 RESOURCE_EXHAUSTED: quota exceeded")


class Echo:
    """A chat model that says which model it is (for the Test button)."""

    def __init__(self, name: str = "default", error: Exception | None = None):
        self.name, self.error = name, error

    def invoke(self, messages, config=None, **kwargs):
        if self.error:
            raise self.error
        return type("Reply", (), {"content": f"OK from {self.name}"})()


# --- FallbackChat: the chosen model, then its backup, then the other provider ---------------


def make(tmp_path: Path, replies: dict[str, object], **choice) -> tuple[FallbackChat, dict]:
    """replies: model name -> reply; "gemini"/"groq" = the providers' .env default models."""
    path = tmp_path / "rag_config.json"
    save_rag_config(RagConfig(**choice), path)
    built = {}

    def factory(provider: str, model: str):
        built[model] = FakeLlm(replies[model])
        return built[model]

    defaults = {p: FakeLlm(replies[p]) for p in ("gemini", "groq") if p in replies}
    return FallbackChat(defaults, path, factory), built


def ask(chat: FallbackChat) -> str:
    return chat.with_structured_output(LlmAnswer).invoke([("human", "q")]).answer


def test_the_model_chosen_in_settings_writes_the_answer(tmp_path):
    chat, built = make(
        tmp_path, {"gemini": QuotaError(), "gemini-pro": ANSWER}, gemini_model="gemini-pro"
    )
    assert ask(chat) == "Twenty days [P1]."
    assert chat.models["gemini"].schemas == []  # the .env default was not used
    assert chat.named("gemini", "gemini-pro") is built["gemini-pro"]  # built once, reused


def test_the_backup_model_takes_over_before_the_other_provider(tmp_path):
    other = LlmAnswer(answerable=True, answer="from Groq [P1]")
    replies = {"gemini": QuotaError(), "gemini-lite": ANSWER, "groq": other}
    chat, built = make(tmp_path, replies, gemini_backup_model="gemini-lite")
    assert chat.attempts(load_rag_config(chat.config_path)) == [
        ("gemini", ""),
        ("gemini", "gemini-lite"),
        ("groq", ""),
    ]
    assert ask(chat) == "Twenty days [P1]."
    assert chat.models["groq"].schemas == []  # never reached


def test_without_fallback_the_backup_is_not_used(tmp_path):
    replies = {"gemini": QuotaError(), "gemini-lite": ANSWER}
    chat, built = make(tmp_path, replies, gemini_backup_model="gemini-lite", chat_fallback=False)
    with pytest.raises(QuotaError):
        ask(chat)
    assert built == {}


# --- Model lists ------------------------------------------------------------------------------


def test_model_list_failures_give_an_empty_list(monkeypatch):
    llm._model_lists.clear()

    def broken(**kwargs):
        raise ConnectionError("offline")

    import google.genai

    monkeypatch.setattr(google.genai, "Client", broken)
    s = Settings(_env_file=None, gemini_api_key="x")
    assert llm.list_chat_models(s, "gemini") == []


def test_only_text_chat_models_are_listed(monkeypatch):
    llm._model_lists.clear()

    class M:
        def __init__(self, name, actions=("generateContent",)):
            self.name, self.supported_actions = f"models/{name}", list(actions)

    class Client:
        def __init__(self, **kwargs):
            self.models = self

        def list(self):
            names = [
                "gemini-3.5-flash-lite",
                "gemini-2.5-pro",
                "gemini-2.5-flash-image",
                "gemini-2.5-flash-preview-tts",
                "gemma-3-27b-it",
            ]
            return [M(n) for n in names] + [M("gemini-embedding-001", ["embedContent"])]

    import google.genai

    monkeypatch.setattr(google.genai, "Client", Client)
    s = Settings(_env_file=None, gemini_api_key="x")
    assert llm.list_chat_models(s, "gemini") == ["gemini-2.5-pro", "gemini-3.5-flash-lite"]
    llm._model_lists.clear()


# --- API ----------------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path, fake_embedder, monkeypatch):
    monkeypatch.setattr(llm, "list_chat_models", lambda s, p: [f"{p}-a", f"{p}-b"])
    settings = Settings(
        _env_file=None,
        online_eval_sample_rate=0,
        gemini_api_key="x",
        groq_api_key="groq-secret-key",
        api_keys="test-key",
        chat_model="gemini-default",
        groq_model="groq-default",
        index_dir=tmp_path / "idx",
        upload_dir=tmp_path / "uploads",
    )
    save_rag_config(RagConfig(similarity_threshold=0.0), tmp_path / "rag_config.json")
    svc = Services(settings, fake_embedder, FakeLlm(), config_path=tmp_path / "rag_config.json")
    app.dependency_overrides[get_services] = lambda: svc
    yield TestClient(app), svc
    app.dependency_overrides.clear()


def test_settings_show_every_model_slot_and_the_locked_embedding_model(client):
    c, svc = client
    info = c.get("/config/models", headers=KEY).json()
    slots = {s["kind"]: s for s in info["slots"]}
    assert slots["gemini"]["current"] == slots["gemini"]["default"] == "gemini-default"
    assert slots["groq"]["options"] == ["groq-a", "groq-b"]
    assert slots["judge"]["current"] == "gemini-default"  # JUDGE_MODEL empty = chat model
    assert info["embedding"] == svc.settings.embedding_model
    assert "groq-secret-key" not in str(info)
    assert c.get("/config/models").status_code == 401  # admin only


def test_a_new_answer_model_needs_a_passing_check_but_the_judge_does_not(client):
    c, svc = client
    r = c.put("/config/models", headers=KEY, json={"gemini_model": "gemini-a"})
    assert r.status_code == 409
    r = c.put("/config/models", headers=KEY, json={"judge_model": "gemini-a"})
    assert r.status_code == 200
    assert load_rag_config(svc.config_path).judge_model == "gemini-a"
    assert svc.judge_name() == "gemini-a"
    # the .env default is stored as "" (a later .env change still applies)
    c.put("/config/models", headers=KEY, json={"judge_model": "gemini-default"})
    assert load_rag_config(svc.config_path).judge_model == ""


def test_the_backup_must_differ_from_the_main_model_and_pass_its_check(client):
    c, _ = client
    same = {"gemini_backup_model": "gemini-default"}
    assert c.put("/config/models", headers=KEY, json=same).status_code == 422
    unchecked = {"gemini_backup_model": "gemini-b"}
    assert c.put("/config/models", headers=KEY, json=unchecked).status_code == 409


def test_the_test_button_calls_exactly_that_model(client):
    c, svc = client
    svc.llm = FallbackChat(
        {"gemini": Echo()},
        svc.config_path,
        lambda p, m: Echo(m, QuotaError() if m == "gemini-b" else None),
    )
    r = c.post("/config/models/test", headers=KEY, json={"kind": "gemini", "model": "gemini-a"})
    assert r.json()["ok"] and r.json()["message"] == "OK from gemini-a"
    r = c.post("/config/models/test", headers=KEY, json={"kind": "gemini", "model": "gemini-b"})
    assert not r.json()["ok"] and "quota" in r.json()["message"]  # no fallback to another model


def _demo_set(c, svc, tmp_path) -> None:
    """One indexed document and a two-question test set using the check's question ids."""
    c.post(
        "/documents",
        headers=KEY,
        files={"file": ("leave.md", b"Employees get twenty days of annual leave.")},
    )
    doc = c.get("/documents", headers=KEY).json()[0]["path"]
    svc.golden_path = tmp_path / "golden.jsonl"
    items = [
        {
            "id": "q06",
            "question": "How many leave days?",
            "answerable": True,
            "reference_answer": "Twenty.",
            "evidence": [{"doc": doc, "quote": "twenty days"}],
        },
        {
            "id": "u01",
            "question": "Which car do we drive?",
            "answerable": False,
            "reference_answer": "",
            "evidence": [],
        },
    ]
    svc.golden_path.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")


def test_a_passing_check_allows_the_model_as_main_or_backup(client, tmp_path, monkeypatch):
    c, svc = client
    _demo_set(c, svc, tmp_path)
    monkeypatch.setattr(models_api, "CHECK_MINIMUM", 2)
    svc.llm.reply = [ANSWER, REFUSE]  # cites the document, then refuses the unanswerable one
    r = c.post("/config/models/check", headers=KEY, json={"kind": "gemini", "model": "gemini-a"})
    assert r.status_code == 202 and r.json()["state"] == "running"
    check = c.get("/config/models", headers=KEY).json()["checks"][0]  # background task done
    assert (check["state"], check["passed"], check["total"], check["ok"]) == ("done", 2, 2, True)
    assert [i["note"] for i in check["items"]] == ["cited the right document", "refused correctly"]
    body = {"gemini_model": "gemini-default", "gemini_backup_model": "gemini-a"}
    assert c.put("/config/models", headers=KEY, json=body).status_code == 200
    assert load_rag_config(svc.config_path).gemini_backup_model == "gemini-a"
    assert (
        c.put("/config/models", headers=KEY, json={"gemini_model": "gemini-a"}).status_code == 200
    )


def test_a_failing_check_keeps_the_model_out(client, tmp_path):
    c, svc = client
    _demo_set(c, svc, tmp_path)
    svc.llm.reply = [ANSWER, ANSWER]  # also "answers" what the documents can't
    c.post("/config/models/check", headers=KEY, json={"kind": "gemini", "model": "gemini-a"})
    check = c.get("/config/models", headers=KEY).json()["checks"][0]
    assert check["passed"] == 1 and not check["ok"]
    assert (
        c.put("/config/models", headers=KEY, json={"gemini_model": "gemini-a"}).status_code == 409
    )


def test_a_quota_error_fails_the_check_with_a_plain_message(client, tmp_path):
    c, svc = client
    _demo_set(c, svc, tmp_path)
    svc.llm.reply = QuotaError()
    c.post("/config/models/check", headers=KEY, json={"kind": "gemini", "model": "gemini-a"})
    check = c.get("/config/models", headers=KEY).json()["checks"][0]
    assert check["state"] == "failed" and "quota" in check["error"].lower()
    # the lock was released: a new check can start
    r = c.post("/config/models/check", headers=KEY, json={"kind": "gemini", "model": "gemini-a"})
    assert r.status_code == 202


def test_the_judge_needs_no_check(client):
    c, _ = client
    r = c.post("/config/models/check", headers=KEY, json={"kind": "judge", "model": "gemini-a"})
    assert r.status_code == 422


# --- Each score records its judge ---------------------------------------------------------------


def test_scores_record_the_judge_model(client):
    _, svc = client

    class FakeEvaluator:
        llm = None

        def score(self, question, answer, contexts, reference):
            return {"faithfulness": 1.0, "answer_relevancy": 0.9}

    svc.evaluator = FakeEvaluator()
    save_rag_config(RagConfig(judge_model="gemini-a"), svc.config_path)
    result = judge(svc, "q", "a [1]", ["ctx"])
    assert result["judge"] == "gemini-a"
    assert svc.answer_evaluator() is svc.evaluator  # a test's fake is kept, never rebuilt
    store = Store(svc.settings.db_path)
    try:
        cols = {r["name"] for r in store.conn.execute("PRAGMA table_info(answer_scores)")}
    finally:
        store.close()
    assert "judge" in cols
