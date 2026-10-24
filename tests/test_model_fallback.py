"""Tests for the answer model (plan G2/G3/G5): Gemini and Groq in the order chosen in
Admin › Settings, with automatic fallback when a provider is out of quota or overloaded.

Fake models only (conftest.FakeLlm, which can raise a scripted provider error); no API calls.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLlm
from rag_platform.api.app import app
from rag_platform.api.deps import Services, get_services
from rag_platform.config import RagConfig, Settings, load_rag_config, save_rag_config
from rag_platform.generation.answer import LlmAnswer
from rag_platform.llm import FallbackChat, provider_problem

KEY = {"X-API-Key": "test-key"}
GUEST = {"X-Guest-Id": "3f1c2a9e-7b4d-4c1a-9e2f-0a1b2c3d4e5f"}
GEMINI = LlmAnswer(answerable=True, answer="from Gemini [P1]")
GROQ = LlmAnswer(answerable=True, answer="from Groq [P1]")


class QuotaError(Exception):
    """Like the providers' rate-limit errors (google: ResourceExhausted, groq: RateLimitError)."""

    def __init__(self):
        super().__init__("429 RESOURCE_EXHAUSTED: quota exceeded")


def make(tmp_path: Path, gemini, groq=GROQ, **choice) -> FallbackChat:
    path = tmp_path / "rag_config.json"
    save_rag_config(RagConfig(**choice), path)
    models = {"gemini": FakeLlm(gemini)}
    if groq is not None:
        models["groq"] = FakeLlm(groq)
    return FallbackChat(models, path)


def ask(chat: FallbackChat) -> str:
    return chat.with_structured_output(LlmAnswer).invoke([("human", "q")]).answer


def test_gemini_first_by_default_and_groq_takes_over_when_gemini_is_out_of_quota(tmp_path):
    assert ask(make(tmp_path, GEMINI)) == "from Gemini [P1]"
    chat = make(tmp_path, QuotaError())
    assert ask(chat) == "from Groq [P1]"
    assert chat.models["gemini"].schemas == [LlmAnswer]  # Gemini was tried first


def test_the_admin_can_put_groq_first(tmp_path):
    chat = make(tmp_path, GEMINI, chat_provider="groq")
    assert chat.order() == ["groq", "gemini"]
    assert ask(chat) == "from Groq [P1]"


def test_without_fallback_only_the_chosen_provider_is_used(tmp_path):
    chat = make(tmp_path, QuotaError(), chat_fallback=False)
    with pytest.raises(QuotaError):
        ask(chat)
    assert chat.models["groq"].schemas == []


def test_other_errors_are_not_hidden_by_switching_provider(tmp_path):
    chat = make(tmp_path, ValueError("bad request: schema not supported"))
    with pytest.raises(ValueError):
        ask(chat)
    assert chat.models["groq"].schemas == []


def test_without_a_groq_key_gemini_is_the_only_provider(tmp_path):
    chat = make(tmp_path, GEMINI, groq=None, chat_provider="groq")  # stale choice
    assert chat.order() == ["gemini"]


def test_when_every_provider_fails_the_last_error_is_raised(tmp_path):
    with pytest.raises(TimeoutError):
        ask(make(tmp_path, QuotaError(), groq=TimeoutError("read timed out")))


def test_which_errors_count_as_a_provider_problem():
    assert provider_problem(QuotaError()) == "quota"
    assert provider_problem(TimeoutError("Timeout")) == "slow"
    assert provider_problem(Exception("Error code: 503 - over capacity")) == "slow"
    assert provider_problem(ValueError("invalid schema")) is None


# --- Admin › Settings "Answer model" -----------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path, fake_embedder):
    def start(groq_key: str = "groq-secret-key") -> tuple[TestClient, Services]:
        settings = Settings(
            _env_file=None,
            online_eval_sample_rate=0,
            gemini_api_key="x",
            groq_api_key=groq_key,
            api_keys="test-key",
            index_dir=tmp_path / "idx",
        )
        save_rag_config(RagConfig(), tmp_path / "rag_config.json")
        svc = Services(settings, fake_embedder, FakeLlm(), config_path=tmp_path / "rag_config.json")
        app.dependency_overrides[get_services] = lambda: svc
        return TestClient(app), svc

    yield start
    app.dependency_overrides.clear()


def test_the_admin_sees_the_providers_and_switches_the_answer_model(client):
    c, svc = client()
    providers = c.get("/config/providers", headers=KEY).json()
    assert [(p["name"], p["configured"]) for p in providers] == [("gemini", True), ("groq", True)]
    assert "groq-secret-key" not in str(providers)  # never the key
    r = c.put("/config/model", headers=KEY, json={"chat_provider": "groq", "chat_fallback": False})
    assert r.status_code == 200
    cfg = load_rag_config(svc.config_path)  # used from the next question
    assert (cfg.chat_provider, cfg.chat_fallback) == ("groq", False)
    assert c.put("/config/model", headers=GUEST, json=r.json()).status_code == 401


def test_a_provider_without_a_key_cannot_be_chosen(client):
    c, _ = client(groq_key="")
    providers = c.get("/config/providers", headers=KEY).json()
    assert providers[1] == {"name": "groq", "model": providers[1]["model"], "configured": False}
    r = c.put("/config/model", headers=KEY, json={"chat_provider": "groq", "chat_fallback": True})
    assert r.status_code == 422
