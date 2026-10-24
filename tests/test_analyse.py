"""Tests for Analyse mode (plan C14, phase 3): a question answered from EVERY document of a
collection with map-reduce (one AI call per document + one to combine).

Checked: one map call per document with global [P] labels, the combined answer's citations,
documents without anything relevant and failed documents are counted (never dropped silently),
the document cap, the free threshold refusal, who may use it and how it counts toward the
hourly limit. Fake models only, no API calls.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLlm
from rag_platform.api import auth, chat
from rag_platform.api.app import app
from rag_platform.api.deps import RateLimiter, Services, get_services
from rag_platform.config import RagConfig, Settings, save_rag_config
from rag_platform.generation import analyse
from rag_platform.generation.analyse import DocumentNote
from rag_platform.generation.answer import BELOW_THRESHOLD, LlmAnswer

GUEST = {"X-Guest-Id": "3f1c2a9e-7b4d-4c1a-9e2f-0a1b2c3d4e5f"}
TOKEN = "alice-google-id-token"
POLICIES = {
    "remote_2025.md": b"Remote work policy 2025: employees may work remotely two days a week.",
    "remote_2026.md": b"Remote work policy 2026: employees may work remotely three days a week.",
    "canteen.md": b"The canteen serves lunch from noon; remote work is not covered here.",
}
QUESTION = "How did the remote work policy change from 2025 to 2026?"
NOTE = DocumentNote(relevant=True, note="Remote work days are listed [P1].")
NOTHING = DocumentNote(relevant=False, note="")


def fake_verify(credential: str, client_id: str) -> dict:
    return {"sub": "1", "email": "alice@example.com", "name": "Alice", "email_verified": True}


@pytest.fixture
def env(tmp_path: Path, fake_embedder, monkeypatch):
    """Alice signed in with a private collection holding the three POLICIES documents."""
    settings = Settings(
        _env_file=None,
        online_eval_sample_rate=0,
        gemini_api_key="x",
        google_client_id="test-client.apps.googleusercontent.com",
        index_dir=tmp_path / "idx",
        upload_dir=tmp_path / "uploads",
    )
    save_rag_config(RagConfig(similarity_threshold=0.0), tmp_path / "rag_config.json")
    svc = Services(settings, fake_embedder, FakeLlm(), config_path=tmp_path / "rag_config.json")
    app.dependency_overrides[get_services] = lambda: svc
    monkeypatch.setattr(auth, "verify_google_token", fake_verify)
    client = TestClient(app)
    session = client.post("/auth/google", json={"credential": TOKEN}).json()["token"]
    alice = {"Authorization": f"Bearer {session}"}
    cid = client.post("/collections", headers=alice, json={"name": "Policies"}).json()["id"]
    for name, data in POLICIES.items():
        r = client.post(
            "/documents", headers=alice, files={"file": (name, data)}, data={"collection_id": cid}
        )
        assert r.status_code == 201, r.text
    store = svc.open_store()
    try:  # the fake embedder scores low: pin a threshold every document passes
        store.update_collection(cid, "Policies", "", 0.01)
    finally:
        store.close()
    yield client, svc, alice, cid
    app.dependency_overrides.clear()


def ask(client, headers, cid, **extra):
    body = {"message": QUESTION, "collection_id": cid, "analyse": True} | extra
    return client.post("/chat", headers=headers, json=body)


def test_every_document_is_read_once_and_the_notes_are_combined_with_citations(env):
    client, svc, alice, cid = env
    svc.llm.reply = {
        DocumentNote: [NOTE, NOTE, NOTHING],
        LlmAnswer: LlmAnswer(answerable=True, answer="Two days [P1] became three [P2] [P9]."),
    }
    r = ask(client, alice, cid)
    assert r.status_code == 200, r.text
    msg = r.json()["assistant_message"]
    assert svc.llm.schemas == [DocumentNote] * 3 + [LlmAnswer]  # map x3, then reduce
    # Each document's passages get their own global labels: [P1] in the first, [P2] in the next.
    assert "[P1]" in svc.llm.prompts[0][1][1] and "[P2]" in svc.llm.prompts[1][1][1]
    assert msg["answerable"] and msg["content"] == "Two days [1] became three [2] ."  # [P9]: gone
    assert len(msg["sources"]) == 2
    assert msg["route"] == "analyse: 3 documents read, 1 without anything on this"


def test_a_failed_document_is_skipped_and_reported(env):
    client, svc, alice, cid = env
    svc.llm.reply = {
        DocumentNote: [NOTE, TimeoutError("slow"), NOTE],
        LlmAnswer: LlmAnswer(answerable=True, answer="Two days [P1]."),
    }
    msg = ask(client, alice, cid).json()["assistant_message"]
    assert msg["answerable"]
    assert msg["route"] == "analyse: 2 documents read, 1 could not be read"


def test_every_document_failing_is_a_provider_error(env):
    client, svc, alice, cid = env
    svc.llm.reply = {DocumentNote: TimeoutError("slow")}
    assert ask(client, alice, cid).status_code == 503


def test_documents_over_the_cap_are_not_read_and_the_answer_says_so(env, monkeypatch):
    client, svc, alice, cid = env
    monkeypatch.setattr(analyse, "MAX_DOCUMENTS", 2)
    svc.llm.reply = {
        DocumentNote: [NOTE, NOTE],
        LlmAnswer: LlmAnswer(answerable=True, answer="Two days [P1]."),
    }
    msg = ask(client, alice, cid).json()["assistant_message"]
    assert svc.llm.schemas.count(DocumentNote) == 2
    assert msg["route"] == "analyse: 2 documents read, 1 more not read (limit 2)"


def test_nothing_relevant_anywhere_is_a_refusal_without_the_combine_call(env):
    client, svc, alice, cid = env
    svc.llm.reply = {DocumentNote: NOTHING}
    msg = ask(client, alice, cid).json()["assistant_message"]
    assert not msg["answerable"] and msg["abstain_reason"] == analyse.NOTHING_RELEVANT
    assert LlmAnswer not in svc.llm.schemas


def test_below_the_collection_threshold_no_ai_call_is_made(env):
    client, svc, alice, cid = env
    store = svc.open_store()
    try:
        store.update_collection(cid, "Policies", "", 0.99)
    finally:
        store.close()
    msg = ask(client, alice, cid).json()["assistant_message"]
    assert msg["abstain_reason"] == BELOW_THRESHOLD
    assert svc.llm.schemas == []


def test_guests_cannot_analyse(env):
    client, svc, _, _ = env
    r = client.post("/chat", headers=GUEST, json={"message": QUESTION, "analyse": True})
    assert r.status_code == 403


def test_an_analysis_counts_as_several_questions(env, monkeypatch):
    client, svc, alice, cid = env
    svc.settings.user_questions_per_hour = 2 * chat.ANALYSE_COST - 1  # room for one, not two
    svc.llm.reply = {DocumentNote: NOTHING}
    assert ask(client, alice, cid).status_code == 200
    r = ask(client, alice, cid)
    assert r.status_code == 429 and "counts as" in r.json()["detail"]


def test_the_limiter_counts_a_cost_and_says_how_long_to_wait():
    limiter = RateLimiter(window=3600)
    assert limiter.hit("k", 5, cost=4) == 0
    assert limiter.hit("k", 5, cost=2) > 0  # 4 + 2 > 5: refused, nothing recorded
    assert limiter.hit("k", 5) == 0  # 5 of 5
    assert limiter.hit("k", 5, cost=6) == 3600  # more than the whole limit: never
