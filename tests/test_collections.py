"""Tests for collections (plan section I, phase 1): named groups of documents, each searched on
its own. The most important property is ISOLATION: a question in one collection must never see
a passage of another - in dense and hybrid search, in chats and their follow-ups.

Fake models only (FakeEmbedder, FakeLlm): no API calls.
"""

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLlm
from rag_platform.api import service
from rag_platform.api.app import app
from rag_platform.api.deps import Services, get_services
from rag_platform.config import DEFAULT_COLLECTION_ID, RagConfig, Settings, save_rag_config
from rag_platform.generation.answer import LlmAnswer
from rag_platform.store.db import Store

KEY = {"X-API-Key": "test-key"}
GUEST = {"X-Guest-Id": "3f1c2a9e-7b4d-4c1a-9e2f-0a1b2c3d4e5f"}
HANDBOOK = b"The default chunk size is 800 characters and the overlap is 100 characters."
LEAVE = b"Employees get twenty days of annual leave per year, plus public holidays."
LOAN = b"The home loan interest rate is 8.5 percent per year, fixed for five years."


@pytest.fixture
def env(tmp_path: Path, fake_embedder):
    settings = Settings(
        _env_file=None,
        online_eval_sample_rate=0,
        gemini_api_key="x",
        api_keys="test-key",
        index_dir=tmp_path / "idx",
        upload_dir=tmp_path / "uploads",
    )
    save_rag_config(RagConfig(similarity_threshold=0.0), tmp_path / "rag_config.json")
    llm = FakeLlm(LlmAnswer(answerable=True, answer="From the documents [1]."))
    svc = Services(settings, fake_embedder, llm, config_path=tmp_path / "rag_config.json")
    app.dependency_overrides[get_services] = lambda: svc
    yield TestClient(app), svc
    app.dependency_overrides.clear()


def upload(client, name: str, data: bytes, collection_id: int = DEFAULT_COLLECTION_ID):
    return client.post(
        "/documents",
        headers=KEY,
        files={"file": (name, data)},
        data={"collection_id": str(collection_id)},
    )


def new_collection(client, name: str, threshold: float | None = 0.0) -> int:
    """Create a collection. Most tests are about isolation, not thresholds, and the fake embedder
    scores low: they pin the threshold to 0 (as the fixture's config does for the demo
    collection). threshold=None keeps the automatic one."""
    r = client.post("/collections", headers=KEY, json={"name": name, "description": "test"})
    assert r.status_code == 201, r.text
    collection_id = r.json()["id"]
    if threshold is not None:
        store = app.dependency_overrides[get_services]().open_store()
        try:
            store.update_collection(collection_id, name, "test", threshold)
        finally:
            store.close()
    return collection_id


def test_every_database_has_the_demo_collection_and_old_data_moves_into_it(tmp_path):
    db = tmp_path / "old.db"
    old = sqlite3.connect(db)  # a database from before collections
    old.executescript(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE,"
        " doc_type TEXT NOT NULL, content_hash TEXT NOT NULL, chunker TEXT NOT NULL,"
        " text TEXT NOT NULL, ingested_at TEXT NOT NULL);"
        "INSERT INTO documents VALUES (1, 'a.md', 'md', 'h', '800/100', 'text', '2026-10-01');"
        "CREATE TABLE conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,"
        " title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
        "INSERT INTO conversations VALUES (1, 'o', 't', '2026-10-01', '2026-10-01');"
    )
    old.close()
    store = Store(db)
    try:
        [demo] = store.list_collections()
        assert demo["id"] == DEFAULT_COLLECTION_ID and demo["documents"] == 1
        assert store.list_documents()[0]["collection_id"] == DEFAULT_COLLECTION_ID
        assert store.get_conversation(1, "o")["collection"] == demo["name"]
    finally:
        store.close()


def test_admins_manage_collections_everyone_can_list_them(env):
    client, _ = env
    listed = client.get("/collections").json()  # open, like /assistant
    assert [(c["id"], c["is_default"]) for c in listed] == [(DEFAULT_COLLECTION_ID, True)]
    hr = new_collection(client, "HR policies")
    assert client.post("/collections", headers=KEY, json={"name": "HR policies"}).status_code == 409
    # A guest has no account: no collections of their own (sign in first).
    assert client.post("/collections", headers=GUEST, json={"name": "Mine"}).status_code == 403
    r = client.put(f"/collections/{hr}", headers=KEY, json={"name": "HR", "description": "Leave"})
    assert r.json()["name"] == "HR" and r.json()["description"] == "Leave"
    assert client.delete(f"/collections/{DEFAULT_COLLECTION_ID}", headers=KEY).status_code == 409
    assert client.delete("/collections/999", headers=KEY).status_code == 404
    assert client.put("/collections/999", headers=KEY, json={"name": "x"}).status_code == 404


def test_a_question_only_searches_its_own_collection(env):
    client, svc = env
    hr, loans = new_collection(client, "HR policies"), new_collection(client, "Loans")
    assert upload(client, "handbook.md", HANDBOOK).status_code == 201
    assert upload(client, "leave.md", LEAVE, hr).status_code == 201
    assert upload(client, "loan.md", LOAN, loans).status_code == 201
    for retrieval in ("dense", "hybrid"):
        cfg = RagConfig(similarity_threshold=0.0, retrieval=retrieval)
        for collection_id, only in [(hr, "leave.md"), (loans, "loan.md")]:
            # A question that matches ANOTHER collection's text best still stays inside its own.
            ans = service.answer(svc, cfg, "What is the home loan interest rate?", collection_id)
            assert ans.context and {Path(c.doc_path).name for c in ans.context} == {only}
    files = {c["name"]: c["documents"] for c in client.get("/collections").json()}
    assert files == {"AI Engineering Handbook": 1, "HR policies": 1, "Loans": 1}
    paths = [d["path"] for d in client.get(f"/documents?collection_id={hr}", headers=KEY).json()]
    assert len(paths) == 1 and paths[0].endswith(f"uploads/{hr}/leave.md")


def test_uploads_go_to_their_collection_and_the_same_name_may_exist_twice(env):
    client, svc = env
    hr = new_collection(client, "HR policies")
    first = upload(client, "notes.md", HANDBOOK).json()["document"]
    second = upload(client, "notes.md", LEAVE, hr).json()["document"]
    assert first["collection_id"] == DEFAULT_COLLECTION_ID and second["collection_id"] == hr
    assert first["path"] != second["path"]
    assert (svc.settings.upload_dir / "notes.md").exists()  # demo collection: as before
    assert (svc.settings.upload_dir / str(hr) / "notes.md").exists()
    assert upload(client, "x.md", LEAVE, 999).status_code == 404  # unknown collection
    only_hr = client.get(f"/documents?collection_id={hr}", headers=KEY).json()
    assert [d["id"] for d in only_hr] == [second["id"]]
    assert len(client.get("/documents", headers=KEY).json()) == 2  # no filter: all

    assert client.delete(f"/documents/{second['id']}", headers=KEY).status_code == 204
    assert not (svc.settings.upload_dir / str(hr) / "notes.md").exists()
    store = svc.open_store()
    try:
        assert store.chunk_ids(collection_id=hr) == []
        assert store.chunk_ids(collection_id=DEFAULT_COLLECTION_ID)  # the other one stays
    finally:
        store.close()


def test_a_chat_stays_in_the_collection_it_started_in(env):
    client, svc = env
    hr = new_collection(client, "HR policies")
    upload(client, "handbook.md", HANDBOOK)
    upload(client, "leave.md", LEAVE, hr)
    first = client.post(
        "/chat", headers=GUEST, json={"message": "How many days?", "collection_id": hr}
    )
    body = first.json()
    assert body["collection_id"] == hr
    assert body["assistant_message"]["sources"][0]["document"].endswith("leave.md")
    # A follow-up that names another collection still searches the conversation's collection.
    follow = client.post(
        "/chat",
        headers=GUEST,
        json={"message": "And holidays?", "conversation_id": body["conversation_id"],
              "collection_id": DEFAULT_COLLECTION_ID},
    )  # fmt: skip
    assert follow.json()["collection_id"] == hr
    assert follow.json()["assistant_message"]["sources"][0]["document"].endswith("leave.md")
    [summary] = client.get("/conversations", headers=GUEST).json()
    assert summary["collection_id"] == hr and summary["collection"] == "HR policies"
    conv = client.get(f"/conversations/{body['conversation_id']}", headers=GUEST).json()
    assert conv["collection"] == "HR policies"
    bad = client.post("/chat", headers=GUEST, json={"message": "Hi there", "collection_id": 999})
    assert bad.status_code == 404


def test_the_welcome_topics_belong_to_the_chosen_collection(env, tmp_path):
    client, svc = env
    svc.assistant_path = tmp_path / "none.json"  # no hand-written topics: generated ones only
    hr = new_collection(client, "HR policies")
    upload(client, "handbook.md", HANDBOOK)
    upload(client, "leave_policy.md", LEAVE, hr)
    topics = client.get(f"/assistant?collection_id={hr}").json()["topics"]
    assert [t["documents"] for t in topics] == [["leave_policy.md"]]
    default = client.get("/assistant").json()["topics"]
    assert [t["documents"] for t in default] == [["handbook.md"]]


def test_deleting_a_collection_removes_its_documents_chats_index_and_files(env):
    client, svc = env
    hr = new_collection(client, "HR policies")
    upload(client, "handbook.md", HANDBOOK)
    upload(client, "leave.md", LEAVE, hr)
    client.post("/chat", headers=GUEST, json={"message": "How many days?", "collection_id": hr})
    index_file = svc.settings.collection_index_path(hr)
    assert index_file.exists()
    assert client.delete(f"/collections/{hr}", headers=GUEST).status_code == 403  # admins only
    assert client.delete(f"/collections/{hr}", headers=KEY).status_code == 204
    assert not index_file.exists()
    assert not (svc.settings.upload_dir / str(hr)).exists()
    assert client.get("/conversations", headers=GUEST).json() == []
    assert [c["id"] for c in client.get("/collections").json()] == [DEFAULT_COLLECTION_ID]
    assert len(client.get("/documents", headers=KEY).json()) == 1  # the demo document stays


def test_a_refusal_points_to_the_collections_that_have_matching_documents(env):
    client, svc = env
    save_rag_config(RagConfig(similarity_threshold=0.9), svc.config_path)
    hr = new_collection(client, "HR policies", threshold=None)  # its automatic threshold
    new_collection(client, "Empty", threshold=None)  # no passages: never suggested
    upload(client, "handbook.md", HANDBOOK)
    upload(client, "leave.md", LEAVE, hr)
    question = LEAVE.decode()  # matches the HR document almost exactly, the handbook not at all
    asked = client.post("/chat", headers=GUEST, json={"message": question})  # demo collection
    bot = asked.json()["assistant_message"]
    assert bot["answerable"] is False and bot["found_in"] == [{"id": hr, "name": "HR policies"}]
    # A question that matches nothing anywhere gets no hint.
    other = client.post("/chat", headers=GUEST, json={"message": "zebra quantum violin"})
    assert other.json()["assistant_message"]["found_in"] == []


def test_each_collection_measures_its_own_threshold_automatically(env):
    client, svc = env
    save_rag_config(RagConfig(similarity_threshold=0.9), svc.config_path)  # strict global value
    hr = new_collection(client, "HR policies", threshold=None)
    empty = {c["id"]: c for c in client.get("/collections").json()}[hr]
    assert empty["threshold_source"] == "default" and empty["threshold"] == 0.9  # nothing yet
    upload(client, "leave.md", LEAVE, hr)  # upload -> measured with the probe questions
    measured = {c["id"]: c for c in client.get("/collections").json()}
    assert measured[hr]["threshold_source"] == "automatic" and measured[hr]["threshold"] < 0.9
    demo = measured[DEFAULT_COLLECTION_ID]
    assert demo["threshold_source"] == "measured" and demo["threshold"] == 0.9  # rag_config
    # A question that the strict global value would refuse is answered in HR with its own value.
    cfg = RagConfig(similarity_threshold=0.9)
    question = "How many days of annual leave do employees get per year?"
    assert service.answer(svc, cfg, question, hr).answerable
    upload(client, "handbook.md", HANDBOOK)
    assert not service.answer(svc, cfg, question, DEFAULT_COLLECTION_ID).answerable


def test_admins_can_override_the_threshold_except_on_the_demo_collection(env):
    client, _ = env
    hr = new_collection(client, "HR policies", threshold=None)
    upload(client, "leave.md", LEAVE, hr)
    r = client.put(
        f"/collections/{hr}", headers=KEY, json={"name": "HR policies", "threshold": 0.7}
    )
    assert r.json()["threshold"] == 0.7 and r.json()["threshold_source"] == "manual"
    back = client.put(f"/collections/{hr}", headers=KEY, json={"name": "HR policies"})
    assert back.json()["threshold_source"] == "automatic"  # no value = automatic again
    demo = client.put(
        f"/collections/{DEFAULT_COLLECTION_ID}",
        headers=KEY,
        json={"name": "Demo", "threshold": 0.5},
    )
    assert demo.status_code == 422  # changes only through the optimizer
    assert (
        client.put(
            f"/collections/{hr}", headers=KEY, json={"name": "x", "threshold": 2}
        ).status_code
        == 422
    )


def test_the_admin_test_question_searches_the_chosen_collection(env):
    # Found by the owner: "Test a question" (Live quality) only searched the demo collection,
    # so a leave question stayed unanswered although HR policies covers it.
    client, svc = env
    upload(client, "handbook.md", HANDBOOK)
    hr = new_collection(client, "HR policies")
    upload(client, "leave.md", LEAVE, hr)
    ask = {"question": "How many days of annual leave do employees get?"}
    demo = client.post("/quality/test", headers=KEY, json=ask).json()  # default: demo collection
    assert {p["document"].split("/")[-1] for p in demo["passages"]} == {"handbook.md"}
    r = client.post("/quality/test", headers=KEY, json=ask | {"collection_id": hr})
    assert r.status_code == 200, r.text
    assert {p["document"].split("/")[-1] for p in r.json()["passages"]} == {"leave.md"}
    assert (
        client.post("/quality/test", headers=KEY, json=ask | {"collection_id": 999}).status_code
        == 404
    )
