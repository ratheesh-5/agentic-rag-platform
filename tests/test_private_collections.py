"""Tests for private collections (plan section I, phase 2): every signed-in user can keep their
own documents, and NOBODY else - other users, guests, admins - can see, search or change them.

Users sign in through a faked Google check (like test_auth.py); everything after that (sessions,
permissions, search, limits) runs for real against a temporary database. No API calls.
"""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLlm
from rag_platform.api import auth
from rag_platform.api.app import app
from rag_platform.api.deps import Services, get_services
from rag_platform.config import RagConfig, Settings, save_rag_config
from rag_platform.generation.answer import LlmAnswer
from rag_platform.store.db import Store

GUEST = {"X-Guest-Id": "3f1c2a9e-7b4d-4c1a-9e2f-0a1b2c3d4e5f"}
TOKENS = {
    "alice-google-id-token": {"sub": "1", "email": "alice@example.com", "name": "Alice"},
    "bob-google-id-token-xx": {"sub": "2", "email": "bob@example.com", "name": "Bob"},
    "boss-google-id-token-x": {"sub": "3", "email": "boss@example.com", "name": "Boss"},
}
RESUME = (
    b"Alice Example works as a Senior AI Engineer at Contoso since 2021 and leads the RAG team."
)
QUESTION = "Where does Alice work as a Senior AI Engineer?"


def fake_verify(credential: str, client_id: str) -> dict:
    return TOKENS[credential] | {"email_verified": True, "picture": ""}


@pytest.fixture
def env(tmp_path: Path, fake_embedder, monkeypatch):
    settings = Settings(
        _env_file=None,
        online_eval_sample_rate=0,
        gemini_api_key="x",
        google_client_id="test-client.apps.googleusercontent.com",
        admin_emails="boss@example.com",
        index_dir=tmp_path / "idx",
        upload_dir=tmp_path / "uploads",
    )
    save_rag_config(RagConfig(similarity_threshold=0.0), tmp_path / "rag_config.json")
    llm = FakeLlm(LlmAnswer(answerable=True, answer="At Contoso [1]."))
    svc = Services(settings, fake_embedder, llm, config_path=tmp_path / "rag_config.json")
    app.dependency_overrides[get_services] = lambda: svc
    monkeypatch.setattr(auth, "verify_google_token", fake_verify)
    client = TestClient(app)
    people = {}
    for name, token in [("alice", "alice-google-id-token"), ("bob", "bob-google-id-token-xx"),
                        ("boss", "boss-google-id-token-x")]:  # fmt: skip
        r = client.post("/auth/google", json={"credential": token})
        people[name] = {"Authorization": f"Bearer {r.json()['token']}"}
    yield client, svc, people
    app.dependency_overrides.clear()


def create(client, headers, name: str, **extra) -> dict:
    r = client.post("/collections", headers=headers, json={"name": name, **extra})
    assert r.status_code == 201, r.text
    return r.json()


def upload(client, headers, collection_id: int, name="resume.md", data=RESUME):
    return client.post(
        "/documents",
        headers=headers,
        files={"file": (name, data)},
        data={"collection_id": str(collection_id)},
    )


def low_threshold(svc, collection_id: int) -> None:
    """The fake embedder scores low; pin the threshold so answers aren't refused."""
    store = svc.open_store()
    try:
        c = store.get_collection(collection_id)
        store.update_collection(collection_id, c["name"], c["description"], 0.3)
    finally:
        store.close()


def test_a_user_keeps_documents_in_a_private_collection_and_asks_questions(env):
    client, svc, people = env
    mine = create(client, people["alice"], "My resume")
    assert mine["is_private"] and mine["can_manage"]
    assert upload(client, people["alice"], mine["id"]).status_code == 201
    low_threshold(svc, mine["id"])
    docs = client.get("/documents", headers=people["alice"]).json()  # her own documents
    assert [d["collection_id"] for d in docs] == [mine["id"]]
    r = client.post(
        "/chat", headers=people["alice"], json={"message": QUESTION, "collection_id": mine["id"]}
    )
    assert r.status_code == 200
    assert r.json()["assistant_message"]["sources"][0]["document"].endswith("resume.md")


def test_nobody_else_can_see_or_touch_a_private_collection(env):
    client, svc, people = env
    mine = create(client, people["alice"], "My resume")
    upload(client, people["alice"], mine["id"])
    doc_id = client.get("/documents", headers=people["alice"]).json()[0]["id"]
    cid = mine["id"]
    for who, headers in [("bob", people["bob"]), ("admin", people["boss"]), ("guest", GUEST),
                         ("nobody", {})]:  # fmt: skip
        listed = client.get("/collections", headers=headers).json()
        assert cid not in [c["id"] for c in listed], who
        assert client.get(f"/assistant?collection_id={cid}", headers=headers).status_code == 404
        if who == "nobody":
            continue
        chat = {"message": QUESTION, "collection_id": cid}
        assert client.post("/chat", headers=headers, json=chat).status_code == 404, who
        assert upload(client, headers, cid).status_code in (403, 404), who
        assert client.delete(f"/documents/{doc_id}", headers=headers).status_code in (403, 404)
        put = client.put(f"/collections/{cid}", headers=headers, json={"name": "x"})
        assert put.status_code in (403, 404), who
        assert client.delete(f"/collections/{cid}", headers=headers).status_code in (403, 404)
    # The admin's document list holds shared documents only - never a user's private ones.
    assert client.get("/documents", headers=people["boss"]).json() == []
    assert client.get(f"/documents?collection_id={cid}", headers=people["boss"]).status_code == 404
    # ... but the admin may know HOW MUCH is stored privately (capacity, cost) - not what.
    totals = client.get("/collections/private-summary", headers=people["boss"]).json()
    assert totals == {"collections": 1, "users": 1, "documents": 1, "passages": 1}
    assert client.get("/collections/private-summary", headers=people["bob"]).status_code == 403


def test_the_ask_in_hint_never_points_to_someone_elses_private_collection(env):
    client, svc, people = env
    save_rag_config(RagConfig(similarity_threshold=0.95), svc.config_path)  # demo refuses
    mine = create(client, people["alice"], "My resume")
    upload(client, people["alice"], mine["id"])
    asked = client.post("/chat", headers=people["bob"], json={"message": RESUME.decode()})
    assert asked.json()["assistant_message"]["found_in"] == []  # Alice's text stays hidden
    own = client.post("/chat", headers=people["alice"], json={"message": RESUME.decode()})
    assert own.json()["assistant_message"]["found_in"] == [{"id": mine["id"], "name": "My resume"}]


def test_live_quality_never_shows_answers_from_private_collections(env):
    client, svc, people = env
    mine = create(client, people["alice"], "My resume")
    upload(client, people["alice"], mine["id"])
    low_threshold(svc, mine["id"])
    chat = {"message": QUESTION, "collection_id": mine["id"]}
    assert client.post("/chat", headers=people["alice"], json=chat).status_code == 200
    report = client.get("/quality/usage?include_admin=true", headers=people["boss"]).json()
    assert QUESTION not in json.dumps(report) and "Contoso" not in json.dumps(report)


def test_names_are_unique_per_owner_and_admins_choose_shared_or_private(env):
    client, _, people = env
    create(client, people["alice"], "My resume")
    create(client, people["bob"], "My resume")  # another owner: fine
    again = client.post("/collections", headers=people["alice"], json={"name": "My resume"})
    assert again.status_code == 409
    shared = create(client, people["boss"], "HR policies")
    assert not shared["is_private"] and shared["can_manage"]
    own = create(client, people["boss"], "Boss notes", private=True)
    assert own["is_private"]
    assert shared["id"] in [c["id"] for c in client.get("/collections", headers=GUEST).json()]
    assert upload(client, people["alice"], shared["id"]).status_code == 403  # shared: admins


def test_per_user_limits_are_checked_before_any_embedding(env):
    client, svc, people = env
    svc.settings.user_max_collections = 1
    svc.settings.user_max_documents = 1
    mine = create(client, people["alice"], "My resume")
    assert (
        client.post("/collections", headers=people["alice"], json={"name": "B"}).status_code == 403
    )
    assert upload(client, people["alice"], mine["id"]).status_code == 201
    second = upload(client, people["alice"], mine["id"], name="cv2.md")
    assert second.status_code == 403 and "at most 1 documents" in second.json()["detail"]
    assert upload(client, people["alice"], mine["id"]).status_code == 201  # replacing is fine
    svc.settings.user_max_documents = 5
    svc.settings.user_max_passages = 1
    embedded = len(svc.embedder.calls)
    big = upload(client, people["alice"], mine["id"], name="long.md", data=b"word " * 2000)
    assert big.status_code == 403 and "passages" in big.json()["detail"]
    assert len(svc.embedder.calls) == embedded  # refused before spending embedding quota
    assert not (svc.settings.upload_dir / str(mine["id"]) / "long.md").exists()


def test_delete_my_account_removes_everything_of_that_user_only(env):
    client, svc, people = env
    mine = create(client, people["alice"], "My resume")
    upload(client, people["alice"], mine["id"])
    low_threshold(svc, mine["id"])
    client.post(
        "/chat", headers=people["alice"], json={"message": QUESTION, "collection_id": mine["id"]}
    )
    bobs = create(client, people["bob"], "Bob's files")
    index_file = svc.settings.collection_index_path(mine["id"])
    assert index_file.exists()

    assert client.delete("/me", headers=GUEST).status_code == 403  # no account to delete
    assert client.delete("/me", headers=people["alice"]).status_code == 204
    assert client.get("/me", headers=people["alice"]).status_code == 401  # signed out for good
    assert not index_file.exists()
    assert not (svc.settings.upload_dir / str(mine["id"])).exists()
    store = svc.open_store()
    try:
        assert store.get_collection(mine["id"]) is None
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM users WHERE email = 'alice@example.com'"
            ).fetchone()[0]
            == 0
        )
        assert store.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
        assert store.get_collection(bobs["id"]) is not None  # Bob is untouched
    finally:
        store.close()


def test_an_old_database_gets_names_unique_per_owner(tmp_path):
    db = tmp_path / "old.db"
    old = sqlite3.connect(db)  # phase 1: names unique for everyone
    old.executescript(
        "CREATE TABLE collections (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL"
        " UNIQUE, description TEXT NOT NULL, owner TEXT NOT NULL DEFAULT '',"
        " created_at TEXT NOT NULL, threshold REAL, auto_threshold REAL);"
        "INSERT INTO collections VALUES (1, 'Demo', '', '', 't', NULL, NULL);"
        "INSERT INTO collections VALUES (7, 'HR policies', '', '', 't', NULL, 0.58);"
    )
    old.close()
    store = Store(db)
    try:
        assert store.get_collection(7)["auto_threshold"] == 0.58  # same ids and values
        store.create_collection("HR policies", "", owner="user-hash")  # now allowed
        with pytest.raises(sqlite3.IntegrityError):
            store.create_collection("HR policies", "")  # but not twice for the same owner
    finally:
        store.close()
