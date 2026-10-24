"""Tests for Sign in with Google (plan B1): sessions, roles, chats per user, sign-out.

Google is never called: `verify_google_token` is replaced by a fake that knows two test tokens
(like the fake LLM). Everything after verification - users, sessions, roles, conversation
ownership, limits - runs for real against a temporary database.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLlm
from rag_platform.api import auth, retention
from rag_platform.api.app import app
from rag_platform.api.deps import Services, get_services, token_hash
from rag_platform.config import RagConfig, Settings, save_rag_config
from rag_platform.generation.answer import LlmAnswer

KEY = {"X-API-Key": "key-a"}
GUEST = {"X-Guest-Id": "3f1c2a9e-7b4d-4c1a-9e2f-0a1b2c3d4e5f"}
ALICE = "google-id-token-for-alice"  # a normal user
BOSS = "google-id-token-for-the-boss"  # email in ADMIN_EMAILS
CLAIMS = {
    ALICE: {"sub": "111", "email": "alice@example.com", "name": "Alice", "picture": "https://p/a"},
    BOSS: {"sub": "222", "email": "Boss@Example.com", "name": "Boss", "picture": ""},
}


def fake_verify(credential: str, client_id: str) -> dict:
    assert client_id == "test-client.apps.googleusercontent.com"  # checked against OUR app
    if credential not in CLAIMS:
        raise ValueError("Token used too late / wrong audience")  # what google-auth raises
    return CLAIMS[credential] | {"email_verified": True}


@pytest.fixture
def env(tmp_path: Path, fake_embedder, monkeypatch):
    settings = Settings(
        _env_file=None,
        online_eval_sample_rate=0,
        gemini_api_key="x",
        api_keys="key-a",
        google_client_id="test-client.apps.googleusercontent.com",
        admin_emails="boss@example.com",
        index_dir=tmp_path / "idx",
        upload_dir=tmp_path / "uploads",
    )
    save_rag_config(RagConfig(similarity_threshold=0.0), tmp_path / "rag_config.json")
    llm = FakeLlm(LlmAnswer(answerable=True, answer="Twenty days [1]."))
    svc = Services(settings, fake_embedder, llm, config_path=tmp_path / "rag_config.json")
    app.dependency_overrides[get_services] = lambda: svc
    monkeypatch.setattr(auth, "verify_google_token", fake_verify)
    client = TestClient(app)
    doc = {"file": ("leave.md", b"Employees get twenty days of annual leave per year.")}
    assert client.post("/documents", headers=KEY, files=doc).status_code == 201
    yield client, svc
    app.dependency_overrides.clear()


def sign_in(client, credential: str, headers: dict | None = None) -> dict:
    r = client.post("/auth/google", json={"credential": credential}, headers=headers or {})
    assert r.status_code == 200, r.text
    return r.json()


def bearer(body: dict) -> dict:
    return {"Authorization": f"Bearer {body['token']}"}


def ask(client, headers: dict):
    return client.post("/chat", headers=headers, json={"message": "How many days of leave?"})


def test_config_tells_the_ui_whether_google_sign_in_is_on(env):
    client, svc = env
    assert client.get("/auth/config").json() == {
        "google_client_id": "test-client.apps.googleusercontent.com"
    }
    svc.settings.google_client_id = ""
    assert client.get("/auth/config").json() == {"google_client_id": ""}
    r = client.post("/auth/google", json={"credential": ALICE})
    assert r.status_code == 503  # not configured


def test_sign_in_gives_a_session_and_chats_follow_the_user_to_any_device(env):
    client, _ = env
    laptop = sign_in(client, ALICE)
    assert laptop["role"] == "user" and laptop["user"]["email"] == "alice@example.com"
    me = client.get("/me", headers=bearer(laptop)).json()
    assert me == {"role": "user", "user": laptop["user"]}
    cid = ask(client, bearer(laptop)).json()["conversation_id"]

    phone = sign_in(client, ALICE)  # signing in again elsewhere = a second session
    assert phone["token"] != laptop["token"]
    listed = client.get("/conversations", headers=bearer(phone)).json()
    assert [c["id"] for c in listed] == [cid]  # same chats on the other device
    assert client.get("/conversations", headers=GUEST).json() == []  # not the guest's


def test_guest_chats_stay_with_the_guest_when_someone_signs_in(env):
    # Found in testing: guest chats were moved into the account that signed in next - on a shared
    # computer, someone else's chats would end up in your account.
    client, _ = env
    cid = ask(client, GUEST).json()["conversation_id"]
    body = sign_in(client, ALICE, headers=GUEST)  # the browser sends its guest id along
    assert "moved_conversations" not in body
    assert client.get("/conversations", headers=bearer(body)).json() == []  # not in the account
    assert [c["id"] for c in client.get("/conversations", headers=GUEST).json()] == [cid]
    r = client.get(f"/conversations/{cid}", headers=bearer(body))
    assert r.status_code == 404  # the account can't open the guest's chat either


def test_admins_are_the_emails_in_admin_emails(env):
    client, _ = env
    boss = sign_in(client, BOSS)
    assert boss["role"] == "admin"  # email compared case-insensitively
    assert client.get("/documents", headers=bearer(boss)).status_code == 200
    alice = sign_in(client, ALICE)
    assert client.get("/documents", headers=bearer(alice)).json() == []  # her own: none yet
    assert client.get("/config", headers=bearer(alice)).status_code == 403  # known, not allowed
    assert client.get("/documents", headers=KEY).status_code == 200  # API keys still work


def test_invalid_google_tokens_are_refused(env, monkeypatch):
    client, _ = env
    r = client.post("/auth/google", json={"credential": "a-token-for-some-other-website"})
    assert r.status_code == 401 and "sign-in failed" in r.json()["detail"]

    def unverified(credential, client_id):
        raise ValueError("Google has not verified this email address")

    monkeypatch.setattr(auth, "verify_google_token", unverified)
    assert client.post("/auth/google", json={"credential": ALICE}).status_code == 401

    def offline(credential, client_id):
        raise ConnectionError("could not fetch Google's keys")

    monkeypatch.setattr(auth, "verify_google_token", offline)
    assert client.post("/auth/google", json={"credential": ALICE}).status_code == 503


def test_the_real_verifier_requires_a_verified_email(monkeypatch):
    from google.oauth2 import id_token

    monkeypatch.setattr(id_token, "verify_oauth2_token", lambda *a: {"email_verified": False})
    with pytest.raises(ValueError, match="not verified"):
        auth.verify_google_token("token", "client")


def test_sign_out_ends_the_session_at_once_and_expired_sessions_stop_working(env):
    client, svc = env
    body = sign_in(client, ALICE)
    assert client.post("/auth/logout", headers=bearer(body)).status_code == 204
    r = client.get("/me", headers=bearer(body))
    assert r.status_code == 401 and "sign in again" in r.json()["detail"]
    assert client.get("/me", headers={"Authorization": "Bearer made-up"}).status_code == 401

    old = sign_in(client, ALICE)
    store = svc.open_store()
    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat(timespec="seconds")
    store.conn.execute("UPDATE sessions SET expires_at = ?", (past,))
    store.conn.commit()
    store.close()
    assert client.get("/me", headers=bearer(old)).status_code == 401
    retention.purge(svc.settings)  # the daily cleanup also removes expired sessions
    store = svc.open_store()
    assert store.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    store.close()


def test_sessions_are_stored_only_as_hashes(env):
    client, svc = env
    body = sign_in(client, ALICE)
    store = svc.open_store()
    stored = [r[0] for r in store.conn.execute("SELECT token_hash FROM sessions")]
    store.close()
    assert stored == [token_hash(body["token"])] and body["token"] not in stored


def test_signed_in_users_have_their_own_hourly_limit(env):
    client, svc = env
    svc.settings.user_questions_per_hour = 1
    alice = bearer(sign_in(client, ALICE))
    assert ask(client, alice).status_code == 200
    assert ask(client, alice).status_code == 429  # her own bucket is used up
    assert ask(client, GUEST).status_code == 200  # guests have a separate (per-IP) bucket
    boss = bearer(sign_in(client, BOSS))
    assert ask(client, boss).status_code == 200 and ask(client, boss).status_code == 200  # admin
