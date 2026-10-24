"""Tests for the admin quality endpoints: release gates, quality checks (default and uploaded test
sets), hiding runs and approving optimizer recommendations. Fake embedder, no API calls."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLlm
from rag_platform.api.app import app
from rag_platform.api.deps import Services, get_services
from rag_platform.config import RagConfig, Settings, load_rag_config, save_rag_config
from rag_platform.evaluation.gates import Gate, check_gates, overall
from rag_platform.store.db import Store

KEY = {"X-API-Key": "admin"}
GUEST = {"X-Guest-Id": "3f1c2a9e-7b4d-4c1a-9e2f-0a1b2c3d4e5f"}
DOC = b"Employees get twenty days of annual leave per year. Remote work needs manager approval."


@pytest.fixture
def admin(tmp_path: Path, fake_embedder):
    settings = Settings(
        _env_file=None,
        online_eval_sample_rate=0,  # no background judge calls in tests
        gemini_api_key="x",
        api_keys="admin",
        index_dir=tmp_path / "idx",
        upload_dir=tmp_path / "uploads",
    )
    save_rag_config(RagConfig(similarity_threshold=0.0), tmp_path / "rag_config.json")
    svc = Services(settings, fake_embedder, FakeLlm(), config_path=tmp_path / "rag_config.json")
    app.dependency_overrides[get_services] = lambda: svc
    client = TestClient(app)
    assert client.post("/documents", headers=KEY, files={"file": ("hr.md", DOC)}).status_code == 201
    doc = client.get("/documents", headers=KEY).json()[0]["path"]
    svc.golden_path = tmp_path / "golden.jsonl"
    svc.golden_path.write_text(_golden(doc), encoding="utf-8")
    yield client, svc, doc
    app.dependency_overrides.clear()


def _golden(doc: str, quote: str = "twenty days of annual leave") -> str:
    rows = [
        {"id": "a1", "question": "How many leave days?", "answerable": True,
         "reference_answer": "Twenty.", "evidence": [{"doc": doc, "quote": quote}]},
        {"id": "u1", "question": "What is the CEO's salary?", "answerable": False,
         "reference_answer": "", "evidence": []},
    ]  # fmt: skip
    return "\n".join(json.dumps(r) for r in rows) + "\n"


# --- release gates (plain code) ----------------------------------------------------------------


def test_gates_pass_fail_and_not_applicable_for_retrieval_only():
    gates = [
        Gate(metric="recall_at_k", label="R", op=">=", threshold=0.85, blocking=True,
             needs_answers=False),
        Gate(metric="faithfulness", label="F", op=">=", threshold=0.9, blocking=True,
             needs_answers=True),
        Gate(metric="latency_p95_s", label="L", op="<=", threshold=4, blocking=False,
             needs_answers=True),
    ]  # fmt: skip
    checks = check_gates({"recall_at_k": 0.9, "faithfulness": 0.5}, True, gates)
    assert [c.status for c in checks] == ["pass", "n/a", "n/a"]  # answers not measured
    assert overall(checks) == "pass"
    checks = check_gates(
        {"recall_at_k": 0.9, "faithfulness": 0.5, "latency_p95_s": 9}, False, gates
    )
    assert [c.status for c in checks] == ["pass", "fail", "warn"]
    assert overall(checks) == "fail"
    assert overall(check_gates({}, False, gates)) == "n/a"


def test_shipped_gates_file_is_valid():
    from rag_platform.evaluation.gates import load_gates

    labels = [g.label for g in load_gates()]
    assert "Recall@5" in labels and "MRR" in labels


# --- quality checks ------------------------------------------------------------------------------


def test_quality_check_with_the_default_test_set(admin):
    client, _, _ = admin
    r = client.post("/eval/runs", headers=KEY)
    assert r.status_code == 201, r.text
    run = r.json()
    assert run["kind"] == "check" and run["current"] and run["test_set"] == "golden.jsonl"
    assert run["metrics"]["recall_at_k"] == 1.0 and run["gates"] in ("pass", "fail", "warn")
    detail = client.get(f"/eval/runs/{run['id']}", headers=KEY).json()
    by_label = {g["label"]: g["status"] for g in detail["gates"]}
    assert by_label["Recall@5"] == "pass" and by_label["Faithfulness"] == "n/a"


def test_quality_check_with_an_uploaded_test_set(admin):
    client, svc, doc = admin
    own = _golden(doc, quote="Remote work needs manager approval")
    r = client.post(
        "/eval/runs", headers=KEY, files={"golden": ("hr_questions.jsonl", own.encode())},
        data={"label": "hr-check"},
    )  # fmt: skip
    assert r.status_code == 201, r.text
    assert r.json()["test_set"] == "hr_questions.jsonl" and r.json()["label"] == "hr-check"
    assert (svc.settings.upload_dir / "test_sets" / "hr_questions.jsonl").exists()


@pytest.mark.parametrize(
    "name,content,code",
    [
        ("q.txt", b"{}", 415),  # wrong extension
        ("q.jsonl", b"not json\n", 422),
        ("q.jsonl", b"", 422),  # no questions
    ],
    ids=["extension", "not-json", "empty"],
)
def test_invalid_test_sets_are_rejected_before_running(admin, name, content, code):
    client, svc, _ = admin
    r = client.post("/eval/runs", headers=KEY, files={"golden": (name, content)})
    assert r.status_code == code, r.text
    assert client.get("/eval/runs", headers=KEY).json() == []  # nothing ran


def test_a_quote_not_in_the_documents_is_explained(admin):
    client, _, doc = admin
    bad = _golden(doc, quote="thirty days of annual leave")
    r = client.post("/eval/runs", headers=KEY, files={"golden": ("q.jsonl", bad.encode())})
    assert r.status_code == 422 and "Invalid test set: q.jsonl:1" in r.json()["detail"]


def test_hide_runs_and_guests_cannot_use_admin_quality_endpoints(admin):
    client, _, _ = admin
    run_id = client.post("/eval/runs", headers=KEY).json()["id"]
    assert (
        client.put(f"/eval/runs/{run_id}/hidden", headers=KEY, json={"hidden": True}).status_code
        == 204
    )
    assert client.get("/eval/runs", headers=KEY).json()[0]["hidden"] is True
    assert (
        client.put("/eval/runs/999/hidden", headers=KEY, json={"hidden": True}).status_code == 404
    )
    for method, path in [("POST", "/eval/runs"), ("GET", "/optimizer/recommendation")]:
        assert client.request(method, path, headers=GUEST).status_code == 401


def test_old_databases_get_the_hidden_column(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE eval_runs (id INTEGER PRIMARY KEY, created_at TEXT, label TEXT,"
        " config TEXT, metrics TEXT, results TEXT)"
    )
    conn.execute("INSERT INTO eval_runs VALUES (1, 't', 'old', '{}', '{}', '[]')")
    conn.commit()
    conn.close()
    store = Store(db)
    assert store.list_eval_runs()[0]["hidden"] is False
    store.close()


# --- optimizer recommendation: approve / reject in the UI ---------------------------------------


def _recommend(svc: Services, changes: dict) -> int:
    base = load_rag_config(svc.config_path).model_dump()
    report = {
        "changes": changes,
        "recommended_config": {**base, **changes},
        "base_config": base,
        "summary": "Use hybrid search.",
    }
    store = svc.open_store()
    try:
        return store.save_recommendation(report)
    finally:
        store.close()


def test_apply_recommendation_writes_the_config(admin):
    client, svc, _ = admin
    assert client.get("/optimizer/recommendation", headers=KEY).json() is None
    rec_id = _recommend(svc, {"retrieval": "hybrid", "similarity_threshold": 0.6})
    rec = client.get("/optimizer/recommendation", headers=KEY).json()
    assert rec["status"] == "pending" and not rec["stale"]
    assert client.post(f"/optimizer/recommendation/{rec_id}/apply", headers=KEY).status_code == 204
    cfg = load_rag_config(svc.config_path)
    assert cfg.retrieval == "hybrid" and cfg.similarity_threshold == 0.6
    assert client.get("/optimizer/recommendation", headers=KEY).json()["status"] == "applied"
    # Decided once: a second click is a conflict, not a second apply.
    assert client.post(f"/optimizer/recommendation/{rec_id}/apply", headers=KEY).status_code == 409


def test_reject_and_stale_recommendations(admin):
    client, svc, _ = admin
    first = _recommend(svc, {"top_k": 3})
    second = _recommend(svc, {"retrieval": "hybrid"})  # supersedes the first
    assert client.post(f"/optimizer/recommendation/{first}/reject", headers=KEY).status_code == 404
    assert client.post(f"/optimizer/recommendation/{second}/reject", headers=KEY).status_code == 204
    assert load_rag_config(svc.config_path).retrieval == "dense"  # unchanged

    third = _recommend(svc, {"retrieval": "hybrid"})
    save_rag_config(RagConfig(similarity_threshold=0.1), svc.config_path)  # config changed since
    assert client.get("/optimizer/recommendation", headers=KEY).json()["stale"] is True
    assert client.post(f"/optimizer/recommendation/{third}/apply", headers=KEY).status_code == 409


def test_check_explains_a_deleted_document_instead_of_a_provider_error(admin):
    client, _, _ = admin
    doc_id = client.get("/documents", headers=KEY).json()[0]["id"]
    assert client.delete(f"/documents/{doc_id}", headers=KEY).status_code == 204
    r = client.post("/eval/runs", headers=KEY)
    assert r.status_code == 409
    assert "no longer indexed: hr.md" in r.json()["detail"]


# --- optimizer started from the UI (background task) -------------------------------------------


@pytest.fixture
def no_job():
    from rag_platform.api import evaluation
    from rag_platform.api.schemas import OptimizerRun

    evaluation._optimizer = OptimizerRun()  # module state: start every test idle
    yield evaluation


def test_optimizer_from_the_ui_with_nothing_to_improve(admin, no_job):
    client, _, _ = admin
    assert client.get("/optimizer/run", headers=KEY).json()["state"] == "idle"
    # TestClient runs the background task before returning, so the run is finished here.
    r = client.post("/optimizer/runs", headers=KEY)
    assert r.status_code == 202 and r.json()["state"] == "running"
    job = client.get("/optimizer/run", headers=KEY).json()
    assert job["state"] == "done" and job["outcome"] == "nothing-to-improve", job
    assert job["test_set"] == "golden.jsonl" and job["finished_at"]
    assert client.get("/optimizer/recommendation", headers=KEY).json() is None


def test_optimizer_from_the_ui_saves_a_recommendation_for_approval(admin, no_job):
    from rag_platform.agents.optimizer_graph.graph import Diagnosis, NextAction, ReportText

    client, svc, doc = admin
    # A second document that matches the question's words better: the evidence drops to rank 2,
    # so the optimizer has something to improve.
    other = b"How many leave days? Leave days are counted per team."
    assert (
        client.post("/documents", headers=KEY, files={"file": ("faq.md", other)}).status_code == 201
    )
    svc.llm = FakeLlm(
        {
            Diagnosis: Diagnosis(findings=["evidence ranked 2nd"], suggestions=["try hybrid"]),
            NextAction: NextAction(action="report", reason="done"),
            ReportText: ReportText(summary="Calibrate", rationale="safe threshold", risks="None"),
        }
    )
    own = _golden(doc).encode()
    r = client.post("/optimizer/runs", headers=KEY, files={"golden": ("hr_q.jsonl", own)})
    assert r.status_code == 202, r.text
    job = client.get("/optimizer/run", headers=KEY).json()
    assert job["state"] == "done" and job["outcome"] == "recommendation", job
    assert any(s.startswith("evaluate: baseline") for s in job["steps"])

    rec = client.get("/optimizer/recommendation", headers=KEY).json()
    assert rec["id"] == job["recommendation_id"] and rec["status"] == "pending"
    assert rec["report"]["test_set"] == "hr_q.jsonl" and rec["report"]["changes"]
    # Nothing changed yet: the human decides in Settings.
    assert load_rag_config(svc.config_path).similarity_threshold == 0.0
    assert (
        client.post(f"/optimizer/recommendation/{rec['id']}/apply", headers=KEY).status_code == 204
    )
    assert load_rag_config(svc.config_path).similarity_threshold > 0.0


def test_only_one_quality_job_at_a_time(admin, no_job):
    client, _, _ = admin
    assert no_job._check_lock.acquire(blocking=False)  # e.g. a quality check is running
    try:
        assert client.post("/optimizer/runs", headers=KEY).status_code == 409
        assert client.post("/eval/runs", headers=KEY).status_code == 409
    finally:
        no_job._check_lock.release()


def test_a_failed_optimizer_run_is_reported_and_frees_the_lock(admin, no_job):
    client, svc, _ = admin
    other = b"How many leave days? Leave days are counted per team."
    client.post("/documents", headers=KEY, files={"file": ("faq.md", other)})
    svc.llm = FakeLlm({})  # no scripted diagnosis: the first chat call fails
    client.post("/optimizer/runs", headers=KEY)
    job = client.get("/optimizer/run", headers=KEY).json()
    assert job["state"] == "failed" and job["error"].startswith("Model provider error")
    assert client.post("/eval/runs", headers=KEY).status_code == 201  # lock released
    assert client.post("/optimizer/runs", headers=GUEST).status_code == 401  # admins only


# --- answer mode switched directly in Settings -------------------------------------------------


def test_admin_switches_the_answer_mode(admin):
    client, svc, _ = admin
    r = client.put("/config/mode", headers=KEY, json={"mode": "auto"})
    assert r.status_code == 200 and r.json()["mode"] == "auto"
    assert load_rag_config(svc.config_path).mode == "auto"  # used from the next question
    assert client.put("/config/mode", headers=KEY, json={"mode": "turbo"}).status_code == 422
    assert client.put("/config/mode", headers=GUEST, json={"mode": "fast"}).status_code == 401


def test_a_recommendation_ignores_the_answer_mode(admin):
    # The optimizer tunes search settings only: switching the mode must not make a waiting
    # recommendation outdated, and applying it must not switch the mode back.
    client, svc, _ = admin
    rec_id = _recommend(svc, {"retrieval": "hybrid"})
    client.put("/config/mode", headers=KEY, json={"mode": "auto"})
    save_rag_config(load_rag_config(svc.config_path, chat_fallback=False), svc.config_path)
    assert client.get("/optimizer/recommendation", headers=KEY).json()["stale"] is False
    assert client.post(f"/optimizer/recommendation/{rec_id}/apply", headers=KEY).status_code == 204
    cfg = load_rag_config(svc.config_path)
    assert cfg.retrieval == "hybrid" and cfg.mode == "auto"
    assert cfg.chat_fallback is False  # the answer model is an admin choice too
