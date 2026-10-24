"""Release gates: pass/fail thresholds for evaluation metrics (the LLM Evaluation Playbook's
"Release gates" table), kept in data/eval/release_gates.json.

Used by the admin UI today (a pass/fail badge per run) and by the CI regression gate later
(Phase 7), so the thresholds live in one file instead of being repeated in code.

A gate that needs real answers (abstention, faithfulness, correctness, latency) is "n/a" for a
retrieval-only run: that run never calls the LLM, so judging it on those metrics would be wrong.
Plain deterministic code - no LLM.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

GATES_FILE = Path("data/eval/release_gates.json")

Status = Literal["pass", "fail", "warn", "n/a"]


class Gate(BaseModel):
    metric: str  # key in an eval run's metrics
    label: str
    op: Literal[">=", "<="]
    threshold: float
    blocking: bool  # False = a miss is only a warning
    needs_answers: bool  # True = not measured by retrieval-only runs


class GateCheck(BaseModel):
    label: str
    metric: str
    op: str
    threshold: float
    blocking: bool
    value: float | None
    status: Status


def load_gates(path: Path = GATES_FILE) -> list[Gate]:
    if not path.exists():
        return []
    return [Gate(**g) for g in json.loads(path.read_text("utf-8"))]


def check_gates(metrics: dict, retrieval_only: bool, gates: list[Gate]) -> list[GateCheck]:
    checks = []
    for g in gates:
        value = metrics.get(g.metric)
        if value is None or (retrieval_only and g.needs_answers):
            status: Status = "n/a"  # not measured by this run
        else:
            ok = value >= g.threshold if g.op == ">=" else value <= g.threshold
            status = "pass" if ok else ("fail" if g.blocking else "warn")
        checks.append(
            GateCheck(**g.model_dump(exclude={"needs_answers"}), value=value, status=status)
        )
    return checks


def overall(checks: list[GateCheck]) -> Status:
    """fail if any blocking gate failed; n/a if nothing was measured; else pass (or warn)."""
    statuses = {c.status for c in checks}
    if "fail" in statuses:
        return "fail"
    if statuses <= {"n/a"}:
        return "n/a"
    return "warn" if "warn" in statuses else "pass"
