"""Workflow B - multi-agent retrieval optimizer (LangGraph, supervisor pattern).

    START -> supervisor --evaluate--> evaluator (tool: baseline experiment) ---> supervisor
                        --diagnose--> diagnostician (LLM: why does retrieval fail?) -> supervisor
                        --optimize--> optimizer (tool-calling agent: runs experiments) -> supervisor
                        --report----> report writer (code picks winner, LLM explains)
                                          -> human approval (interrupt) -> apply -> END
                        --finish----> END (nothing worth improving)

Who decides what (team rule: LLM only where it adds value):
- Code: the order of steps that have no real alternative, the budgets (experiments per round,
  rounds), argument validation, and WHICH experiment wins (experiments.objective).
- LLM: the supervisor's genuine choices (stop vs diagnose; another round vs report), the
  diagnosis, which experiments to try (tool calling), and the written explanation.
- Human: approves or rejects the change. Nothing is applied without approval.

Tool calling (optimizer node): the model is given the RunExperiment schema. It replies with
"tool calls" (name + arguments) instead of text; our loop validates the arguments, runs the
experiment, and sends the metrics back as a ToolMessage. The model sees the results and decides
the next experiment or stops. This is the ReAct pattern (reason -> act -> observe).

Human-in-the-loop: `interrupt()` pauses the graph and returns the report to the caller. The
graph's state is saved by a checkpointer (here in memory), so the CLI can show the report, ask
the user, and resume the same run with `Command(resume=<decision>)`.
"""

import json
import operator
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from rag_platform.agents.optimizer_graph.experiments import ExperimentRunner, objective
from rag_platform.config import RagConfig

# --- Prompts and structured outputs ------------------------------------------------------------

SUPERVISOR_PROMPT = """You supervise the optimization of a retrieval pipeline.
Choose the next action from the allowed list only, based on the current results:
- optimize: run another round of configuration experiments
- report: stop experimenting and write the recommendation
Choose optimize if the last round improved recall or MRR and questions are still failing;
choose report if the last round brought no improvement or nothing is failing any more."""

DIAGNOSE_PROMPT = """You diagnose a retrieval pipeline from evaluation results.
Recall@K = share of the evidence found in the top K chunks; MRR = how high the first relevant
chunk is ranked (1.0 = always first). Failures list questions whose evidence was ranked below 1st
or not found. Explain the likely causes (e.g. exact codes or table rows missed by vector search
-> hybrid search; context too fragmented or too diluted -> chunk size; evidence just below the
cut-off -> top_k) and suggest concrete experiments. Be brief and specific."""

OPTIMIZER_PROMPT = """You optimize a retrieval pipeline by running experiments with the
RunExperiment tool. Each call evaluates one configuration on the golden questions and returns
its metrics. Goal: maximize recall_at_k first, then mrr. Prefer smaller top_k when equal.
Settings you can change (others stay as in the base config):
- retrieval: "dense" (vector) or "hybrid" (vector + BM25 keyword search)
- top_k: 1-10 passages given to the answer generator
- chunk_size: 300-1500 characters, with chunk_overlap 0-300 (must be < chunk_size)
You have a budget of {budget} experiments in this round. Use them on the most promising ideas
(start from the diagnosis), then reply with a one-line summary and no tool call."""

REPORT_PROMPT = """You write the recommendation of a retrieval optimization for an engineer.
The winning configuration was selected by code (highest recall, then MRR). Explain in plain
language what changed, why it is better (cite the numbers), and any risks or trade-offs
(e.g. more passages = more tokens per answer). Be concise."""


class NextAction(BaseModel):
    action: str = Field(description="One of the allowed actions")
    reason: str


class Diagnosis(BaseModel):
    findings: list[str] = Field(description="What the results show, most important first")
    suggestions: list[str] = Field(description="Concrete experiments worth running")


class RunExperiment(BaseModel):
    """Evaluate one retrieval configuration on the golden questions; returns its metrics."""

    reason: str = Field(description="Why this experiment, one sentence")
    retrieval: Literal["dense", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)
    chunk_size: int | None = Field(default=None, ge=300, le=1500)
    chunk_overlap: int | None = Field(default=None, ge=0, le=300)


class ReportText(BaseModel):
    summary: str = Field(description="One or two sentences: the recommendation")
    rationale: str = Field(description="Why it is better, with the numbers")
    risks: str = Field(description="Trade-offs or risks; 'None' if none")


# --- State ---------------------------------------------------------------------------------


class OptState(TypedDict, total=False):
    baseline: dict  # Experiment (as dict) for the current config
    diagnosis: dict
    experiments: Annotated[list[dict], operator.add]  # appended by every optimize round
    rounds: int
    next: str  # supervisor's decision, read by the router
    report: dict
    decision: str  # "approved" / "rejected" / "no-change"
    reindexed: bool
    steps: Annotated[list[str], operator.add]


def _table(exps: list[dict]) -> str:
    """Compact text table of experiments for prompts."""
    lines = []
    for e in exps:
        m = e["metrics"]
        lines.append(
            f"- {e['label']}: changes={e['changes'] or 'none'} recall={m['recall_at_k']} "
            f"mrr={m['mrr']} precision={m['precision_at_k']} "
            f"safe_threshold={m['safe_threshold']} "
            f"unanswerable_blocked={m['unanswerable_below_safe']}"
        )
    return "\n".join(lines)


# --- Graph ---------------------------------------------------------------------------------


def build_optimizer_graph(
    runner: ExperimentRunner,
    llm: BaseChatModel,
    config_path: Path,
    experiments_per_round: int = 3,
    max_rounds: int = 2,
):
    base = runner.base

    def ask(system: str, human: str, schema: type[BaseModel]):
        return llm.with_structured_output(schema).invoke([("system", system), ("human", human)])

    def allowed_actions(state: OptState) -> list[str]:
        """The supervisor may only choose among actions that make sense now (code decides)."""
        if "baseline" not in state:
            return ["evaluate"]
        if "diagnosis" not in state:
            # A fact, not a judgment: if any question's evidence was missed or not ranked first,
            # there is something to improve. (A live run showed the LLM stopping at recall 0.99
            # while MRR was 0.85 - so this decision is made in code.)
            return ["diagnose"] if state["baseline"]["failures"] else ["finish"]
        rounds = state.get("rounds", 0)
        if rounds == 0:
            return ["optimize"]
        return ["report", "optimize"] if rounds < max_rounds else ["report"]

    def supervisor(state: OptState) -> dict:
        options = allowed_actions(state)
        if len(options) == 1:  # no real choice: no LLM call needed
            return {"next": options[0], "steps": [f"supervisor: {options[0]}"]}
        best = max(
            [state["baseline"], *state.get("experiments", [])],
            key=lambda e: objective(e["metrics"]),
        )
        status = (
            f"Baseline:\n{_table([state['baseline']])}\n"
            f"Baseline failures (evidence not ranked first): {len(state['baseline']['failures'])}\n"
            f"Best so far: {best['label']} (still failing: {len(best['failures'])})\n"
            f"Experiments so far:\n{_table(state.get('experiments', [])) or '(none)'}\n"
            f"Rounds done: {state.get('rounds', 0)} of {max_rounds}\nAllowed actions: {options}"
        )
        out: NextAction = ask(SUPERVISOR_PROMPT, status, NextAction)
        action = out.action if out.action in options else options[0]  # never trust blindly
        return {"next": action, "steps": [f"supervisor: {action} ({out.reason})"]}

    def evaluator(state: OptState) -> dict:
        exp = runner.run(base, "opt-baseline")
        m = exp.metrics
        return {
            "baseline": asdict(exp),
            "steps": [f"evaluate: baseline recall={m['recall_at_k']} mrr={m['mrr']}"],
        }

    def diagnostician(state: OptState) -> dict:
        b = state["baseline"]
        failures = json.dumps(b["failures"][:15], indent=1) if b["failures"] else "none"
        human = (
            f"Current config: {json.dumps(b['config'])}\n"
            f"Baseline metrics: {json.dumps(b['metrics'])}\n"
            f"Failures (evidence not ranked first): {failures}"
        )
        out: Diagnosis = ask(DIAGNOSE_PROMPT, human, Diagnosis)
        return {"diagnosis": out.model_dump(), "steps": [f"diagnose: {out.findings[:2]}"]}

    def optimizer(state: OptState) -> dict:
        round_no = state.get("rounds", 0) + 1
        done = [state["baseline"], *state.get("experiments", [])]
        messages = [
            SystemMessage(OPTIMIZER_PROMPT.format(budget=experiments_per_round)),
            HumanMessage(
                f"Base config: {json.dumps(base.model_dump())}\n"
                f"Diagnosis: {json.dumps(state['diagnosis'])}\n"
                f"Results so far:\n{_table(done)}"
            ),
        ]
        model = llm.bind_tools([RunExperiment])
        new: list[dict] = []
        steps: list[str] = []
        # +1 turn so the model can give its closing summary after the last result.
        for _ in range(experiments_per_round + 1):
            ai = model.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                break
            for call in ai.tool_calls:
                if len(new) >= experiments_per_round:
                    content = "Budget exhausted: no more experiments in this round."
                else:
                    content, exp = run_tool(call["args"], f"opt-r{round_no}-e{len(new) + 1}")
                    if exp and exp["label"] in {e["label"] for e in done + new}:
                        content = f"Already run as {exp['label']} (same configuration): {content}"
                        exp = None  # the runner returned the cached result: don't list it twice
                    if exp:
                        new.append(exp)
                        m = exp["metrics"]
                        steps.append(
                            f"experiment {exp['label']}: {exp['changes']}"
                            f" -> recall={m['recall_at_k']} mrr={m['mrr']}"
                        )
                messages.append(ToolMessage(content, tool_call_id=call["id"]))
        return {
            "experiments": new,
            "rounds": round_no,
            "steps": steps or ["optimize: no experiments"],
        }

    def run_tool(args: dict, label: str) -> tuple[str, dict | None]:
        """Validate the model's arguments, run one experiment, return (tool result, experiment)."""
        try:
            call = RunExperiment(**args)
            changes = call.model_dump(exclude={"reason"}, exclude_none=True)
            cfg = RagConfig(**{**base.model_dump(), **changes})
            if cfg.chunk_overlap >= cfg.chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")
        except (ValidationError, ValueError, TypeError) as e:
            return f"Invalid arguments, nothing run: {e}", None
        try:
            exp = asdict(runner.run(cfg, label))
        except Exception as e:  # noqa: BLE001 - e.g. API quota: tell the agent, keep the run alive
            return f"Experiment failed ({type(e).__name__}): {str(e)[:200]}", None
        result = {"label": exp["label"], "changes": exp["changes"], "metrics": exp["metrics"]}
        return json.dumps(result), exp

    def report_writer(state: OptState) -> dict:
        candidates = [state["baseline"], *state.get("experiments", [])]
        best = max(candidates, key=lambda e: objective(e["metrics"]))  # code picks the winner
        recommended = dict(best["config"])
        if best["metrics"]["safe_threshold"] is not None:
            # The similarity gate that never refused an answerable question in this experiment.
            recommended["similarity_threshold"] = best["metrics"]["safe_threshold"]
        changes = {k: v for k, v in recommended.items() if base.model_dump()[k] != v}
        human = (
            f"Base config: {json.dumps(base.model_dump())}\nAll results:\n{_table(candidates)}\n"
            f"Winner: {best['label']}\nRecommended changes: {json.dumps(changes)}"
        )
        text: ReportText = ask(REPORT_PROMPT, human, ReportText)
        report = {
            "winner": best["label"],
            "changes": changes,
            "recommended_config": recommended,
            "base_config": base.model_dump(),  # apply refuses if the config changed meanwhile
            "baseline_metrics": state["baseline"]["metrics"],
            "winner_metrics": best["metrics"],
            "experiments": [
                {
                    "label": e["label"],
                    "changes": e["changes"],
                    "metrics": e["metrics"],
                    "run_id": e["run_id"],
                }
                for e in candidates
            ],
            **text.model_dump(),
        }
        return {"report": report, "steps": [f"report: winner {best['label']}, changes {changes}"]}

    def approval(state: OptState) -> dict:
        if not state["report"]["changes"]:
            return {"decision": "no-change", "steps": ["approval: nothing to change"]}
        approved = interrupt({"report": state["report"]})  # pauses here until the CLI resumes
        decision = "approved" if approved else "rejected"
        return {"decision": decision, "steps": [f"approval: {decision}"]}

    def apply(state: OptState) -> dict:
        cfg = RagConfig(**state["report"]["recommended_config"])
        reindexed = runner.apply(cfg, config_path)
        return {
            "reindexed": reindexed,
            "steps": [f"apply: wrote {config_path}, reindexed={reindexed}"],
        }

    # Routing: plain functions reading the state.
    def route_supervisor(state: OptState) -> str:
        return {"evaluate": "evaluator", "diagnose": "diagnostician", "optimize": "optimizer",
                "report": "report_writer", "finish": END}[state["next"]]  # fmt: skip

    def route_approval(state: OptState) -> str:
        return "apply" if state["decision"] == "approved" else END

    g = StateGraph(OptState)
    for name, fn in [
        ("supervisor", supervisor),
        ("evaluator", evaluator),
        ("diagnostician", diagnostician),
        ("optimizer", optimizer),
        ("report_writer", report_writer),
        ("approval", approval),
        ("apply", apply),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_supervisor,
        ["evaluator", "diagnostician", "optimizer", "report_writer", END],
    )
    for worker in ("evaluator", "diagnostician", "optimizer"):
        g.add_edge(worker, "supervisor")  # workers always report back to the supervisor
    g.add_edge("report_writer", "approval")
    g.add_conditional_edges("approval", route_approval, ["apply", END])
    g.add_edge("apply", END)
    # The checkpointer stores the state between the interrupt and the resume.
    return g.compile(checkpointer=InMemorySaver())
