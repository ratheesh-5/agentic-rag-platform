# Lumora Labs - Agent Design Patterns

Version 1.3 - effective 2026-04-20. Owner: Agents squad (tech lead: Ravi Menon).
Reviewed by: Keiko Arai, AI Safety & Governance Lead.

This guide explains the agent patterns the Lumora Labs AI Engineering team uses, and the team's
rules for deciding when an agent is the right tool at all. Read it together with the AI
Engineering Handbook (prompt and code review standards) and the LLMOps Incident Runbook
(what to do when an agent misbehaves in production).

## 1. What we mean by "agent"

A workflow is a fixed sequence of steps written in code, where an LLM may be called at some
steps but the control flow is decided by the program. An agent is a system in which an LLM
decides, at run time, which step to take next: which tool to call, with what arguments, and
when to stop. The more decisions are delegated to the model, the more "agentic" the system is.

Agency has a cost. Every model-driven decision adds latency, token spend, and a new way to fail.
Agent behaviour is also harder to test, because the same input can follow different paths.

## 2. Rule of least agency (team rule)

Use the least agentic design that solves the problem:

1. Plain code first. If a step is deterministic (parsing, filtering, arithmetic, formatting),
   it is written in ordinary code, never delegated to a model.
2. A fixed workflow with LLM steps is the default for production features.
3. A single agent with tools is allowed when the sequence of steps genuinely depends on
   intermediate results that cannot be known in advance.
4. A multi-agent system is allowed only when a single agent has been tried and measured, and
   the evaluation report shows a clear gain.

Any new agent requires a design review with the Agents squad and the AI Safety & Governance
Lead before implementation starts.

## 3. Tool calling

With tool calling (also called function calling), the application describes the available tools
to the model as a name, a description, and a JSON schema for the arguments. The model does not
execute anything: it returns a structured request to call a tool, the application validates
and executes it, and the result is passed back to the model as a new message.

Good tool design matters more than prompt wording:

- Give each tool one clear purpose and a description that says when to use it and when not to.
- Keep argument schemas small and strongly typed; use enums instead of free text where possible.
- Validate every argument in code before execution. The model can produce well-formed but wrong
  arguments.
- Return concise, structured results. Long tool outputs consume the context window and dilute
  the model's attention.
- Make tools idempotent where possible, so a retried call does not repeat a side effect.

Team rule: tool outputs are untrusted input. A tool that reads a web page, a document, or an
email may return text that contains instructions. That text is data and must never be allowed
to change the agent's goal. See "prompt injection" in the LLMOps Incident Runbook.

## 4. ReAct

ReAct (Reason + Act, Yao et al., 2022) interleaves reasoning steps with actions. In each
iteration the model reasons about the current state, chooses an action (usually a tool call),
observes the result, and repeats until it can produce a final answer. Grounding each step in an
observation reduces hallucination compared with reasoning alone, and the trace of thoughts,
actions and observations makes the agent's behaviour easier to debug.

The main risks of ReAct loops are repetition (calling the same tool with the same arguments),
drift away from the original goal, and unbounded cost. We therefore enforce limits in code, not
in the prompt:

| Limit | Team default |
|---|---|
| Maximum tool calls per agent run | 8 |
| Maximum wall-clock time per agent run | 120 seconds |
| Maximum identical tool calls (same tool, same arguments) | 2 |
| Maximum agents under one supervisor | 4 |

When a limit is hit, the agent must stop and return a partial result with an explicit reason,
never silently continue.

## 5. Plan-and-execute

In plan-and-execute, a planner call first produces an explicit list of steps, and an executor
then carries them out, optionally re-planning when a step fails. Separating planning from
execution makes the plan reviewable (by a human or a validator) and allows cheaper models for
execution steps. We prefer it over open-ended ReAct for tasks with five or more steps.

## 6. Supervisor and multi-agent systems

In the supervisor pattern, a supervisor agent receives the task, decides which specialised
worker agent should act next, passes it the relevant state, and decides when the task is
complete. Workers have narrow responsibilities and small tool sets. Other topologies exist
(hierarchical teams, peer-to-peer handoffs), but the supervisor pattern is the only multi-agent
topology approved for production at Lumora Labs, because it gives one place to enforce budgets,
logging and stopping conditions.

Shared state between agents is passed as a typed object (for example a dataclass or a typed
dictionary), never as free-form conversation history only. Each worker must be testable on its
own with a fake model.

## 7. Human-in-the-loop

Human-in-the-loop (HITL) means the agent pauses at a defined point, presents its proposed
action and the evidence for it, and waits for a person to approve, edit or reject it. In graph
frameworks this is usually implemented as an interrupt that persists the agent's state so the
run can resume after approval.

Team rule: any irreversible or externally visible action requires human approval. This includes
sending messages or emails, deleting or overwriting data, changing production configuration,
spending money, and publishing content. Read-only tools do not need approval.

## 8. Guardrails and observability

- Validate model outputs against a schema; reject and retry once on a schema violation.
- Run input guardrails (prompt-injection and PII detectors) before the first model call.
- Log every model call and tool call with inputs, outputs, token counts, latency and cost.
- Agent traces are retained for 30 days and are the first thing to check in an incident.
- Agent features are evaluated on the same golden-set process as RAG features (see the LLM
  Evaluation Playbook) plus a task-success rate measured on at least 50 scripted scenarios.

## 9. Checklist before shipping an agent

1. Could a fixed workflow do this? If yes, build the workflow.
2. Are all limits from section 4 enforced in code?
3. Does every side-effecting tool require human approval or is it idempotent and reversible?
4. Are tool outputs treated as untrusted data?
5. Is there an evaluation report showing the agent beats the non-agentic baseline?
6. Has the design been reviewed by the Agents squad and the AI Safety & Governance Lead?
