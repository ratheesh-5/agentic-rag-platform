# Lumora Labs - LLMOps Incident Runbook

Version 2.0 - effective 2026-05-04. Owner: Keiko Arai, AI Safety & Governance Lead.
Applies to every production LLM feature built by the AI Engineering team.

## 1. What counts as an AI incident

An AI incident is any event in which an LLM-based feature produces harmful, wrong, insecure or
unexpectedly expensive behaviour in production. Classic outages (a server is down) are handled
by the standard platform runbook; this runbook covers failures specific to models, prompts,
retrieval and agents.

## 2. Severity levels

| Severity | Definition | Examples | Acknowledge within | Status updates |
|---|---|---|---|---|
| AI-SEV1 | Customer harm, data exposure or security breach | PII leaked in a model answer; prompt injection that exfiltrated data or triggered an unapproved action | 15 minutes | every 30 minutes |
| AI-SEV2 | Major quality or availability failure | Hallucination rate above 15%; assistant unavailable; daily spend above 200% of budget | 30 minutes | every hour |
| AI-SEV3 | Partial degradation | Hallucination rate between 8% and 15%; daily spend between 125% and 200% of budget | 4 business hours | daily |
| AI-SEV4 | Minor issue | A single reported bad answer; a flaky evaluation run | next business day | on resolution |

When in doubt, declare the higher severity; it can be downgraded later.

## 3. Error codes

All LLM services log structured error codes with the prefix LUM-. Alerts and dashboards are keyed
on these codes.

| Code | Meaning | First action |
|---|---|---|
| LUM-1001 | Retrieval returned no chunk above the similarity threshold | Normal abstention; investigate only if the rate exceeds 20% of requests in an hour |
| LUM-1002 | Vector index out of sync with the source database | Trigger an index rebuild from the source database |
| LUM-2001 | Model gateway timeout (request exceeded 30 seconds) | Check provider status; gateway retries at most 2 times |
| LUM-2002 | Rate limit returned by the hosted model provider | Enable backoff; consider the fallback tier |
| LUM-3001 | Answer failed citation validation | Answer is replaced by an abstention; review the prompt version |
| LUM-3002 | Guardrail blocked a suspected prompt injection | Review the blocked input; follow playbook 4.2 if repeated |
| LUM-4001 | Daily cost budget exceeded | Follow playbook 4.3 |
| LUM-4002 | Per-request token limit exceeded | Check for runaway agent loops or oversized context |

## 4. Playbooks

### 4.1 Hallucination spike

Trigger: more than 8% of sampled production answers are judged unfaithful over a rolling 24-hour
window (the online judge samples 2% of traffic; see the LLM Evaluation Playbook). Above 15% the
incident is AI-SEV2.

1. Check what changed in the last 72 hours: prompt versions, retrieval configuration, index
   rebuilds, model version or provider, and the document corpus.
2. If a change correlates with the spike, roll back by setting the feature flag
   `rag.config.pinned` to the last known-good configuration.
3. Run the offline evaluation against the golden set and compare with the last release.
4. Check retrieval first: low Recall@5 or many LUM-1001 abstentions usually point to an
   indexing or chunking problem rather than to the model.
5. Record the findings in the incident ticket.

### 4.2 Prompt injection

Trigger: a successful or attempted injection, reported by a user, a guardrail (LUM-3002) or a
trace review. A successful injection that leaked data or caused an action is always AI-SEV1.

1. Disable all side-effecting agent tools with the kill switch `agents.tools.write_enabled=false`.
2. Preserve traces and inputs for the affected sessions; do not delete anything.
3. Notify the AI Safety & Governance Lead within 1 hour.
4. Identify the injection vector (user input, retrieved document, tool output, web page) and
   quarantine the source document if it came from the corpus.
5. Add the attack to the red-team test set before re-enabling tools.

### 4.3 Cost overrun

Trigger: daily spend above 125% of the feature's budget is AI-SEV3; above 200% is AI-SEV2.
Budgets are derived from the "maximum cost per 1K requests" caps in the Model Selection Policy.

1. Break down spend by route and by model tier on the cost dashboard.
2. Look for runaway agent loops (LUM-4002) and abnormal traffic from a single client.
3. Enable response caching for repeated questions.
4. Switch the route to the next cheaper approved tier listed in the Model Selection Policy.
5. If spend is still rising, apply a hard request cap at the Lumora Gateway.

## 5. Escalation path

1. On-call AI engineer (weekly rotation, handover every Monday at 10:00 UTC).
2. Squad tech lead, if an AI-SEV2 is not mitigated within 30 minutes.
3. Head of AI Engineering (Nadia Ferreira) for every AI-SEV1, and for an AI-SEV2 open more than
   2 hours.
4. For AI-SEV1 involving personal data, the Data Protection Officer and Legal are informed by the
   Head of AI Engineering.

## 6. After the incident

A blameless postmortem is required for every AI-SEV1 and AI-SEV2 within 5 business days. It must
include a timeline, the root cause, the evaluation results before and after the fix, and at
least one new golden-set question or red-team case that would have caught the problem.
