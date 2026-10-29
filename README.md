# Agentic RAG Platform

[![CI](https://github.com/ratheesh-5/agentic-rag-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/ratheesh-5/agentic-rag-platform/actions/workflows/ci.yml)

Domain-independent Agentic RAG platform. It answers questions from your documents with
**cited answers**, says **"I don't know"** when the documents don't contain the answer, and
**measures and optimizes its own quality**.

**Stack:** Python 3.12 · uv · LangChain · LangGraph · Gemini · FAISS + SQLite · BM25 · RAGAS ·
FastAPI + Uvicorn · Angular 22 + Angular Material · LangSmith · Docker Compose · GitHub Actions

**New here?** [docs/setup-guide.md](docs/setup-guide.md) ([PDF](docs/setup-guide.pdf)) walks you
from an empty machine to the running app: tools, Gemini key, `.env`, Docker or local development, first steps, troubleshooting.
**Design:** [docs/design.md](docs/design.md) covers the architecture, flow diagrams, data model,
metrics, the agentic workflow and the optimizer.

## Features

- **Collections:** documents are grouped by subject ("AI Engineering Handbook", "HR policies");
  a chat searches only the collection it was started in - each collection has its own search
  index, so unrelated documents never mix. **Private collections:** every signed-in user can
  upload their own documents and ask about them; nobody else - not even admins - sees them.
- **Incremental indexing** of PDF, DOCX, Markdown and text. Only new or changed content is
  embedded, using an embedding cache.
- **Dense or hybrid retrieval**: vector search, or vector + BM25 keyword search fused with
  Reciprocal Rank Fusion.
- **Cited answers** with validated citations, plus explicit abstention at three checkpoints.
- **Agentic mode** (LangGraph): query planning, relevance grading, query rewriting and answer
  verification, with loop limits enforced in code. **Auto mode** answers fast first and hands a
  question to the agent only when the fast answer struggled (cost-aware escalation).
- **Analyse mode** for questions across a whole collection ("compare these policies", "what is
  the overall picture?"): map-reduce - one cited note per document, then one combined answer
  that says how many documents it read.
- **Two AI providers with automatic fallback:** Gemini writes answers, and Groq takes over when
  Gemini is out of quota, overloaded or too slow (or the other way round - chosen in Admin ›
  Settings). The second model was checked on the golden set before it was allowed.
- **Model choice in Settings:** the answer, backup and judge models are picked from the
  providers' live model lists - no `.env` edit, no restart. A **backup model** of the same
  provider is tried before the other provider (limits are per model); a new answer model must
  pass a built-in **quality check** (6 golden questions: cited answers + a correct "I don't
  know") before it can be used, and every score records which judge produced it.
- **Evaluation** on a 50-question golden set: span-based retrieval metrics (Recall@K, MRR, …),
  abstention accuracy, and RAGAS (faithfulness, answer relevancy, context precision/recall,
  answer correctness). Run comparison and resumable runs.
- **Live quality from real chats** (no test data): a sample of answers (1 in 5) is scored in
  the background for faithfulness and relevance; unanswered questions and 👎 answers become
  suggestions (which documents are missing or weak). Admins can test a question on the same
  page and see the answer, its scores and the passages found; their own chats don't count.
- **Multi-agent optimizer**: supervisor, diagnostician, tool-calling retrieval optimizer and
  report writer. It proposes better settings and applies them only after human approval.
- **Chatbot with conversation memory**: follow-up questions are rewritten using earlier turns.
  Conversations and 👍/👎 feedback are stored per user.
- **No registration form:** visitors chat right away as guests (hourly limit); **Sign in with
  Google** for your own chats on every device and private collections; admins are named by
  e-mail (or use an admin key) to manage documents and see evaluation results.
- **Guidance for new users:** the welcome screen lists what the documents cover, with example
  questions; out-of-scope questions get topic suggestions instead of a dead end. The content
  comes from [`assistant.json`](assistant.json), so another knowledge base needs no UI change.
- **Observability:** LangSmith tracing (every request as a tree of steps with prompts, tokens
  and timings) plus a local usage log of every model call, shown on an Admin › Usage page
  (calls, tokens, latency, errors, optional cost) with links to the LangSmith traces; tiles, days, purposes and models filter one list of calls.
- **CI with a quality gate:** every push runs lint, 244 + 134 tests, the UI build, a dependency
  vulnerability audit, Docker image builds and a retrieval quality gate (Recall@5 ≥ 0.85, MRR ≥ 0.70, no drop > 0.02 vs the
  baseline) - without any API key, from a committed embedding snapshot.
- **Docker Compose:** the API and the UI (nginx) start with one command; containers run as
  non-root users, nginx sends security headers (CSP etc.), the API port is local-only.
- **Security basics:** prompt-injection rule for document text plus an upload scan that flags
  instruction-like text, data retention (90 days, "delete all my chats"), Dependabot updates,
  CI audit.
- **REST API** with API-key auth and validated uploads.
- **Web UI** (Angular): a production-style chat (sidebar with conversation history, cited
  answers with a sources popup, feedback, copy) plus admin pages: documents; evaluation with
  one-click quality checks (also on your own uploaded test set) judged against release gates;
  settings where the optimizer's recommendation is applied or rejected.

## Screenshots

The current UI with a sample workspace: the demo handbook under realistic document and
collection names. Answers, passages, similarity scores and evaluation results are the real ones
for these documents; call counts on the Usage page belong to the sample.

**How it fits together** - architecture, then what happens at upload time and at question time
(more diagrams in [docs/design.md](docs/design.md)):

<p>
  <img src="docs/img/screenshots/diagram-architecture.jpg" width="32%" alt="Architecture">
  <img src="docs/img/screenshots/diagram-upload-flow.jpg" width="32%" alt="Upload-time data flow">
  <img src="docs/img/screenshots/diagram-question-flow.jpg" width="32%" alt="Question-time data flow">
</p>

**Chat** - one cited answer with its sources; "I don't know" when the documents don't cover a
question (refused before any AI call); Analyse mode across a whole collection:

![Cited answer with sources](docs/img/screenshots/chat-answer-sources.jpg)
![Refusal with topic suggestions](docs/img/screenshots/chat-refusal.jpg)
![Analyse mode](docs/img/screenshots/chat-analyse-mode.jpg)

<img src="docs/img/screenshots/chat-phone.jpg" width="300" alt="Chat on a phone">

**Documents** - collections, upload with incremental indexing, and the prompt-injection scan
("Check content") on a private collection:

![Documents page after an upload](docs/img/screenshots/documents-upload.jpg)
![Upload flagged by the prompt-injection scan](docs/img/screenshots/documents-injection-scan.jpg)

**Evaluation** - live quality of real chats (scores, suggestions, playground, per document) and
test-set checks against the release gates:

![Live quality](docs/img/screenshots/evaluation-live-quality.jpg)
![Test-set checks](docs/img/screenshots/evaluation-test-set.jpg)

**Settings** - the optimizer's applied recommendation, answer mode, models with quality checks,
measured settings:

![Settings](docs/img/screenshots/settings.jpg)

**Usage** - every model call: tokens, latency, errors, by purpose and by model, LangSmith links:

![Usage](docs/img/screenshots/usage.jpg)

## Quick start with Docker

Requires Docker Desktop and a Gemini API key.

```bash
cp .env.example .env          # set GEMINI_API_KEY and API_KEYS (admin keys)
uv run rag ingest             # once: index the sample corpus (or upload documents in the UI)
docker compose up --build     # UI http://localhost:8080 · API http://localhost:8000/docs
```

If port 8080 is taken: `UI_PORT=8081 docker compose up` (PowerShell:
`$env:UI_PORT=8081; docker compose up --build`). The UI calls the API through nginx on its own
address (`/api/...`), so no CORS setup is needed and a local `rag serve` doesn't interfere. The
index, uploads and `rag_config.json` are mounted from the project folder, so they survive
container restarts.

## Quick start without Docker

Requires [uv](https://docs.astral.sh/uv/) and a Gemini API key.

```bash
uv sync
cp .env.example .env          # then set GEMINI_API_KEY (and API_KEYS for the REST API)
uv run rag check              # verify the key: one chat call + one embedding call
uv run rag ingest             # index data/sample_corpus (6 demo documents)
uv run rag ask "What does error code LUM-2001 mean?"
uv run rag ask "Who must be notified within 1 hour of a prompt injection?" --mode agentic --trace
```

### Evaluate and optimize

```bash
uv run rag eval run --retrieval-only     # retrieval metrics, no chat calls
uv run rag eval run --limit 5            # answers + RAGAS on 5 questions
uv run rag eval list
uv run rag eval compare 1 2
uv run rag optimize                      # agents propose settings; you approve or reject
uv run rag optimize --decision later     # ...or decide later in the UI (Admin › Settings)
```

All settings live in [`rag_config.json`](rag_config.json). Command-line options override them
for a single run. What the chat tells users it covers (topics, example questions) lives in
[`assistant.json`](assistant.json).

### REST API

```bash
uv run rag serve                         # http://127.0.0.1:8000/docs
```

Admins send a key (one of `API_KEYS` in `.env`) in the `X-API-Key` header, or use the
**Authorize** button in `/docs`. Guests send only `X-Guest-Id` (any random UUID) and can use the
chat endpoints, limited to `GUEST_QUESTIONS_PER_HOUR` (default 20) questions per IP.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness (open) |
| `GET /ready` | Readiness: config, database, search index; 503 when one fails (open; Docker health check) |
| `GET /me` | The caller's role (`admin`, `user` = signed in with Google, `guest`) and, if signed in, name/email/picture |
| `GET /auth/config`, `POST /auth/google`, `POST /auth/logout` | Sign in with Google (optional; `GOOGLE_CLIENT_ID`, `ADMIN_EMAILS` in `.env`) |
| `GET /assistant[?collection_id=]` | What a collection covers: topics and example questions (open) |
| `GET /collections`, `POST /collections`, `PUT/DELETE /collections/{id}` | List (shared + your private ones); create, rename, delete (admins: shared; users: their private ones) |
| `DELETE /me` | Delete my account (private collections, chats, sign-in) |
| `POST /chat` | Message in a conversation (memory); starts one if no `conversation_id` (guest or admin) |
| `GET /conversations`, `GET/DELETE /conversations/{id}` | Conversation history |
| `DELETE /conversations` | Delete all of the caller's conversations ("delete my data") |
| `PUT /messages/{id}/feedback` | 👍 (1) / 👎 (-1) / clear (null) on an answer |
| `POST /ask` | One independent question (`mode`, `retrieval`, `top_k` optional); admin, like everything below |
| `GET/POST /documents`, `DELETE /documents/{id}` | List, upload + index, remove |
| `GET /config`, `PUT /config/mode` | Active settings; switch the answer mode (fast / agentic / auto) |
| `GET /eval/runs`, `GET /eval/runs/{id}` | Evaluation runs with release-gate status |
| `POST /eval/runs` | Quality check now (retrieval only, no chat calls); optional `.jsonl` test set |
| `PUT /eval/runs/{id}/hidden` | Hide / show a run |
| `GET /quality/usage` | Live quality from real chats: every answer with its scores, suggestions |
| `POST /quality/test` | Answer and score one test question now (not saved) |
| `GET /optimizer/recommendation`, `POST …/{id}/apply`, `POST …/{id}/reject` | Approve the optimizer's recommendation |
| `POST /optimizer/runs`, `GET /optimizer/run` | Run the optimizer in the background (optional own test set), follow its progress |
| `GET /usage/summary?days=14` | Usage log: calls, tokens, latency, errors, cost per purpose/model/day |
| `GET /usage/trace/{run_id}` | Link to that call's trace in LangSmith |

### Web UI

```bash
cd ui
npm install
npm start                                # http://localhost:4200 (API must be running)
```

Start chatting right away as a guest. With `GOOGLE_CLIENT_ID` set, click **Sign in with Google**:
your chats follow you to every device, and emails in `ADMIN_EMAILS` get the Admin pages. Without
it, click **Admin sign in** and enter one of your `API_KEYS` to manage documents and see
evaluation runs. See [ui/README.md](ui/README.md).

**Google sign-in setup (once, by the app owner - users just click the button):** in
[Google Cloud Console](https://console.cloud.google.com/auth/overview) create the Google Auth
Platform app (Audience: External; add test users while in testing), then **Clients → Create
client → Web application** with the JavaScript origins `http://localhost`,
`http://localhost:4200` and `http://localhost:8081` (or your domain). Put the **client id**
(not the secret) in `.env` as `GOOGLE_CLIENT_ID`, plus `ADMIN_EMAILS`. Open the app as
`localhost`, not `127.0.0.1` (Google checks the origin exactly).

## Observability

- **LangSmith (optional):** create a free account at https://smith.langchain.com, create an API
  key (Settings → API Keys) and add to `.env`:
  `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=…`, `LANGSMITH_PROJECT=rag-platform`.
  The API is traced; CLI batch runs only with `uv run rag --trace-langsmith …`, so evaluation
  runs don't use up the free plan's monthly trace allowance.
- **Usage log (always on):** every model call is recorded locally (purpose, tokens, latency,
  errors) and shown on Admin › Usage. For cost estimates, enter prices per 1M tokens in
  [`model_prices.json`](model_prices.json).
- **Application log:** the API writes one JSON line per request, model call and answer, tied
  together by a request id (also returned as the `X-Request-ID` header). Question and answer text
  is never logged. Read it with `docker compose logs api` (or the `rag serve` console).
- **Data retention:** conversations without activity for 90 days and usage-log rows older than
  90 days are deleted automatically (daily; `CHAT_RETENTION_DAYS` / `USAGE_RETENTION_DAYS`,
  0 = keep). Users can delete all their chats from the sidebar.

## Development

```bash
uv run pytest                            # 244 backend tests, fake models, no API calls
uv run ruff check . && uv run ruff format .
uv run rag gate                          # CI quality gate locally (no API key needed)
cd ui && npm test -- --watch=false       # 134 UI tests (Vitest), HTTP mocked
```

**CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull
request: ruff + pytest, the quality gate, UI tests + production build, Docker image builds.
After a change that alters the indexed texts (e.g. another chunk size), refresh the snapshot
with `uv run rag gate --export` (and `--update-baseline` when the new numbers are accepted).

The Gemini free tier allows 15 chat requests per minute and 500 per day per model, and
100 embedded texts per minute. The code throttles itself to stay within these limits, and
evaluation runs can be resumed with `--resume`.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0–1 | Sample corpus · RAG core | ✅ |
| 2 | Evaluation (golden set, metrics, RAGAS) | ✅ |
| 3 | Agentic RAG (LangGraph) | ✅ |
| 4 | Multi-agent optimizer · hybrid search | ✅ |
| 5 | REST API (FastAPI) | ✅ |
| 6 | Web UI (Angular) · chatbot with conversation memory | ✅ |
| 7 | Observability (LangSmith + usage log), Docker Compose, CI with quality gate | ✅ |
