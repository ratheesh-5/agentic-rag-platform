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
- **REST API** (FastAPI): guests, Google sign-in sessions and admin keys; uploads validated by
  content.
- **Web UI** (Angular): a production-style chat (sidebar with conversation history, cited
  answers with a sources popup, feedback, copy) plus admin pages: documents; evaluation with
  one-click quality checks (also on your own uploaded test set) judged against release gates;
  settings where the optimizer's recommendation is applied or rejected.

## Screenshots

The current UI with a sample workspace: the demo handbook under realistic document and
collection names. Answers, passages, similarity scores and evaluation results are the real ones
for these documents; call counts on the Usage page belong to the sample.

### How it fits together

<img src="docs/img/screenshots/diagram-architecture.jpg" width="720" alt="Architecture">

*Architecture - the Angular UI reaches the FastAPI backend through nginx; the backend holds the
REST API, the RAG pipeline, the LangGraph agents, evaluation and observability; SQLite + FAISS
store the data, Gemini (with Groq as fallback) writes answers and embeddings, LangSmith records
traces.*

<img src="docs/img/screenshots/diagram-upload-flow.jpg" width="720" alt="Upload-time data flow">

*Upload time - every file is validated (type, size, real file signature, prompt-injection
scan), unchanged files are skipped, text is split into passages and only new passages are
embedded; SQLite keeps everything, FAISS is the collection's search index.*

<img src="docs/img/screenshots/diagram-question-flow.jpg" width="720" alt="Question-time data flow">

*Question time - follow-ups are rewritten into a standalone question, hybrid search finds the
top 5 passages, and below the similarity threshold the app says "I don't know" without calling
the LLM; otherwise the LLM writes one answer whose citations are checked in code.*

More diagrams: [docs/design.md](docs/design.md).

### Chat

![Cited answer with sources](docs/img/screenshots/chat-answer-sources.jpg)

*A cited answer built from two documents: the numbers in the text are clickable citations, the
"2 sources" popup lists the documents and previews each passage.*

![Refusal with topic suggestions](docs/img/screenshots/chat-refusal.jpg)

*A question the documents don't cover: an honest "I couldn't find this in the documents" in
0.6 s - refused by the similarity threshold before any LLM call - with topics to try instead.*

![Analyse mode](docs/img/screenshots/chat-analyse-mode.jpg)

*Analyse mode for a question about a whole collection: map-reduce over the documents, one
combined cited answer, and a note on how many documents were read ("6 documents read, 4 without
anything on this").*

<img src="docs/img/screenshots/chat-phone.jpg" width="300" alt="Chat on a phone">

*The same chat on a phone - the layout adapts: the sidebar becomes a ☰ menu.*

### Documents (admin)

![Documents page after an upload](docs/img/screenshots/documents-upload.jpg)

*Collections at the top, drag-and-drop upload, and incremental indexing: re-uploading a changed
file reports "Updated (content changed) · 3 passages · 3 new embeddings" - only the changed text
is embedded again.*

![Upload flagged by the prompt-injection scan](docs/img/screenshots/documents-injection-scan.jpg)

*The prompt-injection scan: a document containing "ignore all previous instructions" is indexed
but flagged with a "Check content" badge for the admin (warn, don't block).*

### Evaluation (admin)

![Live quality](docs/img/screenshots/evaluation-live-quality.jpg)

*Live quality from real chats: 1 in 5 answers is scored in the background for faithfulness and
relevance; unanswered and disliked questions become suggestions; "Test a question" answers and
scores one question on the spot.*

![Test-set checks](docs/img/screenshots/evaluation-test-set.jpg)

*Test-set checks: a quality check runs the 50 golden questions in about a second (search only,
no AI calls) and compares Recall@5 (1.000) and MRR (0.881) with the release gates; every
optimizer experiment is listed too.*

### Settings (admin)

![Settings](docs/img/screenshots/settings.jpg)

*The optimizer's applied recommendation (dense → hybrid search, threshold 0.35 → 0.68), the
answer mode (Fast / Agentic / Auto), the answer provider with automatic fallback, the models with
their quality checks, and the measured search settings.*

### Usage (admin)

![Usage](docs/img/screenshots/usage.jpg)

*Every model call from the local usage log: calls, tokens, latency and errors per day, purpose
and model, each call linked to its LangSmith trace.*

## Keys and accounts

Only a Gemini key is required; everything else is optional. All are free to start with.
Step-by-step instructions are in the [setup guide](docs/setup-guide.md).

| What | Needed for | Where to get it | `.env` setting |
|---|---|---|---|
| **Gemini API key** (required) | Answers and embeddings | [Google AI Studio](https://aistudio.google.com/apikey) - [guide step 3](docs/setup-guide.md#3-get-a-gemini-api-key) | `GEMINI_API_KEY` |
| Admin key | Admin pages and scripts | Any long random text you make yourself - [guide step 5](docs/setup-guide.md#5-create-the-env-settings-file) | `API_KEYS` |
| Groq API key | A second AI provider that takes over when Gemini is busy | [console.groq.com](https://console.groq.com) - [guide step 11](docs/setup-guide.md#11-optional-second-ai-provider-groq) | `GROQ_API_KEY` |
| Google client ID | "Sign in with Google" | [Google Cloud Console](https://console.cloud.google.com/auth/clients) - [guide step 10](docs/setup-guide.md#10-optional-sign-in-with-google) | `GOOGLE_CLIENT_ID`, `ADMIN_EMAILS` |
| LangSmith API key | Traces of every request | [smith.langchain.com](https://smith.langchain.com) - [guide step 12](docs/setup-guide.md#12-optional-langsmith-tracing) | `LANGSMITH_*` |

## Quick start with Docker

Requires Docker Desktop and a Gemini API key.

```bash
cp .env.example .env          # set GEMINI_API_KEY and API_KEYS (admin keys)
docker compose up --build     # UI http://localhost:8080 · API http://localhost:8000/docs
```

Then sign in as admin (one of your `API_KEYS`) and upload documents in **Admin › Documents** -
for example the six demo documents in `data/sample_corpus/`. (With [uv](https://docs.astral.sh/uv/)
installed you can instead index them once with `uv run rag ingest` before starting.)

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
request: ruff + pytest, the quality gate, UI tests + production build, a dependency
vulnerability audit and Docker image builds.
After a change that alters the indexed texts (e.g. another chunk size), refresh the snapshot
with `uv run rag gate --export` (and `--update-baseline` when the new numbers are accepted).

The Gemini free tier allows 15 chat requests per minute and 500 per day per model, and
100 embedded texts per minute. The code throttles itself to stay within these limits, and
evaluation runs can be resumed with `--resume`.

## License

[MIT](LICENSE) © 2026 Ratheesh R
