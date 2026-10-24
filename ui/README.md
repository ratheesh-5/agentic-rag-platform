# Agentic RAG Platform — UI

Angular 22 + Angular Material chat app for the platform's REST API, laid out like common chat
products.

| Area | What it does |
|---|---|
| **Chat** (`/chat`, `/chat?c=5`) | Conversation thread with memory: follow-ups like "and who holds that role?" are understood from earlier turns. Cited answers: click a citation badge or source chip to read the passage, hover for a preview, copy, 👍/👎, typing indicator. New users see what they can ask (topics + example questions from the API's `GET /assistant`), "Related" question chips under an answer (from the cited documents' topics), day separators ("Today, October 13", "Yesterday, ...", then the date) with message times, and topic suggestions when the documents don't cover a question |
| **Sidebar** | New chat, recent conversations (delete on hover), Admin links (admins only); on phones a ☰ menu |
| **Admin › Documents** | Drag-and-drop upload of several `.md/.txt/.pdf/.docx` files (validated and indexed by the API, status per file), summary tiles, searchable list with type filter, delete |
| **Admin › Evaluation** | **Live quality** tab: clickable KPI tiles, **Test a question** (answer, score bars, passages), suggestions, every answer with its scores (filter, search, expand), per document, "Include admin chats" switch. **Test-set checks** tab: status card against the release gates, **Run quality check** (seconds, no model calls) or **Use my test set** (upload `.jsonl`), runs by type, hide old runs, details with the questions to look at |
| **Admin › Usage** | Model calls, tokens, latency, errors and (with prices in `model_prices.json`) cost per day, purpose and model. Tiles, chart days, purposes and models filter one list of calls (sort, search, expand); each call opens its LangSmith trace |
| **Admin › Settings** | Measure → Review → Apply steps, the optimizer's recommendation with **Apply / Reject** (human approval) and its measured effect, the active configuration in plain words (changed settings marked), API status |

**Production vs development:** `npm run build` (production) shows only the chat. In
development (`npm start`) every answer also shows how a follow-up was understood, the agentic
steps and abstention reasons. Answer mode (fast/agentic) and search (dense/hybrid) are backend
settings in `rag_config.json`; the UI never chooses them.

## Run

Requires Node 22.22+ or 24.15+. The API must be running (`uv run rag serve` in the repository
root; set `API_KEYS` in `.env` for admin sign-in).

```bash
npm install
npm start              # http://localhost:4200
```

**No registration.** Anyone can chat right away as a guest: the browser creates a random guest id
once (localStorage), and conversations belong to it. Guests get `GUEST_QUESTIONS_PER_HOUR`
(default 20) questions per hour.

**Sign in with Google** (when the backend has `GOOGLE_CLIENT_ID`): the user signs in on Google's
page; the API returns its own session (kept 7 days in localStorage). Chats then follow the user
to every device, and emails in `ADMIN_EMAILS` get the Admin pages. Open the app as `localhost`
(not `127.0.0.1`): Google checks the page's origin.

**Admin sign in** (fallback without a Google client id): enter one of the backend's `API_KEYS`.
The key is checked by the API, kept only for the browser tab (sessionStorage), and unlocks the
Admin pages.

The API address defaults to `http://127.0.0.1:8000` in development (`npm start`) and to `/api`
in a production build (Docker: nginx forwards `/api` to the API container, same origin, no CORS).
To use another one without rebuilding, run
this in the browser console:

```js
localStorage.setItem('rag.apiBaseUrl', 'http://127.0.0.1:8001')
```

The API allows the UI's origin through `CORS_ORIGINS` (default `http://localhost:4200`).

## Develop

```bash
npm test -- --watch=false   # unit tests (Vitest), HTTP mocked, no API needed
npm run build               # production build -> dist/ui
npx prettier --write "src/**/*.{ts,html,scss}"
```

## Structure

```
src/app/
├── core/       api.ts (typed API client + interceptor adding X-Guest-Id / session / X-API-Key),
│               labels.ts (plain-language names for metrics and settings),
│               api-key.store.ts (guest id, Google session, admin key), google-sign-in.ts,
│               admin-sign-in.ts, admin.guard.ts, confirm.ts (confirmation dialog),
│               conversations.store.ts (sidebar list), models.ts (mirrors the API's schemas.py),
│               errors.ts (user-facing messages)
├── features/   chat/ (the chatbot), documents/, eval-runs/, settings/ (admin; lazy-loaded)
├── app.ts/.html  shell: sidebar (new chat, conversations, admin), toolbar (sign-in / account)
└── app.routes.ts
```

State uses Angular signals (the app is zoneless).
