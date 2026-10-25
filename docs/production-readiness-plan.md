# Production readiness plan (next sessions)

Status 2026-10-12: all 7 phases are pushed (`207b370`). This is the agenda for the next
sessions, taken **step by step, one discussion phase at a time** (see "Discussion phases" at the
end): (A) the approaches we chose, (F) storage and vector databases, (B) security,
(C) performance/optimization, (D) reliability/operations, (E) delivery (Docker images in a
registry, deployment).
Each item: what is missing today → proposed fix. Priorities: **P1** before any real users,
**P2** before scaling, **P3** nice to have. Nothing here is decided yet.

---

## A. Approaches to review together

| Topic | Today | Question to discuss |
|---|---|---|
| Answer mode | ✅ **Done 2026-10-22: auto mode (escalation router).** Fast first; escalate to agentic when the model says not answerable, no valid citation, or the answer covers only part of the question (`complete=false`, same call); threshold refusals are not escalated. Admins switch fast / agentic / auto in Admin › Settings | Still open: measure it - a small run (~15 questions incl. multi-part) fast vs auto vs agentic, answers only (~105 calls) |
| Abstention | similarity threshold 0.68 + model decision + partial answers | Is the balance right for real users (too strict / too lenient)? |
| Storage | SQLite + FAISS files on one machine | When to move to PostgreSQL + pgvector (or a managed vector DB)? |
| Online evaluation | 1 in 5 answers judged by the same model | Separate `JUDGE_MODEL`? Sample rate? |
| Auth | admin API keys + anonymous guests | Move to real sign-in (Entra ID / Google)? |
| Evaluation | golden set (50 Qs) + offline gate | Grow the golden set from real "not answered" questions? |
| Full RAGAS baseline + agentic comparison | still pending (quota) | Run now (≈540 calls over 2 days) or on a paid key? |

---

## F. Storage and vector databases (owner request: discuss first, step by step)

### F1. How it works today (walk-through)
- **Embedding:** each passage (~800 characters) becomes a vector of numbers (Gemini
  `gemini-embedding-001`) that captures its meaning; similar meaning → vectors point in a
  similar direction (cosine similarity).
- **FAISS** (a library inside the API process, not a server): `IndexIDMap(IndexFlatIP)` =
  **exact** search - the question's vector is compared with **every** passage vector. Each vector
  carries the SQLite chunk id. Saved as one file (`data/index/faiss.index`).
- **SQLite** (one file, `data/index/rag.db`) is the **source of truth**: documents, passages
  (text + character offsets), the embedding cache, evaluation runs, conversations, messages,
  scores, usage log, recommendations. FAISS can always be rebuilt from it.
- **BM25** keyword index (hybrid search): built in memory from the passages in SQLite.
- Flow: question → embed → FAISS top hits (+ BM25, fused with RRF) → passage text from SQLite.

### F2. Limits of this design
| Limit | Why it matters |
|---|---|
| Exact search compares with every vector | Fine for thousands of passages; slower at hundreds of thousands+ |
| Files on one machine | One API instance only; no shared index between replicas |
| SQLite allows one writer at a time | Fine for a few admins; not for heavy concurrent writes |
| Index + BM25 reloaded per request (see C1) | Wasted time even at small scale |
| No metadata filtering in the vector search (e.g. by department, access rights) | Needed for per-user document permissions |

### F3. Alternatives to compare (pros/cons, cost, effort)
| Option | What it is | Fits when |
|---|---|---|
| **PostgreSQL + pgvector** | Relational DB with a vector column + vector index (HNSW/IVF) | One database for everything (documents, chats, vectors); familiar SQL, like SQL Server for .NET devs |
| **Qdrant / Weaviate / Milvus** | Dedicated vector database servers | Large collections, rich filtering, high query volume |
| **Chroma** | Lightweight embedded/server vector store | Prototypes; close to what we have |
| **Azure AI Search / Elasticsearch / OpenSearch** | Search engines with built-in hybrid (keyword + vector) search | Enterprise search, Azure deployments, built-in BM25 |
| **Pinecone (managed)** | Hosted vector DB as a service | No servers to run; pay per usage |
| **FAISS with an approximate index (HNSW/IVF)** | Same library, faster search, small accuracy trade-off | Scale up without new infrastructure |

### F4. Questions to decide
1. Expected size (documents/passages) and number of users → do we need to change at all now?
2. One database for everything (pgvector) vs a separate vector DB?
3. Local/Docker only, or a cloud target (e.g. Azure → Azure AI Search or PostgreSQL Flexible
   Server with pgvector)?
4. Do we need per-user document permissions (metadata filters)?
5. Migration path: our design keeps SQLite as the source of truth and FAISS rebuildable, so
   swapping the vector store is mostly one module (`store/vector_index.py`) + `open_index`;
   moving chats/runs to PostgreSQL is a bigger step (`store/db.py`).

## G. Model providers and automatic fallback (future, owner idea 2026-10-13; widened 2026-10-24)

Goal: **resilience and easy configuration** - keep answering when a key expires or a quota runs
out, and switch provider without code changes. Not for comparing providers.
Owner, 2026-10-24: friends and relatives will try the app (later also through the APK, E5); they
should not have to wait for one free quota. The admin picks the answer model in Settings
(Gemini, another free provider, a self-hosted open model, later a paid one) and the backend
falls back automatically. **G2, G3, G5 done 2026-10-24** (Gemini + Groq `openai/gpt-oss-120b`,
Settings › Answer model, fallback on quota / overload / timeout; Groq checked on 10 golden
questions: 9/10). G1 (several keys) and G6 (Ollama) later; G4 partly done (the usage log names
the model of every call).

| # | Item | How | Effort |
|---|---|---|---|
| G1 | Several keys per provider | `GEMINI_API_KEYS=key1,key2`; on quota / invalid-key errors retry with the next key. Same model -> embeddings keep working (no re-index). Keys only help if they belong to different Google Cloud projects (same project = shared quota) | small |
| G2 | Provider fallback for chat | Primary chat model + ordered fallbacks (e.g. Gemini -> Claude -> OpenAI) via LangChain `with_fallbacks`, configured by name (`provider:model`); only `llm.py` changes. Embeddings stay on ONE provider (switching = deliberate re-index) | small-medium |
| G3 | Admin › Settings "Model providers" card | Configured providers/keys shown as ✓ (never the key), active provider + fallback order (saved in a config file like rag_config.json), last error per provider, "Test connection" (one tiny call). Keys stay in `.env` / secret store - never entered in the browser | medium |
| G4 | Usage page | Which provider/key served each call, how often a fallback was used | small |
| G5 | Free second provider + quality check before use | Add one OpenAI-compatible free provider (first candidate **Groq**: no card, says it doesn't train on API data; or OpenRouter `:free` models) as a fallback through LangChain. A model is offered in Settings only after it passes the golden set (structured output `{answerable, answer, complete}` + valid `[P]` citations - small open models often fail this). Each provider adds its own free quota | small-medium |
| G6 | Self-hosted open model (Ollama) on the SERVER | Ollama next to the API (Docker service or another machine) as the last fallback - no quota, no data leaves the server. CPU-only is slow (a 7B model ~8-12 tokens/s on 4 cores; a 1-3B model faster but weaker); fine for testing and a few users. Free hosting options checked 2026-10-24: **Oracle Cloud Always Free** Ampere A1 (reportedly cut to 2 OCPU / 12 GB in 2026, no GPU, idle instances can be reclaimed, sign-up/capacity not guaranteed); Hugging Face Spaces CPU Basic (2 vCPU / 16 GB, sleeps when idle; reports that free Docker Spaces need a paid plan now) - not a reliable server. Owner's own PC with Ollama remains the simplest test host | medium |
| G7 | Model choice in Admin › Settings, with a quality check before use (owner, 2026-10-25) | **Built 2026-10-25.** Dropdowns for the Gemini answer model, the Groq model, an optional **backup model per provider** (tried before the other provider - limits are per model) and the judge model (lists read live from Google / Groq, filtered to chat models, cached); saved in `rag_config.json` like the mode (empty = `.env` default), used from the next question without a restart. **Test** = one tiny call. A new answer model must first pass a **quality check** (6 golden questions in the background: cited answers + an off-topic refusal, at least 5 of 6) - learned when Groq dropped the citations. Judge: Gemini only, warning that scores before/after a switch aren't comparable; each score records its judge. Embedding model shown locked (a change means re-indexing) | medium |

**Research 2026-10-24 (verify again before building - free terms change often):**

| Provider | Free limits (reported) | Your data | Fit |
|---|---|---|---|
| Gemini (today) | ~15 requests/min, ~500/day per model and project | Free tier: Google may use prompts/answers to improve products, human review possible (not in EEA/CH/UK); paid tier: not used | keep as main |
| Groq | ~30 requests/min; daily cap per model reported between 1,000 and 14,400; no card | Says it does not train on API data; logs up to 30 days, zero-retention option | best free fallback |
| OpenRouter `:free` | 20/min, **50/day** (1,000/day after buying $10 credits once) | depends on the model's host | too small without credits |
| Mistral | free "Experiment" plan reportedly replaced in Oct 2026, limits no longer published; free use may require opting into training | unclear | skip for now |
| Ollama (self-hosted) | no limit, hardware-bound | stays on the server | last fallback / testing |

Privacy: with users' private collections (section I), free tiers that may use data for training
should be flagged in Settings ("free tier: data may be used by the provider").

**Not planned: a model running ON the phone.** The APK (E5) is only the UI; search and answers run
on the server (Python, FAISS, SQLite). An on-device model would mean rewriting the backend for
Android, a 1-4 GB download per phone and slow, weak answers.

Not in scope: automatic embedding-provider switching (needs re-embedding every document, a new
FAISS index and a new CI gate snapshot) - only as an explicit admin "re-index" action, if ever.

## B. Security

| # | Gap today | Proposed fix | Prio |
|---|---|---|---|
| B1 | ✅ Done 2026-10-23 (owner live-tested with real Google accounts): `POST /auth/google` verifies the Google ID token, own 7-day sessions (hashed, revocable), roles user/admin (`ADMIN_EMAILS`), chats per user on every device, guest chats stay with the guest (not moved into an account), 50 questions/hour per user; API keys kept for scripts. UI: Google button, account menu, session in localStorage, CSP for Google (was: admin = shared API keys in `.env`, stored in the browser's sessionStorage) | Real sign-in (OIDC: Microsoft Entra ID or Google) with roles; at least hashed keys + rotation. **Owner idea 2026-10-21: "Sign in with Google"** for users (Gmail accounts) - identity only: conversations owned by the Google user id instead of a browser guest id (history on every device), per-user question limits, admin role by allow-listed emails. The Gemini key stays **one server-side key**; Google offers no way to create an AI Studio key for each user at sign-in, and asking users to paste their own key is a poor experience plus key storage | P1 |
| B2 | No HTTPS in Docker (plain HTTP) | TLS at the edge (reverse proxy / cloud ingress), HSTS | P1 |
| B3 | ✅ Done 2026-10-13: CSP, nosniff, X-Frame-Options, Referrer- and Permissions-Policy, server_tokens off (was: nginx sends no security headers) | `Content-Security-Policy`, `X-Content-Type-Options`, `X-Frame-Options`/`frame-ancestors`, `Referrer-Policy` | P1 |
| B4 | ✅ Done 2026-10-13: API as uid 10001, nginx-unprivileged, no-new-privileges, cap_drop ALL; still open: read-only root filesystem (was: containers run as **root**) | Non-root `USER`, read-only filesystem where possible, drop capabilities | P1 |
| B5 | ✅ Done 2026-10-13: API port on 127.0.0.1 only, nginx overwrites X-Forwarded-For, API trusts only nginx's fixed IP - verified: faked IPs ignored (was: port 8000 published and `FORWARDED_ALLOW_IPS="*"` → spoofable guest limit) | Don't publish 8000 in production (only nginx); trust only the proxy's network | P1 |
| B6 | ◐ Prompt rule done 2026-10-13 (`UNTRUSTED_PASSAGES` in answer, grader, verifier; local test: a planted "ignore all previous instructions" passage was not followed - also not without the rule, so it is defence in depth). 2026-10-23: detection at upload done - instruction-like lines are flagged (warn only, document still indexed), shown as a "Check content" badge in Admin › Documents and logged. Still open: output checks (was: indirect prompt injection via uploaded documents) | Treat passages as data in the prompt (already partly), detect instruction-like text at upload, admin review of uploads, output checks | P1 |
| B7 | Uploads: type/size/signature checks only | Malware scanning, DOCX/ZIP bomb limits, per-admin upload quotas, audit of who uploaded what | P2 |
| B8 | Secrets in a local `.env` | Secret store (e.g. Azure Key Vault / GitHub environments), never in images (already true) | P1 (cloud) |
| B9 | Guest rate limit is in memory, per process (resets on restart, not shared) | Shared store (Redis) or gateway rate limiting; per-guest + per-IP + global budget | P2 |
| B10 | ✅ Done 2026-10-23: daily cleanup (conversations inactive > 90 days with messages/feedback/scores, usage log > 90 days; `CHAT_RETENTION_DAYS` / `USAGE_RETENTION_DAYS`, 0 = keep), "Delete all" conversations in the sidebar, retention note under the chat input. Still open: PII redaction before LangSmith tracing (was: personal data kept forever, reviewed 2026-10-13, fine for the demo): conversations/messages, online scores + feedback, usage log and eval runs are never deleted automatically; the UI's 7/14/30-day Usage filter and the 200/500-row lists are display limits only. Guest history lives as long as the browser keeps its localStorage guest id. LangSmith traces expire by LangSmith's own retention | Retention policy + delete job (e.g. chats/usage older than 90 days), PII redaction before tracing, privacy notice, "delete my data" | P1 |
| B11 | No audit log of admin actions (upload, delete, apply config) | Append-only audit table + Admin view | P2 |
| B12 | ✅ Done 2026-10-13: Dependabot (uv, npm, actions, docker) + CI audit job (pip-audit, npm audit); 2 advisories without a fix reviewed and ignored; still open: image scanning (Trivy) (was: no dependency / image scanning) | Dependabot (pip, npm, GitHub Actions, Docker), `pip-audit`, `npm audit`, Trivy in CI | P1 |
| B13 | Model cost abuse (a guest could drive cost) | Daily/monthly budget caps with alerts (the incident runbook's "cost overrun"), kill switch | P2 |

## C. Performance and optimization

| # | Gap today | Proposed fix | Prio |
|---|---|---|---|
| C1 | ~~Every request reloads the FAISS index and rebuilds the BM25 index~~ | ✅ Done 2026-10-13: `SearchCache` in the API keeps both in memory; a 0.02 ms stamp (chunk count, max chunk id, FAISS file time) triggers a rebuild only after documents change. Measured before: 4.6 ms/request at 80 passages, growing linearly | P1 |
| C2 | API image is 1.7 GB (RAGAS + eval tools are in the runtime) | Split dependency groups: runtime vs eval/optimizer; slimmer image | P2 |
| C3 | Free-tier throttling, judge shares the chat limit. ✅ Partly done 2026-10-13: burst of 2 (`CHAT_BURST`) - a question's rewrite + answer no longer wait ~8.6 s in our own limiter (questions sent within ~10 s of each other still wait) | Paid tier or separate `JUDGE_MODEL`; queue background scoring | P2 |
| C4 | No answer cache | Semantic cache for repeated questions (with invalidation on document change) | P3 |
| C5 | No reranking | Optional cross-encoder rerank of the top 20 (measure with the gate first) | P3 |
| C9 | Context noise: fast mode always sends the top 5 passages, even when one holds the answer (seen 2026-10-13: "What does HyDE do?" → only [1] relevant, cited correctly). Fixed top-k favours recall over precision; the 0.68 threshold only checks the best passage | Options: relative score cutoff (drop passages far below the best; few lines, tune with the quality check) · reranker (C5) · agentic mode already grades and drops irrelevant passages. Measure Recall@5 / Precision@5 before and after | P3 |
| C10 | Multi-hop questions (found 2026-10-15 with HR test documents): "Who approves parental leave, and what is that person's email?" needs the role from one document, then the email from another. The planner plans all searches up front, before it knows the role, so the directory passage is never retrieved | **Iterative retrieval** in agentic mode: after grading, an LLM step asks "is a fact still missing? if so, which search?" and loops once more (budget in code, like rewrites). ~1-2 extra calls only for such questions | P3 |
| C11 | The similarity threshold (0.68) is calibrated on the demo corpus only. With other documents, good passages can score lower (HR test 2026-10-15: a table of names 0.60, a keyword query 0.655), so fast mode refuses answerable questions. Agentic mode no longer applies it after grading | Calibrate per corpus: run the quality check on a small test set for each new document collection and take its `safe_threshold`. Bigger step (owner discussion 2026-10-16): **collections** - unrelated domains (e.g. HR policies vs AI handbook) become separate knowledge bases, each with its own settings (`rag_config`), test set and optimizer run; the chat searches the chosen collection (user picks, or a router). Today one settings set is a compromise over all documents (seen: Brightwave docs lowered the Lumora baseline Recall@5 1.000 → 0.9875). **Decided 2026-10-24: collections with owners** - shared (admin) + private (per user) collections, isolated search; see section I | P1 (next feature) |
| C12 | ~~The optimizer could only be started from the CLI~~ | ✅ Done 2026-10-21: Admin › Settings › *Find better settings* runs the optimizer in the background with the default or an uploaded test set (`POST /optimizer/runs`, progress via `GET /optimizer/run`, shares the quality-check lock); the recommendation shows which test set it was measured on. Placed on Settings (not Evaluation) so the result appears where it is approved | P3 |
| C13 | Writing a test set for new documents by hand is slow (owner question 2026-10-21: "can uploading create the test set?"). Automatic generation was rejected earlier (design.md: generated questions copy the document's wording, so scores look better than for real users, and an unchecked test set can lead to wrong settings) | **Draft helper, human-reviewed:** Admin › Documents › "Draft test questions" - the LLM proposes ~5 questions per document with the evidence quote (+ 1-2 unanswerable ones), asked to use user-like wording; the admin edits, deletes or confirms each, then saves a `.jsonl` test set (quotes validated like today). Never runs automatically, never used unreviewed. Cost ~1 chat call per document. Optional later: RAGAS test-set generator instead of an own prompt | P3 |
| C14 | Analysis questions over several documents ("compare these case documents", "what is the overall picture?") get only the top 5 passages, often from 1-2 documents (owner idea 2026-10-24) | **Analyse mode** within one collection: map-reduce - per document, find what is relevant and write a short cited note (1 call per document), then combine the notes into one answer (comparisons, conflicts, overall picture), sources grouped by document. ~1 call per document + 1, so a document limit (e.g. 10). Builds on C11 (needs collections) | P2 |
| C15 | Documents with images (owner question 2026-10-24): only the **text** of PDF/DOCX is indexed; images are skipped, and a **scanned** PDF (page photos) has no text at all, so nothing in it can be found | **Level 1 - OCR** of scanned pages (Gemini reads the page image, or local Tesseract; 1 call per scanned page, at upload only). **Level 2 - image descriptions**: at upload Gemini describes each chart/diagram/screenshot ("Bar chart: leave days per year...") and the description is indexed like text, cited as "image on page N" (1 call per image, at upload). Ingestion-only change (loaders + one describe step); search and answers stay the same. Needs a per-document image limit on the free tier. Level 3 (multimodal embeddings + a model that looks at images when answering) only if levels 1-2 aren't enough. Planned after phase 3 (Analyse mode) | P2 |
| C6 | Single process, SQLite write lock | PostgreSQL + pgvector, several API replicas behind the proxy | P2 (scale) |
| C7 | Sync endpoints only | Async model calls where it helps throughput | P3 |
| C8 | No streaming (answers validated first) | Revisit: stream after validation per sentence, or show progress steps | P3 |

## D. Reliability and operations

| # | Gap today | Proposed fix | Prio |
|---|---|---|---|
| D1 | ✅ Done 2026-10-23: JSON lines on stdout per request / model call / answer with a request id (also in the `X-Request-ID` header) and the LangSmith run id; no question or answer text (was: no application logging, print/uvicorn only) | Structured JSON logs with request id, linked to the LangSmith run id | P1 |
| D2 | ✅ Done 2026-10-23: `GET /ready` (config, database, search index; key checked by startup; no model call) + Docker healthchecks for API and UI; the UI waits for a healthy API (was: no Docker health checks; `/health` doesn't check dependencies) | `/health` (liveness) + `/ready` (DB, index, model key); `healthcheck` in compose | P1 |
| D3 | Schema changes by hand (`Store._migrate`) | Versioned migrations (e.g. Alembic or a migrations table) | P2 |
| D4 | No backups of SQLite/index/uploads | Scheduled backups + restore test (FAISS is rebuildable; SQLite + uploads are not) | P1 |
| D5 | No alerts | Alerts on error rate, latency p95, cost per day, "not answered" spike (LangSmith or cloud monitor) | P2 |
| D6 | No load test | k6/Locust test for chat + upload; know the limits | P2 |
| D7 | One environment | dev / staging / prod config, staging gets the same CI gate | P2 |
| D8 | CI not yet watched | Confirm the first GitHub Actions run is green; branch protection with required checks | P1 |

## E. Delivery

| # | Item | Proposal | Prio |
|---|---|---|---|
| E1 | Docker images in a registry ("different repo") | Publish to GitHub Container Registry on version tags (`v1.0.0`): `ghcr.io/ratheesh-5/agentic-rag-platform-api` / `-ui`, using GitHub's built-in token; compose can then `pull` instead of build | P2 |
| E2 | Versioning | Semantic versions + CHANGELOG + GitHub releases | P2 |
| E3 | Cloud deployment | e.g. Azure Container Apps (or App Service for Containers) with infrastructure as code (Bicep/Terraform), managed TLS, secrets in Key Vault | P3 |
| E4 | Answer-quality gate in CI | Nightly (not per push) small RAGAS run with a secret key, compared with the baseline | P3 |
| E5 | No mobile app (owner idea 2026-10-21) | **Android APK** from the existing Angular UI with **Capacitor** (wraps the web app; one code base). Needs: the API reachable over HTTPS (E3 + B2), Google Sign-In in the app (B1, native plugin), a mobile layout check (the chat already works on phones). Cheaper first step: a **PWA** (installable from the browser, no store) | P3 |

---

## H. Product use cases (owner idea 2026-10-21, with the mobile app E5)

The core - answer from a set of documents, with sources, or say "I don't know" - fits any job where
people need a precise answer from long documents and must be able to check it.

| # | Use case | Example question | Why this app fits |
|---|---|---|---|
| H1 | **Company policy assistant** (employees) | "What's the hotel limit in London?" | Long HR/travel policies nobody reads; tested with the Brightwave HR documents |
| H2 | **Personal document vault** | "Does my health policy cover day-care surgery?" | Insurance, rental, loan, warranty papers; the cited source is the proof |
| H3 | **Field technicians / maintenance** | "What does error E-42 mean and what's the first fix?" | Manuals on the phone; like the demo's LUM error codes |
| H4 | **Students** | "Which topics are in unit 3?" | Course notes, syllabus, exam rules - with the page it came from |
| H5 | **Customer support staff** | "What's the return window for product X?" | Consistent answers citing the official text |
| H6 | **Government schemes / public rules** | "Am I eligible for this scheme?" | Official PDFs; answers in the user's language (Gemini is multilingual) |

**Needed before any of them:** private collections per user or company (C11, section I), Google sign-in
(B1), HTTPS hosting (E3, B2), a paid Gemini key for real traffic (free tier ≈ 15 questions/minute
for everyone). **Portfolio demo pick:** H2 or H1 - easy to explain, everyone has the problem, and
"answer with source or say I don't know" is what makes them trustworthy.

---

## I. Collections with owners (owner decision 2026-10-24)

Goal: users upload their own documents into collections ("My loan", "Land case 2024") and ask
questions - also analysis questions across the collection - without ever seeing another user's
data; the same API later serves the web UI and a mobile app (E5). Today all documents are one
shared pool and only admins upload.

**Model (owner went with these recommendations):**

| Space | Who uploads | Who can search it |
|---|---|---|
| **Shared collections** | Admins | Everyone, guests included (today's knowledge base) |
| **My collections** (private) | Each signed-in user | Only that user (optionally plus shared ones) |

- Guests: shared collections only; uploading needs Google sign-in (B1).
- Admins manage shared collections and see usage/cost, but **cannot read users' private
  documents** (only counts and sizes) - trust and privacy law.

**Must be right from the start:**
1. Every document and passage carries an **owner** and a **collection** (conversations already
   have an owner).
2. **Isolation is enforced on the server inside search** (filter to allowed passages *before*
   ranking - FAISS, BM25 and caches alike), never by hiding results in the UI; tests prove user B
   can never receive user A's text.
3. Uploads by anyone: per-user limits (documents, storage, questions per day; B13), the upload
   scan (B6) and later malware scanning (B7).
4. Deleting an account deletes all its documents, passages, collections and chats (B10).
5. Storage: one FAISS index + SQLite is fine for a demo; many users with own documents → revisit
   PostgreSQL + pgvector (C6, section F), where "filter by owner" is a WHERE clause.
6. Mobile app (E5): just another client of the same API; needs HTTPS hosting (B2/E3). If the API
   enforces ownership, every client is safe.

**Phases** (owner, 2026-10-24; one at a time: design → owner approval → build + tests → owner
review → push):

| Phase | What it adds | Size |
|---|---|---|
| **1. Collections (shared)** ✅ done 2026-10-24 | Admins create collections and upload into them; the chat gets a collection picker and searches only the chosen collection; today's documents become the first collection(s). Built with one FAISS + BM25 index per collection (isolation by construction, tested); the owner's HR test documents moved to "HR policies". Added after owner testing: "Searching in" above the question box, "Ask in <collection>" after a refusal, collection names in the sidebar, and an **automatic similarity threshold per collection** (off-topic probe questions; a resume's questions were refused under the global 0.68) - part of phase 4 brought forward | Medium |
| **2. Private collections** ✅ done 2026-10-24 | Signed-in users create their own collections and upload; strict isolation (tests prove user B never sees user A's text - 404 everywhere); admins see totals only, never content; per-user limits (5 collections, 20 documents, 500 passages, checked before embedding); "My documents" page; "Delete my account" removes collections, files, chats and the account | Medium-Large |
| **3. Analyse mode (C14)** ✅ done 2026-10-24 | Questions across a whole collection (compare, overall picture): per-question toggle "Analyse all documents" (signed-in users and admins); map-reduce - code picks each document's best 4 passages (best 10 documents), 1 cited note per document, 1 call combines them; the answer says how many documents were read / had nothing / were over the limit / failed; counts as 5 questions toward the hourly limit | Medium |
| **4. Quality per collection** (on hold, owner 2026-10-24: measurement only, today's flow works; resume later as 4a test sets + quality check, 4b settings + optimizer) | Own settings per collection (threshold, chunk size - measured by the optimizer, not guessed), own test set (C13 drafts, owner reviews), own optimizer run | Medium |
| **5. Before real users** (optional) | Storage review (PostgreSQL + pgvector), HTTPS hosting (B2/E3), Android app (E5) | Large |

Phases 1-3 are enough for the portfolio; 4 is a strong extra; 5 only when publishing. Chunking stays
one global setting (800/100) in phases 1-3 (each document already stores the chunking it used);
per-collection chunking comes in phase 4. D8 can be done between any phases.

---

## Discussion phases (one per session or step)
1. **Phase 1 - Storage and vector databases (F):** ✅ discussed 2026-10-13. Decision: **keep
   SQLite + FAISS** (fits the expected size) and add C1 (done). Revisit pgvector when going
   multi-user/cloud or beyond ~10k passages.
2. **Phase 2 - Approaches (A):** review the main design choices.
3. **Phase 3 - Quick wins (no API cost):** ✅ C1, B3, B4, B5, B6 (prompt rule), B12 done 2026-10-13; D2 + D1 + B10 + B6 upload scan done 2026-10-23; D8 next (B1 done; discuss full vs light branch protection).
4. **Phase 4 - Security before real users:** ✅ B1, B5, B6, B10 done; B2 open (with hosting).
5. **Phase 5 - Delivery:** E1 publish Docker images (registry), E2 versions, E3 cloud target.
6. **Later - Model providers and fallback (G):** G1 + G2 first (resilience), then G3 + G4.
7. **Phase 6 - Scale and cost:** storage migration if decided in phase 1 (C6), C2, C3, B9, B13,
   D4, D5.

## Suggested order
1. **Quick wins (no API cost):** C1 in-memory index, B3 headers, B4 non-root, D2 health checks,
   B12 Dependabot + audits, D1 logging, D8 confirm CI.
2. **Security before real users:** B1 sign-in, B5 network, B6 prompt injection, B10 retention.
3. **Delivery:** E1 publish images, E2 versions.
4. **Scale / cost:** C2, C3, B9, B13, D4, D5, then C6 if needed.
