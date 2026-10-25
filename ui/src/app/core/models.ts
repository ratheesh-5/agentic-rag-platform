// TypeScript shapes of the REST API's requests and responses.
// They mirror src/rag_platform/api/schemas.py (the API's pydantic models) - keep both in sync.

/** fast = 1 model call · agentic = LangGraph agent · auto = fast first, agentic when needed */
export type Mode = 'fast' | 'agentic' | 'auto';
/** Chat providers that can write answers (Admin › Settings "Answer model"). */
export type ProviderName = 'gemini' | 'groq';
export type Retrieval = 'dense' | 'hybrid';

/** GET /assistant: what the chatbot covers (assistant.json in the backend). */
export interface Topic {
  title: string;
  description: string;
  examples: string[];
  documents?: string[]; // source files; the API hides a topic when none of them is indexed
  generated?: boolean; // added by the API for an uploaded document no topic covers
}

export interface AssistantProfile {
  name: string;
  description: string;
  topics: Topic[];
  chat_retention_days: number; // chats without activity this long are deleted; 0 = kept
}

/** A user signed in with Google (name, email and photo from their Google account). */
export interface UserInfo {
  email: string;
  name: string;
  picture: string; // photo URL, '' if none
}

/** GET /me: admin (admin email or valid X-API-Key), user (signed in with Google) or guest. */
export interface Me {
  role: 'admin' | 'user' | 'guest';
  user: UserInfo | null;
}

/** GET /auth/config: the Google client id for the sign-in button ('' = Google sign-in off). */
export interface AuthConfig {
  google_client_id: string;
}

/** POST /auth/google: our own session after Google confirmed who the user is. */
export interface SignInResponse {
  token: string; // sent as "Authorization: Bearer <token>"
  expires_at: string;
  role: 'admin' | 'user';
  user: UserInfo;
}

export interface Source {
  number: number; // the [n] used in the answer text
  document: string;
  start: number;
  end: number;
  score: number;
  text: string;
}

// --- Chat (conversations with memory) ---

export interface ChatRequest {
  message: string;
  conversation_id?: number; // omitted = start a new conversation
  collection_id?: number; // the collection a NEW conversation searches (default: the demo one)
  // Analyse mode: answer from every document of the collection (signed-in users and admins).
  analyse?: boolean;
  // No mode/search here: those are backend settings (rag_config.json); the API refuses guest
  // overrides. Other API clients can still send them as admins (see api/schemas.py).
}

export interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  // assistant messages only:
  answerable?: boolean | null;
  sources?: Source[];
  standalone_question?: string | null; // how a follow-up was understood (memory)
  abstain_reason?: string;
  steps?: string[];
  route?: string; // auto mode: "fast" or "auto → agentic (reason)"
  feedback?: 1 | -1 | null;
  faithfulness?: number | null; // online-eval score (sampled answers only)
  answer_relevancy?: number | null;
  seconds?: number | null; // assistant: how long the answer took
  // "I couldn't find this" here, but these collections have matching documents.
  found_in?: { id: number; name: string }[];
}

export interface ChatResponse {
  conversation_id: number;
  collection_id: number;
  title: string;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
}

export interface ConversationSummary {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  messages: number;
  collection_id: number;
  collection: string; // the collection's name
}

export interface Conversation extends Omit<ConversationSummary, 'messages'> {
  messages: ChatMessage[];
}

/** GET /collections: a named group of documents; a chat searches exactly one. */
export interface Collection {
  id: number;
  name: string;
  description: string;
  documents: number;
  passages: number;
  is_default: boolean; // the demo collection (evaluation uses it; can't be deleted)
  is_private: boolean; // only its owner sees it
  can_manage: boolean; // the caller may upload into / rename / delete it
  /** Similarity threshold used for its questions, and where it comes from. */
  threshold: number;
  threshold_source: 'measured' | 'manual' | 'automatic' | 'default';
  threshold_override: number | null; // an admin's manual value
}

/** POST /collections, PUT /collections/{id} (admin). */
export interface CollectionIn {
  name: string;
  description: string;
  threshold?: number | null; // manual override; null/omitted = automatic
  private?: boolean; // admins: a private collection of their own instead of a shared one
}

/** GET /collections/private-summary (admin): totals only, never names or content. */
export interface PrivateSummary {
  collections: number;
  users: number;
  documents: number;
  passages: number;
}

export interface DocumentInfo {
  id: number;
  path: string;
  doc_type: string;
  collection_id: number;
  chunks: number;
  chunker: string;
  ingested_at: string;
  warnings: string[]; // lines that look like instructions to an AI (prompt-injection scan)
}

export interface UploadResponse {
  document: DocumentInfo;
  status: 'added' | 'updated' | 'unchanged';
  embeddings_new: number;
  embeddings_cached: number;
}

export type MetricValue = number | string | boolean | null;

export type GateStatus = 'pass' | 'fail' | 'warn' | 'n/a';

/** full = answers + RAGAS (CLI) · check = retrieval-only quality check · experiment = optimizer */
export type RunKind = 'full' | 'check' | 'experiment';

export interface EvalRunSummary {
  id: number;
  created_at: string;
  label: string;
  metrics: Record<string, MetricValue>;
  kind: RunKind;
  hidden: boolean;
  current: boolean; // measured with the active configuration
  gates: GateStatus; // release gates, overall
  test_set: string;
}

/** One release gate (data/eval/release_gates.json) checked against a run. */
export interface GateCheck {
  label: string;
  metric: string;
  op: '>=' | '<=';
  threshold: number;
  blocking: boolean;
  value: number | null;
  status: GateStatus;
}

/** Per-question result stored with a run (only the fields the UI uses). */
export interface QuestionResult {
  id: string;
  question: string;
  answerable: boolean;
  predicted_answerable: boolean;
  retrieval: { rr: number; recall: number } | null;
}

export interface EvalRun extends EvalRunSummary {
  config: Record<string, unknown> & { rag?: RagConfig };
  results: QuestionResult[];
}

/** GET /eval/runs/{id} returns the gate checks under "gates" (a list, not the summary status). */
export type EvalRunDetail = Omit<EvalRun, 'gates'> & { gates: GateCheck[] };

/** GET /quality/usage: live quality from real chats (online evaluation). */
export interface UsageAnswer {
  message_id: number;
  question: string;
  answer: string;
  documents: string[];
  faithfulness: number | null;
  answer_relevancy: number | null;
  created_at: string;
}

/** One stored chat answer in the per-question list. */
export interface RecentAnswer extends UsageAnswer {
  answerable: boolean;
  abstain_reason: string;
  feedback: number | null;
  by_admin: boolean; // asked while signed in as admin (team testing)
}

export interface UsageSummary {
  sample_rate: number; // share of answers scored automatically (0.2 = 1 in 5)
  include_admin: boolean;
  admin_chats: number; // answers asked while signed in as admin
  recent: RecentAnswer[]; // newest first
  totals: {
    questions: number;
    answered: number;
    unanswered: number;
    escalated: number; // auto mode: handed from fast to agentic
    thumbs_up: number;
    thumbs_down: number;
    scored: number;
    faithfulness: number | null;
    answer_relevancy: number | null;
  };
  unanswered: {
    question: string;
    understood_as: string | null;
    reason: string;
    last_asked: string;
    count: number;
  }[];
  thumbs_down: UsageAnswer[];
  low_scores: UsageAnswer[];
  documents: {
    document: string;
    answers: number;
    thumbs_down: number;
    faithfulness: number | null;
    answer_relevancy: number | null;
  }[];
}

/** POST /quality/test: one admin test question, answered and scored on the spot. */
export interface QualityTestResult {
  question: string;
  answerable: boolean;
  answer: string;
  abstain_reason: string;
  sources: Source[];
  passages: { document: string; score: number; cited: boolean; text: string }[];
  faithfulness: number | null;
  answer_relevancy: number | null;
  judge_calls: number;
  judge_model: string; // who scored it (Settings › Models)
  score_error: string | null;
  seconds: number;
}

/** GET /usage/summary: the usage log aggregated (Admin › Usage). */
export interface UsageStats {
  calls: number;
  embed_texts: number;
  input_tokens: number;
  output_tokens: number;
  errors: number;
  avg_latency_ms: number | null;
  p95_latency_ms: number | null;
  cost: number | null; // only for models priced in model_prices.json
}

/** One model call from the usage log. */
export interface UsageCall {
  id: number;
  created_at: string;
  model: string;
  purpose: string; // answer, rewrite, judge, agent: <node>, embed, ...
  kind: 'chat' | 'embed';
  latency_ms: number;
  input_tokens: number | null;
  output_tokens: number | null;
  texts: number | null; // embeddings: texts in the request
  error: string | null;
  run_id: string | null; // LangSmith run id (trace link)
  cost: number | null;
}

export interface UsageReport {
  days: number;
  totals: UsageStats;
  priced_models: string[];
  by_purpose: (UsageStats & { purpose: string })[];
  by_model: (UsageStats & { model: string })[];
  per_day: (UsageStats & { date: string })[];
  calls: UsageCall[]; // every call of the period, newest first (up to 500)
  recent_errors: { created_at: string; model: string; purpose: string; error: string }[];
  tracing: { enabled: boolean; project: string; sampling_rate: number; url: string | null };
}

/** The optimizer's latest recommendation, approved or rejected on the Settings page. */
export interface Recommendation {
  id: number;
  created_at: string;
  status: 'pending' | 'applied' | 'rejected' | 'superseded';
  decided_at: string | null;
  stale: boolean; // the configuration changed since: its numbers no longer apply
  report: {
    changes: Partial<RagConfig>;
    recommended_config: RagConfig;
    base_config?: RagConfig; // the settings it was measured against
    baseline_metrics: Record<string, number | null>;
    winner_metrics: Record<string, number | null>;
    summary?: string;
    rationale?: string;
    risks?: string;
    test_set?: string; // the test set it was measured on (runs started from the UI)
  };
}

/** The optimizer started from the UI (background run; the Settings page polls it). */
export interface OptimizerRun {
  state: 'idle' | 'running' | 'done' | 'failed';
  test_set: string;
  steps: string[];
  outcome: '' | 'recommendation' | 'no-change' | 'nothing-to-improve';
  recommendation_id: number | null;
  error: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface RagConfig {
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  retrieval: Retrieval;
  hybrid_candidates: number;
  similarity_threshold: number;
  mode: Mode;
  max_rewrites: number;
  max_regenerations: number;
  chat_provider: ProviderName; // writes answers first
  chat_fallback: boolean; // the other provider takes over on quota / overload errors
  // Settings › Models: chosen models ('' = the .env default) and same-provider backups ('' = none)
  gemini_model: string;
  groq_model: string;
  judge_model: string;
  gemini_backup_model: string;
  groq_backup_model: string;
}

/** Settings › Models: a model slot (answer model per provider, or the judge). */
export type ModelKind = 'gemini' | 'groq' | 'judge';

export interface ModelSlot {
  kind: ModelKind;
  configured: boolean; // the provider's API key is set on the server
  current: string; // the model in use
  default: string; // the .env model
  backup: string; // answer models: same-provider backup ('' = none)
  options: string[]; // models the provider offers today ([] = list not available)
}

/** Quality check of one answer model: a few demo questions answered with exactly that model. */
export interface ModelCheck {
  kind: 'gemini' | 'groq';
  model: string;
  state: 'running' | 'done' | 'failed';
  passed: number;
  total: number;
  ok: boolean;
  error: string | null;
  started_at: string;
  finished_at: string | null;
  items: { id: string; question: string; ok: boolean; note: string }[];
}

export interface ModelsInfo {
  slots: ModelSlot[];
  embedding: string; // shown only: changing it means re-indexing every document
  checks: ModelCheck[];
  check_minimum: number;
}

/** PUT /config/models body: '' = the .env default (models) or none (backups). */
export interface ModelsChange {
  gemini_model: string;
  groq_model: string;
  judge_model: string;
  gemini_backup_model: string;
  groq_backup_model: string;
}

export interface ModelTestResult {
  ok: boolean;
  message: string;
  seconds: number;
}

/** A chat provider and whether its key is set on the server (the key itself is never sent). */
export interface ChatProvider {
  name: ProviderName;
  model: string;
  configured: boolean;
}
