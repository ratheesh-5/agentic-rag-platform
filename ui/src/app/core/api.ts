import { HttpClient, HttpHeaders, HttpInterceptorFn } from '@angular/common/http';
import { Injectable, InjectionToken, inject, isDevMode } from '@angular/core';
import { Observable, catchError, throwError } from 'rxjs';

import { ApiKeyStore } from './api-key.store';
import {
  AssistantProfile,
  AuthConfig,
  ChatProvider,
  ChatRequest,
  ChatResponse,
  Collection,
  CollectionIn,
  Conversation,
  ConversationSummary,
  DocumentInfo,
  EvalRunDetail,
  EvalRunSummary,
  Me,
  Mode,
  OptimizerRun,
  PrivateSummary,
  ProviderName,
  QualityTestResult,
  RagConfig,
  Recommendation,
  SignInResponse,
  UploadResponse,
  UsageReport,
  UsageSummary,
} from './models';

/**
 * Where the REST API runs:
 * - development (`npm start`): http://127.0.0.1:8000 (`uv run rag serve`),
 * - production build (Docker): `/api` on the UI's own address - nginx forwards it to the API
 *   container, so there is no CORS and no clash with a local API on port 8000.
 * Can be overridden without rebuilding via localStorage 'rag.apiBaseUrl' (e.g. API on another
 * port or host); tests override it by providing this token.
 */
export const API_BASE_URL = new InjectionToken<string>('API_BASE_URL', {
  providedIn: 'root',
  factory: () =>
    (
      localStorage.getItem('rag.apiBaseUrl') ?? (isDevMode() ? 'http://127.0.0.1:8000' : '/api')
    ).replace(/\/+$/, ''),
});

/**
 * Adds who we are to every request sent to our API - and only to our API: X-Guest-Id (always;
 * a signed-in user's chats belong to the account, the guest's stay with the guest), the session
 * token as "Authorization: Bearer" when signed in with Google, and X-API-Key when an admin key
 * was entered. A request that already carries X-API-Key keeps it (admin sign-in checks a key
 * before storing it). A 401 on a request that sent the session token means the session ended
 * (expired or signed out elsewhere): it is dropped, so the browser continues as a guest.
 * A functional interceptor - .NET analogy: a DelegatingHandler on the HttpClient.
 */
export const apiKeyInterceptor: HttpInterceptorFn = (req, next) => {
  const store = inject(ApiKeyStore);
  const base = inject(API_BASE_URL);
  if (!req.url.startsWith(base)) {
    return next(req);
  }
  const headers: Record<string, string> = { 'X-Guest-Id': store.guestId };
  const session = store.session();
  if (session) {
    headers['Authorization'] = `Bearer ${session.token}`;
  }
  if (store.key() && !req.headers.has('X-API-Key')) {
    headers['X-API-Key'] = store.key();
  }
  return next(req.clone({ setHeaders: headers })).pipe(
    catchError((err) => {
      if (session && err?.status === 401 && store.session()?.token === session.token) {
        store.clearSession();
      }
      return throwError(() => err);
    }),
  );
};

/** Typed client for the REST API: one method per endpoint. */
@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly base = inject(API_BASE_URL);

  health(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.base}/health`);
  }

  /** What a collection covers: topics and example questions for new users. */
  assistant(collectionId?: number): Observable<AssistantProfile> {
    const params = collectionId ? { collection_id: collectionId } : undefined;
    return this.http.get<AssistantProfile>(`${this.base}/assistant`, { params });
  }

  collections(): Observable<Collection[]> {
    return this.http.get<Collection[]>(`${this.base}/collections`);
  }

  createCollection(body: CollectionIn): Observable<Collection> {
    return this.http.post<Collection>(`${this.base}/collections`, body);
  }

  updateCollection(id: number, body: CollectionIn): Observable<Collection> {
    return this.http.put<Collection>(`${this.base}/collections/${id}`, body);
  }

  /** Admin: how much users store privately (totals only). */
  privateSummary(): Observable<PrivateSummary> {
    return this.http.get<PrivateSummary>(`${this.base}/collections/private-summary`);
  }

  /** "Delete my account": private collections, chats and the sign-in (signed-in users). */
  deleteAccount(): Observable<void> {
    return this.http.delete<void>(`${this.base}/me`);
  }

  /** Deletes the collection with its documents and chats (whoever may manage it). */
  deleteCollection(id: number): Observable<void> {
    return this.http.delete<void>(`${this.base}/collections/${id}`);
  }

  /** The caller's role. With `key`, checks that key instead of the stored one (admin sign-in). */
  me(key?: string): Observable<Me> {
    const headers = key ? new HttpHeaders({ 'X-API-Key': key }) : undefined;
    return this.http.get<Me>(`${this.base}/me`, { headers });
  }

  /** Is Google sign-in configured? (open endpoint) */
  authConfig(): Observable<AuthConfig> {
    return this.http.get<AuthConfig>(`${this.base}/auth/config`);
  }

  /** Exchange the ID token from Google's button for our own session. */
  signInWithGoogle(credential: string): Observable<SignInResponse> {
    return this.http.post<SignInResponse>(`${this.base}/auth/google`, { credential });
  }

  /** End the session on the server (the token stops working at once). */
  logout(): Observable<void> {
    return this.http.post<void>(`${this.base}/auth/logout`, {});
  }

  chat(request: ChatRequest): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.base}/chat`, request);
  }

  conversations(): Observable<ConversationSummary[]> {
    return this.http.get<ConversationSummary[]>(`${this.base}/conversations`);
  }

  conversation(id: number): Observable<Conversation> {
    return this.http.get<Conversation>(`${this.base}/conversations/${id}`);
  }

  deleteConversation(id: number): Observable<void> {
    return this.http.delete<void>(`${this.base}/conversations/${id}`);
  }

  /** "Delete my data": every conversation of this browser (guest id) or admin key. */
  deleteAllConversations(): Observable<void> {
    return this.http.delete<void>(`${this.base}/conversations`);
  }

  feedback(messageId: number, rating: 1 | -1 | null): Observable<void> {
    return this.http.put<void>(`${this.base}/messages/${messageId}/feedback`, { rating });
  }

  /** Documents of one collection (or all, without an id). */
  documents(collectionId?: number): Observable<DocumentInfo[]> {
    const params = collectionId ? { collection_id: collectionId } : undefined;
    return this.http.get<DocumentInfo[]>(`${this.base}/documents`, { params });
  }

  upload(file: File, collectionId?: number): Observable<UploadResponse> {
    const form = new FormData(); // multipart/form-data, field name "file" as the API expects
    form.append('file', file, file.name);
    if (collectionId) {
      form.append('collection_id', String(collectionId));
    }
    return this.http.post<UploadResponse>(`${this.base}/documents`, form);
  }

  deleteDocument(id: number): Observable<void> {
    return this.http.delete<void>(`${this.base}/documents/${id}`);
  }

  config(): Observable<RagConfig> {
    return this.http.get<RagConfig>(`${this.base}/config`);
  }

  /** Switch the answer mode (admin); used from the next question. */
  setMode(mode: Mode): Observable<RagConfig> {
    return this.http.put<RagConfig>(`${this.base}/config/mode`, { mode });
  }

  /** The chat providers the server knows and whether each is set up (admin). */
  chatProviders(): Observable<ChatProvider[]> {
    return this.http.get<ChatProvider[]>(`${this.base}/config/providers`);
  }

  /** Answer model (admin): which provider writes answers first, fallback on/off. */
  setModel(chat_provider: ProviderName, chat_fallback: boolean): Observable<RagConfig> {
    return this.http.put<RagConfig>(`${this.base}/config/model`, { chat_provider, chat_fallback });
  }

  evalRuns(): Observable<EvalRunSummary[]> {
    return this.http.get<EvalRunSummary[]>(`${this.base}/eval/runs`);
  }

  evalRun(id: number): Observable<EvalRunDetail> {
    return this.http.get<EvalRunDetail>(`${this.base}/eval/runs/${id}`);
  }

  /** Retrieval-only quality check of the active config; optional own test set (.jsonl). */
  runQualityCheck(testSet?: File): Observable<EvalRunSummary> {
    const form = new FormData();
    if (testSet) {
      form.append('golden', testSet, testSet.name);
      form.append('label', 'check-' + testSet.name.replace(/\.jsonl$/i, '').slice(0, 50));
    }
    return this.http.post<EvalRunSummary>(`${this.base}/eval/runs`, form);
  }

  setRunHidden(id: number, hidden: boolean): Observable<void> {
    return this.http.put<void>(`${this.base}/eval/runs/${id}/hidden`, { hidden });
  }

  /** Live quality from real chats: totals, suggestions, per-document numbers (no LLM calls). */
  usage(includeAdmin = false): Observable<UsageSummary> {
    const query = includeAdmin ? '?include_admin=true' : '';
    return this.http.get<UsageSummary>(`${this.base}/quality/usage${query}`);
  }

  /** Admin playground: answer one question and score it now (not saved). */
  testQuestion(question: string): Observable<QualityTestResult> {
    return this.http.post<QualityTestResult>(`${this.base}/quality/test`, { question });
  }

  /** Usage log: calls, tokens, latency, errors, cost for the last `days` days. */
  usageSummary(days: number): Observable<UsageReport> {
    return this.http.get<UsageReport>(`${this.base}/usage/summary?days=${days}`);
  }

  /** Link to one model call's trace in LangSmith (looked up on demand). */
  traceUrl(runId: string): Observable<{ url: string }> {
    return this.http.get<{ url: string }>(`${this.base}/usage/trace/${runId}`);
  }

  recommendation(): Observable<Recommendation | null> {
    return this.http.get<Recommendation | null>(`${this.base}/optimizer/recommendation`);
  }

  decideRecommendation(id: number, decision: 'apply' | 'reject'): Observable<void> {
    return this.http.post<void>(`${this.base}/optimizer/recommendation/${id}/${decision}`, null);
  }

  /** Start the optimizer in the background; optional own test set (.jsonl). */
  startOptimizer(testSet?: File): Observable<OptimizerRun> {
    const form = new FormData();
    if (testSet) {
      form.append('golden', testSet, testSet.name);
    }
    return this.http.post<OptimizerRun>(`${this.base}/optimizer/runs`, form);
  }

  /** Progress and outcome of the optimizer started from the UI. */
  optimizerRun(): Observable<OptimizerRun> {
    return this.http.get<OptimizerRun>(`${this.base}/optimizer/run`);
  }
}
