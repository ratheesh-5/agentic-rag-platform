import {
  Component,
  DestroyRef,
  ElementRef,
  computed,
  effect,
  inject,
  input,
  isDevMode,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import { TextFieldModule } from '@angular/cdk/text-field';
import { DatePipe, NgTemplateOutlet } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatMenuModule } from '@angular/material/menu';
import { MatTooltipModule } from '@angular/material/tooltip';
import { Router } from '@angular/router';

import { ApiService } from '../../core/api';
import { ApiKeyStore } from '../../core/api-key.store';
import { CollectionsStore } from '../../core/collections.store';
import { ConversationsStore, InFlight } from '../../core/conversations.store';
import { describeError } from '../../core/errors';
import { AssistantProfile, ChatMessage, Source } from '../../core/models';

/**
 * The chatbot: a conversation thread with memory (the backend rewrites follow-up questions
 * using earlier turns), cited answers, sources, feedback and copy.
 *
 * New users learn what they can ask from the backend's assistant profile (GET /assistant):
 * topics with example questions on the welcome screen and as suggestions when a question is
 * outside the documents. After an answer, "Related" chips offer the other example questions of
 * the cited documents' topics.
 *
 * Answer mode and search method are backend settings (rag_config.json), never chosen here.
 * Production builds show only what an end user needs; developer details (how a follow-up was
 * understood, agentic steps, abstention reasons) appear only in development (`npm start`).
 */
/** Abbreviations shown in capitals in document names (file names are usually lower case). */
const ACRONYMS: Record<string, string> = {
  ai: 'AI',
  api: 'API',
  faq: 'FAQ',
  hr: 'HR',
  llm: 'LLM',
  llmops: 'LLMOps',
  mlops: 'MLOps',
  rag: 'RAG',
  sql: 'SQL',
};

@Component({
  selector: 'app-chat',
  imports: [
    DatePipe, // full date in the message-time tooltip
    NgTemplateOutlet, // one template for every collection in the picker menu
    FormsModule,
    TextFieldModule, // cdkTextareaAutosize: the input grows with the text
    MatButtonModule,
    MatIconModule,
    MatMenuModule, // the collection picker of a new chat
    MatTooltipModule,
  ],
  templateUrl: './chat.html',
  styleUrl: './chat.scss',
})
export class Chat {
  private readonly api = inject(ApiService);
  private readonly router = inject(Router);
  readonly conversations = inject(ConversationsStore);
  readonly collections = inject(CollectionsStore);

  /**
   * Conversation id from the URL query (/chat?c=5), bound by withComponentInputBinding;
   * undefined = new chat. A query parameter (not /chat/5) keeps this same component alive when
   * a new conversation gets its id, so nothing reloads or flickers.
   */
  readonly c = input<string>();
  /** Developer details on/off. A signal so tests can switch it off like a production build. */
  readonly dev = signal(isDevMode());

  /** What the collection covers (null until loaded; the chat works without it). */
  readonly profile = signal<AssistantProfile | null>(null);
  /**
   * The open conversation's collection (null for a new chat). A conversation keeps the
   * collection it started in; a new chat uses the one picked on the welcome screen.
   */
  private readonly openCollection = signal<{ id: number; name: string } | null>(null);
  /** The collection this chat searches: the open conversation's, else the selected one. */
  readonly collectionId = computed(
    () => this.openCollection()?.id ?? this.collections.selectedId(),
  );
  /** The picker's groups: the user's private collections and the shared ones. */
  readonly mine = computed(() => this.collections.list().filter((c) => c.is_private));
  readonly shared = computed(() => this.collections.list().filter((c) => !c.is_private));
  readonly collectionName = computed(
    () => this.openCollection()?.name ?? this.collections.selected()?.name ?? '',
  );

  readonly messages = signal<ChatMessage[]>([]);
  readonly draft = signal('');
  readonly pending = signal(false);
  /**
   * "Analyse all documents" for the next question (per question: off again after sending).
   * Reads every document of the collection - for comparisons and overviews; signed-in users
   * and admins only (it costs several AI calls).
   */
  readonly analyse = signal(false);
  /** The question being answered is an analysis (expected to take longer). */
  readonly analysing = signal(false);
  private readonly keyStore = inject(ApiKeyStore);
  readonly canAnalyse = computed(() => !!this.keyStore.session() || this.keyStore.isAdmin());
  /** Seconds since the question was sent (shown while waiting, so a slow answer isn't "frozen"). */
  readonly waited = signal(0);
  private waitTimer?: ReturnType<typeof setInterval>;
  readonly error = signal('');
  /** A short note above the chat, e.g. why a linked conversation could not be opened. */
  readonly notice = signal('');
  /**
   * Questions that got no answer (temporary message id -> why). They stay in the chat with
   * Retry / Edit instead of vanishing - the server saves a question only after it is answered.
   */
  readonly failed = signal<Map<number, string>>(new Map());
  readonly title = signal('');
  /** Which source passage is open, as "messageId:number" ('' = none). */
  readonly openSource = signal('');
  /** The passage number pointed at in the Sources popup: previewed at the popup's bottom. */
  readonly hovered = signal<Source | null>(null);
  readonly copiedId = signal<number | null>(null);

  /** Id of the newest assistant message: only it shows topic suggestions after a refusal. */
  readonly lastAnswerId = computed(
    () => [...this.messages()].reverse().find((m) => m.role === 'assistant')?.id ?? null,
  );

  private conversationId: number | null = null;
  /**
   * Counts conversation switches. A reply that arrives after the user opened another chat (or
   * "New chat") belongs to the old screen: it is saved on the server but must not be shown here.
   */
  private view = 0;
  private readonly thread = viewChild<ElementRef<HTMLElement>>('thread');

  constructor() {
    // Leaving the chat page (e.g. to Settings) while waiting: this page must ignore the answer
    // (it would otherwise jump back to the chat). The store keeps the question; the next chat
    // page shows it. Bumping `view` makes the waiting callbacks stand down, as for a chat switch.
    inject(DestroyRef).onDestroy(() => {
      this.view++;
      this.stopWaiting();
    });
    // Topics and example questions of the collection this chat searches (reloaded when the user
    // picks another collection or opens a chat from another one).
    effect(() => {
      const id = this.collectionId();
      untracked(() =>
        this.api.assistant(id).subscribe({
          next: (p) => this.profile.set(p),
          error: () => this.profile.set(null), // optional help text; never block the chat
        }),
      );
    });
    // Signed in, signed out, session expired, or another account: the open chat belonged to the
    // previous person (guest chats stay with the guest, account chats with the account), so start
    // a fresh one (found in testing: their messages stayed on screen).
    const keys = inject(ApiKeyStore);
    let signedIn = keys.session()?.token ?? null;
    effect(() => {
      const now = keys.session()?.token ?? null;
      untracked(() => {
        if (now !== signedIn) {
          this.startOver();
        }
        signedIn = now;
      });
    });
    // Load the conversation whenever the URL's id changes (or start fresh for /chat).
    effect(() => {
      const raw = this.c();
      untracked(() => this.open(raw ? Number(raw) : null));
    });
  }

  private open(id: number | null): void {
    if (id !== null && id === this.conversationId) {
      return; // we just created it ourselves: the messages are already on screen
    }
    this.conversationId = id;
    this.view++;
    this.pending.set(false); // a question still running elsewhere doesn't block this chat
    this.stopWaiting();
    this.error.set('');
    if (id !== null) {
      this.notice.set(''); // (kept on the fresh chat a 404 sends the user to - see below)
    }
    this.failed.set(new Map());
    this.openSource.set('');
    if (id === null) {
      this.openCollection.set(null);
      this.messages.set([]);
      this.title.set('');
      this.resume(null);
      return;
    }
    this.api.conversation(id).subscribe({
      next: (conv) => {
        this.messages.set(conv.messages);
        this.title.set(conv.title);
        this.openCollection.set({ id: conv.collection_id, name: conv.collection });
        this.resume(id);
        this.scrollToEnd();
      },
      error: (err) => {
        if (err?.status !== 404) {
          this.error.set(describeError(err));
          return;
        }
        // Not this person's chat (another account's link, or the URL kept after signing out) or
        // deleted. Keeping the dead id would send every new question into it and fail (found in
        // testing), so continue in a fresh chat with a clean URL instead.
        this.conversationId = null;
        this.notice.set(
          "That conversation isn't available (it belongs to another account or was deleted). " +
            'You are in a new chat.',
        );
        this.router.navigate(['/chat'], { replaceUrl: true }); // replace: "Back" won't return
      },
    });
  }

  /** The user's question that an assistant message answered (for "Ask in <collection>"). */
  questionBefore(answer: ChatMessage): string {
    const list = this.messages();
    for (let i = list.indexOf(answer) - 1; i >= 0; i--) {
      if (list[i].role === 'user') {
        return list[i].content;
      }
    }
    return '';
  }

  /** "Ask in HR policies": the same question in a new chat in that collection. */
  askIn(collectionId: number, question: string): void {
    this.collections.select(collectionId);
    this.open(null); // a fresh chat now (searches the selected collection)
    this.router.navigate(['/chat']);
    if (question) {
      this.send(question);
    }
  }

  /** A fresh "New chat" screen (and URL) for the next person. */
  private startOver(): void {
    this.open(null);
    this.router.navigate(['/chat']);
  }

  /** Coming back to a chat whose question is still being answered (or failed meanwhile). */
  private resume(id: number | null): void {
    const waiting = this.conversations.pendingFor(id);
    if (waiting) {
      this.attach(waiting);
    }
  }

  send(text?: string): void {
    const message = (text ?? this.draft()).trim();
    if (!message || this.pending()) {
      return;
    }
    this.draft.set('');
    this.error.set('');
    this.notice.set('');
    this.showAllTopics.set(false); // a new refusal starts with the short topic list again
    // The store sends it, so the request survives leaving this page (see InFlight).
    const analyse = this.analyse() && this.canAnalyse();
    this.analyse.set(false);
    this.attach(this.conversations.ask(message, this.conversationId, this.collectionId(), analyse));
  }

  /**
   * Show a question that is being answered (just sent, or found again after coming back) and
   * wait for its answer. The question appears at once with its temporary id.
   */
  private attach(waiting: InFlight): void {
    const temp = waiting.question;
    this.analysing.set(waiting.analyse);
    this.messages.update((list) => [...list, temp]);
    if (waiting.error) {
      this.markFailed(waiting, waiting.error); // failed while the user was on another page
      return;
    }
    this.pending.set(true);
    this.startWaiting(waiting.startedAt);
    this.scrollToEnd();
    const view = this.view; // the screen this question was asked on

    waiting.answer$.subscribe({
      next: (res) => {
        if (view !== this.view) {
          return; // saved in its own chat; shown there when the user opens it again
        }
        this.conversations.forget(waiting);
        const saved = new Set([res.user_message.id, res.assistant_message.id]);
        this.messages.update((list) => [
          // Without the temporary question - and without the saved pair, in case the chat was
          // reloaded from the server after the answer had already been stored there.
          ...list.filter((m) => m !== temp && !saved.has(m.id)),
          res.user_message,
          res.assistant_message,
        ]);
        this.pending.set(false);
        this.stopWaiting();
        this.title.set(res.title);
        if (!this.openCollection()) {
          const id = res.collection_id;
          this.openCollection.set({ id, name: this.collections.nameOf(id) });
        }
        const isNew = this.conversationId === null;
        this.conversationId = res.conversation_id;
        if (isNew) {
          // Give the new conversation its own URL (shareable, survives reload).
          this.router.navigate(['/chat'], { queryParams: { c: res.conversation_id } });
        }
        this.scrollToEnd();
      },
      error: (err) => {
        if (view !== this.view) {
          return; // the user has moved on; shown when they open this chat again
        }
        this.markFailed(waiting, describeError(err));
      },
    });
  }

  /** The question stays in the chat, marked as not answered, with Retry and Edit. */
  private markFailed(waiting: InFlight, reason: string): void {
    this.conversations.forget(waiting);
    this.failed.update((map) => new Map(map).set(waiting.question.id, reason));
    this.analyse.set(waiting.analyse); // Retry / Edit ask it the same way again
    this.pending.set(false);
    this.stopWaiting();
  }

  /** Under an analysed answer: "Analysis: 5 documents read, 2 without anything on this". */
  reading(m: ChatMessage): string {
    return m.route?.startsWith('analyse: ') ? 'Analysis: ' + m.route.slice(9) : '';
  }

  /** Send a failed question again, exactly as typed. */
  retry(message: ChatMessage): void {
    this.dropFailed(message);
    this.send(message.content);
  }

  /** Put a failed question back into the input box to change it before sending. */
  editFailed(message: ChatMessage): void {
    this.dropFailed(message);
    this.draft.set(message.content);
  }

  private dropFailed(message: ChatMessage): void {
    this.messages.update((list) => list.filter((m) => m !== message));
    this.failed.update((map) => {
      const next = new Map(map);
      next.delete(message.id);
      return next;
    });
  }

  /** Count the seconds since the question was sent (also when coming back to it later). */
  private startWaiting(startedAt: number): void {
    this.stopWaiting();
    const tick = () => this.waited.set(Math.floor((Date.now() - startedAt) / 1000));
    tick();
    this.waitTimer = setInterval(tick, 1000);
  }

  private stopWaiting(): void {
    clearInterval(this.waitTimer);
  }

  /** Message time ("7:31 PM"); the day is shown by the separator above (see dayLabel). */
  when(iso: string): string {
    return new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }

  /** True if message i starts a new day, so a day separator goes above it (like Teams). */
  newDay(i: number): boolean {
    const list = this.messages();
    return (
      i === 0 ||
      new Date(list[i - 1].created_at).toDateString() !==
        new Date(list[i].created_at).toDateString()
    );
  }

  /** Day separator text: "Today, October 13", "Yesterday, October 12", "Monday, October 11"
   *  (+ year if not this year). */
  dayLabel(iso: string, now = new Date()): string {
    const d = new Date(iso);
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    const year = d.getFullYear() !== now.getFullYear() ? { year: 'numeric' as const } : {};
    const date = d.toLocaleDateString([], { month: 'long', day: 'numeric', ...year });
    if (d.toDateString() === now.toDateString()) return `Today, ${date}`;
    if (d.toDateString() === yesterday.toDateString()) return `Yesterday, ${date}`;
    return d.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric', ...year });
  }

  /**
   * Follow-up suggestions under the newest answer (like "Related" in search/chat products):
   * the example questions of the topics whose documents the answer cited, minus the question
   * just asked. No model call - topics come from GET /assistant, sources from the answer.
   */
  /** Topic chips shown after a refusal before "+N more" (a wall of 13 chips is hard to scan). */
  static readonly TOPIC_LIMIT = 6;
  readonly showAllTopics = signal(false);
  readonly visibleTopics = computed(() => {
    const topics = this.profile()?.topics ?? [];
    return this.showAllTopics() ? topics : topics.slice(0, Chat.TOPIC_LIMIT);
  });
  readonly hiddenTopics = computed(
    () => (this.profile()?.topics?.length ?? 0) - this.visibleTopics().length,
  );

  /**
   * Welcome screen: hand-written topics (assistant.json) as cards, every other document as a
   * compact chip in "More documents" - 100 uploads must not mean 100 cards. A knowledge base with
   * only uploads shows its first few as cards.
   */
  static readonly DOC_LIMIT = 8;
  readonly showAllDocs = signal(false);
  readonly topicCards = computed(() => {
    const topics = this.profile()?.topics ?? [];
    const written = topics.filter((t) => !t.generated);
    return written.length ? written : topics.slice(0, Chat.TOPIC_LIMIT);
  });
  readonly otherDocs = computed(() => {
    const cards = new Set(this.topicCards());
    return (this.profile()?.topics ?? []).filter((t) => !cards.has(t));
  });
  readonly visibleDocs = computed(() =>
    this.showAllDocs() ? this.otherDocs() : this.otherDocs().slice(0, Chat.DOC_LIMIT),
  );

  readonly related = computed<string[]>(() => {
    const list = this.messages();
    const answer = list.at(-1);
    if (answer?.role !== 'assistant' || !answer.answerable || !answer.sources?.length) return [];
    const cited = new Set(answer.sources.map((s) => this.fileName(s.document)));
    const norm = (q: string) =>
      q
        .toLowerCase()
        .replace(/[\W_]+/g, ' ')
        .trim();
    const asked = new Set(
      [list.at(-2)?.content, answer.standalone_question].filter((q) => !!q).map((q) => norm(q!)),
    );
    const questions = (this.profile()?.topics ?? [])
      .filter((t) => t.documents?.some((d) => cited.has(d)))
      .flatMap((t) => t.examples)
      .filter((q) => !asked.has(norm(q)));
    return [...new Set(questions)].slice(0, 3);
  });

  /** Enter sends; Shift+Enter makes a new line (like common chat apps). */
  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  rate(message: ChatMessage, rating: 1 | -1): void {
    const next = message.feedback === rating ? null : rating; // clicking again clears it
    this.api.feedback(message.id, next).subscribe({
      next: () =>
        this.messages.update((list) =>
          list.map((m) => (m.id === message.id ? { ...m, feedback: next } : m)),
        ),
      error: (err) => this.error.set(describeError(err)),
    });
  }

  copy(message: ChatMessage): void {
    navigator.clipboard?.writeText(message.content);
    this.copiedId.set(message.id);
    setTimeout(() => this.copiedId.set(null), 1500);
  }

  toggleSource(message: ChatMessage, source: Source): void {
    const key = `${message.id}:${source.number}`;
    this.openSource.set(this.openSource() === key ? '' : key);
  }

  /** Answer text split into plain text and citation markers, so "[2]" can be a link to source 2.
   * .NET analogy: Regex.Split with a capture group - the captured numbers stay in the result. */
  segments(message: ChatMessage): { text: string; cite?: Source }[] {
    return message.content.split(/(\[\d+\])/).map((part) => {
      const n = /^\[(\d+)\]$/.exec(part)?.[1];
      const cite = n ? message.sources?.find((s) => s.number === Number(n)) : undefined;
      return cite ? { text: part, cite } : { text: part };
    });
  }

  /**
   * Tooltip for a source: readable document name + the start of the passage (tells same-file
   * sources apart). No server path (an internal detail) and no Markdown marks like "## " or "**".
   */
  preview(source: Source): string {
    const name = `${this.docTitle(source.document)} (${this.docType(source.document)})`;
    return `${name}\n${this.snippet(source, 160)}`;
  }

  /** Start of a passage for the Sources popup's preview box. */
  excerpt(source: Source): string {
    return this.snippet(source, 220);
  }

  /**
   * A cited answer's documents, each with its passages, in citation order (one line per
   * document in the Sources popup: "RAG design guide · 1 2 3").
   */
  docsOf(message: ChatMessage): { document: string; sources: Source[] }[] {
    const docs = new Map<string, Source[]>(); // a Map keeps insertion order
    for (const s of this.orderedSources(message)) {
      docs.set(s.document, [...(docs.get(s.document) ?? []), s]);
    }
    return [...docs].map(([document, sources]) => ({ document, sources }));
  }

  /** The passage as plain text, shortened: no Markdown marks or table pipes. */
  private snippet(source: Source, max: number): string {
    const text = source.text
      .replace(/^#+\s*/gm, '') // heading marks at the start of a line
      .replace(/\*\*/g, '') // bold marks
      .replace(/^[\s|:-]*\|[\s|:-]*$/gm, '') // table separator rows like |---|---|
      .replace(/\s*\|\s*/g, ' · ') // table cells: "| a | b |" -> "a · b"
      .replace(/\s+/g, ' ')
      .replace(/^[\s·]+|[\s·]+$/g, '') // no dot at the start or end
      .replace(/(\s·)+/g, ' ·'); // no doubled dots between rows
    return text.length > max ? text.slice(0, max) + '…' : text;
  }

  sourceText(message: ChatMessage): Source | undefined {
    const [id, num] = this.openSource().split(':').map(Number);
    return id === message.id ? message.sources?.find((s) => s.number === num) : undefined;
  }

  fileName(path: string): string {
    return path.split('/').pop() ?? path;
  }

  /**
   * The number shown for a source: 1, 2, ... in the order the answer first cites them. The stored
   * number points into the passages the model saw, so an answer citing only passage 2 would show
   * a lone "2" - the display renumbers, the stored answer stays unchanged.
   */
  label(message: ChatMessage, source: Source): number {
    const known = new Set((message.sources ?? []).map((s) => s.number));
    const order = [...new Set(message.content.match(/\[\d+\]/g) ?? [])]
      .map((c) => Number(c.slice(1, -1)))
      .filter((n) => known.has(n)); // a number without a source is shown as plain text
    const i = order.indexOf(source.number);
    return i >= 0 ? i + 1 : order.length + 1 + (message.sources ?? []).indexOf(source);
  }

  /** Sources in the order of their display numbers. */
  orderedSources(message: ChatMessage): Source[] {
    return [...(message.sources ?? [])].sort(
      (a, b) => this.label(message, a) - this.label(message, b),
    );
  }

  /**
   * "data/uploads/model_selection_policy_v1.docx" -> "Model selection policy v1";
   * known abbreviations keep their capitals: "llmops_incident_runbook.md" -> "LLMOps incident runbook".
   */
  docTitle(path: string): string {
    const words = this.fileName(path)
      .replace(/\.[^.]+$/, '')
      .split(/[_\-\s]+/)
      .filter((w) => w)
      .map((w) => ACRONYMS[w.toLowerCase()] ?? w);
    const name = words.join(' ');
    return name.charAt(0).toUpperCase() + name.slice(1);
  }

  /** "PDF", "DOCX", "MD", "TXT". */
  docType(path: string): string {
    return (/\.([^./]+)$/.exec(path)?.[1] ?? '').toUpperCase();
  }

  docIcon(path: string): string {
    const type = this.docType(path);
    return type === 'PDF' ? 'picture_as_pdf' : type === 'DOCX' ? 'description' : 'article';
  }

  private scrollToEnd(): void {
    // After Angular has rendered the new messages, scroll the thread to the bottom.
    setTimeout(() => {
      const el = this.thread()?.nativeElement;
      el?.scrollTo?.({ top: el.scrollHeight, behavior: 'smooth' }); // ?.() = only if supported
    });
  }
}
