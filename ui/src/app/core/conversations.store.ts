import { Injectable, effect, inject, signal, untracked } from '@angular/core';
import { Observable, shareReplay } from 'rxjs';

import { ApiService } from './api';
import { ApiKeyStore } from './api-key.store';
import { describeError } from './errors';
import { ChatMessage, ChatResponse, ConversationSummary } from './models';

/**
 * A question that is still being answered. It lives here, not in the chat page: the page is
 * destroyed when the user opens Settings or Documents while waiting, and an unanswered question
 * is not on the server yet (it is saved together with its answer). Coming back, the chat page
 * finds it here and shows it again with "Thinking…" - or its answer, or its error.
 */
export interface InFlight {
  conversationId: number | null; // null = a new chat (it gets an id with the answer)
  question: ChatMessage; // shown while waiting (temporary negative id)
  startedAt: number; // Date.now() when sent: the "Thinking… N s" counter continues from here
  answer$: Observable<ChatResponse>; // the one shared request (late subscribers get the result)
  analyse: boolean; // asked in Analyse mode (reads every document: slower)
  error?: string; // set when it failed: shown with Retry / Edit when the chat is opened
  key: string; // who asked + which chat (see ConversationsStore.key)
}

/**
 * The caller's conversation list, shared by the sidebar (shows it) and the chat page (adds to
 * it), plus the questions still being answered. One instance for the whole app
 * (providedIn: 'root' = a singleton service, like AddSingleton in .NET).
 */
@Injectable({ providedIn: 'root' })
export class ConversationsStore {
  private readonly api = inject(ApiService);
  private readonly keys = inject(ApiKeyStore);
  /** In-progress questions per person and conversation ("new" for a chat without an id yet). */
  private readonly inFlight = new Map<string, InFlight>();

  readonly list = signal<ConversationSummary[]>([]);

  constructor() {
    // Conversations belong to the signed-in user, else to this browser's guest id. Reload now
    // and whenever the sign-in changes (signed in, signed out, or the session expired).
    effect(() => {
      this.keys.session(); // the only dependency
      untracked(() => this.reload());
    });
  }

  reload(): void {
    this.api.conversations().subscribe({
      next: (list) => this.list.set(list),
      error: () => this.list.set([]), // the page shows the error; the sidebar just stays empty
    });
  }

  remove(id: number): void {
    this.api.deleteConversation(id).subscribe({ next: () => this.reload() });
  }

  removeAll(): void {
    this.api.deleteAllConversations().subscribe({ next: () => this.reload() });
  }

  /**
   * Send a question. The request runs here, independent of any page: it finishes (and is saved
   * on the server) even if the user navigates away. shareReplay(1) = one HTTP call that every
   * subscriber shares; a page that subscribes after it finished still gets the answer.
   */
  ask(
    message: string,
    conversationId: number | null,
    collectionId?: number,
    analyse = false,
  ): InFlight {
    const key = this.key(conversationId);
    const entry: InFlight = {
      key,
      conversationId,
      question: {
        id: -Date.now(),
        role: 'user',
        content: message,
        created_at: new Date().toISOString(),
      },
      startedAt: Date.now(),
      analyse,
      answer$: this.api
        .chat({
          ...(conversationId !== null
            ? { message, conversation_id: conversationId } // keeps its own collection
            : { message, collection_id: collectionId }),
          ...(analyse ? { analyse } : {}),
        })
        .pipe(shareReplay(1)),
    };
    this.inFlight.set(key, entry);
    entry.answer$.subscribe({
      next: () => {
        // A new chat now has its own id on the server and appears in the sidebar; nothing is
        // left to show on the "New chat" screen. (For an existing chat the entry stays until the
        // chat page has shown the answer - see forget.)
        if (conversationId === null) {
          this.forget(entry);
        }
        this.reload(); // new title / new order in the sidebar
      },
      error: (err) => (entry.error = describeError(err)), // kept until the chat shows it
    });
    return entry;
  }

  /** The question still being answered (or just failed) in this conversation, if any. */
  pendingFor(conversationId: number | null): InFlight | undefined {
    return this.inFlight.get(this.key(conversationId));
  }

  /** The chat page has shown the answer or the error: nothing left to remember. */
  forget(entry: InFlight): void {
    if (this.inFlight.get(entry.key) === entry) {
      this.inFlight.delete(entry.key);
    }
  }

  /**
   * Waiting questions belong to the person who asked: after signing out (or in as someone else)
   * the next person never sees them. The session token identifies a signed-in user; "guest" is
   * this browser's guest (an admin key doesn't change whose chats these are).
   */
  private key(conversationId: number | null): string {
    const who = this.keys.session()?.token ?? 'guest';
    return `${who}:${conversationId ?? 'new'}`;
  }
}
