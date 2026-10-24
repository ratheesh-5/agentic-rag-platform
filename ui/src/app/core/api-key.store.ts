import { Injectable, computed, signal } from '@angular/core';

import { Me, SignInResponse } from './models';

/** What the browser keeps after "Sign in with Google": our session token and who the user is. */
export type Session = Pick<SignInResponse, 'token' | 'expires_at' | 'role' | 'user'>;

/**
 * Who this browser is to the API:
 *
 * - Guest id: a random id generated once and kept in localStorage. Sent as X-Guest-Id on every
 *   request, so the API knows which conversations are "mine". Anyone can chat as a guest.
 * - Session (Sign in with Google): our own session token from POST /auth/google, kept in
 *   localStorage so the user stays signed in for SESSION_DAYS (7) across browser restarts.
 *   Sent as "Authorization: Bearer ..."; chats then belong to the account (every device), and
 *   users whose email is in ADMIN_EMAILS get the Admin pages. Sign-out ends it on the server.
 * - Admin key (optional fallback when Google sign-in is not configured): one of API_KEYS,
 *   entered via "Admin sign in". Sent as X-API-Key. Kept in sessionStorage: it survives a reload
 *   but is gone when the tab closes.
 *
 * A browser app can't hide values from its own user, which is fine: the guest id only protects
 * the guest's own chat history, the session token only the user's own account, and the CSP keeps
 * other sites' scripts out.
 */
@Injectable({ providedIn: 'root' })
export class ApiKeyStore {
  private static readonly storageKey = 'rag.apiKey';
  private static readonly guestKey = 'rag.guestId';
  private static readonly sessionKey = 'rag.session';

  /** Admin key ('' = none). A signal, so the UI updates when it changes. */
  readonly key = signal<string>(sessionStorage.getItem(ApiKeyStore.storageKey) ?? '');
  /** The Google sign-in session, or null (guest / admin key only). */
  readonly session = signal<Session | null>(ApiKeyStore.loadSession());
  /** Admin pages: an admin user (email in ADMIN_EMAILS) or an accepted admin key. */
  readonly isAdmin = computed(() => !!this.key() || this.session()?.role === 'admin');
  /** The signed-in user (name, email, photo), or null. */
  readonly user = computed(() => this.session()?.user ?? null);

  /** This browser's guest id (created on first use). */
  readonly guestId: string = ApiKeyStore.loadGuestId();

  set(key: string): void {
    const trimmed = key.trim();
    sessionStorage.setItem(ApiKeyStore.storageKey, trimmed);
    this.key.set(trimmed);
  }

  clear(): void {
    sessionStorage.removeItem(ApiKeyStore.storageKey);
    this.key.set('');
  }

  setSession(session: Session): void {
    const { token, expires_at, role, user } = session; // store only these fields
    const value: Session = { token, expires_at, role, user };
    localStorage.setItem(ApiKeyStore.sessionKey, JSON.stringify(value));
    this.session.set(value);
  }

  /** Refresh role and profile from GET /me (ADMIN_EMAILS or the Google profile may change). */
  update(me: Me): void {
    const current = this.session();
    if (current && me.user && me.role !== 'guest') {
      this.setSession({ ...current, role: me.role, user: me.user });
    }
  }

  clearSession(): void {
    localStorage.removeItem(ApiKeyStore.sessionKey);
    this.session.set(null);
  }

  private static loadSession(): Session | null {
    try {
      const session = JSON.parse(localStorage.getItem(ApiKeyStore.sessionKey) ?? 'null');
      // Expired sessions are dropped right away (the API would refuse them anyway).
      return session && new Date(session.expires_at) > new Date() ? session : null;
    } catch {
      return null; // unreadable value: start as a guest
    }
  }

  private static loadGuestId(): string {
    let id = localStorage.getItem(ApiKeyStore.guestKey);
    if (!id) {
      id = crypto.randomUUID(); // 122 random bits: unguessable, so nobody else gets this history
      localStorage.setItem(ApiKeyStore.guestKey, id);
    }
    return id;
  }
}
