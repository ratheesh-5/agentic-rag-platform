import { BreakpointObserver } from '@angular/cdk/layout';
import { Component, computed, effect, inject, signal, untracked } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { MatButtonModule } from '@angular/material/button';
import { MatDialog } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatMenuModule } from '@angular/material/menu';
import { MatSidenav, MatSidenavModule } from '@angular/material/sidenav';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { filter, map } from 'rxjs';

import { AdminSignIn } from './core/admin-sign-in';
import { ApiService } from './core/api';
import { ApiKeyStore } from './core/api-key.store';
import { ConfirmService } from './core/confirm';
import { ConversationsStore } from './core/conversations.store';
import { GoogleSignIn } from './core/google-sign-in';

/**
 * App shell, laid out like common chat products: a sidebar with "New chat" and recent
 * conversations (plus the Admin pages once an admin signed in); a toolbar with "Sign in with
 * Google" (or the account menu once signed in; "Admin sign in" with a key when the server has
 * no Google client id); the routed page on the right. Anyone can chat right away as a guest.
 * On narrow screens the sidebar becomes a slide-over menu.
 */
@Component({
  selector: 'app-root',
  imports: [
    GoogleSignIn,
    MatButtonModule,
    MatIconModule,
    MatMenuModule,
    MatSidenavModule,
    MatToolbarModule,
    MatTooltipModule,
    RouterLink,
    RouterLinkActive,
    RouterOutlet,
  ],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  readonly keyStore = inject(ApiKeyStore);
  readonly conversations = inject(ConversationsStore);
  private readonly api = inject(ApiService);
  private readonly router = inject(Router);
  private readonly dialog = inject(MatDialog);
  private readonly confirm = inject(ConfirmService);

  /** True on phones/narrow windows: sidebar slides over the content instead of staying open. */
  readonly narrow = toSignal(
    inject(BreakpointObserver)
      .observe('(max-width: 800px)')
      .pipe(map((state) => state.matches)),
    { initialValue: false },
  );

  /** Id of the conversation in the URL (/chat?c=5), to highlight it in the sidebar. */
  readonly activeId = toSignal(
    this.router.events.pipe(
      filter((e) => e instanceof NavigationEnd),
      map(() => Number(this.router.parseUrl(this.router.url).queryParams['c']) || null),
    ),
    { initialValue: null },
  );

  /** Google sign-in client id from the API ('' = off: the admin-key dialog is offered). */
  readonly googleClientId = signal('');

  constructor() {
    // Sign-out, an expired session or a removed admin key: leave pages this person may no
    // longer use (Admin pages need an admin; My documents needs a sign-in) - found in testing:
    // after signing out the My documents page stayed open with buttons that no longer worked.
    effect(() => {
      const admin = this.keyStore.isAdmin();
      const signedIn = admin || !!this.keyStore.session();
      untracked(() => {
        const url = this.router.url;
        if ((!admin && url.startsWith('/admin')) || (!signedIn && url.startsWith('/documents'))) {
          this.router.navigate(['/chat']);
        }
      });
    });
    this.api.authConfig().subscribe({
      next: (cfg) => this.googleClientId.set(cfg.google_client_id),
      error: () => this.googleClientId.set(''), // API down: guests can still try to chat
    });
    // Refresh role and profile of a stored session (an expired one is dropped by the
    // interceptor's 401 handling).
    if (this.keyStore.session()) {
      this.api.me().subscribe({ next: (me) => this.keyStore.update(me), error: () => {} });
    }
  }

  /** Chats from 2+ collections: show each chat's collection under its title. */
  readonly mixedCollections = computed(
    () => new Set(this.conversations.list().map((c) => c.collection_id)).size > 1,
  );

  readonly adminPages = [
    { path: '/admin/documents', label: 'Documents', icon: 'description' },
    { path: '/admin/evaluation', label: 'Evaluation', icon: 'analytics' },
    { path: '/admin/usage', label: 'Usage', icon: 'monitoring' },
    { path: '/admin/settings', label: 'Settings', icon: 'settings' },
  ];

  signIn(): void {
    this.dialog.open(AdminSignIn, { width: '420px', maxWidth: '92vw' });
  }

  signOut(): void {
    if (this.keyStore.session()) {
      this.api.logout().subscribe({ error: () => {} }); // ends the session on the server
      this.keyStore.clearSession();
    }
    this.keyStore.clear(); // pages the person may no longer use close (constructor effect)
  }

  /** On narrow screens, close the slide-over menu after picking something. */
  closeOnNarrow(nav: MatSidenav): void {
    if (this.narrow()) {
      nav.close();
    }
  }

  /** "Delete my account": private collections, chats and the sign-in - after a red check. */
  deleteAccount(): void {
    this.confirm
      .ask({
        title: 'Delete your account?',
        message:
          'Your private collections with their documents, all your chats and your sign-in are ' +
          'removed for good. Shared collections are not affected.',
        action: 'Delete account',
        icon: 'person_remove',
        danger: true,
      })
      .subscribe((yes) => {
        if (!yes) {
          return;
        }
        this.api.deleteAccount().subscribe({
          next: () => {
            this.keyStore.clearSession();
            this.router.navigate(['/chat']);
          },
        });
      });
  }

  /** "Delete my data" (privacy): all of this browser's conversations at once. */
  deleteAllConversations(): void {
    this.confirm
      .ask({
        title: 'Delete all conversations?',
        message:
          this.conversations.list().length === 1
            ? 'Your conversation and its answers are removed for good.'
            : `All ${this.conversations.list().length} conversations and their answers are removed for good.`,
        action: 'Delete all',
        icon: 'delete_sweep',
        danger: true,
      })
      .subscribe((yes) => {
        if (yes) {
          this.conversations.removeAll();
          this.router.navigate(['/chat']);
        }
      });
  }

  deleteConversation(event: Event, id: number, title: string): void {
    event.preventDefault(); // the button sits inside the conversation link
    event.stopPropagation();
    this.confirm
      .ask({
        title: 'Delete this conversation?',
        message: `"${title}" and its answers are removed for good.`,
        action: 'Delete',
        icon: 'delete',
        danger: true,
      })
      .subscribe((yes) => {
        if (!yes) {
          return;
        }
        this.conversations.remove(id);
        if (this.activeId() === id) {
          this.router.navigate(['/chat']);
        }
      });
  }
}
