import {
  AfterViewInit,
  Component,
  ElementRef,
  inject,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';

import { ApiService } from './api';
import { ApiKeyStore } from './api-key.store';
import { describeError } from './errors';
import { SignInResponse } from './models';

/** The parts of Google Identity Services (window.google.accounts.id) this component uses. */
interface GoogleIdentity {
  initialize(options: {
    client_id: string;
    callback: (response: { credential: string }) => void;
  }): void;
  renderButton(parent: HTMLElement, options: Record<string, string | number>): void;
}

const GIS_SCRIPT = 'https://accounts.google.com/gsi/client';
let loading: Promise<GoogleIdentity> | null = null;

/** Load Google's sign-in script once per page (allowed by the CSP in nginx.conf). */
function loadGoogleIdentity(): Promise<GoogleIdentity> {
  loading ??= new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = GIS_SCRIPT;
    script.async = true;
    script.onload = () => {
      const id = (window as unknown as { google?: { accounts: { id: GoogleIdentity } } }).google
        ?.accounts.id;
      if (id) {
        resolve(id);
      } else {
        reject(new Error('Google sign-in did not load'));
      }
    };
    script.onerror = () => {
      loading = null; // allow a retry later
      reject(new Error('Google sign-in could not be loaded'));
    };
    document.head.appendChild(script);
  });
  return loading;
}

/**
 * "Sign in with Google" button (plan B1). Google draws the button and runs the sign-in on its
 * own pages (we never see a password) and hands us an ID token; the API checks it and returns
 * our own session, stored in ApiKeyStore. Shown only when the API reports a client id.
 */
@Component({
  selector: 'app-google-sign-in',
  template: `
    <div #button class="google-button"></div>
    @if (error()) {
      <span class="error-text" role="alert">{{ error() }}</span>
    }
  `,
  styles: `
    :host {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .error-text {
      color: var(--mat-sys-error);
      font: var(--mat-sys-body-small);
    }
  `,
})
export class GoogleSignIn implements AfterViewInit {
  private readonly api = inject(ApiService);
  private readonly store = inject(ApiKeyStore);

  readonly clientId = input.required<string>();
  /** Emitted after a successful sign-in (the session is already stored). */
  readonly signedIn = output<SignInResponse>();
  readonly error = signal('');
  private readonly button = viewChild.required<ElementRef<HTMLElement>>('button');

  ngAfterViewInit(): void {
    loadGoogleIdentity().then(
      (google) => {
        google.initialize({
          client_id: this.clientId(),
          callback: ({ credential }) => this.exchange(credential),
        });
        google.renderButton(this.button().nativeElement, {
          type: 'standard',
          theme: 'filled_blue', // Google's white 'outline' style vanished on the light toolbar
          size: 'medium',
          shape: 'pill',
          text: 'signin_with',
        });
      },
      () => this.error.set('Google sign-in is unavailable.'),
    );
  }

  /** Google's ID token -> our session (POST /auth/google). */
  exchange(credential: string): void {
    this.error.set('');
    this.api.signInWithGoogle(credential).subscribe({
      next: (res) => {
        this.store.setSession(res);
        this.signedIn.emit(res);
      },
      error: (err) => this.error.set(describeError(err)),
    });
  }
}
