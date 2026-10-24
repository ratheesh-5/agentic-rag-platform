import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';

import { ApiService } from './api';
import { ApiKeyStore } from './api-key.store';
import { describeError } from './errors';

/**
 * "Admin sign in" dialog: the admin pastes one of API_KEYS. The key is checked with GET /me
 * before it is stored, so a typo never "half signs in". Closes with true on success.
 */
@Component({
  selector: 'app-admin-sign-in',
  imports: [FormsModule, MatButtonModule, MatDialogModule, MatFormFieldModule, MatInputModule],
  template: `
    <h2 mat-dialog-title>Admin sign in</h2>
    <mat-dialog-content>
      <p class="hint">Enter an admin key (one of <code>API_KEYS</code> in the backend's .env).</p>
      <mat-form-field appearance="outline" class="full">
        <mat-label>Admin key</mat-label>
        <input
          matInput
          type="password"
          autocomplete="off"
          cdkFocusInitial
          [ngModel]="key()"
          (ngModelChange)="key.set($event); error.set('')"
          (keydown.enter)="signIn()"
        />
      </mat-form-field>
      @if (error()) {
        <div class="error-text" role="alert">{{ error() }}</div>
      }
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button mat-dialog-close>Cancel</button>
      <button mat-flat-button (click)="signIn()" [disabled]="!key().trim() || busy()">
        Sign in
      </button>
    </mat-dialog-actions>
  `,
  styles: `
    .hint {
      margin-top: 0;
    }
    .error-text {
      color: var(--mat-sys-error);
      font: var(--mat-sys-body-small);
    }
  `,
})
export class AdminSignIn {
  private readonly api = inject(ApiService);
  private readonly store = inject(ApiKeyStore);
  private readonly dialog = inject(MatDialogRef<AdminSignIn, boolean>);

  readonly key = signal('');
  readonly error = signal('');
  readonly busy = signal(false);

  signIn(): void {
    const key = this.key().trim();
    if (!key || this.busy()) {
      return;
    }
    this.busy.set(true);
    this.api.me(key).subscribe({
      next: (me) => {
        this.busy.set(false);
        if (me.role === 'admin') {
          this.store.set(key);
          this.dialog.close(true);
        } else {
          this.error.set('This key is not an admin key.');
        }
      },
      error: (err) => {
        this.busy.set(false);
        this.error.set(err?.status === 401 ? 'Invalid admin key.' : describeError(err));
      },
    });
  }
}
