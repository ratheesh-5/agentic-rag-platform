import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';

import { MatCheckboxModule } from '@angular/material/checkbox';

import { ApiService } from './api';
import { ApiKeyStore } from './api-key.store';
import { describeError } from './errors';
import { Collection } from './models';

/**
 * Create a collection, or rename one (when opened with an existing collection). Saves through
 * the API and closes with the saved collection; a taken name (409) is shown in the dialog.
 */
@Component({
  selector: 'app-collection-dialog',
  imports: [
    FormsModule,
    MatButtonModule,
    MatCheckboxModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
  ],
  template: `
    <h2 mat-dialog-title>{{ editing ? 'Rename collection' : 'New collection' }}</h2>
    <mat-dialog-content>
      <p class="hint">
        A collection groups documents on one subject. A chat searches only one collection.
        @if (!editing && !keys.isAdmin()) {
          <b>Private:</b> only you can see and search it.
        }
      </p>
      <mat-form-field appearance="outline" class="full">
        <mat-label>Name</mat-label>
        <input
          matInput
          maxlength="60"
          cdkFocusInitial
          [ngModel]="name()"
          (ngModelChange)="name.set($event); error.set('')"
          (keydown.enter)="save()"
        />
      </mat-form-field>
      <mat-form-field appearance="outline" class="full">
        <mat-label>Description (optional)</mat-label>
        <input
          matInput
          maxlength="300"
          [ngModel]="description()"
          (ngModelChange)="description.set($event)"
        />
      </mat-form-field>
      @if (!editing?.is_default) {
        <mat-form-field appearance="outline" class="full">
          <mat-label>Similarity threshold (optional)</mat-label>
          <input
            matInput
            type="number"
            min="0.3"
            max="0.95"
            step="0.01"
            placeholder="automatic"
            [ngModel]="threshold()"
            (ngModelChange)="threshold.set($event)"
          />
          <mat-hint>Leave empty: measured automatically after every upload (recommended).</mat-hint>
        </mat-form-field>
      }
      @if (!editing && keys.isAdmin() && keys.user()) {
        <mat-checkbox [ngModel]="private()" (ngModelChange)="private.set($event)">
          Private - only I can see it (otherwise shared with everyone)
        </mat-checkbox>
      }
      @if (error()) {
        <div class="error-text" role="alert">{{ error() }}</div>
      }
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button mat-dialog-close>Cancel</button>
      <button mat-flat-button (click)="save()" [disabled]="!name().trim() || busy()">
        {{ editing ? 'Save' : 'Create' }}
      </button>
    </mat-dialog-actions>
  `,
  styles: `
    .hint {
      margin-top: 0;
    }
    .full {
      width: 100%;
    }
    .error-text {
      color: var(--mat-sys-error);
      font: var(--mat-sys-body-small);
    }
  `,
})
export class CollectionDialog {
  private readonly api = inject(ApiService);
  private readonly dialog = inject(MatDialogRef<CollectionDialog, Collection>);
  readonly keys = inject(ApiKeyStore);
  /** The collection to rename, or null to create a new one. */
  readonly editing = inject<Collection | null>(MAT_DIALOG_DATA, { optional: true });

  readonly name = signal(this.editing?.name ?? '');
  readonly description = signal(this.editing?.description ?? '');
  /** Manual threshold; '' / null = automatic (the normal case). */
  readonly threshold = signal<number | string | null>(this.editing?.threshold_override ?? null);
  /** Admins only: create a private collection instead of a shared one. */
  readonly private = signal(false);
  readonly error = signal('');
  readonly busy = signal(false);

  save(): void {
    const raw = this.threshold();
    const body = {
      name: this.name().trim(),
      description: this.description().trim(),
      threshold: raw === null || raw === '' ? null : Number(raw),
      ...(this.private() ? { private: true } : {}),
    };
    if (!body.name || this.busy()) {
      return;
    }
    this.busy.set(true);
    const request = this.editing
      ? this.api.updateCollection(this.editing.id, body)
      : this.api.createCollection(body);
    request.subscribe({
      next: (saved) => this.dialog.close(saved),
      error: (err) => {
        this.busy.set(false);
        this.error.set(
          err?.status === 409 ? 'A collection with this name already exists.' : describeError(err),
        );
      },
    });
  }
}
