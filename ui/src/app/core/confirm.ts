import { Component, Injectable, inject } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MAT_DIALOG_DATA, MatDialog, MatDialogModule } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { Observable, map } from 'rxjs';

/** What a confirmation asks: a short title, one explaining sentence and the action's name. */
export interface ConfirmOptions {
  title: string;
  message: string;
  action: string; // label of the confirm button, e.g. "Delete" or "Switch"
  icon?: string; // Material icon shown before the title
  danger?: boolean; // red confirm button for actions that remove something
}

/**
 * Material confirmation dialog: replaces the browser's plain `confirm()` box, which can't be
 * styled and looks out of place. Opened through `ConfirmService`; closes with true on confirm.
 */
@Component({
  selector: 'app-confirm-dialog',
  imports: [MatButtonModule, MatDialogModule, MatIconModule],
  template: `
    <h2 mat-dialog-title class="title">
      @if (data.icon) {
        <mat-icon [class.danger-icon]="data.danger">{{ data.icon }}</mat-icon>
      }
      {{ data.title }}
    </h2>
    <mat-dialog-content>
      <p class="message">{{ data.message }}</p>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button [mat-dialog-close]="false">Cancel</button>
      <button
        mat-flat-button
        cdkFocusInitial
        [class.danger]="data.danger"
        [mat-dialog-close]="true"
      >
        {{ data.action }}
      </button>
    </mat-dialog-actions>
  `,
  styles: `
    .title {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      font: var(--mat-sys-title-large);
      overflow-wrap: anywhere; // long file names wrap instead of pushing the dialog wider
    }
    // Material puts an invisible ::before strut in the title for text baseline alignment; in a
    // flex row it becomes an extra item and pushes the icon right of the message text.
    .title::before {
      display: none;
    }
    .title mat-icon {
      flex: none;
      margin-top: 2px; // centred on the first title line
      color: var(--mat-sys-primary);
    }
    .title mat-icon.danger-icon {
      color: var(--mat-sys-error);
    }
    .message {
      margin: 0;
      font: var(--mat-sys-body-large);
      color: var(--mat-sys-on-surface-variant);
    }
    .danger {
      --mat-button-filled-container-color: var(--mat-sys-error);
      --mat-button-filled-label-text-color: var(--mat-sys-on-error);
    }
  `,
})
export class ConfirmDialog {
  readonly data = inject<ConfirmOptions>(MAT_DIALOG_DATA);
}

/**
 * `ask()` opens the dialog and emits true (confirmed) or false (cancelled, Escape or a click
 * outside). Like an awaitable `MessageBox.Show` in .NET, but asynchronous: callers subscribe.
 */
@Injectable({ providedIn: 'root' })
export class ConfirmService {
  private readonly dialog = inject(MatDialog);

  ask(options: ConfirmOptions): Observable<boolean> {
    return this.dialog
      .open(ConfirmDialog, { data: options, width: '440px', maxWidth: '92vw' })
      .afterClosed()
      .pipe(map((result) => result === true));
  }
}
