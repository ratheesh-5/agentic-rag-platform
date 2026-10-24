import { DatePipe } from '@angular/common';
import { Component, computed, effect, inject, signal, untracked } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { MatDialog } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTooltipModule } from '@angular/material/tooltip';

import { ApiService } from '../../core/api';
import { ApiKeyStore } from '../../core/api-key.store';
import { CollectionDialog } from '../../core/collection-dialog';
import { CollectionsStore } from '../../core/collections.store';
import { ConfirmService } from '../../core/confirm';
import { describeError } from '../../core/errors';
import { Collection, DocumentInfo, PrivateSummary, UploadResponse } from '../../core/models';
import { ShowRows } from '../../core/show-rows.directive';

/** One file in the upload queue (several files can be dropped at once). */
export interface UploadItem {
  name: string;
  state: 'waiting' | 'uploading' | 'done' | 'warning' | 'failed';
  detail: string;
}

const TYPE_ICONS: Record<string, string> = {
  pdf: 'picture_as_pdf',
  docx: 'article',
  md: 'description',
  txt: 'notes',
};

/**
 * Admin › Documents: the knowledge the assistant answers from, one collection at a time (chips
 * at the top; "New collection", rename, delete). Uploads go into the selected collection.
 * Upload by drag-and-drop or the file picker (several files are uploaded one after the other);
 * the API validates, splits, embeds and indexes each one - usable from the next question.
 * List with search and type filter; delete asks first.
 */
@Component({
  selector: 'app-documents',
  imports: [
    DatePipe,
    FormsModule,
    MatButtonModule,
    MatChipsModule,
    MatIconModule,
    MatProgressBarModule,
    MatTooltipModule,
    ShowRows,
  ],
  templateUrl: './documents.html',
  styleUrl: './documents.scss',
  host: { class: 'page' },
})
export class Documents {
  private readonly api = inject(ApiService);
  readonly keys = inject(ApiKeyStore); // the template shows admin-only parts
  private readonly confirm = inject(ConfirmService);
  private readonly dialog = inject(MatDialog);
  readonly collections = inject(CollectionsStore);

  /** The collections this caller may manage: admins the shared ones (+ their own private
   * ones), every signed-in user their own private ones. Others aren't listed here at all. */
  readonly managed = computed(() => this.collections.list().filter((c) => c.can_manage));
  /** The collection shown: the selected one if the caller may manage it, else the first one. */
  readonly current = computed(() => {
    const selected = this.collections.selected();
    return selected?.can_manage ? selected : (this.managed()[0] ?? null);
  });
  /** Admins: how much users keep privately (totals only, never content). */
  readonly privateTotals = signal<PrivateSummary | null>(null);

  /** File types the API accepts (it re-checks them; this only filters the file picker). */
  readonly accept = '.md,.txt,.pdf,.docx';

  readonly documents = signal<DocumentInfo[]>([]);
  readonly uploads = signal<UploadItem[]>([]);
  readonly dragging = signal(false);
  readonly search = signal('');
  readonly type = signal('all');
  readonly message = signal('');
  readonly error = signal('');

  readonly busy = computed(() => this.uploads().some((u) => u.state === 'uploading'));
  readonly totalChunks = computed(() => this.documents().reduce((sum, d) => sum + d.chunks, 0));
  readonly lastIndexed = computed(
    () =>
      this.documents()
        .map((d) => d.ingested_at)
        .sort()
        .at(-1) ?? null,
  );
  /** File types present, with counts: for the tiles and the filter chips. */
  readonly types = computed(() => {
    const counts = new Map<string, number>();
    this.documents().forEach((d) => counts.set(d.doc_type, (counts.get(d.doc_type) ?? 0) + 1));
    return [...counts.entries()].map(([name, count]) => ({ name, count }));
  });
  readonly visible = computed(() => {
    const term = this.search().trim().toLowerCase();
    return this.documents()
      .filter((d) => this.type() === 'all' || d.doc_type === this.type())
      .filter((d) => !term || d.path.toLowerCase().includes(term))
      .sort((a, b) => b.ingested_at.localeCompare(a.ingested_at)); // newest first
  });

  constructor() {
    // Load now and again whenever the API key or the selected collection changes (the effect
    // re-runs when one of the signals it reads changes).
    effect(() => {
      this.keys.key();
      this.current()?.id;
      untracked(() => this.load()); // reads inside load() must not trigger reloads
    });
    if (this.keys.isAdmin()) {
      this.api.privateSummary().subscribe({
        next: (t) => this.privateTotals.set(t),
        error: () => this.privateTotals.set(null), // optional information
      });
    }
  }

  load(): void {
    this.error.set('');
    const collection = this.current();
    if (!collection) {
      this.documents.set([]); // no collection of your own yet: "New collection" first
      return;
    }
    this.api.documents(collection.id).subscribe({
      next: (docs) => this.documents.set(docs),
      error: (err) => this.error.set(describeError(err)),
    });
  }

  /** Where a collection's similarity threshold comes from, in plain words. */
  thresholdLabel(source: Collection['threshold_source']): string {
    return {
      measured: 'measured with the test set',
      manual: 'set by an admin',
      automatic: 'automatic',
      default: 'global value (measured after the first upload)',
    }[source];
  }

  thresholdHelp(source: Collection['threshold_source']): string {
    const what =
      'Questions whose best passage scores below this get "I couldn\'t find this" without an AI call. ';
    return (
      what +
      {
        measured:
          'Measured on this collection with the test set (changes through Settings › optimizer).',
        manual: 'Set by an admin in Rename. Empty it there to measure it automatically again.',
        automatic:
          'Measured after every upload: just above how well clearly unrelated questions match.',
        default: 'The global value until the first document is uploaded.',
      }[source]
    );
  }

  /** "New collection" or "Rename": the dialog saves; the new one becomes the selected one. */
  editCollection(existing: Collection | null = null): void {
    this.dialog
      .open(CollectionDialog, { data: existing, width: '440px', maxWidth: '92vw' })
      .afterClosed()
      .subscribe((saved?: Collection) => {
        if (saved) {
          this.collections.select(saved.id);
          this.collections.reload();
        }
      });
  }

  /** Delete the selected collection (not the demo one) after a red confirmation. */
  deleteCollection(collection: Collection): void {
    this.confirm
      .ask({
        title: `Delete the collection "${collection.name}"?`,
        message:
          `Its ${collection.documents} document(s) and all chats in it are removed for good. ` +
          'Other collections are not affected.',
        action: 'Delete collection',
        icon: 'delete',
        danger: true,
      })
      .subscribe((yes) => {
        if (!yes) {
          return;
        }
        this.api.deleteCollection(collection.id).subscribe({
          next: () => {
            this.message.set(`Collection "${collection.name}" deleted.`);
            const next = this.managed().find((c) => c.id !== collection.id);
            this.collections.select(next?.id ?? 1);
            this.collections.reload();
          },
          error: (err) => this.error.set(describeError(err)),
        });
      });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const files = [...(input.files ?? [])];
    input.value = ''; // allow selecting the same file again
    this.uploadAll(files);
  }

  // Drag and drop: preventDefault on dragover is what allows dropping (browser rule).
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(true);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(false);
    this.uploadAll([...(event.dataTransfer?.files ?? [])]);
  }

  /** Queue the files and upload them one after the other (the API indexes one at a time). */
  uploadAll(files: File[]): void {
    if (!files.length || this.busy() || !this.current()) {
      return; // never upload without a collection (the server would refuse it)
    }
    this.message.set('');
    this.error.set('');
    this.uploads.set(files.map((f) => ({ name: f.name, state: 'waiting', detail: '' })));
    this.next(files, 0);
  }

  private next(files: File[], i: number): void {
    if (i >= files.length) {
      this.load();
      this.collections.reload(); // document counts in the chips
      return;
    }
    this.setUpload(i, 'uploading', 'Validating and indexing…');
    this.api.upload(files[i], this.current()?.id).subscribe({
      next: (res) => {
        // Indexed either way; a suspicious document is only flagged for the admin to check.
        const state = res.document.warnings.length ? 'warning' : 'done';
        this.setUpload(i, state, this.describeUpload(res));
        this.next(files, i + 1);
      },
      error: (err) => {
        this.setUpload(i, 'failed', describeError(err));
        this.next(files, i + 1);
      },
    });
  }

  private setUpload(i: number, state: UploadItem['state'], detail: string): void {
    this.uploads.update((list) => list.map((u, j) => (j === i ? { ...u, state, detail } : u)));
  }

  /** "Added · 3 passages · 3 new embeddings" in plain words. */
  describeUpload(res: UploadResponse): string {
    const what = {
      added: 'Added',
      updated: 'Updated (content changed)',
      unchanged: 'Already indexed, nothing changed',
    }[res.status];
    const passages = `${res.document.chunks} passage${res.document.chunks === 1 ? '' : 's'}`;
    const emb = res.embeddings_new
      ? ` · ${res.embeddings_new} new embedding(s)` +
        (res.embeddings_cached ? `, ${res.embeddings_cached} reused` : '')
      : '';
    const warn = res.document.warnings.length
      ? ` · Check it: text that looks like instructions to the AI ("${res.document.warnings[0]}")`
      : '';
    return `${what} · ${passages}${emb}${warn}`;
  }

  /** Tooltip for the warning badge: why it is shown and the suspicious lines. */
  warningText(doc: DocumentInfo): string {
    return (
      'Contains text that looks like instructions to the AI (possible prompt injection). ' +
      'It is indexed; the assistant treats passages as data, but check the document: ' +
      doc.warnings.map((w) => `"${w}"`).join(' · ')
    );
  }

  remove(doc: DocumentInfo): void {
    this.confirm
      .ask({
        title: `Remove ${this.fileName(doc)}?`,
        message: 'The assistant can no longer answer from this document.',
        action: 'Remove',
        icon: 'delete',
        danger: true,
      })
      .subscribe((yes) => yes && this.delete(doc));
  }

  private delete(doc: DocumentInfo): void {
    this.error.set('');
    this.api.deleteDocument(doc.id).subscribe({
      next: () => {
        this.message.set(`${this.fileName(doc)} removed.`);
        this.load();
      },
      error: (err) => this.error.set(describeError(err)),
    });
  }

  fileName(doc: DocumentInfo): string {
    return doc.path.split('/').pop() ?? doc.path;
  }

  /** Where a document came from: uploaded in the UI or part of the sample corpus (CLI). */
  source(doc: DocumentInfo): string {
    return doc.path.includes('uploads/') ? 'Uploaded' : 'Sample corpus';
  }

  icon(type: string): string {
    return TYPE_ICONS[type] ?? 'draft';
  }
}
