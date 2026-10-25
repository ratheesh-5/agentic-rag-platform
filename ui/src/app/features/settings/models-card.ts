import { Component, DestroyRef, computed, inject, input, output, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { MatTooltipModule } from '@angular/material/tooltip';

import { ApiService } from '../../core/api';
import { ConfirmService } from '../../core/confirm';
import { describeError } from '../../core/errors';
import {
  ModelCheck,
  ModelKind,
  ModelSlot,
  ModelsChange,
  ModelsInfo,
  ModelTestResult,
  RagConfig,
} from '../../core/models';

/** One dropdown of the card: an answer model, a backup model or the judge. */
interface Row {
  key: keyof ModelsChange;
  kind: ModelKind;
  label: string;
  help: string;
  backup: boolean;
  slot: ModelSlot;
}

/**
 * Admin › Settings › Models (plan G7): choose the answer models, a backup model per provider and
 * the judge from the providers' current model lists - no .env edit, no restart.
 *
 * A new answer or backup model must pass a quality check first (a few demo questions answered
 * with exactly that model; the server enforces it too). "Test" is one tiny call. The judge needs
 * no check, but scores of different judges aren't comparable, so switching it asks first. The
 * embedding model is only shown: changing it means re-indexing every document.
 */
@Component({
  selector: 'app-models-card',
  imports: [MatButtonModule, MatFormFieldModule, MatIconModule, MatSelectModule, MatTooltipModule],
  templateUrl: './models-card.html',
  styleUrl: './models-card.scss',
})
export class ModelsCard {
  private readonly api = inject(ApiService);
  private readonly confirm = inject(ConfirmService);

  /** Automatic fallback is on: backup models are used only then, so they are locked when off. */
  readonly fallback = input(true);
  /** The saved configuration, so the Settings page shows the new models at once. */
  readonly saved = output<RagConfig>();

  static readonly POLL_MS = 3000;
  readonly info = signal<ModelsInfo | null>(null);
  readonly draft = signal<ModelsChange | null>(null);
  readonly tests = signal<Record<string, ModelTestResult | 'running'>>({});
  readonly saving = signal(false);
  readonly error = signal('');
  readonly notice = signal('');
  private poll?: ReturnType<typeof setInterval>;

  readonly rows = computed<Row[]>(() => {
    const info = this.info();
    if (!info) {
      return [];
    }
    const slot = (kind: ModelKind) => info.slots.find((s) => s.kind === kind)!;
    const rows: Row[] = [];
    for (const kind of ['gemini', 'groq'] as const) {
      const name = kind === 'gemini' ? 'Gemini' : 'Groq';
      rows.push(
        {
          key: `${kind}_model`,
          kind,
          label: `${name} answer model`,
          help: `Writes the answers when ${name} is used.`,
          backup: false,
          slot: slot(kind),
        },
        {
          key: `${kind}_backup_model`,
          kind,
          label: `${name} backup model`,
          help: `Tried next when the ${name} answer model is out of quota or overloaded (limits are per model). Needs automatic fallback.`,
          backup: true,
          slot: slot(kind),
        },
      );
    }
    rows.push({
      key: 'judge_model',
      kind: 'judge',
      label: 'Judge model (Gemini)',
      help: 'Scores answers for live quality. Scores from different judges are not comparable.',
      backup: false,
      slot: slot('judge'),
    });
    return rows;
  });

  /** Something differs from what is in use. */
  readonly changed = computed(() => {
    const draft = this.draft();
    return !!draft && this.rows().some((r) => draft[r.key] !== this.inUse(r));
  });

  /** Every changed answer / backup model may be used (default, in use, or passed its check). */
  readonly allowed = computed(() => {
    const draft = this.draft();
    return !!draft && this.rows().every((r) => !this.needsCheck(r, draft[r.key]));
  });

  constructor() {
    this.load();
    inject(DestroyRef).onDestroy(() => this.stopPolling());
  }

  load(): void {
    this.api.models().subscribe({
      next: (info) => {
        const first = !this.info();
        this.info.set(info);
        if (first) {
          this.resetDraft();
        }
        if (info.checks.some((c) => c.state === 'running')) {
          this.startPolling();
        } else {
          this.stopPolling();
        }
      },
      error: (err) => this.error.set(describeError(err)),
    });
  }

  resetDraft(): void {
    const draft = {} as ModelsChange;
    for (const r of this.rows()) {
      draft[r.key] = this.inUse(r);
    }
    this.draft.set(draft);
  }

  /** The model a row has now ('' = no backup). */
  inUse(r: Row): string {
    return r.backup ? r.slot.backup : r.slot.current;
  }

  /** Dropdown entries: the provider's list plus the models in use and the .env default. */
  options(r: Row): string[] {
    const names = new Set([...r.slot.options, r.slot.default, r.slot.current]);
    if (r.slot.backup) {
      names.add(r.slot.backup);
    }
    if (r.backup) {
      names.delete(this.draft()?.[`${r.kind}_model` as keyof ModelsChange] ?? '');
    }
    return [...names].filter(Boolean).sort();
  }

  set(r: Row, model: string): void {
    this.draft.update((d) => (d ? { ...d, [r.key]: model } : d));
    this.notice.set('');
  }

  checkOf(kind: ModelKind, model: string): ModelCheck | undefined {
    return this.info()?.checks.find((c) => c.kind === kind && c.model === model);
  }

  testOf(kind: ModelKind, model: string): ModelTestResult | 'running' | undefined {
    return this.tests()[`${kind}:${model}`];
  }

  /** A new answer / backup model without a passing check can't be used yet. */
  needsCheck(r: Row, model: string): boolean {
    if (r.kind === 'judge' || !model || model === this.inUse(r)) {
      return false;
    }
    if (model === r.slot.default || model === r.slot.current || model === r.slot.backup) {
      return false;
    }
    return !this.checkOf(r.kind, model)?.ok;
  }

  test(r: Row, model: string): void {
    const key = `${r.kind}:${model}`;
    this.tests.update((t) => ({ ...t, [key]: 'running' }));
    this.api.testModel(r.kind, model).subscribe({
      next: (res) => this.tests.update((t) => ({ ...t, [key]: res })),
      error: (err) =>
        this.tests.update((t) => ({
          ...t,
          [key]: { ok: false, message: describeError(err), seconds: 0 },
        })),
    });
  }

  check(r: Row, model: string): void {
    this.confirm
      .ask({
        title: `Check ${model}?`,
        message: `It answers ${this.info()?.check_minimum ?? 5}+ of 6 demo questions correctly to pass: up to 6 AI calls on this model's quota.`,
        action: 'Run check',
        icon: 'fact_check',
      })
      .subscribe((yes) => {
        if (!yes) {
          return;
        }
        this.error.set('');
        this.api.checkModel(r.kind, model).subscribe({
          next: () => this.load(),
          error: (err) => this.error.set(describeError(err)),
        });
      });
  }

  save(): void {
    const draft = this.draft();
    const judge = this.rows().find((r) => r.kind === 'judge');
    if (!draft || !judge) {
      return;
    }
    if (draft.judge_model === this.inUse(judge)) {
      this.store(draft);
      return;
    }
    this.confirm
      .ask({
        title: `Use ${draft.judge_model} as the judge?`,
        message:
          'New scores will come from this judge. Scores from different judges are not comparable; every score keeps the name of its judge.',
        action: 'Use judge',
        icon: 'gavel',
      })
      .subscribe((yes) => yes && this.store(draft));
  }

  private store(draft: ModelsChange): void {
    this.saving.set(true);
    this.error.set('');
    this.api.setModels(draft).subscribe({
      next: (cfg) => {
        this.saving.set(false);
        this.notice.set('Models saved. Used from the next question.');
        this.saved.emit(cfg);
        this.info.set(null); // reload and take the saved models as the new draft
        this.load();
      },
      error: (err) => {
        this.saving.set(false);
        this.error.set(describeError(err));
      },
    });
  }

  private startPolling(): void {
    if (!this.poll) {
      this.poll = setInterval(() => this.load(), ModelsCard.POLL_MS);
    }
  }

  private stopPolling(): void {
    clearInterval(this.poll);
    this.poll = undefined;
  }
}
