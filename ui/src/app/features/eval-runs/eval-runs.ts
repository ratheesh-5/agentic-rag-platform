import { DatePipe } from '@angular/common';
import { Component, computed, effect, inject, signal, untracked } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTooltipModule } from '@angular/material/tooltip';

import { ApiService } from '../../core/api';
import { ApiKeyStore } from '../../core/api-key.store';
import { describeError } from '../../core/errors';
import { METRICS, SETTINGS, fmt } from '../../core/labels';
import { LiveQuality } from './live-quality';
import {
  EvalRunDetail,
  EvalRunSummary,
  GateCheck,
  GateStatus,
  RagConfig,
  RunKind,
} from '../../core/models';

type Filter = 'all' | RunKind;

/**
 * Admin › Evaluation: is the assistant good enough, and did a change make it better or worse?
 * Tab 1 "Live quality" (live-quality.ts): measured from real chats. Tab 2 "Test-set checks"
 * (this component's own content):
 *
 * - "Run quality check" = retrieval-only evaluation of the ACTIVE configuration (no chat calls,
 *   seconds), on the default test set or an uploaded one (.jsonl, e.g. for your own documents).
 * - Each run is judged against the release gates (data/eval/release_gates.json).
 * - Old test runs can be hidden; full evaluations (answers + RAGAS) still come from the CLI.
 */
@Component({
  selector: 'app-eval-runs',
  imports: [
    DatePipe,
    MatButtonModule,
    MatChipsModule,
    MatExpansionModule,
    MatIconModule,
    MatProgressBarModule,
    MatSlideToggleModule,
    MatTabsModule,
    MatTooltipModule,
    LiveQuality,
  ],
  templateUrl: './eval-runs.html',
  styleUrl: './eval-runs.scss',
  host: { class: 'page' },
})
export class EvalRuns {
  private readonly api = inject(ApiService);
  private readonly keys = inject(ApiKeyStore);

  readonly fmt = fmt;
  readonly metricInfo = METRICS;
  readonly settingInfo = SETTINGS;
  /** Headline tiles, in this order. */
  readonly kpis = ['recall_at_k', 'mrr', 'abstention_accuracy', 'answer_correctness'];
  /** Metrics listed in the detail panel. */
  readonly detailMetrics = [
    'recall_at_k',
    'mrr',
    'precision_at_k',
    'abstention_accuracy',
    'false_abstention_rate',
    'unanswerable_answered_rate',
    'faithfulness',
    'answer_relevancy',
    'answer_correctness',
    'latency_p95_s',
    'safe_threshold',
  ];
  /** Search-only runs (quality checks, experiments) never generate answers: answer metrics such
   * as made-up answers or latency would be meaningless there, so only these are listed. */
  readonly searchMetrics = ['recall_at_k', 'mrr', 'precision_at_k', 'safe_threshold'];
  readonly filters: { value: Filter; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'check', label: 'Quality checks' },
    { value: 'experiment', label: 'Optimizer experiments' },
    { value: 'full', label: 'Full evaluations' },
  ];
  readonly kindLabel: Record<RunKind, string> = {
    full: 'Full evaluation',
    check: 'Quality check',
    experiment: 'Optimizer experiment',
  };

  readonly runs = signal<EvalRunSummary[]>([]); // newest first
  readonly filter = signal<Filter>('all');
  readonly showHidden = signal(false);
  readonly selected = signal<EvalRunDetail | null>(null);
  /** Gate checks of the headline run (with values and n/a for unmeasured metrics). */
  readonly headline = signal<EvalRunDetail | null>(null);
  readonly running = signal(false);
  readonly error = signal('');
  readonly notice = signal('');

  readonly hiddenCount = computed(() => this.runs().filter((r) => r.hidden).length);
  readonly visible = computed(() =>
    this.runs().filter(
      (r) =>
        (this.showHidden() || !r.hidden) && (this.filter() === 'all' || r.kind === this.filter()),
    ),
  );
  /** The newest visible run made with the active configuration (not an optimizer try). */
  readonly latest = computed(
    () =>
      this.runs().find((r) => r.current && !r.hidden && r.kind !== 'experiment') ??
      this.runs().find((r) => r.current && !r.hidden) ??
      null,
  );

  constructor() {
    effect(() => {
      this.keys.key(); // reload when the admin key changes
      untracked(() => this.load());
    });
    // Load the headline run's gate details whenever it changes.
    effect(() => {
      const run = this.latest();
      untracked(() => {
        if (!run) {
          this.headline.set(null);
          return;
        }
        this.api.evalRun(run.id).subscribe({ next: (d) => this.headline.set(d) });
      });
    });
  }

  load(): void {
    this.error.set('');
    this.api.evalRuns().subscribe({
      next: (runs) => this.runs.set([...runs].reverse()),
      error: (err) => this.error.set(describeError(err)),
    });
  }

  runCheck(testSet?: File): void {
    this.running.set(true);
    this.error.set('');
    this.notice.set('');
    this.api.runQualityCheck(testSet).subscribe({
      next: (run) => {
        this.running.set(false);
        const verdict =
          run.gates === 'fail'
            ? 'below the release gates'
            : run.gates === 'n/a'
              ? 'finished'
              : 'meets the release gates';
        this.notice.set(
          `Quality check #${run.id} on ${run.test_set}: ${verdict} ` +
            `(Recall@5 ${fmt(run.metrics['recall_at_k'])}, MRR ${fmt(run.metrics['mrr'])}).`,
        );
        this.load();
        this.open(run);
      },
      error: (err) => {
        this.running.set(false);
        this.error.set(describeError(err));
      },
    });
  }

  onFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = ''; // the same file can be chosen again
    if (file) {
      this.runCheck(file);
    }
  }

  open(run: EvalRunSummary): void {
    if (this.selected()?.id === run.id) {
      this.selected.set(null); // second click closes
      return;
    }
    this.api.evalRun(run.id).subscribe({
      next: (detail) => this.selected.set(detail),
      error: (err) => this.error.set(describeError(err)),
    });
  }

  toggleHidden(run: EvalRunSummary, event: Event): void {
    event.stopPropagation(); // the button sits inside a clickable row
    this.api.setRunHidden(run.id, !run.hidden).subscribe({
      next: () =>
        this.runs.update((list) =>
          list.map((r) => (r.id === run.id ? { ...r, hidden: !run.hidden } : r)),
        ),
      error: (err) => this.error.set(describeError(err)),
    });
  }

  metricsFor(detail: EvalRunDetail): string[] {
    return detail.kind === 'full' ? this.detailMetrics : this.searchMetrics;
  }

  gate(detail: EvalRunDetail | null, metric: string): GateCheck | undefined {
    return detail?.gates.find((g) => g.metric === metric);
  }

  statusIcon(status: GateStatus): string {
    return {
      pass: 'check_circle',
      fail: 'cancel',
      warn: 'warning',
      'n/a': 'remove_circle_outline',
    }[status];
  }

  headlineText(status: GateStatus): string {
    return {
      pass: 'Meets the release gates',
      warn: 'Meets the blocking gates (with warnings)',
      fail: 'Below the release gates',
      'n/a': 'No gate measured yet',
    }[status];
  }

  /** Answerable questions whose answer passage was not ranked first: what to look at. */
  missed(detail: EvalRunDetail): { question: string; where: string }[] {
    return detail.results
      .filter((r) => r.answerable && r.retrieval && r.retrieval.rr < 1)
      .map((r) => ({
        question: r.question,
        where: r.retrieval!.rr ? `found at rank ${Math.round(1 / r.retrieval!.rr)}` : 'not found',
      }));
  }

  settingsUsed(detail: EvalRunDetail): { label: string; value: unknown }[] {
    const rag = (detail.config['rag'] ?? {}) as Partial<RagConfig>;
    return Object.entries(rag).map(([key, value]) => ({
      label: SETTINGS[key]?.label ?? key,
      value,
    }));
  }
}
