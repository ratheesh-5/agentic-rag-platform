import { DatePipe, NgTemplateOutlet, PercentPipe } from '@angular/common';
import {
  Component,
  ElementRef,
  computed,
  effect,
  inject,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatTooltipModule } from '@angular/material/tooltip';

import { ApiService } from '../../core/api';
import { ApiKeyStore } from '../../core/api-key.store';
import { CollectionsStore } from '../../core/collections.store';
import { describeError } from '../../core/errors';
import { fmt } from '../../core/labels';
import { ShowRows } from '../../core/show-rows.directive';
import { QualityTestResult, RecentAnswer, UsageSummary } from '../../core/models';

/** Filters of the per-question list (also set by clicking a KPI tile). */
export type AnswerFilter = 'all' | 'answered' | 'unanswered' | 'low' | 'disliked' | 'scored';

const LOW = 0.7; // below this a score is "weak" (same rule as the backend's low-score list)

/**
 * "Live quality" tab: quality measured from what users really ask - no test data needed.
 *
 * - KPI tiles (click one to filter the answers below).
 * - "Test a question": runs the live pipeline on one question and scores it on the spot
 *   (always scored, never saved - so testing doesn't distort the numbers).
 * - Suggestions: unanswered questions, disliked and low-scored answers.
 * - Answers: every answer with its scores, filterable and searchable.
 * - By document: which documents answers come from and how well they score.
 * Chats made while signed in as admin are hidden unless "Include admin chats" is on.
 */
@Component({
  selector: 'app-live-quality',
  imports: [
    DatePipe,
    NgTemplateOutlet, // renders the reusable score-bar template (<ng-template #meter>)
    PercentPipe,
    FormsModule,
    MatButtonModule,
    MatChipsModule,
    MatIconModule,
    MatProgressBarModule,
    MatSlideToggleModule,
    MatTooltipModule,
    ShowRows, // lists show 5 rows, the rest scrolls
  ],
  templateUrl: './live-quality.html',
  styleUrl: './live-quality.scss',
})
export class LiveQuality {
  private readonly api = inject(ApiService);
  private readonly keys = inject(ApiKeyStore);
  private readonly answersList = viewChild<ElementRef<HTMLElement>>('answersList');

  readonly fmt = fmt;
  readonly usage = signal<UsageSummary | null>(null);
  readonly includeAdmin = signal(false);
  readonly error = signal('');

  // Per-question list state.
  readonly filter = signal<AnswerFilter>('all');
  readonly search = signal('');
  readonly expanded = signal<number | null>(null);
  readonly filters: { value: AnswerFilter; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'answered', label: 'Answered' },
    { value: 'unanswered', label: 'Not answered' },
    { value: 'scored', label: 'Scored' },
    { value: 'low', label: 'Low score' },
    { value: 'disliked', label: 'Disliked' },
  ];

  // "Test a question" state.
  readonly examples = signal<string[]>([]);
  readonly testQuestion = signal('');
  /**
   * The collection the test question searches - like a new chat, one collection at a time
   * (found in testing: a leave question was "not in the documents" because the test box only
   * searched the demo collection while the leave policy lives in HR policies).
   */
  readonly collections = inject(CollectionsStore);
  private readonly chosenCollection = signal<number | null>(null);
  readonly testCollection = computed(
    () => this.chosenCollection() ?? this.collections.selectedId(),
  );
  readonly testing = signal(false);
  readonly testResult = signal<QualityTestResult | null>(null);
  readonly testError = signal('');
  readonly openPassage = signal<number | null>(null);

  /** Which suggestion list is shown in the Suggestions card. */
  readonly tip = signal<'unanswered' | 'disliked' | 'low'>('unanswered');

  readonly answers = computed(() => {
    const term = this.search().trim().toLowerCase();
    return (this.usage()?.recent ?? []).filter(
      (a) => this.matches(a, this.filter()) && (!term || a.question.toLowerCase().includes(term)),
    );
  });

  constructor() {
    effect(() => {
      this.keys.key(); // reload when the admin key changes
      this.includeAdmin(); // ...or the admin-chats switch
      untracked(() => this.load());
    });
    // Example questions for the test box: the chosen collection's topics (GET /assistant).
    effect(() => {
      const collectionId = this.testCollection();
      untracked(() =>
        this.api.assistant(collectionId).subscribe({
          next: (p) => this.examples.set(p.topics.map((t) => t.examples[0]).filter(Boolean)),
          error: () => this.examples.set([]),
        }),
      );
    });
  }

  load(): void {
    this.error.set('');
    this.api.usage(this.includeAdmin()).subscribe({
      next: (u) => this.usage.set(u),
      error: (err) => this.error.set(describeError(err)),
    });
  }

  /** KPI tile click: filter the answers list and scroll to it. */
  show(filter: AnswerFilter): void {
    this.filter.set(filter);
    this.expanded.set(null);
    this.answersList()?.nativeElement.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
  }

  toggle(id: number): void {
    this.expanded.set(this.expanded() === id ? null : id);
  }

  /** Search another collection: the last result belonged to the previous one. */
  chooseCollection(id: number): void {
    this.chosenCollection.set(id);
    this.testResult.set(null);
    this.testError.set('');
  }

  runTest(question?: string): void {
    const q = (question ?? this.testQuestion()).trim();
    if (q.length < 2 || this.testing()) {
      return;
    }
    this.testQuestion.set(q);
    this.testing.set(true);
    this.testError.set('');
    this.testResult.set(null);
    this.openPassage.set(null);
    this.api.testQuestion(q, this.testCollection()).subscribe({
      next: (r) => {
        this.testing.set(false);
        this.testResult.set(r);
      },
      error: (err) => {
        this.testing.set(false);
        this.testError.set(describeError(err));
      },
    });
  }

  /** "1 in 5 answers" for 0.2, "every answer" for 1, "no answers (scoring is off)" for 0. */
  sampleText(rate: number): string {
    if (rate <= 0) {
      return 'no answers (scoring is off)';
    }
    return rate >= 1 ? 'every answer' : `1 in ${Math.round(1 / rate)} answers`;
  }

  /** Score colour class: good / weak / none. */
  level(value: number | null | undefined): string {
    return value === null || value === undefined ? 'none' : value >= LOW ? 'good' : 'weak';
  }

  /** Width of a 0..1 score bar in percent. */
  width(value: number | null | undefined): number {
    return Math.max(0, Math.min(1, value ?? 0)) * 100;
  }

  count(filter: AnswerFilter): number {
    return (this.usage()?.recent ?? []).filter((a) => this.matches(a, filter)).length;
  }

  private matches(a: RecentAnswer, filter: AnswerFilter): boolean {
    const scores = [a.faithfulness, a.answer_relevancy];
    switch (filter) {
      case 'answered':
        return a.answerable;
      case 'unanswered':
        return !a.answerable;
      case 'scored':
        return scores.some((s) => s !== null);
      case 'low':
        return scores.some((s) => s !== null && s < LOW);
      case 'disliked':
        return a.feedback === -1;
      default:
        return true;
    }
  }
}
