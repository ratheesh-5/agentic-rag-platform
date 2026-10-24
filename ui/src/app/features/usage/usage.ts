import { DatePipe, DecimalPipe } from '@angular/common';
import {
  Component,
  ElementRef,
  WritableSignal,
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
import { MatTooltipModule } from '@angular/material/tooltip';

import { ApiService } from '../../core/api';
import { ApiKeyStore } from '../../core/api-key.store';
import { describeError } from '../../core/errors';
import { UsageCall, UsageReport, UsageStats } from '../../core/models';
import { ShowRows } from '../../core/show-rows.directive';

type ChartMetric = 'calls' | 'tokens' | 'cost';
type Status = 'all' | 'errors' | 'slow';
type Sort = 'newest' | 'slowest' | 'tokens';

/**
 * Admin › Usage: what the assistant costs to run - model calls, tokens, latency, errors and
 * (if prices are entered in model_prices.json) cost, per purpose, model and day.
 *
 * Interactive like Live quality: KPI tiles, chart bars and the purpose/model rows all filter
 * ONE calls list below (active filters show as removable chips); the list sorts, searches,
 * shows 5 rows and expands a call to its details and its LangSmith trace.
 * Data: the local usage log (every call, no limits).
 */
@Component({
  selector: 'app-usage',
  imports: [
    DatePipe,
    DecimalPipe,
    FormsModule,
    MatButtonModule,
    MatChipsModule,
    MatIconModule,
    MatTooltipModule,
    ShowRows,
  ],
  templateUrl: './usage.html',
  styleUrl: './usage.scss',
  host: { class: 'page' },
})
export class Usage {
  private readonly api = inject(ApiService);
  private readonly keys = inject(ApiKeyStore);
  private readonly callsCard = viewChild<ElementRef<HTMLElement>>('callsCard');

  readonly ranges = [7, 14, 30];
  readonly days = signal(14);
  readonly metric = signal<ChartMetric>('calls');
  readonly report = signal<UsageReport | null>(null);
  readonly error = signal('');
  readonly opening = signal<number | null>(null); // call whose trace link is loading

  // Filters of the calls list (set by tiles, bars, rows; cleared by the chips).
  readonly day = signal<string | null>(null);
  readonly purpose = signal<string | null>(null);
  readonly model = signal<string | null>(null);
  readonly status = signal<Status>('all');
  readonly sort = signal<Sort>('newest');
  readonly search = signal('');
  readonly expanded = signal<number | null>(null);

  readonly hasFilters = computed(
    () =>
      !!(this.day() || this.purpose() || this.model() || this.search()) || this.status() !== 'all',
  );

  /** "Slow" = at or above the p95 latency of the period (the slowest ~5% of calls). */
  readonly slowLimit = computed(() => this.report()?.totals.p95_latency_ms ?? null);

  readonly calls = computed(() => {
    const r = this.report();
    if (!r) {
      return [];
    }
    const term = this.search().trim().toLowerCase();
    const slow = this.slowLimit();
    const list = r.calls.filter(
      (c) =>
        (!this.day() || c.created_at.startsWith(this.day()!)) &&
        (!this.purpose() || c.purpose === this.purpose()) &&
        (!this.model() || c.model === this.model()) &&
        (this.status() !== 'errors' || !!c.error) &&
        (this.status() !== 'slow' ||
          (slow !== null && c.kind === 'chat' && c.latency_ms >= slow)) &&
        (!term || `${c.purpose} ${c.model} ${c.error ?? ''}`.toLowerCase().includes(term)),
    );
    const tokens = (c: UsageCall) => (c.input_tokens ?? 0) + (c.output_tokens ?? 0);
    const by: Record<Sort, (a: UsageCall, b: UsageCall) => number> = {
      newest: () => 0, // already newest first from the API
      slowest: (a, b) => b.latency_ms - a.latency_ms,
      tokens: (a, b) => tokens(b) - tokens(a),
    };
    return [...list].sort(by[this.sort()]);
  });

  /** Longest latency in the list, so the latency bars are relative to it. */
  readonly maxLatency = computed(() => Math.max(1, ...this.calls().map((c) => c.latency_ms)));

  /** Bars of the per-day chart: height relative to the busiest day. */
  readonly bars = computed(() => {
    const r = this.report();
    if (!r) {
      return [];
    }
    const value = (d: UsageStats) =>
      this.metric() === 'calls'
        ? d.calls
        : this.metric() === 'tokens'
          ? d.input_tokens + d.output_tokens
          : (d.cost ?? 0);
    const max = Math.max(...r.per_day.map(value), 0);
    return r.per_day.map((d) => ({
      date: d.date,
      value: value(d),
      // Days without calls get no bar; small non-zero days stay visible (at least 2%).
      height: max && value(d) ? Math.max(2, (value(d) / max) * 100) : 0,
      label: this.metric() === 'cost' ? this.money(d.cost) : String(value(d)),
    }));
  });

  /** Share of all calls per purpose, for the bars in the breakdown. */
  readonly maxPurposeCalls = computed(() =>
    Math.max(1, ...(this.report()?.by_purpose.map((p) => p.calls + p.embed_texts) ?? [1])),
  );

  constructor() {
    effect(() => {
      this.keys.key(); // reload when the admin key changes
      this.days(); // ...or the time range
      untracked(() => this.load());
    });
  }

  load(): void {
    this.error.set('');
    this.api.usageSummary(this.days()).subscribe({
      next: (r) => {
        this.report.set(r);
        // A day outside the new range would filter everything away.
        if (this.day() && !r.per_day.some((d) => d.date === this.day())) {
          this.day.set(null);
        }
      },
      error: (err) => this.error.set(describeError(err)),
    });
  }

  /** Tile / row / bar click: set (or toggle off) one filter and scroll to the calls list. */
  show(kind: 'status' | 'day' | 'purpose' | 'model', value: string): void {
    const targets: Record<string, WritableSignal<string | null>> = {
      day: this.day,
      purpose: this.purpose,
      model: this.model,
    };
    if (kind === 'status') {
      const s = value as Status;
      this.status.set(this.status() === s && s !== 'all' ? 'all' : s);
    } else {
      targets[kind].set(targets[kind]() === value ? null : value);
    }
    this.expanded.set(null);
    this.callsCard()?.nativeElement.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
  }

  /** "Model calls" tile: everything, newest first. */
  showAll(): void {
    this.clearFilters();
    this.sort.set('newest');
    this.show('status', 'all');
  }

  /** "Tokens" tile: biggest calls first (keeps the other filters). */
  showBiggest(): void {
    this.sort.set('tokens');
    this.callsCard()?.nativeElement.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
  }

  clearFilters(): void {
    this.day.set(null);
    this.purpose.set(null);
    this.model.set(null);
    this.status.set('all');
    this.search.set('');
  }

  toggle(id: number): void {
    this.expanded.set(this.expanded() === id ? null : id);
  }

  /** Open one call's trace in LangSmith (the API looks the link up on demand). */
  openTrace(call: UsageCall, event?: Event): void {
    event?.stopPropagation();
    if (!call.run_id) {
      return;
    }
    this.opening.set(call.id);
    this.api.traceUrl(call.run_id).subscribe({
      next: ({ url }) => {
        this.opening.set(null);
        window.open(url, '_blank', 'noopener');
      },
      error: (err) => {
        this.opening.set(null);
        this.error.set(describeError(err));
      },
    });
  }

  /** 1234 -> "1.2 s"; null -> "–". */
  seconds(ms: number | null): string {
    return ms === null ? '–' : `${(ms / 1000).toFixed(1)} s`;
  }

  /** 12345 -> "12.3k". */
  compact(n: number): string {
    return n >= 1_000_000
      ? `${(n / 1_000_000).toFixed(1)}M`
      : n >= 1000
        ? `${(n / 1000).toFixed(1)}k`
        : String(n);
  }

  /** 1 -> "1 call", 3 -> "3 calls". */
  plural(n: number, word: string): string {
    return `${n} ${word}${n === 1 ? '' : 's'}`;
  }

  money(cost: number | null): string {
    return cost === null ? '–' : `$${cost < 0.01 ? cost.toFixed(4) : cost.toFixed(2)}`;
  }

  /** What a call did, in short: tokens for chat calls, texts for embeddings. */
  size(c: UsageCall): string {
    return c.kind === 'embed'
      ? this.plural(c.texts ?? 0, 'text')
      : `${c.input_tokens ?? '–'} in / ${c.output_tokens ?? '–'} out tokens`;
  }

  /** "agent: grade" -> an icon for the purpose. */
  purposeIcon(purpose: string): string {
    if (purpose.startsWith('agent')) return 'account_tree';
    return (
      {
        answer: 'chat',
        rewrite: 'edit_note',
        judge: 'gavel',
        embed: 'scatter_plot',
        chat: 'forum',
      }[purpose] ?? 'bolt'
    );
  }
}
