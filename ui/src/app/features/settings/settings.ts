import { DatePipe } from '@angular/common';
import { Component, DestroyRef, computed, effect, inject, signal, untracked } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleGroup, MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSlideToggle, MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatTooltipModule } from '@angular/material/tooltip';

import { API_BASE_URL, ApiService } from '../../core/api';
import { ApiKeyStore } from '../../core/api-key.store';
import { ConfirmService } from '../../core/confirm';
import { describeError } from '../../core/errors';
import { METRICS, SETTINGS, SETTING_GROUPS, fmt } from '../../core/labels';
import {
  ChatProvider,
  Mode,
  OptimizerRun,
  ProviderName,
  RagConfig,
  Recommendation,
} from '../../core/models';

/** Icon per settings group. */
const GROUP_ICONS: Record<string, string> = {
  'Document chunking': 'view_agenda',
  Search: 'travel_explore',
  Answering: 'smart_toy',
};

/**
 * Admin › Settings: the active configuration (rag_config.json) and the optimizer's
 * recommendation, which an admin applies or rejects here - the human-approval step of the
 * multi-agent optimizer. Settings are not edited by hand: every change is measured first
 * (the optimizer runs the experiments), so a change can never silently make answers worse.
 * The optimizer can be started here (default or own test set); it runs in the background on the
 * server and this page polls its progress every few seconds.
 *
 * Layout: a 3-step "how settings change" strip (measure -> review -> apply) that highlights
 * where we are, the recommendation card, then the active settings as readable tiles.
 */
@Component({
  selector: 'app-settings',
  imports: [
    DatePipe,
    MatButtonModule,
    MatButtonToggleModule,
    MatIconModule,
    MatProgressBarModule,
    MatSlideToggleModule,
    MatTooltipModule,
  ],
  templateUrl: './settings.html',
  styleUrl: './settings.scss',
  host: { class: 'page' },
})
export class Settings {
  private readonly api = inject(ApiService);
  private readonly keys = inject(ApiKeyStore);
  private readonly confirm = inject(ConfirmService);
  readonly baseUrl = inject(API_BASE_URL);

  readonly fmt = fmt;
  readonly metricInfo = METRICS;
  readonly command = 'uv run rag optimize --decision later';
  /** How often the page asks for the optimizer's progress while it runs. */
  static readonly POLL_MS = 3000;
  /** Before/after metrics shown for a recommendation. */
  readonly compareMetrics = ['recall_at_k', 'mrr', 'safe_threshold'];

  readonly healthy = signal<boolean | null>(null); // null = not checked yet
  readonly config = signal<RagConfig | null>(null);
  readonly recommendation = signal<Recommendation | null>(null);
  readonly deciding = signal(false);
  readonly showDetails = signal(false); // details of an already-decided recommendation
  readonly copied = signal(false);
  readonly error = signal('');
  readonly notice = signal('');
  /** Answer modes the admin can switch between directly (not through a recommendation). */
  readonly modes: Mode[] = ['fast', 'agentic', 'auto'];
  readonly switching = signal(false);
  /** Answer model card: the chat providers and whether each is set up on the server. */
  readonly providers = signal<ChatProvider[]>([]);
  readonly providerNames: Record<ProviderName, string> = { gemini: 'Gemini', groq: 'Groq' };
  /** Fallback needs a second provider that is set up. */
  readonly canFallBack = computed(() => this.providers().filter((p) => p.configured).length > 1);
  /** The optimizer started from this page (null until loaded). */
  readonly job = signal<OptimizerRun | null>(null);
  readonly optimizing = computed(() => this.job()?.state === 'running');
  private poll?: ReturnType<typeof setInterval>;

  /** Which step of measure -> review -> apply we are at (1..3). */
  readonly step = computed(() => {
    const status = this.recommendation()?.status;
    return status === 'pending' ? 2 : status === 'applied' ? 3 : 1;
  });

  /** Settings changed by the last APPLIED recommendation get a "changed" badge. */
  readonly changedKeys = computed(() => {
    const rec = this.recommendation();
    return rec?.status === 'applied' ? new Set(Object.keys(rec.report.changes)) : new Set();
  });

  /** The configuration in display groups (Document chunking, Search, Answering). */
  readonly groups = computed(() => {
    const cfg = this.config();
    if (!cfg) {
      return [];
    }
    return SETTING_GROUPS.map((group) => ({
      group,
      icon: GROUP_ICONS[group] ?? 'tune',
      items: Object.entries(cfg)
        .filter(([key]) => key !== 'mode' && SETTINGS[key]?.group === group) // mode: "Your choices"
        .map(([key, value]) => ({ key, value, ...SETTINGS[key] })),
    }));
  });

  /** The recommendation's changes: setting, value it was measured against, recommended value. */
  readonly changes = computed(() => {
    const rec = this.recommendation();
    const cfg = this.config();
    if (!rec || !cfg) {
      return [];
    }
    // "from" = the settings the recommendation was measured against (after applying, the
    // current config already holds the new values).
    const base = (rec.report.base_config ?? cfg) as unknown as Record<string, unknown>;
    return Object.entries(rec.report.changes).map(([key, value]) => ({
      label: SETTINGS[key]?.label ?? key,
      help: SETTINGS[key]?.help ?? '',
      from: this.display(key, base[key]),
      to: this.display(key, value),
    }));
  });

  constructor() {
    effect(() => {
      this.keys.key(); // reload when the admin key changes
      untracked(() => this.load());
    });
    inject(DestroyRef).onDestroy(() => this.stopPolling()); // leaving the page stops polling
  }

  load(): void {
    this.error.set('');
    this.api.health().subscribe({
      next: () => this.healthy.set(true),
      error: () => this.healthy.set(false),
    });
    this.api.config().subscribe({
      next: (cfg) => this.config.set(cfg),
      error: (err) => this.error.set(describeError(err)),
    });
    this.api.chatProviders().subscribe({
      next: (list) => this.providers.set(list),
      error: () => this.providers.set([]), // the card then shows only the stored choice
    });
    this.api.recommendation().subscribe({
      next: (rec) => this.recommendation.set(rec),
      error: (err) => this.error.set(describeError(err)),
    });
    this.api.optimizerRun().subscribe({
      next: (job) => {
        this.job.set(job);
        if (job?.state === 'running') {
          this.startPolling(); // e.g. started in another tab, or the page was reloaded
        }
      },
      error: () => this.job.set(null), // progress is optional; the page works without it
    });
  }

  /**
   * Switch the answer mode after a confirmation that names its cost. The mode is a cost /
   * quality choice, so it is changed directly here - unlike search settings, which change only
   * through a measured recommendation.
   */
  changeMode(mode: Mode, toggle?: MatButtonToggleGroup): void {
    const cfg = this.config();
    if (!cfg || mode === cfg.mode) {
      return;
    }
    // Cancelled or failed: show the active mode again. Setting the group directly is needed:
    // the [value] binding still holds the same mode, so Angular sees no change to re-render.
    const revert = () => {
      if (toggle) {
        toggle.value = cfg.mode;
      }
    };
    const cost: Record<Mode, string> = {
      fast: '1 AI call per question; no planning, grading or verification.',
      agentic: 'About 4 AI calls per question: slower, and the free quota runs out faster.',
      auto: 'Fast first (1 call); questions the fast answer struggles with are handed to the agent (about 5 calls).',
    };
    const name = this.display('mode', mode);
    this.confirm
      .ask({
        title: `Switch to ${name} mode?`,
        message: cost[mode],
        action: `Use ${name}`,
        icon: 'tune',
      })
      .subscribe((yes) => {
        if (yes) {
          this.saveMode(mode, revert);
        } else {
          revert();
        }
      });
  }

  private saveMode(mode: Mode, revert: () => void): void {
    this.switching.set(true);
    this.error.set('');
    this.api.setMode(mode).subscribe({
      next: (updated) => {
        this.switching.set(false);
        this.config.set(updated);
        this.notice.set(`Answer mode: ${this.display('mode', mode)}. Used from the next question.`);
      },
      error: (err) => {
        this.switching.set(false);
        revert();
        this.error.set(describeError(err));
      },
    });
  }

  /** The model name of a provider, e.g. "openai/gpt-oss-120b" for Groq. */
  modelOf(name: ProviderName): string {
    return this.providers().find((p) => p.name === name)?.model ?? '';
  }

  /**
   * Answer model: which provider writes answers first, and whether the other one takes over
   * when it is out of quota or overloaded. Applies from the next question (no confirmation:
   * unlike the mode, it doesn't change the cost per question). When saving fails the control
   * is put back by hand: its binding still holds the old value, so Angular won't re-render it.
   */
  changeProvider(provider: ProviderName, toggle?: MatButtonToggleGroup): void {
    const cfg = this.config();
    if (cfg && provider !== cfg.chat_provider) {
      this.saveModel(
        provider,
        cfg.chat_fallback,
        () => toggle && (toggle.value = cfg.chat_provider),
      );
    }
  }

  changeFallback(on: boolean, slide?: MatSlideToggle): void {
    const cfg = this.config();
    if (cfg) {
      this.saveModel(cfg.chat_provider, on, () => slide && (slide.checked = cfg.chat_fallback));
    }
  }

  private saveModel(provider: ProviderName, fallback: boolean, revert: () => void): void {
    this.switching.set(true);
    this.error.set('');
    this.api.setModel(provider, fallback).subscribe({
      next: (updated) => {
        this.switching.set(false);
        this.config.set(updated);
        const other = provider === 'gemini' ? 'groq' : 'gemini';
        this.notice.set(
          `Answers: ${this.providerNames[provider]}` +
            (fallback && this.canFallBack() ? `, then ${this.providerNames[other]}` : '') +
            '. Used from the next question.',
        );
      },
      error: (err) => {
        this.switching.set(false);
        revert();
        this.error.set(describeError(err));
      },
    });
  }

  /** Start the optimizer in the background (default test set, or the admin's own file). */
  optimize(testSet?: File): void {
    this.confirm
      .ask({
        title: 'Run the optimizer?',
        message:
          'It makes a few AI calls (diagnosis, choosing experiments, report) and can take a few ' +
          'minutes. Nothing changes until you apply its recommendation.',
        action: 'Run optimizer',
        icon: 'auto_fix_high',
      })
      .subscribe((yes) => yes && this.startOptimizer(testSet));
  }

  private startOptimizer(testSet?: File): void {
    this.error.set('');
    this.notice.set('');
    this.api.startOptimizer(testSet).subscribe({
      next: (job) => {
        this.job.set(job);
        this.startPolling();
      },
      error: (err) => this.error.set(describeError(err)),
    });
  }

  onFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = ''; // the same file can be chosen again
    if (file) {
      this.optimize(file);
    }
  }

  private startPolling(): void {
    this.stopPolling();
    this.poll = setInterval(() => {
      this.api.optimizerRun().subscribe({
        next: (job) => {
          this.job.set(job);
          if (job.state !== 'running') {
            this.stopPolling();
            if (job.outcome === 'recommendation') {
              this.load(); // the new recommendation appears for review
            }
          }
        },
        error: (err) => {
          this.stopPolling();
          this.error.set(describeError(err));
        },
      });
    }, Settings.POLL_MS);
  }

  private stopPolling(): void {
    clearInterval(this.poll);
    this.poll = undefined;
  }

  /** The optimizer's result in plain words. */
  outcomeText(job: OptimizerRun): string {
    switch (job.outcome) {
      case 'recommendation':
        return `Finished: a recommendation is waiting for your review above (test set ${job.test_set}).`;
      case 'no-change':
        return `Finished: the current settings are already the best on ${job.test_set}. Nothing to change.`;
      default:
        return `Finished: every test question in ${job.test_set} already ranks its answer first. Nothing to improve.`;
    }
  }

  decide(decision: 'apply' | 'reject'): void {
    const rec = this.recommendation();
    if (!rec) {
      return;
    }
    const apply = decision === 'apply';
    this.confirm
      .ask(
        apply
          ? {
              title: 'Apply the recommendation?',
              message: 'The assistant uses the recommended configuration from the next question.',
              action: 'Apply',
              icon: 'check_circle',
            }
          : {
              title: 'Reject the recommendation?',
              message: 'The configuration stays as it is.',
              action: 'Reject',
              icon: 'block',
              danger: true,
            },
      )
      .subscribe((yes) => yes && this.saveDecision(rec, decision));
  }

  private saveDecision(rec: Recommendation, decision: 'apply' | 'reject'): void {
    this.deciding.set(true);
    this.error.set('');
    this.api.decideRecommendation(rec.id, decision).subscribe({
      next: () => {
        this.deciding.set(false);
        this.notice.set(
          decision === 'apply'
            ? 'Applied. Run a quality check on the Evaluation page to confirm the result.'
            : 'Rejected. The configuration is unchanged.',
        );
        this.load();
      },
      error: (err) => {
        this.deciding.set(false);
        this.error.set(describeError(err));
      },
    });
  }

  copyCommand(): void {
    navigator.clipboard?.writeText(this.command);
    this.copied.set(true);
    setTimeout(() => this.copied.set(false), 1500);
  }

  /** Signed difference for a metric, e.g. "+0.031". */
  delta(rec: Recommendation, key: string): string {
    const before = rec.report.baseline_metrics[key];
    const after = rec.report.winner_metrics[key];
    if (typeof before !== 'number' || typeof after !== 'number') {
      return '';
    }
    const d = after - before;
    return (d >= 0 ? '+' : '') + d.toFixed(3);
  }

  /** A setting's value in plain words: "800 chars", "Hybrid", "0.68". */
  display(key: string, value: unknown): string {
    if (value === null || value === undefined) {
      return '–';
    }
    const units: Record<string, string> = {
      chunk_size: 'chars',
      chunk_overlap: 'chars',
      top_k: 'passages',
      hybrid_candidates: 'per search',
    };
    if (typeof value === 'string') {
      return value.charAt(0).toUpperCase() + value.slice(1); // dense -> Dense
    }
    if (key === 'similarity_threshold' && typeof value === 'number') {
      return value.toFixed(2);
    }
    return units[key] ? `${value} ${units[key]}` : String(value);
  }
}
