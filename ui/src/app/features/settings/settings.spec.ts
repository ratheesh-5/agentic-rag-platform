// Settings page with a mocked backend: grouped configuration and the optimizer recommendation
// (apply / reject = the human-approval step, stale recommendations cannot be applied) and the
// optimizer started from the page (background run, progress polled).
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { Component, input } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { of } from 'rxjs';

import { API_BASE_URL } from '../../core/api';
import { ConfirmService } from '../../core/confirm';
import { ChatProvider, OptimizerRun, RagConfig, Recommendation } from '../../core/models';
import { ModelsCard } from './models-card';
import { Settings } from './settings';

/** The Models card has its own spec (models-card.spec.ts); here it is a stand-in. */
@Component({ selector: 'app-models-card', template: '' })
class ModelsCardStub {
  readonly fallback = input(true);
}

const BASE = 'http://api.test';
const CONFIG: RagConfig = {
  chunk_size: 800,
  chunk_overlap: 100,
  top_k: 5,
  retrieval: 'dense',
  hybrid_candidates: 20,
  similarity_threshold: 0.35,
  mode: 'fast',
  max_rewrites: 2,
  max_regenerations: 1,
  chat_provider: 'gemini',
  chat_fallback: true,
  gemini_model: '',
  groq_model: '',
  judge_model: '',
  gemini_backup_model: '',
  groq_backup_model: '',
};
const PROVIDERS: ChatProvider[] = [
  { name: 'gemini', model: 'gemini-3.1-flash-lite', configured: true },
  { name: 'groq', model: 'openai/gpt-oss-120b', configured: true },
];

function recommendation(extra: Partial<Recommendation> = {}): Recommendation {
  return {
    id: 4,
    created_at: '2026-10-12T10:00:00+00:00',
    status: 'pending',
    decided_at: null,
    stale: false,
    report: {
      changes: { retrieval: 'hybrid', similarity_threshold: 0.68 },
      recommended_config: { ...CONFIG, retrieval: 'hybrid', similarity_threshold: 0.68 },
      baseline_metrics: { recall_at_k: 0.988, mrr: 0.85, safe_threshold: 0.68 },
      winner_metrics: { recall_at_k: 1.0, mrr: 0.881, safe_threshold: 0.68 },
      summary: 'Switch to hybrid search.',
      rationale: 'Recall reaches 1.0.',
      risks: 'None',
    },
    ...extra,
  };
}

const IDLE: OptimizerRun = {
  state: 'idle',
  test_set: '',
  steps: [],
  outcome: '',
  recommendation_id: null,
  error: '',
  started_at: null,
  finished_at: null,
};

describe('Settings', () => {
  let fixture: ComponentFixture<Settings>;
  let http: HttpTestingController;
  let page: Settings;
  const text = () => fixture.nativeElement.textContent as string;

  async function load(
    rec: Recommendation | null,
    job: OptimizerRun = IDLE,
    providers: ChatProvider[] = PROVIDERS,
  ) {
    http.expectOne(`${BASE}/health`).flush({ status: 'ok' });
    http.expectOne(`${BASE}/config`).flush(CONFIG);
    http.expectOne(`${BASE}/config/providers`).flush(providers);
    http.expectOne(`${BASE}/optimizer/recommendation`).flush(rec);
    http.expectOne(`${BASE}/optimizer/run`).flush(job);
    await fixture.whenStable();
  }

  beforeEach(async () => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: API_BASE_URL, useValue: BASE },
      ],
    });
    TestBed.overrideComponent(Settings, {
      remove: { imports: [ModelsCard] },
      add: { imports: [ModelsCardStub] },
    });
    fixture = TestBed.createComponent(Settings);
    page = fixture.componentInstance;
    http = TestBed.inject(HttpTestingController);
    await fixture.whenStable();
  });

  afterEach(() => {
    vi.useRealTimers();
    http.verify();
  });

  it('explains the configuration in groups with plain names', async () => {
    await load(null);
    for (const label of ['Document chunking', 'Search', 'Answering', 'Search method']) {
      expect(text()).toContain(label);
    }
    expect(text()).toContain('No recommendation yet');
  });

  it('shows what a recommendation changes and its measured effect', async () => {
    await load(recommendation());
    expect(text()).toContain('Switch to hybrid search.');
    expect(page.changes()).toEqual([
      expect.objectContaining({ label: 'Search method', from: 'Dense', to: 'Hybrid' }),
      expect.objectContaining({ label: 'Similarity threshold', from: '0.35', to: '0.68' }),
    ]);
    expect(page.delta(recommendation(), 'recall_at_k')).toBe('+0.012');
    expect(page.step()).toBe(2); // measure -> REVIEW -> apply
    expect(text()).toContain('A recommendation is waiting for you');
  });

  it('after applying: "from" comes from the measured base config, changed settings are marked', async () => {
    const applied = recommendation({ status: 'applied', decided_at: '2026-10-12T11:00:00+00:00' });
    applied.report.base_config = CONFIG; // measured against dense / 0.35
    http.expectOne(`${BASE}/health`).flush({ status: 'ok' });
    // The active config already holds the new values.
    http
      .expectOne(`${BASE}/config`)
      .flush({ ...CONFIG, retrieval: 'hybrid', similarity_threshold: 0.68 });
    http.expectOne(`${BASE}/config/providers`).flush(PROVIDERS);
    http.expectOne(`${BASE}/optimizer/recommendation`).flush(applied);
    http.expectOne(`${BASE}/optimizer/run`).flush(IDLE);
    await fixture.whenStable();
    expect(page.changes()[0]).toEqual(expect.objectContaining({ from: 'Dense', to: 'Hybrid' }));
    expect(page.step()).toBe(3);
    expect(page.changedKeys().has('retrieval')).toBe(true);
    expect(text()).toContain('changed');
  });

  it('switches the answer mode directly, after a confirmation that names the cost', async () => {
    await load(null);
    expect(fixture.nativeElement.querySelector('.mode-switch')).toBeTruthy(); // Fast/Agentic/Auto
    const ask = vi.spyOn(TestBed.inject(ConfirmService), 'ask').mockReturnValue(of(true));
    page.changeMode('auto');
    expect(ask.mock.calls[0][0].title).toBe('Switch to Auto mode?');
    const req = http.expectOne(`${BASE}/config/mode`);
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ mode: 'auto' });
    req.flush({ ...CONFIG, mode: 'auto' });
    await fixture.whenStable();
    expect(page.config()?.mode).toBe('auto');
    expect(page.notice()).toContain('Used from the next question');
  });

  it('keeps the mode when the switch is cancelled or fails', async () => {
    await load(null);
    vi.spyOn(TestBed.inject(ConfirmService), 'ask')
      .mockReturnValueOnce(of(false))
      .mockReturnValueOnce(of(true));
    // Click the real toggle: after Cancel the highlighted button must go back to "Fast" (the
    // [value] binding alone doesn't re-render it - the bug this test guards against).
    const host = fixture.nativeElement as HTMLElement;
    const button = (label: string) =>
      [...host.querySelectorAll('.mode-switch button')].find((b) =>
        b.textContent?.includes(label),
      ) as HTMLButtonElement;
    const selected = () =>
      host.querySelector('.mode-switch .mat-button-toggle-checked')?.textContent?.trim();
    expect(selected()).toBe('Fast');
    button('Agentic').click(); // cancelled: nothing is sent
    await fixture.whenStable();
    http.expectNone(`${BASE}/config/mode`);
    expect(page.config()?.mode).toBe('fast');
    expect(selected()).toBe('Fast');
    button('Agentic').click(); // confirmed, but the API refuses
    http
      .expectOne(`${BASE}/config/mode`)
      .flush({ detail: 'nope' }, { status: 500, statusText: 'Server Error' });
    await fixture.whenStable();
    expect(page.config()?.mode).toBe('fast');
    expect(selected()).toBe('Fast');
    expect(page.error()).toBeTruthy();
  });

  it('switches the answer model and its fallback directly; keeps them when saving fails', async () => {
    await load(null);
    expect(text()).toContain('Answer model');
    expect(text()).toContain('Model: gemini-3.1-flash-lite');
    expect(page.answerModel({ ...CONFIG, gemini_model: 'gemini-pro' }, 'gemini')).toBe(
      'gemini-pro',
    );
    page.changeProvider('groq');
    const req = http.expectOne(`${BASE}/config/model`);
    expect(req.request.body).toEqual({ chat_provider: 'groq', chat_fallback: true });
    req.flush({ ...CONFIG, chat_provider: 'groq' });
    await fixture.whenStable();
    expect(page.notice()).toBe('Answers: Groq, then Gemini. Used from the next question.');

    const slide = { checked: false };
    page.changeFallback(false, slide as never);
    http
      .expectOne(`${BASE}/config/model`)
      .flush({ detail: 'nope' }, { status: 500, statusText: 'Server Error' });
    expect(slide.checked).toBe(true); // put back: fallback is still on
    expect(page.config()?.chat_fallback).toBe(true);
  });

  it('without a second provider, fallback can still use a backup model', async () => {
    await load(null, IDLE, [PROVIDERS[0], { ...PROVIDERS[1], configured: false }]);
    expect(page.canFallBack()).toBe(false);
    expect(text()).toContain('its backup model (if chosen below) answers instead');
    expect(text()).toContain('add GROQ_API_KEY to .env');
    const slide = fixture.nativeElement.querySelector('mat-slide-toggle button');
    expect(slide.disabled).toBe(false); // the switch stays usable for the backup model
  });

  it('shows settings in plain words', () => {
    http.match(() => true).forEach((r) => r.flush(null));
    expect(page.display('chunk_size', 800)).toBe('800 chars');
    expect(page.display('retrieval', 'hybrid')).toBe('Hybrid');
    expect(page.display('similarity_threshold', 0.68)).toBe('0.68');
    expect(page.display('max_rewrites', 2)).toBe('2');
  });

  it('applies after confirmation and reloads', async () => {
    await load(recommendation());
    vi.spyOn(TestBed.inject(ConfirmService), 'ask').mockReturnValue(of(true));
    page.decide('apply');
    http.expectOne(`${BASE}/optimizer/recommendation/4/apply`).flush(null);
    expect(page.notice()).toContain('Applied');
    await load(recommendation({ status: 'applied', decided_at: '2026-10-12T11:00:00+00:00' }));
    expect(text()).toContain('Recommendation applied');
  });

  it('does nothing when the confirmation is cancelled', async () => {
    await load(recommendation());
    vi.spyOn(TestBed.inject(ConfirmService), 'ask').mockReturnValue(of(false));
    page.decide('reject');
    http.expectNone(`${BASE}/optimizer/recommendation/4/reject`);
  });

  it('cannot apply a stale recommendation', async () => {
    await load(recommendation({ stale: true }));
    expect(text()).toContain('configuration changed after this recommendation');
    const apply = [...fixture.nativeElement.querySelectorAll('button')].find((b: HTMLElement) =>
      b.textContent?.includes('Apply'),
    ) as HTMLButtonElement;
    expect(apply.disabled).toBe(true);
  });

  it('starts the optimizer, shows its progress and then the new recommendation', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] }); // only the polling clock
    await load(null);
    vi.spyOn(TestBed.inject(ConfirmService), 'ask').mockReturnValue(of(true)); // the page explains the AI cost first
    page.optimize();
    const start = http.expectOne(`${BASE}/optimizer/runs`);
    expect((start.request.body as FormData).has('golden')).toBe(false); // default test set
    start.flush({ ...IDLE, state: 'running', test_set: 'golden.jsonl' });
    await fixture.whenStable();
    expect(text()).toContain('Optimizing on golden.jsonl');

    vi.advanceTimersByTime(Settings.POLL_MS); // still running: progress lines appear
    http.expectOne(`${BASE}/optimizer/run`).flush({
      ...IDLE,
      state: 'running',
      test_set: 'golden.jsonl',
      steps: ['evaluate: baseline recall=0.98 mrr=0.85'],
    });
    await fixture.whenStable();
    expect(text()).toContain('evaluate: baseline recall=0.98');

    vi.advanceTimersByTime(Settings.POLL_MS); // done: the page reloads the recommendation
    http.expectOne(`${BASE}/optimizer/run`).flush({
      ...IDLE,
      state: 'done',
      test_set: 'golden.jsonl',
      outcome: 'recommendation',
      recommendation_id: 4,
    });
    const done = { ...IDLE, state: 'done' as const, outcome: 'recommendation' as const };
    await load(recommendation(), { ...done, test_set: 'golden.jsonl' });
    expect(text()).toContain('A recommendation is waiting for you');
    vi.advanceTimersByTime(Settings.POLL_MS * 2);
    http.expectNone(`${BASE}/optimizer/run`); // polling stopped
  });

  it("runs the optimizer on the admin's own test set", async () => {
    await load(null);
    vi.spyOn(TestBed.inject(ConfirmService), 'ask').mockReturnValue(of(true));
    page.optimize(new File(['{}'], 'hr_questions.jsonl'));
    const req = http.expectOne(`${BASE}/optimizer/runs`);
    expect(((req.request.body as FormData).get('golden') as File).name).toBe('hr_questions.jsonl');
    req.flush({ ...IDLE, state: 'done', test_set: 'hr_questions.jsonl', outcome: 'no-change' });
    await fixture.whenStable();
    expect(text()).toContain('already the best on hr_questions.jsonl');
  });

  it('does not start the optimizer when the cost question is cancelled', async () => {
    await load(null);
    vi.spyOn(TestBed.inject(ConfirmService), 'ask').mockReturnValue(of(false));
    page.optimize();
    http.expectNone(`${BASE}/optimizer/runs`);
  });

  it('keeps showing a run that is still going after a page reload, and its errors', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] }); // only the polling clock
    await load(null, { ...IDLE, state: 'running', test_set: 'golden.jsonl' });
    expect(page.optimizing()).toBe(true);
    vi.advanceTimersByTime(Settings.POLL_MS);
    http.expectOne(`${BASE}/optimizer/run`).flush({
      ...IDLE,
      state: 'failed',
      error: 'The AI service is slow or overloaded right now.',
    });
    await fixture.whenStable();
    expect(text()).toContain('slow or overloaded');
    expect(page.optimizing()).toBe(false);
  });

  it('hides the optimizer while a recommendation waits for a decision', async () => {
    await load(recommendation());
    expect(text()).not.toContain('Find better settings');
  });

  it('offers the optimizer again when the waiting recommendation is stale', async () => {
    await load(recommendation({ stale: true }));
    expect(text()).toContain('Run the optimizer again below');
    expect(text()).toContain('Find better settings'); // ...and "below" really exists
  });
});
