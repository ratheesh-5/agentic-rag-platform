// Usage page with a mocked backend: totals, the per-day chart, and the interactive calls list
// (tiles, bars and rows filter it; sort, search, expand, LangSmith trace links).
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { API_BASE_URL } from '../../core/api';
import { UsageCall, UsageReport, UsageStats } from '../../core/models';
import { Usage } from './usage';

const BASE = 'http://api.test';

function stats(extra: Partial<UsageStats> = {}): UsageStats {
  return {
    calls: 0,
    embed_texts: 0,
    input_tokens: 0,
    output_tokens: 0,
    errors: 0,
    avg_latency_ms: null,
    p95_latency_ms: null,
    cost: null,
    ...extra,
  };
}

function call(id: number, extra: Partial<UsageCall> = {}): UsageCall {
  return {
    id,
    created_at: '2026-10-12T10:00:00+00:00',
    model: 'gemini-x',
    purpose: 'answer',
    kind: 'chat',
    latency_ms: 4000,
    input_tokens: 1000,
    output_tokens: 100,
    texts: null,
    error: null,
    run_id: `run-${id}`,
    cost: null,
    ...extra,
  };
}

function report(extra: Partial<UsageReport> = {}): UsageReport {
  return {
    days: 14,
    totals: stats({
      calls: 3,
      embed_texts: 2,
      input_tokens: 12000,
      output_tokens: 900,
      avg_latency_ms: 4200,
      p95_latency_ms: 9000,
      errors: 1,
    }),
    priced_models: [],
    by_purpose: [
      { purpose: 'answer', ...stats({ calls: 2, input_tokens: 9000, avg_latency_ms: 4000 }) },
      { purpose: 'embed', ...stats({ embed_texts: 2, avg_latency_ms: 700 }) },
    ],
    by_model: [{ model: 'gemini-x', ...stats({ calls: 3 }) }],
    per_day: [
      { date: '2026-10-11', ...stats({ calls: 1 }) },
      { date: '2026-10-12', ...stats({ calls: 2 }) },
      { date: '2026-10-13', ...stats() },
    ],
    calls: [
      call(4, {
        kind: 'embed',
        purpose: 'embed',
        model: 'emb',
        input_tokens: null,
        output_tokens: null,
        texts: 2,
        latency_ms: 700,
        run_id: null,
      }),
      call(3, { latency_ms: 9100, input_tokens: 5000 }),
      call(2, { purpose: 'agent: grade', error: 'ResourceExhausted: 429', latency_ms: 300 }),
      call(1, { created_at: '2026-10-11T09:00:00+00:00' }),
    ],
    recent_errors: [],
    tracing: {
      enabled: true,
      project: 'rag-platform',
      sampling_rate: 1,
      url: 'https://smith.langchain.com',
    },
    ...extra,
  };
}

describe('Usage', () => {
  let fixture: ComponentFixture<Usage>;
  let http: HttpTestingController;
  let page: Usage;
  const text = () => fixture.nativeElement.textContent as string;
  const ids = () => page.calls().map((c) => c.id);

  beforeEach(async () => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: API_BASE_URL, useValue: BASE },
      ],
    });
    fixture = TestBed.createComponent(Usage);
    page = fixture.componentInstance;
    http = TestBed.inject(HttpTestingController);
    await fixture.whenStable();
    http.expectOne(`${BASE}/usage/summary?days=14`).flush(report());
    await fixture.whenStable();
  });

  afterEach(() => http.verify());

  it('shows totals in plain units, the tracing status and every call newest first', () => {
    expect(text()).toContain('12.9k'); // tokens in + out
    expect(text()).toContain('4.2 s'); // average latency
    expect(text()).toContain('add prices in model_prices.json');
    expect(text()).toContain('LangSmith tracing is on');
    expect(text()).toContain('Calls (4)');
    expect(ids()).toEqual([4, 3, 2, 1]);
    expect(page.size(page.calls()[0])).toBe('2 texts'); // embeddings show texts, not tokens
  });

  it('filters from the tiles: errors, slow calls, everything', () => {
    page.show('status', 'errors');
    expect(ids()).toEqual([2]);
    page.show('status', 'slow'); // >= p95 (9000 ms)
    expect(ids()).toEqual([3]);
    page.show('status', 'slow'); // clicking the active tile again clears it
    expect(ids()).toEqual([4, 3, 2, 1]);
    page.show('purpose', 'answer');
    page.showAll();
    expect(page.hasFilters()).toBe(false);
  });

  it('filters by a clicked day, purpose and model, and combines them', () => {
    page.show('day', '2026-10-11');
    expect(ids()).toEqual([1]);
    page.show('day', '2026-10-11'); // toggle off
    page.show('purpose', 'answer');
    page.show('model', 'gemini-x');
    expect(ids()).toEqual([3, 1]);
    page.clearFilters();
    expect(ids()).toEqual([4, 3, 2, 1]);
  });

  it('sorts by latency or tokens and searches errors', () => {
    page.sort.set('slowest');
    expect(ids()).toEqual([3, 1, 4, 2]);
    page.showBiggest();
    expect(ids()[0]).toBe(3);
    page.sort.set('newest');
    page.search.set('429');
    expect(ids()).toEqual([2]);
  });

  it('scales the day bars and gives empty days no bar', () => {
    expect(page.bars().map((b) => b.height)).toEqual([50, 100, 0]);
  });

  it('expands a call and opens its LangSmith trace', async () => {
    page.toggle(3);
    await fixture.whenStable();
    expect(text()).toContain('Open trace in LangSmith');
    const open = vi.spyOn(window, 'open').mockReturnValue(null);
    page.openTrace(page.calls()[1]);
    http.expectOne(`${BASE}/usage/trace/run-3`).flush({ url: 'https://smith.langchain.com/x' });
    expect(open).toHaveBeenCalledWith('https://smith.langchain.com/x', '_blank', 'noopener');
  });

  it('reloads for another time range', async () => {
    page.days.set(7);
    await fixture.whenStable();
    http.expectOne(`${BASE}/usage/summary?days=7`).flush(report({ days: 7 }));
  });

  it('formats numbers and money', () => {
    expect(page.compact(1_250_000)).toBe('1.3M');
    expect(page.money(0.00123)).toBe('$0.0012');
    expect(page.money(null)).toBe('–');
    expect(page.plural(1, 'call')).toBe('1 call');
  });
});
