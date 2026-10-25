// Evaluation page with a mocked backend: headline gates, quality check (default and uploaded
// test set), filters and hiding runs.
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { MatTabGroup } from '@angular/material/tabs';
import { By } from '@angular/platform-browser';

import { API_BASE_URL } from '../../core/api';
import { EvalRunDetail, EvalRunSummary } from '../../core/models';
import { EvalRuns } from './eval-runs';

const BASE = 'http://api.test';
const EMPTY_USAGE = {
  sample_rate: 0.2,
  include_admin: false,
  admin_chats: 0,
  recent: [],
  totals: {
    questions: 0,
    answered: 0,
    unanswered: 0,
    thumbs_up: 0,
    thumbs_down: 0,
    scored: 0,
    faithfulness: null,
    answer_relevancy: null,
  },
  unanswered: [],
  thumbs_down: [],
  low_scores: [],
  documents: [],
};

function run(id: number, extra: Partial<EvalRunSummary> = {}): EvalRunSummary {
  return {
    id,
    created_at: '2026-10-12T10:00:00+00:00',
    label: `run-${id}`,
    metrics: { recall_at_k: 0.988, mrr: 0.85, n_questions: 50 },
    kind: 'check',
    hidden: false,
    current: false,
    gates: 'pass',
    test_set: 'golden.jsonl',
    ...extra,
  };
}

function detail(summary: EvalRunSummary): EvalRunDetail {
  return {
    ...summary,
    config: { rag: { chunk_size: 800 } as never },
    results: [
      {
        id: 'q1',
        question: 'Who leads the Retrieval squad?',
        answerable: true,
        predicted_answerable: true,
        retrieval: { rr: 0.5, recall: 1 },
      },
    ],
    gates: [
      {
        label: 'Recall@5',
        metric: 'recall_at_k',
        op: '>=',
        threshold: 0.85,
        blocking: true,
        value: 0.988,
        status: 'pass',
      },
      {
        label: 'Abstention accuracy',
        metric: 'abstention_accuracy',
        op: '>=',
        threshold: 0.9,
        blocking: true,
        value: 0.8,
        status: 'n/a',
      },
    ],
  };
}

describe('EvalRuns', () => {
  let fixture: ComponentFixture<EvalRuns>;
  let http: HttpTestingController;
  let page: EvalRuns;
  const text = () => fixture.nativeElement.textContent as string;

  beforeEach(async () => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: API_BASE_URL, useValue: BASE },
      ],
    });
    fixture = TestBed.createComponent(EvalRuns);
    page = fixture.componentInstance;
    http = TestBed.inject(HttpTestingController);
    await fixture.whenStable();
    // The page opens on "Live quality" (tested in live-quality.spec.ts): answer its request,
    // then switch to the "Test-set checks" tab that these tests are about.
    http.match(`${BASE}/quality/usage`).forEach((r) => r.flush(EMPTY_USAGE));
    http.match((r) => r.url === `${BASE}/collections`).forEach((r) => r.flush([]));
    await fixture.whenStable();
    http.match((r) => r.url === `${BASE}/assistant`).forEach((r) => r.flush({ topics: [] }));
    fixture.debugElement.query(By.directive(MatTabGroup)).componentInstance.selectedIndex = 1;
    await fixture.whenStable();
  });

  afterEach(() => http.verify());

  async function loadRuns(runs: EvalRunSummary[]) {
    http.expectOne(`${BASE}/eval/runs`).flush(runs); // oldest first, like the API
    await fixture.whenStable();
  }

  it('headlines the newest run of the active config, with gates in plain language', async () => {
    const current = run(2, { current: true });
    await loadRuns([run(1, { kind: 'experiment' }), current]);
    http.expectOne(`${BASE}/eval/runs/2`).flush(detail(current));
    await fixture.whenStable();
    expect(text()).toContain('Meets the release gates');
    expect(text()).toContain('0.988');
    expect(text()).toContain('needs a full evaluation'); // abstention not measured by a check
  });

  it('runs a quality check and opens the new run', async () => {
    await loadRuns([]);
    page.runCheck();
    const req = http.expectOne(`${BASE}/eval/runs`);
    expect(req.request.method).toBe('POST');
    expect((req.request.body as FormData).has('golden')).toBe(false); // default test set
    req.flush(run(3, { current: true }));
    await fixture.whenStable();
    expect(page.notice()).toContain('meets the release gates');
    await loadRuns([run(3, { current: true })]);
    http.match(`${BASE}/eval/runs/3`).forEach((r) => r.flush(detail(run(3))));
  });

  it('sends an uploaded test set with the check', async () => {
    await loadRuns([]);
    page.runCheck(new File(['{}'], 'hr_questions.jsonl'));
    const req = http.expectOne((r) => r.method === 'POST');
    const form = req.request.body as FormData;
    expect((form.get('golden') as File).name).toBe('hr_questions.jsonl');
    expect(form.get('label')).toBe('check-hr_questions');
    req.flush({ detail: 'Invalid test set: quote not found' }, { status: 422, statusText: 'x' });
    expect(page.error()).toContain('Invalid test set');
  });

  it('filters by type and hides runs', async () => {
    await loadRuns([run(1, { kind: 'experiment' }), run(2), run(3, { hidden: true })]);
    expect(page.visible().map((r) => r.id)).toEqual([2, 1]); // hidden left out, newest first
    page.filter.set('experiment');
    expect(page.visible().map((r) => r.id)).toEqual([1]);
    page.filter.set('all');
    page.showHidden.set(true);
    expect(page.visible().map((r) => r.id)).toEqual([3, 2, 1]);

    page.toggleHidden(page.runs()[1], new Event('click'));
    const req = http.expectOne(`${BASE}/eval/runs/2/hidden`);
    expect(req.request.body).toEqual({ hidden: true });
    req.flush(null);
    expect(page.runs().find((r) => r.id === 2)!.hidden).toBe(true);
  });

  it('lists only search metrics for search-only runs', async () => {
    await loadRuns([]);
    const check = detail(run(1));
    expect(page.metricsFor(check)).not.toContain('unanswerable_answered_rate');
    expect(page.metricsFor({ ...check, kind: 'full' })).toContain('unanswerable_answered_rate');
  });

  it('lists the questions whose answer passage was not ranked first', async () => {
    await loadRuns([run(1)]);
    expect(page.missed(detail(run(1)))).toEqual([
      { question: 'Who leads the Retrieval squad?', where: 'found at rank 2' },
    ]);
  });
});
