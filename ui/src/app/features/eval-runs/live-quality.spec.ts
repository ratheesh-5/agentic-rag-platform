// Live quality tab with a mocked backend: totals, clickable tiles that filter the answers list,
// "Test a question", the admin-chats switch and suggestions (GET /quality/usage, POST /quality/test).
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { API_BASE_URL } from '../../core/api';
import { Collection, QualityTestResult, RecentAnswer, UsageSummary } from '../../core/models';
import { LiveQuality } from './live-quality';

const BASE = 'http://api.test';
const collection = (id: number, name: string): Collection => ({
  id,
  name,
  description: '',
  documents: 5,
  passages: 30,
  is_default: id === 1,
  is_private: false,
  can_manage: true,
  threshold: 0.68,
  threshold_source: 'measured',
  threshold_override: null,
});
const COLLECTIONS = [collection(1, 'AI Engineering Handbook'), collection(2, 'HR policies')];

function answer(id: number, extra: Partial<RecentAnswer> = {}): RecentAnswer {
  return {
    message_id: id,
    question: `question ${id}`,
    answer: `answer ${id} [1].`,
    documents: ['rag_design_guide.pdf'],
    faithfulness: null,
    answer_relevancy: null,
    created_at: '2026-10-12T10:00:00+00:00',
    answerable: true,
    abstain_reason: '',
    feedback: null,
    by_admin: false,
    ...extra,
  };
}

const USAGE: UsageSummary = {
  sample_rate: 0.2,
  include_admin: false,
  admin_chats: 5,
  totals: {
    questions: 4,
    answered: 3,
    unanswered: 1,
    escalated: 2,
    thumbs_up: 1,
    thumbs_down: 1,
    scored: 2,
    faithfulness: 0.95,
    answer_relevancy: 0.62,
  },
  recent: [
    answer(4, { question: 'What is LangGraph?', answerable: false, documents: [] }),
    answer(3, { faithfulness: 0.9, answer_relevancy: 0.4 }),
    answer(2, { feedback: -1 }),
    answer(1, { faithfulness: 1, answer_relevancy: 0.9, feedback: 1 }),
  ],
  unanswered: [
    {
      question: 'What is LangGraph?',
      understood_as: null,
      reason: 'no passage above similarity threshold',
      last_asked: '2026-10-12T10:00:00+00:00',
      count: 3,
    },
  ],
  thumbs_down: [answer(2, { feedback: -1 })],
  low_scores: [answer(3, { faithfulness: 0.9, answer_relevancy: 0.4 })],
  documents: [
    {
      document: 'rag_design_guide.pdf',
      answers: 3,
      thumbs_down: 1,
      faithfulness: 0.95,
      answer_relevancy: 0.62,
    },
  ],
};

describe('LiveQuality', () => {
  let fixture: ComponentFixture<LiveQuality>;
  let http: HttpTestingController;
  let page: LiveQuality;
  const text = () => fixture.nativeElement.textContent as string;

  async function load(usage: UsageSummary = USAGE) {
    http.expectOne(`${BASE}/quality/usage`).flush(usage);
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
    fixture = TestBed.createComponent(LiveQuality);
    page = fixture.componentInstance;
    http = TestBed.inject(HttpTestingController);
    // The test box searches one collection (the selected one) and shows its example questions.
    http.expectOne((r) => r.url === `${BASE}/collections`).flush(COLLECTIONS);
    await fixture.whenStable();
    const topics = http.expectOne((r) => r.url === `${BASE}/assistant`);
    expect(topics.request.params.get('collection_id')).toBe('1');
    topics.flush({
      name: 'x',
      description: '',
      topics: [{ title: 'RAG', description: '', examples: ['What does HyDE do?'] }],
    });
    await fixture.whenStable();
  });

  afterEach(() => http.verify());

  it('shows totals, scores and the sample rate in plain words', async () => {
    await load();
    expect(text()).toContain('1 in 5 answers');
    expect(text()).toContain('25%'); // 1 of 4 not answered
    expect(text()).toContain('0.950');
    expect(text()).toContain('Include admin chats (5)');
    expect(text()).toContain('2 escalated to agentic'); // auto mode handed 2 to the agent
    // Relevance 0.62 is below 0.7: its tile is marked weak.
    expect(fixture.nativeElement.querySelector('.kpi.weak').textContent).toContain('relevance');
  });

  it('filters the answers list from a KPI tile, the chips and the search box', async () => {
    await load();
    expect(page.answers().length).toBe(4);
    page.show('unanswered');
    expect(page.answers().map((a) => a.message_id)).toEqual([4]);
    page.show('low');
    expect(page.answers().map((a) => a.message_id)).toEqual([3]); // relevance 0.4
    page.show('disliked');
    expect(page.answers().map((a) => a.message_id)).toEqual([2]);
    page.show('all');
    page.search.set('langgraph');
    expect(page.answers().map((a) => a.message_id)).toEqual([4]);
    expect(page.count('scored')).toBe(2);
  });

  it('expands an answer to show its text', async () => {
    await load();
    page.toggle(3);
    await fixture.whenStable();
    expect(text()).toContain('answer 3 [1].');
    page.toggle(3);
    expect(page.expanded()).toBeNull();
  });

  it('reloads with admin chats when the switch is on', async () => {
    await load();
    page.includeAdmin.set(true);
    await fixture.whenStable();
    http.expectOne(`${BASE}/quality/usage?include_admin=true`).flush(USAGE);
  });

  it("tests a question in the chosen collection, with that collection's example questions", async () => {
    await load();
    expect(text()).toContain('HR policies');
    page.chooseCollection(2);
    await fixture.whenStable();
    const topics = http.expectOne((r) => r.url === `${BASE}/assistant`);
    expect(topics.request.params.get('collection_id')).toBe('2');
    topics.flush({ name: 'x', description: '', topics: [] });
    page.runTest('How many days of annual leave do employees get?');
    const req = http.expectOne(`${BASE}/quality/test`);
    expect(req.request.body.collection_id).toBe(2);
    req.flush({ detail: 'x' }, { status: 503, statusText: 'x' });
  });

  it('tests a question: answer, scores and passages, nothing saved', async () => {
    await load();
    page.runTest('What does HyDE do?');
    const req = http.expectOne(`${BASE}/quality/test`);
    expect(req.request.body).toEqual({ question: 'What does HyDE do?', collection_id: 1 });
    const result: QualityTestResult = {
      question: 'What does HyDE do?',
      answerable: true,
      answer: 'HyDE embeds a hypothetical answer [1].',
      abstain_reason: '',
      sources: [],
      passages: [
        { document: 'data/x/rag_design_guide.pdf', score: 0.8, cited: true, text: 'HyDE…' },
      ],
      faithfulness: 1,
      answer_relevancy: 0.95,
      judge_calls: 4,
      score_error: null,
      seconds: 9.5,
    };
    req.flush(result);
    await fixture.whenStable();
    expect(text()).toContain('Answered from the documents');
    expect(text()).toContain('HyDE embeds a hypothetical answer');
    expect(text()).toContain('rag_design_guide.pdf');
    expect(text()).toContain('cited');
  });

  it('shows a test error and keeps the question', async () => {
    await load();
    page.testQuestion.set('What does HyDE do?');
    page.runTest();
    http
      .expectOne(`${BASE}/quality/test`)
      .flush({ detail: 'Model quota exceeded' }, { status: 429, statusText: 'Too Many' });
    expect(page.testError()).toContain('quota');
    expect(page.testQuestion()).toBe('What does HyDE do?');
  });

  it('lists suggestions: unanswered questions (with count) and disliked answers', async () => {
    await load();
    expect(text()).toContain('×3');
    expect(text()).toContain('Disliked (1)');
  });

  it('explains the empty state and points to the admin switch', async () => {
    await load({ ...USAGE, totals: { ...USAGE.totals, questions: 0 }, recent: [] });
    expect(text()).toContain('No questions from users yet');
    expect(text()).toContain('your own 5');
  });
});
