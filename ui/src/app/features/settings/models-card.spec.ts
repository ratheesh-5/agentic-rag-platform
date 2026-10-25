// Settings › Models card with a mocked backend: model slots from the providers' lists, Test,
// the quality check that a new answer / backup model needs before it can be saved, the judge
// confirmation and the locked embedding model.
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { of } from 'rxjs';

import { API_BASE_URL } from '../../core/api';
import { ConfirmService } from '../../core/confirm';
import { ModelCheck, ModelsInfo } from '../../core/models';
import { ModelsCard } from './models-card';

const BASE = 'http://api.test';

function info(checks: ModelCheck[] = []): ModelsInfo {
  return {
    slots: [
      {
        kind: 'gemini',
        configured: true,
        current: 'gemini-lite',
        default: 'gemini-lite',
        backup: '',
        options: ['gemini-flash', 'gemini-lite', 'gemini-pro'],
      },
      {
        kind: 'groq',
        configured: false,
        current: 'groq-default',
        default: 'groq-default',
        backup: '',
        options: [],
      },
      {
        kind: 'judge',
        configured: true,
        current: 'gemini-lite',
        default: 'gemini-lite',
        backup: '',
        options: ['gemini-flash', 'gemini-lite'],
      },
    ],
    embedding: 'gemini-embedding-001',
    checks,
    check_minimum: 5,
  };
}

function check(extra: Partial<ModelCheck> = {}): ModelCheck {
  return {
    kind: 'gemini',
    model: 'gemini-pro',
    state: 'done',
    passed: 6,
    total: 6,
    ok: true,
    error: null,
    started_at: '2026-10-25T10:00:00+00:00',
    finished_at: '2026-10-25T10:01:00+00:00',
    items: [],
    ...extra,
  };
}

describe('ModelsCard', () => {
  let fixture: ComponentFixture<ModelsCard>;
  let http: HttpTestingController;
  let card: ModelsCard;
  const text = () => fixture.nativeElement.textContent as string;
  const row = (key: string) => card.rows().find((r) => r.key === key)!;

  async function load(data: ModelsInfo = info()) {
    http.expectOne(`${BASE}/config/models`).flush(data);
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
    fixture = TestBed.createComponent(ModelsCard);
    card = fixture.componentInstance;
    http = TestBed.inject(HttpTestingController);
    await fixture.whenStable();
  });

  afterEach(() => {
    vi.useRealTimers();
    http.verify();
  });

  it('shows every slot, a provider without a key and the locked embedding model', async () => {
    await load();
    for (const label of ['Gemini answer model', 'Gemini backup model', 'Judge model']) {
      expect(text()).toContain(label);
    }
    expect(text()).toContain('Not set up: add its API key');
    expect(text()).toContain('gemini-embedding-001');
    expect(card.changed()).toBe(false);
  });

  it('backup models are locked while automatic fallback is off', async () => {
    fixture.componentRef.setInput('fallback', false);
    await load();
    expect(text()).toContain('Turn on Automatic fallback to use a backup model.');
    const selects = fixture.nativeElement.querySelectorAll('mat-select');
    expect(selects[0].getAttribute('aria-disabled')).toBe('false'); // answer model: free to change
    expect(selects[1].getAttribute('aria-disabled')).toBe('true'); // its backup: locked
  });

  it('a backup cannot be the main model', async () => {
    await load();
    expect(card.options(row('gemini_backup_model'))).toEqual(['gemini-flash', 'gemini-pro']);
  });

  it('a new answer model needs a passing check before it can be saved', async () => {
    await load();
    card.set(row('gemini_model'), 'gemini-pro');
    await fixture.whenStable();
    expect(card.changed()).toBe(true);
    expect(card.allowed()).toBe(false);
    expect(text()).toContain('Run a passing quality check');

    vi.spyOn(TestBed.inject(ConfirmService), 'ask').mockReturnValue(of(true));
    card.check(row('gemini_model'), 'gemini-pro');
    const req = http.expectOne(`${BASE}/config/models/check`);
    expect(req.request.body).toEqual({ kind: 'gemini', model: 'gemini-pro' });
    req.flush(check({ state: 'running', passed: 0, total: 0, ok: false }));
    http.expectOne(`${BASE}/config/models`).flush(info([check()])); // reloaded: passed
    await fixture.whenStable();
    expect(text()).toContain('Quality 6/6 · passed');
    expect(card.allowed()).toBe(true);

    card.save();
    const put = http.expectOne(`${BASE}/config/models`);
    expect(put.request.method).toBe('PUT');
    expect(put.request.body.gemini_model).toBe('gemini-pro');
    put.flush({});
    http.expectOne(`${BASE}/config/models`).flush(info([check()]));
    expect(card.notice()).toContain('Models saved');
  });

  it('a failed check keeps the model out', async () => {
    await load(info([check({ passed: 3, ok: false })]));
    card.set(row('gemini_backup_model'), 'gemini-pro');
    await fixture.whenStable();
    expect(text()).toContain('Quality 3/6 · not enough');
    expect(card.allowed()).toBe(false);
  });

  it('polls while a check runs', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] }); // only the polling clock
    await load(info([check({ state: 'running', ok: false })]));
    vi.advanceTimersByTime(ModelsCard.POLL_MS);
    http.expectOne(`${BASE}/config/models`).flush(info([check()]));
    vi.advanceTimersByTime(ModelsCard.POLL_MS);
    http.expectNone(`${BASE}/config/models`); // finished: polling stopped
  });

  it('Test calls exactly the chosen model and shows the result', async () => {
    await load();
    card.test(row('gemini_model'), 'gemini-flash');
    expect(card.testOf('gemini', 'gemini-flash')).toBe('running');
    http
      .expectOne(`${BASE}/config/models/test`)
      .flush({ ok: false, message: 'Out of quota for this model right now.', seconds: 0.4 });
    await fixture.whenStable();
    card.set(row('gemini_model'), 'gemini-flash');
    await fixture.whenStable();
    expect(text()).toContain('Out of quota for this model right now.');
  });

  it('the judge needs no check but asks before switching', async () => {
    await load();
    const ask = vi.spyOn(TestBed.inject(ConfirmService), 'ask').mockReturnValue(of(false));
    card.set(row('judge_model'), 'gemini-flash');
    expect(card.allowed()).toBe(true);
    card.save();
    expect(ask).toHaveBeenCalled();
    http.expectNone(`${BASE}/config/models`); // cancelled: nothing saved
  });
});
