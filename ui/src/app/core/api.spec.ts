// Tests for the API client, the guest-id / admin-key interceptor and error messages.
// HttpTestingController intercepts HttpClient: no real request ever leaves the test
// (.NET analogy: a mocked HttpMessageHandler).
import { HttpErrorResponse, provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { API_BASE_URL, ApiService, apiKeyInterceptor } from './api';
import { ApiKeyStore } from './api-key.store';
import { describeError } from './errors';

const BASE = 'http://api.test';

describe('ApiService + apiKeyInterceptor', () => {
  let api: ApiService;
  let http: HttpTestingController;
  let keys: ApiKeyStore;

  beforeEach(() => {
    sessionStorage.clear();
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptors([apiKeyInterceptor])),
        provideHttpClientTesting(),
        { provide: API_BASE_URL, useValue: BASE },
      ],
    });
    api = TestBed.inject(ApiService);
    http = TestBed.inject(HttpTestingController);
    keys = TestBed.inject(ApiKeyStore);
  });

  afterEach(() => http.verify()); // fails if a request was made but not expected

  it('adds X-API-Key to API requests once a key is set', () => {
    keys.set('  secret  ');
    api.documents().subscribe();
    const req = http.expectOne(`${BASE}/documents`);
    expect(req.request.headers.get('X-API-Key')).toBe('secret'); // trimmed
    req.flush([]);
  });

  it('sends only the guest id when not signed in as admin', () => {
    api.health().subscribe();
    const req = http.expectOne(`${BASE}/health`);
    expect(req.request.headers.has('X-API-Key')).toBe(false);
    expect(req.request.headers.get('X-Guest-Id')).toBe(keys.guestId);
    req.flush({ status: 'ok' });
  });

  it('keeps the same guest id across app restarts (localStorage)', () => {
    expect(keys.guestId).toMatch(/^[0-9a-f-]{36}$/);
    expect(localStorage.getItem('rag.guestId')).toBe(keys.guestId);
  });

  it('never sends the key to other hosts', () => {
    keys.set('secret');
    TestBed.inject(ApiService)['http'].get('https://elsewhere.example/x').subscribe();
    const req = http.expectOne('https://elsewhere.example/x');
    expect(req.request.headers.has('X-API-Key')).toBe(false);
    expect(req.request.headers.has('X-Guest-Id')).toBe(false);
    req.flush({});
  });

  it('uploads as multipart form data with a "file" field', () => {
    const file = new File(['# hello'], 'note.md', { type: 'text/markdown' });
    api.upload(file).subscribe();
    const req = http.expectOne(`${BASE}/documents`);
    expect(req.request.method).toBe('POST');
    expect((req.request.body as FormData).get('file')).toBeInstanceOf(File);
    req.flush({});
  });

  it('keeps the key in sessionStorage and clears it', () => {
    keys.set('abc');
    expect(sessionStorage.getItem('rag.apiKey')).toBe('abc');
    keys.clear();
    expect(keys.key()).toBe('');
    expect(sessionStorage.getItem('rag.apiKey')).toBeNull();
  });
});

describe('describeError', () => {
  const err = (status: number, detail?: unknown) =>
    new HttpErrorResponse({ status, error: detail === undefined ? null : { detail } });

  it('explains common API errors', () => {
    expect(describeError(err(0))).toContain('Cannot reach the API');
    expect(describeError(err(401))).toContain('Admin sign-in');
    expect(describeError(err(502))).toContain('can’t be reached right now'); // API down
    expect(describeError(err(504))).toContain('can’t be reached right now'); // API too slow
    expect(describeError(err(415, 'Unsupported file type .exe'))).toBe(
      'Unsupported file type .exe',
    );
    expect(describeError(err(429))).toContain('quota');
    expect(describeError(err(500))).toBe('Request failed (500).');
  });

  it('falls back when FastAPI returns a validation list instead of a message', () => {
    expect(describeError(err(422, [{ msg: 'too short' }]))).toBe('The request was not valid.');
  });
});
