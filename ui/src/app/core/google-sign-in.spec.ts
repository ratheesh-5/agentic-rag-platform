// "Sign in with Google": Google's ID token is exchanged for our own session, which is kept in
// localStorage (7 days) and dropped when it has expired. Google's script is never loaded here.
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { API_BASE_URL, apiKeyInterceptor } from './api';
import { ApiKeyStore } from './api-key.store';
import { GoogleSignIn } from './google-sign-in';
import { SignInResponse } from './models';

const BASE = 'http://api.test';
const RESPONSE: SignInResponse = {
  token: 'session-token',
  expires_at: '2999-01-01T00:00:00+00:00',
  role: 'user',
  user: { email: 'alice@example.com', name: 'Alice', picture: '' },
};

describe('Sign in with Google', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    localStorage.removeItem('rag.session');
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptors([apiKeyInterceptor])),
        provideHttpClientTesting(),
        { provide: API_BASE_URL, useValue: BASE },
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function button() {
    const fixture = TestBed.createComponent(GoogleSignIn);
    fixture.componentRef.setInput('clientId', 'client.apps.googleusercontent.com');
    return fixture.componentInstance; // not rendered: Google's script stays out of tests
  }

  it("exchanges Google's ID token for our session and remembers it", () => {
    const page = button();
    const signedIn = vi.fn();
    page.signedIn.subscribe(signedIn);
    page.exchange('google-id-token');
    const req = http.expectOne(`${BASE}/auth/google`);
    expect(req.request.body).toEqual({ credential: 'google-id-token' });
    req.flush(RESPONSE);
    const store = TestBed.inject(ApiKeyStore);
    expect(store.user()?.email).toBe('alice@example.com');
    expect(store.isAdmin()).toBe(false);
    expect(JSON.parse(localStorage.getItem('rag.session')!).token).toBe('session-token');
    expect(signedIn).toHaveBeenCalledWith(RESPONSE);
  });

  it('shows why a sign-in failed', () => {
    const page = button();
    page.exchange('bad-token');
    http
      .expectOne(`${BASE}/auth/google`)
      .flush({ detail: 'Google sign-in failed.' }, { status: 401, statusText: 'Unauthorized' });
    expect(page.error()).toBeTruthy();
    expect(TestBed.inject(ApiKeyStore).session()).toBeNull();
  });

  it('forgets a stored session that has expired', () => {
    const expired = { ...RESPONSE, expires_at: '2020-01-01T00:00:00+00:00' };
    localStorage.setItem('rag.session', JSON.stringify(expired));
    expect(TestBed.inject(ApiKeyStore).session()).toBeNull();
  });
});
