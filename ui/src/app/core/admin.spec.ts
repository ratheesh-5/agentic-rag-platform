// Admin sign-in dialog (key checked with GET /me before it is stored) and the admin route guard.
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { MatDialogRef } from '@angular/material/dialog';
import { Router, UrlTree, provideRouter } from '@angular/router';

import { AdminSignIn } from './admin-sign-in';
import { adminGuard } from './admin.guard';
import { API_BASE_URL, apiKeyInterceptor } from './api';
import { ApiKeyStore } from './api-key.store';

const BASE = 'http://api.test';

describe('Admin sign in', () => {
  const close = vi.fn();

  beforeEach(() => {
    sessionStorage.clear();
    close.mockReset();
    TestBed.configureTestingModule({
      providers: [
        provideRouter([]),
        provideHttpClient(withInterceptors([apiKeyInterceptor])),
        provideHttpClientTesting(),
        { provide: API_BASE_URL, useValue: BASE },
        { provide: MatDialogRef, useValue: { close } },
      ],
    });
  });

  it('stores the key only after the API confirms it is an admin key', () => {
    const page = TestBed.createComponent(AdminSignIn).componentInstance;
    const http = TestBed.inject(HttpTestingController);
    page.key.set(' good-key ');
    page.signIn();
    const req = http.expectOne(`${BASE}/me`);
    expect(req.request.headers.get('X-API-Key')).toBe('good-key');
    req.flush({ role: 'admin' });
    expect(TestBed.inject(ApiKeyStore).key()).toBe('good-key');
    expect(close).toHaveBeenCalledWith(true);
  });

  it('rejects a wrong key and stays a guest', () => {
    const page = TestBed.createComponent(AdminSignIn).componentInstance;
    page.key.set('typo');
    page.signIn();
    TestBed.inject(HttpTestingController)
      .expectOne(`${BASE}/me`)
      .flush({ detail: 'Invalid or missing API key' }, { status: 401, statusText: 'Unauthorized' });
    expect(page.error()).toBe('Invalid admin key.');
    expect(TestBed.inject(ApiKeyStore).isAdmin()).toBe(false);
    expect(close).not.toHaveBeenCalled();
  });

  it('guards the admin pages: guests are sent to the chat', () => {
    const run = () => TestBed.runInInjectionContext(() => adminGuard({} as never, [], {} as never));
    const redirect = run() as UrlTree;
    expect(TestBed.inject(Router).serializeUrl(redirect)).toBe('/chat');
    TestBed.inject(ApiKeyStore).set('admin-key');
    expect(run()).toBe(true);
  });
});
