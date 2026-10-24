// App shell: guests chat right away; "Sign in with Google" (or the admin-key dialog when the
// server has no Google client id); the Admin pages appear only for admins.
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { App } from './app';
import { API_BASE_URL, apiKeyInterceptor } from './core/api';
import { ApiKeyStore, Session } from './core/api-key.store';
import { ConfirmService } from './core/confirm';
import { ConversationsStore } from './core/conversations.store';
import { ConversationSummary } from './core/models';

const BASE = 'http://api.test';
const CONV: ConversationSummary = {
  id: 3,
  title: 'What is the chunk size?',
  created_at: '',
  updated_at: '',
  messages: 2,
  collection_id: 1,
  collection: 'AI Engineering Handbook',
};
const ALICE: Session = {
  token: 'session-token',
  expires_at: '2999-01-01T00:00:00+00:00',
  role: 'user',
  user: { email: 'alice@example.com', name: 'Alice', picture: '' },
};

describe('App', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    sessionStorage.clear();
    localStorage.removeItem('rag.session');
    TestBed.configureTestingModule({
      providers: [
        provideRouter([]),
        provideHttpClient(withInterceptors([apiKeyInterceptor])),
        provideHttpClientTesting(),
        { provide: API_BASE_URL, useValue: BASE },
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  /** Start the app: the API answers the sign-in config and the conversation list. */
  async function start(clientId = '', conversations: ConversationSummary[] = [CONV]) {
    const fixture = TestBed.createComponent(App);
    http.expectOne(`${BASE}/auth/config`).flush({ google_client_id: clientId });
    await fixture.whenStable(); // runs the effect that loads the conversations
    http.expectOne(`${BASE}/conversations`).flush(conversations);
    await fixture.whenStable();
    return fixture;
  }

  const text = (fixture: ComponentFixture<App>) => fixture.nativeElement.textContent as string;

  it('lets a guest chat: conversations listed, no Admin pages, an Admin sign in button', async () => {
    const fixture = await start();
    expect(text(fixture)).toContain('New chat');
    expect(text(fixture)).toContain('What is the chunk size?');
    expect(text(fixture)).toContain('Admin sign in'); // no Google client id on the server
    expect(text(fixture)).not.toContain('Evaluation');
  });

  it("shows each chat's collection in the sidebar once chats come from 2+ collections", async () => {
    const fixture = await start('', [CONV, { ...CONV, id: 4 }]);
    expect(fixture.nativeElement.querySelector('.conversation-collection')).toBeNull();
    const hr = { ...CONV, id: 5, collection_id: 2, collection: 'HR policies' };
    TestBed.inject(ConversationsStore).list.set([CONV, hr]);
    await fixture.whenStable();
    const labels = [...fixture.nativeElement.querySelectorAll('.conversation-collection')];
    expect(labels.map((l) => (l as HTMLElement).textContent?.trim())).toEqual([
      'AI Engineering Handbook',
      'HR policies',
    ]);
  });

  it('offers Sign in with Google when the server has a client id', async () => {
    const fixture = await start('client.apps.googleusercontent.com');
    expect(fixture.nativeElement.querySelector('app-google-sign-in')).toBeTruthy();
    expect(text(fixture)).not.toContain('Admin sign in');
  });

  it('a signed-in user sees the account menu; sign-out ends the session on the server', async () => {
    TestBed.inject(ApiKeyStore).setSession(ALICE);
    const fixture = TestBed.createComponent(App);
    http.expectOne(`${BASE}/auth/config`).flush({ google_client_id: 'cid' });
    const me = http.expectOne(`${BASE}/me`); // role and profile are refreshed at start
    expect(me.request.headers.get('Authorization')).toBe('Bearer session-token');
    me.flush({ role: 'user', user: ALICE.user });
    await fixture.whenStable();
    http.expectOne(`${BASE}/conversations`).flush([CONV]);
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelector('.avatar.initial').textContent).toContain('A');
    expect(text(fixture)).not.toContain('Evaluation'); // a user, not an admin

    fixture.componentInstance.signOut();
    const logout = http.expectOne(`${BASE}/auth/logout`);
    expect(logout.request.headers.get('Authorization')).toBe('Bearer session-token');
    logout.flush(null);
    await fixture.whenStable();
    http.expectOne(`${BASE}/conversations`).flush([]); // back to this browser's guest chats
    expect(TestBed.inject(ApiKeyStore).session()).toBeNull();
    expect(localStorage.getItem('rag.session')).toBeNull();
  });

  it('a signed-in user gets "My documents" and can delete the account after a red check', async () => {
    TestBed.inject(ApiKeyStore).setSession(ALICE);
    const fixture = TestBed.createComponent(App);
    http.expectOne(`${BASE}/auth/config`).flush({ google_client_id: 'cid' });
    http.expectOne(`${BASE}/me`).flush({ role: 'user', user: ALICE.user });
    await fixture.whenStable();
    http.expectOne(`${BASE}/conversations`).flush([]);
    await fixture.whenStable();
    expect(text(fixture)).toContain('My documents');
    expect(text(fixture)).not.toContain('Sign in to upload');
    const ask = vi.spyOn(TestBed.inject(ConfirmService), 'ask').mockReturnValue(of(true));
    vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);
    fixture.componentInstance.deleteAccount();
    expect(ask.mock.calls[0][0].danger).toBe(true);
    const del = http.expectOne(`${BASE}/me`);
    expect(del.request.method).toBe('DELETE');
    del.flush(null);
    await fixture.whenStable();
    http.match(`${BASE}/conversations`).forEach((r) => r.flush([]));
    expect(TestBed.inject(ApiKeyStore).session()).toBeNull(); // signed out for good
  });

  it('a guest is told to sign in to upload own documents', async () => {
    const fixture = await start('cid', []);
    expect(text(fixture)).toContain('Sign in to upload your own documents');
    expect(text(fixture)).not.toContain('My documents');
  });

  it('an admin email gets the Admin pages from its role', async () => {
    TestBed.inject(ApiKeyStore).setSession({ ...ALICE, role: 'admin' });
    const fixture = TestBed.createComponent(App);
    http.expectOne(`${BASE}/auth/config`).flush({ google_client_id: 'cid' });
    http.expectOne(`${BASE}/me`).flush({ role: 'admin', user: ALICE.user });
    await fixture.whenStable();
    http.expectOne(`${BASE}/conversations`).flush([]);
    await fixture.whenStable();
    expect(text(fixture)).toContain('Evaluation');
    expect(text(fixture)).toContain('Admin');
  });

  it('drops an expired session on a 401 and continues as a guest', async () => {
    TestBed.inject(ApiKeyStore).setSession(ALICE);
    const fixture = TestBed.createComponent(App);
    http.expectOne(`${BASE}/auth/config`).flush({ google_client_id: 'cid' });
    http.expectOne(`${BASE}/me`).flush({}, { status: 401, statusText: 'Unauthorized' });
    await fixture.whenStable();
    expect(TestBed.inject(ApiKeyStore).session()).toBeNull();
    http.match(`${BASE}/conversations`).forEach((r) => r.flush([]));
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelector('app-google-sign-in')).toBeTruthy();
  });

  it('deletes all conversations after confirmation (privacy: "delete my data")', async () => {
    const fixture = await start('', [CONV, { ...CONV, id: 4 }]);
    const button = fixture.nativeElement.querySelector('.delete-all') as HTMLButtonElement;
    const ask = vi.spyOn(TestBed.inject(ConfirmService), 'ask').mockReturnValue(of(true));
    const navigate = vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);
    button.click();
    expect(ask.mock.calls[0][0].message).toContain('All 2 conversations');
    const del = http.expectOne((r) => r.method === 'DELETE' && r.url === `${BASE}/conversations`);
    del.flush(null);
    http.expectOne(`${BASE}/conversations`).flush([]); // the sidebar reloads: empty
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelector('.delete-all')).toBeNull(); // nothing left
    expect(navigate).toHaveBeenCalledWith(['/chat']); // back to a new chat
  });

  it('signing out on My documents (or the session ending) goes back to the chat', async () => {
    TestBed.inject(ApiKeyStore).setSession(ALICE);
    const fixture = TestBed.createComponent(App);
    http.expectOne(`${BASE}/auth/config`).flush({ google_client_id: 'cid' });
    http.expectOne(`${BASE}/me`).flush({ role: 'user', user: ALICE.user });
    await fixture.whenStable();
    http.match(`${BASE}/conversations`).forEach((r) => r.flush([]));
    const router = TestBed.inject(Router);
    vi.spyOn(router, 'url', 'get').mockReturnValue('/documents');
    const navigate = vi.spyOn(router, 'navigate').mockResolvedValue(true);
    TestBed.inject(ApiKeyStore).clearSession(); // e.g. signed out in another tab, or expired
    await fixture.whenStable();
    expect(navigate).toHaveBeenCalledWith(['/chat']);
    http.match(`${BASE}/conversations`).forEach((r) => r.flush([]));
  });

  it('shows the Admin pages with an admin key, and hides them again on sign out', async () => {
    const fixture = await start('', []);
    TestBed.inject(ApiKeyStore).set('admin-key');
    await fixture.whenStable();
    for (const label of ['Documents', 'Evaluation', 'Settings']) {
      expect(text(fixture)).toContain(label);
    }
    expect(text(fixture)).not.toContain('Admin sign in');

    fixture.componentInstance.signOut();
    await fixture.whenStable();
    expect(text(fixture)).not.toContain('Evaluation');
    expect(text(fixture)).toContain('Admin sign in');
  });
});
