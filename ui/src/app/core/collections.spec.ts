// Collections: the selected collection is remembered (falls back to the demo one if it is gone),
// and the create / rename dialog saves through the API and shows a taken name.
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';

import { API_BASE_URL } from './api';
import { CollectionDialog } from './collection-dialog';
import { ApiKeyStore } from './api-key.store';
import { CollectionsStore } from './collections.store';
import { Collection } from './models';

const BASE = 'http://api.test';
const DEMO: Collection = {
  id: 1,
  name: 'AI Engineering Handbook',
  description: '',
  documents: 6,
  passages: 78,
  is_default: true,
  threshold: 0.68,
  threshold_source: 'measured',
  threshold_override: null,
  is_private: false,
  can_manage: true,
};
const HR: Collection = { ...DEMO, id: 4, name: 'HR policies', is_default: false };

describe('Collections', () => {
  const close = vi.fn();
  let http: HttpTestingController;

  function setup(data: Collection | null = null) {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: API_BASE_URL, useValue: BASE },
        { provide: MatDialogRef, useValue: { close } },
        { provide: MAT_DIALOG_DATA, useValue: data },
      ],
    });
    http = TestBed.inject(HttpTestingController);
  }

  beforeEach(() => {
    localStorage.removeItem('rag.collectionId');
    close.mockReset();
  });

  afterEach(() => http.verify());

  it('remembers the selected collection and falls back to the demo one when it is gone', () => {
    setup();
    const store = TestBed.inject(CollectionsStore);
    http.expectOne((r) => r.url === `${BASE}/collections`).flush([DEMO, HR]);
    expect(store.selectedId()).toBe(1); // nothing chosen yet: the demo collection
    store.select(4);
    expect(store.selected()?.name).toBe('HR policies');
    expect(localStorage.getItem('rag.collectionId')).toBe('4');
    store.list.set([DEMO]); // HR was deleted meanwhile
    expect(store.selectedId()).toBe(1);
    localStorage.removeItem('rag.collectionId');
  });

  it('creates a collection and reports a taken name', () => {
    setup();
    const dialog = TestBed.createComponent(CollectionDialog).componentInstance;
    dialog.name.set(' HR policies ');
    dialog.save();
    const taken = http.expectOne(`${BASE}/collections`);
    // No threshold typed in: null = measured automatically (the normal case).
    expect(taken.request.body).toEqual({ name: 'HR policies', description: '', threshold: null });
    taken.flush({ detail: 'exists' }, { status: 409, statusText: 'Conflict' });
    expect(dialog.error()).toContain('already exists');
    expect(close).not.toHaveBeenCalled();
    dialog.name.set('HR');
    dialog.save();
    http.expectOne(`${BASE}/collections`).flush(HR);
    expect(close).toHaveBeenCalledWith(HR);
  });

  it('an admin signed in with Google may make a new collection private', () => {
    setup();
    TestBed.inject(ApiKeyStore).setSession({
      token: 't',
      expires_at: '2999-01-01T00:00:00+00:00',
      role: 'admin',
      user: { email: 'boss@example.com', name: 'Boss', picture: '' },
    });
    const dialog = TestBed.createComponent(CollectionDialog).componentInstance;
    dialog.name.set('Boss notes');
    dialog.private.set(true);
    dialog.save();
    const req = http.expectOne((r) => r.url === `${BASE}/collections`);
    expect(req.request.body.private).toBe(true);
    req.flush({ ...HR, is_private: true });
    localStorage.removeItem('rag.session');
  });

  it('renames an existing collection', () => {
    setup(HR);
    const dialog = TestBed.createComponent(CollectionDialog).componentInstance;
    expect(dialog.name()).toBe('HR policies');
    dialog.name.set('People policies');
    dialog.threshold.set('0.62'); // an admin's override (the number input gives a string)
    dialog.save();
    const req = http.expectOne(`${BASE}/collections/4`);
    expect(req.request.method).toBe('PUT');
    expect(req.request.body.threshold).toBe(0.62);
    req.flush({ ...HR, name: 'People policies' });
    expect(close).toHaveBeenCalled();
  });
});
