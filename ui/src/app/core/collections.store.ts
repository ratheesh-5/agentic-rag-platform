import { Injectable, computed, effect, inject, signal, untracked } from '@angular/core';

import { ApiService } from './api';
import { ApiKeyStore } from './api-key.store';
import { Collection } from './models';

/**
 * The collections (named groups of documents) and the one this browser works with. A new chat
 * searches the selected collection; the Documents page shows and uploads into it. The choice is
 * remembered in localStorage (a convenience: the server never relies on it).
 * One instance for the whole app (providedIn: 'root' = a singleton, like AddSingleton in .NET).
 */
@Injectable({ providedIn: 'root' })
export class CollectionsStore {
  private static readonly storageKey = 'rag.collectionId';
  private readonly api = inject(ApiService);

  readonly list = signal<Collection[]>([]);
  private readonly chosen = signal<number | null>(CollectionsStore.load());

  /** The selected collection's id: the remembered one if it still exists, else the demo one. */
  readonly selectedId = computed(() => {
    const list = this.list();
    const id = this.chosen();
    if (list.some((c) => c.id === id)) {
      return id as number;
    }
    return list.find((c) => c.is_default)?.id ?? list[0]?.id ?? 1;
  });
  readonly selected = computed(() => this.list().find((c) => c.id === this.selectedId()) ?? null);

  constructor() {
    this.reload();
    // Private collections belong to the signed-in user: reload when the person changes (the
    // effect's first run is the start, already loaded above).
    const keys = inject(ApiKeyStore);
    let signedIn = keys.session()?.token ?? null;
    effect(() => {
      const now = keys.session()?.token ?? null;
      untracked(() => {
        if (now !== signedIn) {
          signedIn = now;
          this.reload();
        }
      });
    });
  }

  reload(): void {
    this.api.collections().subscribe({
      next: (list) => this.list.set(list),
      error: () => this.list.set([]), // the pages show their own errors
    });
  }

  select(id: number): void {
    this.chosen.set(id);
    try {
      localStorage.setItem(CollectionsStore.storageKey, String(id));
    } catch {
      // storage blocked (private mode): the choice simply isn't remembered
    }
  }

  nameOf(id: number): string {
    return this.list().find((c) => c.id === id)?.name ?? '';
  }

  private static load(): number | null {
    try {
      return Number(localStorage.getItem(CollectionsStore.storageKey)) || null;
    } catch {
      return null;
    }
  }
}
