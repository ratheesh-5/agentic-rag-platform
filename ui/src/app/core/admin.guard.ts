import { inject } from '@angular/core';
import { CanMatchFn, Router } from '@angular/router';

import { ApiKeyStore } from './api-key.store';

/**
 * Admin pages exist only for signed-in admins; guests are sent to the chat.
 * (The API enforces this too - the guard just keeps guests out of pages that would fail.)
 * .NET analogy: [Authorize(Roles = "Admin")] on a Razor page, but client-side.
 */
export const adminGuard: CanMatchFn = () =>
  inject(ApiKeyStore).isAdmin() || inject(Router).parseUrl('/chat');

/** "My documents" (private collections): signed-in users and admins; guests go to the chat. */
export const signedInGuard: CanMatchFn = () => {
  const keys = inject(ApiKeyStore);
  return !!keys.session() || keys.isAdmin() || inject(Router).parseUrl('/chat');
};
