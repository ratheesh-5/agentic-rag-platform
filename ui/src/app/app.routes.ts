import { Routes } from '@angular/router';

import { adminGuard, signedInGuard } from './core/admin.guard';

// Pages are lazy-loaded: each page's code is downloaded only when it is first opened.
export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'chat' },
  // The chatbot. /chat = new conversation, /chat?c=5 = conversation 5.
  {
    path: 'chat',
    title: 'Knowledge Assistant',
    loadComponent: () => import('./features/chat/chat').then((m) => m.Chat),
  },
  // My documents: a signed-in user's private collections (the same page as Admin › Documents,
  // showing only the collections the caller may manage).
  {
    path: 'documents',
    canMatch: [signedInGuard],
    title: 'My documents',
    loadComponent: () => import('./features/documents/documents').then((m) => m.Documents),
  },
  // Admin pages (for operators, not end users): only after "Admin sign in".
  {
    path: 'admin',
    canMatch: [adminGuard],
    children: [
      {
        path: 'documents',
        title: 'Documents · Admin',
        loadComponent: () => import('./features/documents/documents').then((m) => m.Documents),
      },
      {
        path: 'evaluation',
        title: 'Evaluation · Admin',
        loadComponent: () => import('./features/eval-runs/eval-runs').then((m) => m.EvalRuns),
      },
      {
        path: 'usage',
        title: 'Usage · Admin',
        loadComponent: () => import('./features/usage/usage').then((m) => m.Usage),
      },
      {
        path: 'settings',
        title: 'Settings · Admin',
        loadComponent: () => import('./features/settings/settings').then((m) => m.Settings),
      },
    ],
  },
  { path: '**', redirectTo: 'chat' },
];
