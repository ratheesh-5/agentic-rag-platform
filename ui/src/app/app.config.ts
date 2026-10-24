import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { ApplicationConfig, provideBrowserGlobalErrorListeners } from '@angular/core';
import {
  provideRouter,
  withComponentInputBinding,
  withNavigationErrorHandler,
} from '@angular/router';

import { routes } from './app.routes';
import { apiKeyInterceptor } from './core/api';
import { reloadOnNewVersion } from './core/new-version';

// Application-wide providers (.NET analogy: the service registrations in Program.cs).
export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    // A tab still running an old version reloads itself when a page's code is gone after a
    // deploy (see core/new-version.ts).
    provideRouter(
      routes,
      withComponentInputBinding(),
      withNavigationErrorHandler((error) => {
        reloadOnNewVersion(error);
      }),
    ),
    // HttpClient for the REST API; the interceptor adds the X-API-Key header.
    provideHttpClient(withInterceptors([apiKeyInterceptor])),
  ],
};
