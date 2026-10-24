import { NavigationError } from '@angular/router';

/**
 * After a new version is deployed, a browser tab that was already open still runs the OLD app.
 * Pages are lazy-loaded (their code is downloaded on first visit), and every build renames
 * those files - so the old app asks for files that no longer exist, and a click on a page it
 * hasn't opened yet silently does nothing (found in testing 2026-10-23: Documents and
 * Evaluation "didn't open" after a rebuild, Settings and Usage did - they were loaded before).
 *
 * The fix used by most single-page apps: when a page's code can't be downloaded, reload the
 * browser at the address the user wanted. The reload fetches the new version; signed-in state
 * survives (it is in browser storage). At most one reload per 10 seconds, so a real network
 * problem can't cause an endless reload loop.
 */

// Browser messages for a failed `import()` of a page's code: Chrome/Edge, Firefox, Safari.
const STALE_CODE = /dynamically imported module|Importing a module script failed/i;
const LAST_RELOAD = 'rag.reloadedForNewVersion';
const MIN_SECONDS_BETWEEN_RELOADS = 10;

export function isStaleCodeError(error: unknown): boolean {
  return STALE_CODE.test(String((error as Error)?.message ?? error));
}

/**
 * Router navigation-error handler (registered in app.config.ts). Returns true when it reloaded.
 * `reload` and `now` are parameters so tests can check the decision without reloading.
 */
export function reloadOnNewVersion(
  event: NavigationError,
  reload: (url: string) => void = (url) => location.assign(url),
  now: number = Date.now(),
): boolean {
  if (!isStaleCodeError(event.error)) {
    return false;
  }
  const last = Number(sessionStorage.getItem(LAST_RELOAD) ?? 0);
  if (now - last < MIN_SECONDS_BETWEEN_RELOADS * 1000) {
    return false; // reloaded just now and it still fails: a real network problem, don't loop
  }
  sessionStorage.setItem(LAST_RELOAD, String(now));
  reload(event.url);
  return true;
}
