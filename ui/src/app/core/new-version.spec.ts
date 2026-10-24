// An open tab that runs an old version reloads itself when a page's code is gone after a deploy.
import { NavigationError } from '@angular/router';

import { isStaleCodeError, reloadOnNewVersion } from './new-version';

const gone = (url = '/admin/documents') =>
  new NavigationError(
    1,
    url,
    new TypeError(
      'Failed to fetch dynamically imported module: http://localhost:8081/chunk-D54c.js',
    ),
  );

describe('reloadOnNewVersion', () => {
  beforeEach(() => sessionStorage.clear());

  it('recognises the browsers\' "page code is gone" errors, nothing else', () => {
    expect(isStaleCodeError(gone().error)).toBe(true);
    expect(isStaleCodeError(new Error('error loading dynamically imported module'))).toBe(true);
    expect(isStaleCodeError(new Error('Importing a module script failed.'))).toBe(true);
    expect(isStaleCodeError(new Error('Cannot match any routes'))).toBe(false);
  });

  it('reloads at the page the user wanted', () => {
    const reload = vi.fn();
    expect(reloadOnNewVersion(gone('/admin/evaluation'), reload, 100_000)).toBe(true);
    expect(reload).toHaveBeenCalledWith('/admin/evaluation');
  });

  it('never reloads in a loop: at most once per 10 seconds', () => {
    const reload = vi.fn();
    reloadOnNewVersion(gone(), reload, 100_000);
    expect(reloadOnNewVersion(gone(), reload, 105_000)).toBe(false); // still failing: stop
    expect(reloadOnNewVersion(gone(), reload, 111_000)).toBe(true); // later: try again
    expect(reload).toHaveBeenCalledTimes(2);
  });

  it('leaves other navigation errors alone', () => {
    const reload = vi.fn();
    const other = new NavigationError(1, '/x', new Error('Cannot match any routes'));
    expect(reloadOnNewVersion(other, reload)).toBe(false);
    expect(reload).not.toHaveBeenCalled();
  });
});
