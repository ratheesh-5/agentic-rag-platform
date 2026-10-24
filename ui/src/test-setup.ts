// Runs before every unit test (angular.json › test › setupFiles).
// Every test starts signed out and with nothing remembered: browser storage is shared by the test
// files that run in the same worker, so an admin key or Google session left by one file (e.g. the
// chat tests) made the Documents page ask for admin-only data in another - found when CI (fewer
// CPUs, so more files per worker) failed while local runs passed.
import { beforeEach } from 'vitest';

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});
