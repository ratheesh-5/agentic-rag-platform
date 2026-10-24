import { HttpErrorResponse } from '@angular/common/http';

/**
 * Turns an HTTP error into a message a user can act on. The API returns {"detail": "..."}
 * with meaningful status codes (401 key, 413/415/422 upload, 429 quota or guest limit,
 * 503 provider).
 */
export function describeError(error: unknown): string {
  if (!(error instanceof HttpErrorResponse)) {
    return 'Unexpected error.';
  }
  const detail = typeof error.error?.detail === 'string' ? error.error.detail : '';
  switch (error.status) {
    case 0:
      // Status 0 = no usable response: the API is down, OR the browser blocked the response
      // because the API doesn't allow this page's origin (CORS). The browser can't tell us which.
      return (
        'Cannot reach the API. Check that `uv run rag serve` is running (restart it after ' +
        'updating the code) and that CORS_ORIGINS in .env includes this page’s address.'
      );
    case 401:
      return 'Admin sign-in required, or the admin key is no longer valid. Sign in again.';
    case 422:
      // FastAPI validation errors are a list; upload errors are a plain message.
      return detail || 'The request was not valid.';
    case 429:
      return detail || 'Model quota exceeded. Try again in a minute.';
    case 502:
    case 504:
      // From the web server in front of the API: the API is down or didn't answer in time.
      return 'The assistant can’t be reached right now. Please try again in a moment.';
    default:
      return detail || `Request failed (${error.status}).`;
  }
}
