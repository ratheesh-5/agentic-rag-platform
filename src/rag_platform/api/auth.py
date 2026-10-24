"""Sign in with Google (security plan B1): users get their chats on every device, admins are
named people instead of "whoever has the key".

The flow (OpenID Connect, the identity layer on top of OAuth 2.0):
1. The browser shows Google's "Sign in with Google" button (Google Identity Services). The user
   signs in on Google's own page - we never see a password - and the browser receives an ID
   token: a JWT signed by Google saying who the user is (id "sub", email, name, picture).
2. The UI sends it to POST /auth/google. `verify_google_token` checks Google's signature, that
   the token was issued FOR THIS APP (audience = GOOGLE_CLIENT_ID; a token for another site is
   refused), that it has not expired and that Google verified the email.
3. We create or update the user and start OUR OWN session: a random 256-bit token, stored only
   as a hash, valid SESSION_DAYS (7). Why not reuse Google's token: it expires after an hour and
   can't be revoked by us; our session survives browser restarts and sign-out ends it at once.
4. Every later request sends "Authorization: Bearer <token>" (see deps.current_caller).
Admins: users whose email is in ADMIN_EMAILS, decided on each request (changing .env takes
effect at once). Guest chats are NOT moved into the account: on a shared computer they would
land in whoever signs in next (found in testing 2026-10-23); they stay with the guest.

.NET analogy: AddAuthentication().AddGoogle() / an Entra ID app registration - the client id
identifies the app, users bring their own accounts; our session ~ the auth cookie it issues.
Tests replace `verify_google_token` (no network), like the fake LLM.
"""

import logging
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials

from rag_platform.api.deps import (
    Services,
    _bearer,
    get_services,
    token_hash,
)
from rag_platform.api.schemas import AuthConfig, GoogleSignIn, SignInResponse, UserInfo

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger("rag_platform.api")


def verify_google_token(credential: str, client_id: str) -> dict:
    """Check a Google ID token and return its claims; ValueError if it is not valid for us.

    google-auth downloads Google's public signing keys and checks signature, audience, issuer
    and expiry (like JwtBearer validation with Google's metadata in .NET).
    """
    from google.auth.transport.requests import Request  # lazy: only needed at sign-in
    from google.oauth2 import id_token

    claims = id_token.verify_oauth2_token(credential, Request(), client_id)
    if not claims.get("email_verified"):
        raise ValueError("Google has not verified this email address")
    return claims


@router.get("/config", response_model=AuthConfig)
def auth_config(svc: Services = Depends(get_services)) -> AuthConfig:
    """Open: the UI asks whether Google sign-in is on (the client id is public by design)."""
    return AuthConfig(google_client_id=svc.settings.google_client_id)


@router.post("/google", response_model=SignInResponse)
def sign_in_with_google(
    body: GoogleSignIn,
    svc: Services = Depends(get_services),
) -> SignInResponse:
    s = svc.settings
    if not s.google_client_id:
        raise HTTPException(503, "Google sign-in is not configured (set GOOGLE_CLIENT_ID)")
    try:
        claims = verify_google_token(body.credential, s.google_client_id)
    except ValueError as e:  # bad signature, other app's token, expired, unverified email
        log.warning("google sign-in refused", extra={"reason": type(e).__name__})
        raise HTTPException(401, "Google sign-in failed. Please try again.") from e
    except Exception as e:  # noqa: BLE001 - e.g. Google's keys could not be downloaded
        log.warning("google sign-in unavailable", extra={"error": type(e).__name__})
        raise HTTPException(503, "Google sign-in is unavailable right now.") from e

    token = secrets.token_urlsafe(32)  # 256 random bits (like RandomNumberGenerator in .NET)
    expires = datetime.now(UTC) + timedelta(days=s.session_days)
    expires_at = expires.isoformat(timespec="seconds")
    store = svc.open_store()
    try:
        user_id = store.upsert_user(
            claims["sub"], claims["email"], claims.get("name", ""), claims.get("picture", "")
        )
        store.create_session(token_hash(token), user_id, expires_at)
    finally:
        store.close()
    is_admin = claims["email"].lower() in s.admin_email_list
    # The log names the user only by id (D1: no personal data such as the email in logs).
    log.info("signed in", extra={"user_id": user_id, "admin": is_admin})
    return SignInResponse(
        token=token,
        expires_at=expires_at,
        role="admin" if is_admin else "user",
        user=UserInfo(
            email=claims["email"], name=claims.get("name", ""), picture=claims.get("picture", "")
        ),
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def sign_out(
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
    svc: Services = Depends(get_services),
) -> None:
    """End the session on the server: the token stops working at once, even if copied."""
    if bearer is None:
        return
    store = svc.open_store()
    try:
        store.delete_session(token_hash(bearer.credentials))
    finally:
        store.close()
