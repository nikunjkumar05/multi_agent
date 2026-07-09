"""
JWT Bearer token authentication.

Behaviour
---------
* If ``settings.jwt_secret`` is still the default placeholder
  ("change-me-in-production"), auth is DISABLED — all requests pass through.
  This keeps the dev/local experience friction-free.
* Once a real secret is set in .env, every protected route requires a valid
  HS256 JWT in the ``Authorization: Bearer <token>`` header.

Usage in routes
---------------
    from fastapi import Depends
    from api.middleware.auth import require_auth

    @router.post("/execute", dependencies=[Depends(require_auth)])
    async def execute(...): ...

Generating a token (development helper)
----------------------------------------
    python -c "
    import jwt, datetime
    print(jwt.encode({'sub': 'dev', 'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)},
                     'your-secret-here', algorithm='HS256'))
    "
"""

from __future__ import annotations

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import DEFAULT_JWT_SECRET

_bearer = HTTPBearer(auto_error=False)


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """
    FastAPI dependency — raises HTTP 401 if authentication fails.
    Import and add as ``dependencies=[Depends(require_auth)]``.
    """
    from core.config import settings

    # Dev mode: skip auth when the secret has not been changed
    if settings.jwt_secret == DEFAULT_JWT_SECRET:
        return

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
