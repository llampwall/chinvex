"""Authentication middleware for gateway."""

import secrets
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import load_gateway_config

security = HTTPBearer()


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Verify bearer token using constant-time comparison.

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        Token string if valid

    Raises:
        HTTPException: 500 if token not configured, 401 if invalid
    """
    config = load_gateway_config()
    expected = config.token

    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Gateway token not configured (CHINVEX_API_TOKEN missing)"
        )

    provided = credentials.credentials
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )

    return provided
