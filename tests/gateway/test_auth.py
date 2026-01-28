import os
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from chinvex.gateway.auth import verify_token


def test_verify_token_success():
    """Should accept valid token."""
    os.environ["CHINVEX_API_TOKEN"] = "valid_token_123"

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token_123")

    # Should not raise
    result = verify_token(creds)
    assert result == "valid_token_123"

    del os.environ["CHINVEX_API_TOKEN"]


def test_verify_token_invalid():
    """Should reject invalid token."""
    os.environ["CHINVEX_API_TOKEN"] = "valid_token_123"

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong_token")

    with pytest.raises(HTTPException) as exc_info:
        verify_token(creds)

    assert exc_info.value.status_code == 401
    assert "Invalid authentication token" in exc_info.value.detail

    del os.environ["CHINVEX_API_TOKEN"]


def test_verify_token_not_configured():
    """Should fail when token not configured."""
    os.environ.pop("CHINVEX_API_TOKEN", None)

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="any_token")

    with pytest.raises(HTTPException) as exc_info:
        verify_token(creds)

    assert exc_info.value.status_code == 500
    assert "not configured" in exc_info.value.detail
