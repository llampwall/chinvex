"""Test webhook signature generation."""
import pytest
import hmac
import hashlib
import json


def test_generate_signature_creates_hmac():
    """Test that signature is HMAC-SHA256."""
    from chinvex.webhook_signature import generate_signature

    payload = {"event": "test"}
    secret = "test_secret"

    signature = generate_signature(payload, secret)

    # Should be sha256=<hex>
    assert signature.startswith("sha256=")
    assert len(signature) == 71  # "sha256=" + 64 hex chars


def test_verify_signature_validates_correctly():
    """Test that signature verification works."""
    from chinvex.webhook_signature import generate_signature, verify_signature

    payload = {"event": "test", "data": "value"}
    secret = "test_secret"

    signature = generate_signature(payload, secret)

    assert verify_signature(payload, signature, secret) is True
    assert verify_signature(payload, "sha256=invalid", secret) is False
    assert verify_signature(payload, signature, "wrong_secret") is False


def test_signature_is_deterministic():
    """Test that same payload + secret = same signature."""
    from chinvex.webhook_signature import generate_signature

    payload = {"event": "test"}
    secret = "test_secret"

    sig1 = generate_signature(payload, secret)
    sig2 = generate_signature(payload, secret)

    assert sig1 == sig2
