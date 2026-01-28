"""Webhook signature generation and verification."""
import hmac
import hashlib
import json


def generate_signature(payload: dict, secret: str) -> str:
    """
    Generate HMAC-SHA256 signature for webhook payload.

    Returns signature in format: "sha256=<hex>"
    """
    # Serialize payload to canonical JSON
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')

    # Generate HMAC
    signature = hmac.new(
        secret.encode('utf-8'),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()

    return f"sha256={signature}"


def verify_signature(payload: dict, signature: str, secret: str) -> bool:
    """
    Verify webhook signature.

    Uses constant-time comparison to prevent timing attacks.
    """
    expected = generate_signature(payload, secret)

    # Constant-time comparison
    return hmac.compare_digest(expected, signature)
