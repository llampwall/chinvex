"""Webhook notification implementation."""
import ipaddress
import socket
from urllib.parse import urlparse
from pathlib import Path
import time
import requests


def validate_webhook_url(url: str) -> bool:
    """
    Validate webhook URL for security.

    Requirements:
    - HTTPS only
    - No private IPs (127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    - No localhost
    """
    try:
        parsed = urlparse(url)

        # HTTPS required
        if parsed.scheme != 'https':
            return False

        # Block localhost
        if parsed.hostname in ('localhost', '127.0.0.1', '::1'):
            return False

        # Resolve hostname and check IP
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))

            # Block private IPs
            if ip.is_private or ip.is_loopback:
                return False
        except (socket.gaierror, ValueError):
            return False

        return True

    except Exception:
        return False


def sanitize_source_uri(source_uri: str) -> str:
    """
    Sanitize source_uri to filename only.

    Prevents leaking directory structure via webhooks.
    """
    return Path(source_uri).name


def send_webhook(url: str, payload: dict, secret: str | None = None, retry_count: int = 2, retry_delay_sec: int = 5) -> bool:
    """
    Send webhook notification with retry logic.

    Returns True if successful, False otherwise.
    """
    # Validate URL
    if not validate_webhook_url(url):
        print(f"Invalid webhook URL: {url}")
        return False

    # Add signature if secret provided
    headers = {"Content-Type": "application/json"}
    if secret:
        from .webhook_signature import generate_signature
        signature = generate_signature(payload, secret)
        headers["X-Chinvex-Signature"] = signature

    # Retry loop
    for attempt in range(retry_count + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=10
            )

            if response.status_code < 300:
                return True
            else:
                print(f"Webhook failed with status {response.status_code}")

        except requests.RequestException as e:
            print(f"Webhook request failed: {e}")

        # Retry delay
        if attempt < retry_count:
            time.sleep(retry_delay_sec)

    return False


def create_watch_hit_payload(watch_id: str, query: str, hits: list[dict]) -> dict:
    """
    Create webhook payload for watch hit event.

    Includes snippet (first 200 chars) only, not full chunk text.
    Sanitizes source_uri to filename only.
    """
    formatted_hits = []
    for hit in hits:
        formatted_hits.append({
            "chunk_id": hit["chunk_id"],
            "score": hit["score"],
            "snippet": hit.get("text", "")[:200],
            "source": sanitize_source_uri(hit.get("source_uri", "unknown"))
        })

    return {
        "event": "watch_hit",
        "watch_id": watch_id,
        "query": query,
        "hits": formatted_hits
    }
