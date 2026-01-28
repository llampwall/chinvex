#!/usr/bin/env python3
"""
P2 Gateway Acceptance Test Suite
Run this against a running gateway to verify all functionality.
"""

import os
import sys
import requests
from typing import Callable

BASE_URL = os.environ.get("CHINVEX_GATEWAY_URL", "http://localhost:7778")
TOKEN = os.environ.get("CHINVEX_API_TOKEN")

if not TOKEN:
    print("Error: CHINVEX_API_TOKEN not set")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}


def test(name: str, test_fn: Callable, expected_status: int = 200) -> bool:
    """Run a test and print result."""
    try:
        result = test_fn()

        if isinstance(result, requests.Response):
            passed = result.status_code == expected_status
            if not passed:
                print(f"[FAIL] {name}")
                print(f"  Expected: {expected_status}, Got: {result.status_code}")
                print(f"  Response: {result.text[:200]}")
            else:
                print(f"[PASS] {name}")
        else:
            passed = bool(result)
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}")

        return passed
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return False


# Test suite
results = []

# 2.5.1: Health endpoint (no auth)
results.append(test(
    "2.5.1: Health endpoint accessible",
    lambda: requests.get(f"{BASE_URL}/health"),
    200
))

# 2.5.2: Auth required
results.append(test(
    "2.5.2: Auth required for evidence",
    lambda: requests.post(
        f"{BASE_URL}/v1/evidence",
        json={"context": "Chinvex", "query": "test"}
    ),
    403
))

# 2.5.3: Valid token accepted
results.append(test(
    "2.5.3: Valid token accepted",
    lambda: requests.post(
        f"{BASE_URL}/v1/evidence",
        headers=HEADERS,
        json={"context": "Chinvex", "query": "test"}
    ),
    200
))

# 2.5.4: Invalid token rejected
results.append(test(
    "2.5.4: Invalid token rejected",
    lambda: requests.post(
        f"{BASE_URL}/v1/evidence",
        headers={"Authorization": "Bearer invalid_token", "Content-Type": "application/json"},
        json={"context": "Chinvex", "query": "test"}
    ),
    401
))

# 2.5.9: Unknown context
results.append(test(
    "2.5.9: Unknown context returns 404",
    lambda: requests.post(
        f"{BASE_URL}/v1/evidence",
        headers=HEADERS,
        json={"context": "NonExistent", "query": "test"}
    ),
    404
))

# 2.5.11: Invalid context name
results.append(test(
    "2.5.11: Invalid context name rejected",
    lambda: requests.post(
        f"{BASE_URL}/v1/evidence",
        headers=HEADERS,
        json={"context": "../../../etc/passwd", "query": "test"}
    ),
    422  # Pydantic validation error
))

# Summary
print("\n" + "="*50)
print(f"Tests passed: {sum(results)}/{len(results)}")
print("="*50)

sys.exit(0 if all(results) else 1)
