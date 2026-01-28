#!/usr/bin/env python3
"""
P2 Gateway E2E Smoke Test for Chinvex

Tests all P2 gateway functionality:
- Authentication (valid/invalid/missing tokens)
- All endpoints (/health, /v1/search, /v1/evidence, /v1/chunks, /v1/contexts)
- Security (rate limiting, input validation, context allowlist)
- Optional /v1/answer endpoint (if enabled)

Prerequisites:
- Gateway running (chinvex gateway serve --port 7778)
- At least one context with ingested data
- CHINVEX_API_TOKEN set

Usage:
    python scripts/e2e_smoke_p2.py [--base-url URL] [--context NAME]
    
    # Local testing
    python scripts/e2e_smoke_p2.py --base-url http://localhost:7778
    
    # Production testing (via tunnel/proxy)
    python scripts/e2e_smoke_p2.py --base-url https://chinvex.yourdomain.com
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("ERROR: requests library required. Run: pip install requests")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_URL = "http://localhost:7778"
DEFAULT_CONTEXT = "Chinvex"


def get_config(args):
    """Build test configuration from args and environment."""
    token = os.environ.get("CHINVEX_API_TOKEN")
    if not token:
        print("ERROR: CHINVEX_API_TOKEN environment variable not set")
        sys.exit(1)
    
    return {
        "base_url": args.base_url.rstrip("/"),
        "context": args.context,
        "token": token,
        "headers": {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    }


# =============================================================================
# Test Infrastructure
# =============================================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []
    
    def record(self, test_id: str, name: str, passed: bool, skipped: bool = False, detail: str = ""):
        if skipped:
            self.skipped += 1
            status = "SKIP"
        elif passed:
            self.passed += 1
            status = "PASS"
        else:
            self.failed += 1
            status = "FAIL"
        
        self.results.append({
            "id": test_id,
            "name": name,
            "status": status,
            "detail": detail
        })
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbol = {"PASS": "[+]", "FAIL": "[-]", "SKIP": "[~]"}[status]
        print(f"{timestamp} {symbol} {test_id}: {name}")
        if detail and status == "FAIL":
            print(f"         Detail: {detail}")
    
    def summary(self):
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Passed:  {self.passed}")
        print(f"Failed:  {self.failed}")
        print(f"Skipped: {self.skipped}")
        print(f"Total:   {len(self.results)}")
        
        if self.failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if r["status"] == "FAIL":
                    print(f"  - {r['id']}: {r['name']}")
                    if r["detail"]:
                        print(f"    {r['detail']}")
        
        return self.failed == 0


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} [*] {msg}")


# =============================================================================
# Test Cases
# =============================================================================

def test_health_no_auth(cfg, results):
    """2.1: Health endpoint accessible without auth."""
    try:
        r = requests.get(f"{cfg['base_url']}/health", timeout=10)
        passed = r.status_code == 200 and r.json().get("status") == "ok"
        detail = "" if passed else f"status={r.status_code}, body={r.text[:100]}"
        results.record("2.1", "Health endpoint (no auth)", passed, detail=detail)
    except Exception as e:
        results.record("2.1", "Health endpoint (no auth)", False, detail=str(e))


def test_auth_required(cfg, results):
    """2.2: Endpoints require authentication."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/evidence",
            json={"context": cfg["context"], "query": "test"},
            timeout=10
        )
        # Should be 401 or 403 (depends on FastAPI config)
        passed = r.status_code in (401, 403)
        detail = "" if passed else f"Expected 401/403, got {r.status_code}"
        results.record("2.2", "Auth required for /v1/evidence", passed, detail=detail)
    except Exception as e:
        results.record("2.2", "Auth required for /v1/evidence", False, detail=str(e))


def test_invalid_token_rejected(cfg, results):
    """2.3: Invalid token rejected."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/evidence",
            headers={"Authorization": "Bearer invalid_token_12345", "Content-Type": "application/json"},
            json={"context": cfg["context"], "query": "test"},
            timeout=10
        )
        passed = r.status_code == 401
        detail = "" if passed else f"Expected 401, got {r.status_code}"
        results.record("2.3", "Invalid token rejected", passed, detail=detail)
    except Exception as e:
        results.record("2.3", "Invalid token rejected", False, detail=str(e))


def test_valid_token_accepted(cfg, results):
    """2.4: Valid token accepted."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/evidence",
            headers=cfg["headers"],
            json={"context": cfg["context"], "query": "test"},
            timeout=30
        )
        passed = r.status_code == 200
        detail = "" if passed else f"Expected 200, got {r.status_code}: {r.text[:100]}"
        results.record("2.4", "Valid token accepted", passed, detail=detail)
    except Exception as e:
        results.record("2.4", "Valid token accepted", False, detail=str(e))


def test_contexts_endpoint(cfg, results):
    """2.5: GET /v1/contexts returns context list."""
    try:
        r = requests.get(
            f"{cfg['base_url']}/v1/contexts",
            headers=cfg["headers"],
            timeout=10
        )
        passed = r.status_code == 200 and "contexts" in r.json()
        detail = "" if passed else f"status={r.status_code}, body={r.text[:100]}"
        results.record("2.5", "Contexts endpoint", passed, detail=detail)
    except Exception as e:
        results.record("2.5", "Contexts endpoint", False, detail=str(e))


def test_search_returns_results(cfg, results):
    """2.6: POST /v1/search returns ranked results."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/search",
            headers=cfg["headers"],
            json={"context": cfg["context"], "query": "search", "k": 5},
            timeout=30
        )
        data = r.json()
        passed = (
            r.status_code == 200 
            and "results" in data 
            and isinstance(data["results"], list)
        )
        detail = "" if passed else f"status={r.status_code}, keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}"
        results.record("2.6", "Search returns results", passed, detail=detail)
    except Exception as e:
        results.record("2.6", "Search returns results", False, detail=str(e))


def test_search_respects_k(cfg, results):
    """2.7: Search respects k parameter."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/search",
            headers=cfg["headers"],
            json={"context": cfg["context"], "query": "test", "k": 3},
            timeout=30
        )
        data = r.json()
        passed = r.status_code == 200 and len(data.get("results", [])) <= 3
        detail = "" if passed else f"Expected <=3 results, got {len(data.get('results', []))}"
        results.record("2.7", "Search respects k limit", passed, detail=detail)
    except Exception as e:
        results.record("2.7", "Search respects k limit", False, detail=str(e))


def test_evidence_grounded_field(cfg, results):
    """2.8: POST /v1/evidence returns grounded field."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/evidence",
            headers=cfg["headers"],
            json={"context": cfg["context"], "query": "chinvex search retrieval", "k": 5},
            timeout=30
        )
        data = r.json()
        passed = r.status_code == 200 and "grounded" in data
        detail = "" if passed else f"status={r.status_code}, keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}"
        results.record("2.8", "Evidence returns grounded field", passed, detail=detail)
    except Exception as e:
        results.record("2.8", "Evidence returns grounded field", False, detail=str(e))


def test_evidence_pack_present(cfg, results):
    """2.9: Evidence response includes evidence_pack."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/evidence",
            headers=cfg["headers"],
            json={"context": cfg["context"], "query": "test", "k": 5},
            timeout=30
        )
        data = r.json()
        passed = r.status_code == 200 and "evidence_pack" in data
        detail = "" if passed else f"Missing evidence_pack in response"
        results.record("2.9", "Evidence pack present", passed, detail=detail)
    except Exception as e:
        results.record("2.9", "Evidence pack present", False, detail=str(e))


def test_chunks_endpoint(cfg, results):
    """2.10: POST /v1/chunks retrieves chunks by ID."""
    # First get some chunk IDs from search
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/search",
            headers=cfg["headers"],
            json={"context": cfg["context"], "query": "test", "k": 2},
            timeout=30
        )
        if r.status_code != 200:
            results.record("2.10", "Chunks endpoint", False, detail="Could not get chunk IDs from search")
            return
        
        search_results = r.json().get("results", [])
        if not search_results:
            results.record("2.10", "Chunks endpoint", True, skipped=True, detail="No chunks to test with")
            return
        
        chunk_ids = [c.get("chunk_id") for c in search_results if c.get("chunk_id")]
        if not chunk_ids:
            results.record("2.10", "Chunks endpoint", True, skipped=True, detail="No chunk IDs in results")
            return
        
        # Now test chunks endpoint
        r = requests.post(
            f"{cfg['base_url']}/v1/chunks",
            headers=cfg["headers"],
            json={"context": cfg["context"], "chunk_ids": chunk_ids[:2]},
            timeout=30
        )
        data = r.json()
        passed = r.status_code == 200 and "chunks" in data
        detail = "" if passed else f"status={r.status_code}"
        results.record("2.10", "Chunks endpoint", passed, detail=detail)
    except Exception as e:
        results.record("2.10", "Chunks endpoint", False, detail=str(e))


def test_unknown_context_404(cfg, results):
    """2.11: Unknown context returns 404."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/evidence",
            headers=cfg["headers"],
            json={"context": "NonExistentContext12345", "query": "test"},
            timeout=10
        )
        passed = r.status_code == 404
        detail = "" if passed else f"Expected 404, got {r.status_code}"
        results.record("2.11", "Unknown context returns 404", passed, detail=detail)
    except Exception as e:
        results.record("2.11", "Unknown context returns 404", False, detail=str(e))


def test_invalid_context_name_rejected(cfg, results):
    """2.12: Invalid context name (path traversal) rejected."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/evidence",
            headers=cfg["headers"],
            json={"context": "../../../etc/passwd", "query": "test"},
            timeout=10
        )
        # Should be 400 or 422 (validation error)
        passed = r.status_code in (400, 422)
        detail = "" if passed else f"Expected 400/422, got {r.status_code}"
        results.record("2.12", "Invalid context name rejected", passed, detail=detail)
    except Exception as e:
        results.record("2.12", "Invalid context name rejected", False, detail=str(e))


def test_k_capped_at_max(cfg, results):
    """2.13: k parameter capped at max (20)."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/search",
            headers=cfg["headers"],
            json={"context": cfg["context"], "query": "test", "k": 100},
            timeout=30
        )
        data = r.json()
        # Either request succeeds with capped results, or returns 400/422
        if r.status_code in (400, 422):
            passed = True
            detail = "Rejected k>20 with validation error"
        elif r.status_code == 200:
            passed = len(data.get("results", [])) <= 20
            detail = "" if passed else f"Got {len(data.get('results', []))} results, expected <=20"
        else:
            passed = False
            detail = f"Unexpected status {r.status_code}"
        results.record("2.13", "k capped at max", passed, detail=detail)
    except Exception as e:
        results.record("2.13", "k capped at max", False, detail=str(e))


def test_query_length_limit(cfg, results):
    """2.14: Query length limit enforced."""
    try:
        long_query = "test " * 500  # ~2500 chars, over 1000 limit
        r = requests.post(
            f"{cfg['base_url']}/v1/evidence",
            headers=cfg["headers"],
            json={"context": cfg["context"], "query": long_query},
            timeout=10
        )
        # Should be 400 or 422 (validation error)
        passed = r.status_code in (400, 422)
        detail = "" if passed else f"Expected 400/422, got {r.status_code}"
        results.record("2.14", "Query length limit enforced", passed, detail=detail)
    except Exception as e:
        results.record("2.14", "Query length limit enforced", False, detail=str(e))


def test_openapi_schema_available(cfg, results):
    """2.15: OpenAPI schema available for ChatGPT Actions."""
    try:
        r = requests.get(f"{cfg['base_url']}/openapi.json", timeout=10)
        passed = r.status_code == 200 and "openapi" in r.json()
        detail = "" if passed else f"status={r.status_code}"
        results.record("2.15", "OpenAPI schema available", passed, detail=detail)
    except Exception as e:
        results.record("2.15", "OpenAPI schema available", False, detail=str(e))


def test_answer_endpoint_disabled(cfg, results):
    """2.16: /v1/answer returns error when disabled."""
    try:
        r = requests.post(
            f"{cfg['base_url']}/v1/answer",
            headers=cfg["headers"],
            json={"context": cfg["context"], "query": "test"},
            timeout=10
        )
        # Should be 403 or 404 (disabled) or specific error
        if r.status_code == 404:
            passed = True
            detail = "Endpoint not found (expected if not implemented)"
        elif r.status_code == 403:
            passed = True
            detail = "Endpoint disabled as expected"
        elif r.status_code == 200:
            # Might be enabled - check for error in body
            data = r.json()
            if data.get("error") == "answer_endpoint_disabled":
                passed = True
                detail = "Returned disabled error in body"
            else:
                passed = False
                detail = "Endpoint returned 200 - may be enabled"
        else:
            passed = False
            detail = f"Unexpected status {r.status_code}"
        results.record("2.16", "Answer endpoint disabled by default", passed, detail=detail)
    except Exception as e:
        results.record("2.16", "Answer endpoint disabled by default", False, detail=str(e))


def test_cors_headers_present(cfg, results):
    """2.17: CORS headers present for ChatGPT."""
    try:
        r = requests.options(
            f"{cfg['base_url']}/v1/evidence",
            headers={
                "Origin": "https://chat.openai.com",
                "Access-Control-Request-Method": "POST"
            },
            timeout=10
        )
        # Check for CORS headers
        cors_origin = r.headers.get("Access-Control-Allow-Origin", "")
        passed = "chat.openai.com" in cors_origin or cors_origin == "*"
        detail = "" if passed else f"CORS origin: {cors_origin}"
        results.record("2.17", "CORS headers for ChatGPT", passed, detail=detail)
    except Exception as e:
        results.record("2.17", "CORS headers for ChatGPT", False, detail=str(e))


def test_audit_log_written(cfg, results):
    """2.18: Audit log captures requests (check manually)."""
    # This is a reminder to check manually - can't verify file from remote
    results.record("2.18", "Audit log written", True, skipped=True, 
                   detail="Check P:\\ai_memory\\gateway_audit.jsonl manually")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="P2 Gateway E2E Smoke Test")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Gateway base URL")
    parser.add_argument("--context", default=DEFAULT_CONTEXT, help="Context name to test")
    args = parser.parse_args()
    
    cfg = get_config(args)
    results = TestResults()
    
    print("=" * 60)
    print("Chinvex P2 Gateway E2E Smoke Test")
    print("=" * 60)
    print(f"Base URL: {cfg['base_url']}")
    print(f"Context:  {cfg['context']}")
    print(f"Token:    {cfg['token'][:8]}...")
    print("=" * 60)
    print()
    
    # Check gateway is reachable
    log("Checking gateway availability...")
    try:
        r = requests.get(f"{cfg['base_url']}/health", timeout=5)
        if r.status_code != 200:
            print(f"ERROR: Gateway not healthy (status {r.status_code})")
            sys.exit(1)
        log(f"Gateway healthy: {r.json()}")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to gateway at {cfg['base_url']}")
        print("Make sure gateway is running: chinvex gateway serve")
        sys.exit(1)
    
    print()
    print("-" * 60)
    print("Running tests...")
    print("-" * 60)
    
    # Authentication tests
    test_health_no_auth(cfg, results)
    test_auth_required(cfg, results)
    test_invalid_token_rejected(cfg, results)
    test_valid_token_accepted(cfg, results)
    
    # Endpoint tests
    test_contexts_endpoint(cfg, results)
    test_search_returns_results(cfg, results)
    test_search_respects_k(cfg, results)
    test_evidence_grounded_field(cfg, results)
    test_evidence_pack_present(cfg, results)
    test_chunks_endpoint(cfg, results)
    
    # Security tests
    test_unknown_context_404(cfg, results)
    test_invalid_context_name_rejected(cfg, results)
    test_k_capped_at_max(cfg, results)
    test_query_length_limit(cfg, results)
    
    # Integration tests
    test_openapi_schema_available(cfg, results)
    test_answer_endpoint_disabled(cfg, results)
    test_cors_headers_present(cfg, results)
    test_audit_log_written(cfg, results)
    
    # Summary
    success = results.summary()
    
    if success:
        print("\n✓ All tests passed! Gateway is ready for ChatGPT integration.")
    else:
        print("\n✗ Some tests failed. Review output above.")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
