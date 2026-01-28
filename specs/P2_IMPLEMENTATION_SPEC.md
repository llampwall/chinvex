# Chinvex P2 Implementation Spec

**Version:** 2.0  
**Date:** 2026-01-27  
**Status:** Ready for implementation  
**Depends on:** P1 complete

---

## Overview

P2 exposes Chinvex as a remote, TLS-reachable API that ChatGPT Actions can call. The gateway is **read-only** and **token-authenticated**. Chinvex remains a retrieval + grounding engine; synthesis is optional and flag-gated.

### P2 Scope

1. **P2.1** Gateway service (FastAPI endpoints)
2. **P2.2** OpenAPI schema for ChatGPT Actions
3. **P2.3** Security hardening (allowlist, caps, rate limiting, audit)
4. **P2.4** Deployment (Cloudflare Tunnel or Caddy)
5. **P2.5** Acceptance tests

### Non-Goals (P2)

- Writeback / ingestion over the internet (ingest stays local)
- Cross-context search
- Chunking improvements
- Watch history / notifications
- Archive tier
- Multi-user auth flows
- Web UI

### Deferred to P3

- Chunking strategy improvements (overlap, semantic boundaries, code-aware splitting)
- Watch history + webhook notifications
- Cross-context search (`--all` flag)
- Archive tier for old content

---

## 1. Gateway Service (P2.1)

### Purpose

Expose a remote, TLS-reachable API that ChatGPT Actions can call to retrieve Chinvex results with token auth and strict read-only scope.

### Architecture

```
┌─────────────────┐      HTTPS       ┌─────────────────┐
│   ChatGPT       │ ───────────────► │  Chinvex        │
│   (Actions)     │  Bearer token    │  Gateway        │
└─────────────────┘                  │  (FastAPI)      │
                                     └────────┬────────┘
┌─────────────────┐      MCP                  │
│   Claude/Codex  │ ──────────────────────────┤
│   (MCP)         │                           │
└─────────────────┘                  ┌────────▼────────┐
                                     │  Chinvex Core   │
┌─────────────────┐      CLI         │  (same backend) │
│   Terminal      │ ──────────────────────────┘
└─────────────────┘
```

All interfaces (Gateway, MCP, CLI) call the same underlying search/retrieval functions.

### Tech Stack

- **FastAPI** — lightweight, async, auto-generates OpenAPI spec
- **Uvicorn** — ASGI server
- **Cloudflare Tunnel** or **Caddy** — TLS termination

### Endpoints

#### Required Endpoints

##### `GET /health`

Health check. No auth required.

**Response:**
```json
{
  "status": "ok",
  "version": "0.2.0",
  "contexts_available": 3
}
```

##### `POST /v1/search`

Raw hybrid search. Returns ranked chunks without grounding check.

**Request:**
```json
{
  "context": "Chinvex",
  "query": "hybrid retrieval scoring",
  "k": 10,
  "source_types": ["repo", "chat", "codex_session"],
  "no_recency": false
}
```

**Response:**
```json
{
  "context": "Chinvex",
  "query": "hybrid retrieval scoring",
  "results": [
    {
      "chunk_id": "abc123",
      "text": "def blend_scores(fts_norm, vec_norm)...",
      "source_uri": "C:\\Code\\chinvex\\src\\scoring.py",
      "source_type": "repo",
      "scores": {
        "fts": 0.82,
        "vector": 0.75,
        "blended": 0.79,
        "rank": 0.79
      },
      "metadata": {
        "line_start": 42,
        "line_end": 67,
        "updated_at": "2026-01-26T10:00:00Z"
      }
    }
  ],
  "total_results": 10
}
```

##### `POST /v1/evidence`

Search with grounding check. Returns evidence pack + grounding status. **This is the primary endpoint for ChatGPT Actions.**

**Request:**
```json
{
  "context": "Chinvex",
  "query": "How does score blending work?",
  "k": 8
}
```

**Response (grounded=true):**
```json
{
  "context": "Chinvex",
  "query": "How does score blending work?",
  "grounded": true,
  "evidence_pack": {
    "chunks": [
      {
        "chunk_id": "abc123",
        "text": "def blend_scores(fts_norm, vec_norm):\n    FTS_W, VEC_W = 0.6, 0.4\n    ...",
        "source_uri": "C:\\Code\\chinvex\\src\\scoring.py",
        "source_type": "repo",
        "range": {
          "line_start": 42,
          "line_end": 67
        },
        "score": 0.89
      }
    ]
  },
  "retrieval_debug": {
    "k": 8,
    "chunks_retrieved": 8,
    "chunks_above_threshold": 5
  }
}
```

**Response (grounded=false):**
```json
{
  "context": "Chinvex",
  "query": "What is the airspeed velocity of an unladen swallow?",
  "grounded": false,
  "evidence_pack": {
    "chunks": []
  },
  "retrieval_debug": {
    "k": 8,
    "chunks_retrieved": 3,
    "chunks_above_threshold": 0
  },
  "message": "No retrieved content supports a direct answer to this query."
}
```

##### `POST /v1/chunks`

Fetch specific chunks by ID. Useful for follow-up retrieval.

**Request:**
```json
{
  "context": "Chinvex",
  "chunk_ids": ["abc123", "def456"]
}
```

**Response:**
```json
{
  "context": "Chinvex",
  "chunks": [
    {
      "chunk_id": "abc123",
      "text": "...",
      "source_uri": "...",
      "source_type": "repo",
      "metadata": {...}
    }
  ]
}
```

##### `GET /v1/contexts`

List available contexts. Respects allowlist.

**Response:**
```json
{
  "contexts": [
    {
      "name": "Chinvex",
      "aliases": ["chindex"],
      "updated_at": "2026-01-27T04:00:00Z"
    },
    {
      "name": "Personal",
      "aliases": [],
      "updated_at": "2026-01-27T03:00:00Z"
    }
  ]
}
```

#### Optional Endpoint (Flag-Gated)

##### `POST /v1/answer`

Full synthesis with LLM. **Disabled by default.** Enable with `GATEWAY_ENABLE_SERVER_LLM=true`.

**Request:**
```json
{
  "context": "Chinvex",
  "query": "How does score blending work?",
  "k": 8,
  "grounded": true
}
```

**Response:**
```json
{
  "schema_version": 1,
  "context": "Chinvex",
  "query": "How does score blending work?",
  "grounded": true,
  "answer": "Score blending combines normalized FTS and vector scores using weighted addition. FTS gets 60% weight, vector gets 40%. This happens after both score types are min-max normalized within the candidate set.",
  "citations": [
    {
      "chunk_id": "abc123",
      "source_uri": "C:\\Code\\chinvex\\src\\scoring.py",
      "range": {"line_start": 42, "line_end": 67}
    }
  ],
  "evidence_pack": {
    "chunks": [...]
  },
  "errors": []
}
```

**When disabled:**
```json
{
  "error": "answer_endpoint_disabled",
  "message": "Server-side synthesis is disabled. Use /v1/evidence instead."
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `context` | string | Context name to search |
| `query` | string | Natural language query |
| `k` | int | Max chunks to retrieve (default 8, max 20) |
| `source_types` | string[] | Filter by source type (optional) |
| `no_recency` | bool | Disable recency decay (default false) |
| `grounded` | bool | Whether evidence supports an answer |
| `evidence_pack` | object | Retrieved chunks with metadata |
| `chunk_id` | string | Stable chunk identifier |
| `score` | float | Final rank score (0-1) |

---

## 2. OpenAPI Schema for ChatGPT Actions (P2.2)

### Purpose

ChatGPT Actions requires an OpenAPI schema describing the endpoints. FastAPI auto-generates this.

### Access

```
GET /openapi.json
```

### ChatGPT Actions Import

1. Go to chat.openai.com → Create GPT → Configure → Actions
2. Click "Create new action"
3. Click "Import from URL"
4. Enter: `https://your-domain/openapi.json`
5. Configure authentication (see below)

### Authentication Configuration in ChatGPT

- **Authentication type:** API Key
- **Auth Type:** Bearer
- **Header name:** Authorization
- **API Key:** Your `CHINVEX_API_TOKEN` value

### Schema Design Rules

Keep schemas boring and stable. ChatGPT will hallucinate if the schema is loose.

- Use explicit types (no `any` or `object` without properties)
- Provide descriptions for all fields
- Use enums where values are constrained
- Keep parameter descriptions under 300 chars (OpenAI limit)
- Keep endpoint descriptions under 300 chars

### Example OpenAPI Fragment

```yaml
openapi: 3.1.0
info:
  title: Chinvex Memory API
  version: 0.2.0
  description: Query personal knowledge base with grounded retrieval

paths:
  /v1/evidence:
    post:
      operationId: getEvidence
      summary: Search for evidence to answer a question
      description: Returns relevant chunks with grounding status. Use this for most queries.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [context, query]
              properties:
                context:
                  type: string
                  description: Context name (e.g., "Chinvex", "Personal")
                query:
                  type: string
                  description: Natural language question
                k:
                  type: integer
                  default: 8
                  maximum: 20
                  description: Max chunks to retrieve
      responses:
        '200':
          description: Evidence pack with grounding status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/EvidenceResponse'
        '401':
          description: Unauthorized
        '404':
          description: Context not found

components:
  schemas:
    EvidenceResponse:
      type: object
      properties:
        context:
          type: string
        query:
          type: string
        grounded:
          type: boolean
          description: True if evidence supports an answer
        evidence_pack:
          type: object
          properties:
            chunks:
              type: array
              items:
                $ref: '#/components/schemas/Chunk'
        message:
          type: string
          description: Present when grounded=false
    
    Chunk:
      type: object
      properties:
        chunk_id:
          type: string
        text:
          type: string
        source_uri:
          type: string
        source_type:
          type: string
          enum: [repo, chat, codex_session, note]
        score:
          type: number
        range:
          type: object
          properties:
            line_start:
              type: integer
            line_end:
              type: integer

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer

security:
  - bearerAuth: []
```

### GPT System Prompt

Include this in your custom GPT's instructions:

```
You have access to the user's personal knowledge base via the Chinvex API.

CRITICAL RULES:
1. When the user asks about their projects, decisions, code, or past work, ALWAYS call getEvidence first.
2. If grounded=false, say "I couldn't find information about that in your memory." Do NOT make up an answer.
3. If grounded=true, synthesize an answer using ONLY the returned chunks. Cite sources.
4. Never claim to know something that isn't in the evidence pack.

Available contexts: Ask the user which context to search, or use "Personal" as default.

When citing, use format: [source_uri:line_start-line_end]
```

---

## 3. Security Hardening (P2.3)

### Purpose

Don't regret exposing this. Start read-only and harden.

### Authentication

#### Bearer Token

```
Authorization: Bearer <token>
```

- Token set via `CHINVEX_API_TOKEN` environment variable
- Constant-time comparison to prevent timing attacks
- All endpoints except `/health` require auth
- Return `401 Unauthorized` on missing/invalid token

```python
import secrets

def verify_token(provided: str, expected: str) -> bool:
    """Constant-time comparison."""
    return secrets.compare_digest(provided, expected)
```

#### Token Generation

```bash
chinvex gateway token generate
# Output: CHINVEX_API_TOKEN=a1b2c3d4e5f6...

chinvex gateway token rotate
# Generates new token, invalidates old
```

Store token securely. Add to environment or secrets manager.

### Context Allowlist

Gateway only serves contexts explicitly listed in config. Prevents "query arbitrary disk path" attacks.

```json
{
  "gateway": {
    "context_allowlist": ["Chinvex", "Personal"]
  }
}
```

- If allowlist is empty/missing: serve all contexts (dev mode)
- If allowlist is set: only those contexts are accessible
- Request for unlisted context returns `404 Not Found` (not `403`, to avoid enumeration)

### Request Limits

#### Parameter Caps

| Parameter | Max Value | Default |
|-----------|-----------|---------|
| `k` | 20 | 8 |
| `chunk_ids` (array length) | 20 | - |
| `query` length | 1000 chars | - |

Requests exceeding limits return `400 Bad Request` with explanation.

#### Response Caps

| Field | Max Size |
|-------|----------|
| `text` per chunk | 5000 chars (truncate, add `"[truncated]"`) |
| `chunks` array | 20 items |
| Total response | 1 MB |

#### Rate Limiting

Simple token-bucket per API token:

```json
{
  "gateway": {
    "rate_limit": {
      "requests_per_minute": 60,
      "requests_per_hour": 500
    }
  }
}
```

Return `429 Too Many Requests` with `Retry-After` header when exceeded.

For P2, implement in-memory. Redis-backed rate limiting is P3.

### Audit Log

Append-only log of all requests:

**File:** `P:\ai_memory\gateway_audit.jsonl`

```jsonl
{"ts": "2026-01-27T10:00:00Z", "request_id": "abc123", "endpoint": "/v1/evidence", "context": "Chinvex", "query_hash": "sha256:...", "status": 200, "latency_ms": 142, "client_ip": "..."}
{"ts": "2026-01-27T10:00:01Z", "request_id": "def456", "endpoint": "/v1/search", "context": "Personal", "query_hash": "sha256:...", "status": 401, "latency_ms": 2, "client_ip": "..."}
```

**Fields:**
- `ts`: ISO8601 timestamp
- `request_id`: UUID for correlation
- `endpoint`: Path called
- `context`: Context requested (if applicable)
- `query_hash`: SHA256 of query (not raw text, for privacy)
- `status`: HTTP status code
- `latency_ms`: Response time
- `client_ip`: Requester IP (for abuse detection)

**Do not log:**
- Raw query text
- Full response bodies
- Token values

### CORS

Allow only trusted origins:

```json
{
  "gateway": {
    "cors_origins": [
      "https://chat.openai.com",
      "https://chatgpt.com"
    ]
  }
}
```

For development, can add `http://localhost:*`.

### Input Validation

- Validate `context` is alphanumeric + underscore only
- Validate `query` is UTF-8, no null bytes
- Validate `chunk_ids` are hex strings of expected length
- Reject requests with unexpected fields (strict schema)

```python
CONTEXT_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,50}$')
CHUNK_ID_PATTERN = re.compile(r'^[a-f0-9]{12}$')

def validate_context(name: str) -> bool:
    return bool(CONTEXT_PATTERN.match(name))
```

---

## 4. Deployment (P2.4)

### Overview

Gateway runs on the same machine as the indexes. TLS is handled by a reverse proxy or tunnel.

### Option A: Cloudflare Tunnel (Recommended)

Fastest path to HTTPS. No port forwarding, no dynamic DNS.

#### Setup

1. Install cloudflared:
   ```bash
   # Windows
   winget install cloudflare.cloudflared
   
   # Or download from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/
   ```

2. Login and create tunnel:
   ```bash
   cloudflared tunnel login
   cloudflared tunnel create chinvex
   ```

3. Configure tunnel (`~/.cloudflared/config.yml`):
   ```yaml
   tunnel: chinvex
   credentials-file: ~/.cloudflared/<tunnel-id>.json
   
   ingress:
     - hostname: chinvex.yourdomain.com
       service: http://localhost:7778
     - service: http_status:404
   ```

4. Add DNS record:
   ```bash
   cloudflared tunnel route dns chinvex chinvex.yourdomain.com
   ```

5. Run tunnel:
   ```bash
   cloudflared tunnel run chinvex
   ```

#### PM2 Integration

```bash
pm2 start "cloudflared tunnel run chinvex" --name chinvex-tunnel
pm2 start "chinvex gateway serve --port 7778" --name chinvex-gateway
pm2 save
```

### Option B: Caddy Reverse Proxy

If you already have Caddy running (like for godex).

#### Caddyfile

```
chinvex.yourdomain.com {
    reverse_proxy localhost:7778
    
    header {
        X-Content-Type-Options nosniff
        X-Frame-Options DENY
        X-XSS-Protection "1; mode=block"
    }
    
    log {
        output file /var/log/caddy/chinvex.log
    }
}
```

#### Start Services

```bash
pm2 start "chinvex gateway serve --port 7778" --name chinvex-gateway
# Caddy already running from godex setup
caddy reload
```

### Gateway CLI

```bash
# Start gateway server
chinvex gateway serve [--host 0.0.0.0] [--port 7778]

# Generate/rotate token
chinvex gateway token generate
chinvex gateway token rotate

# Check status
chinvex gateway status
```

### Configuration

**File:** `P:\ai_memory\gateway.json` (or in context.json)

```json
{
  "gateway": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 7778,
    "token_env": "CHINVEX_API_TOKEN",
    "context_allowlist": ["Chinvex", "Personal"],
    "cors_origins": ["https://chat.openai.com", "https://chatgpt.com"],
    "rate_limit": {
      "requests_per_minute": 60,
      "requests_per_hour": 500
    },
    "enable_server_llm": false
  }
}
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CHINVEX_API_TOKEN` | Bearer token for auth |
| `GATEWAY_ENABLE_SERVER_LLM` | Enable `/v1/answer` endpoint (default: false) |
| `CHINVEX_GATEWAY_CONFIG` | Path to gateway.json (optional) |

### Health Check Script

For monitoring:

```bash
#!/bin/bash
# health_check.sh
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" https://chinvex.yourdomain.com/health)
if [ "$RESPONSE" != "200" ]; then
    echo "Gateway unhealthy: $RESPONSE"
    exit 1
fi
echo "Gateway healthy"
```

### Startup Order

1. Ensure Ollama is running (for embeddings)
2. Start gateway: `chinvex gateway serve`
3. Start tunnel/proxy: `cloudflared tunnel run chinvex` or Caddy
4. Verify: `curl https://chinvex.yourdomain.com/health`

---

## 5. Acceptance Tests (P2.5)

### Prerequisites

- Gateway running locally on port 7778
- Tunnel/proxy configured (for HTTPS tests)
- At least one context with ingested data
- `CHINVEX_API_TOKEN` set

### Test Suite

#### Authentication Tests

```bash
# Test 2.5.1: Health endpoint (no auth required)
curl -s https://chinvex.yourdomain.com/health
# Expected: {"status": "ok", ...}

# Test 2.5.2: Unauthorized request rejected
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://chinvex.yourdomain.com/v1/evidence \
  -H "Content-Type: application/json" \
  -d '{"context": "Chinvex", "query": "test"}'
# Expected: 401

# Test 2.5.3: Valid token accepted
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://chinvex.yourdomain.com/v1/evidence \
  -H "Authorization: Bearer $CHINVEX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": "Chinvex", "query": "test"}'
# Expected: 200

# Test 2.5.4: Invalid token rejected
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://chinvex.yourdomain.com/v1/evidence \
  -H "Authorization: Bearer invalid_token_here" \
  -H "Content-Type: application/json" \
  -d '{"context": "Chinvex", "query": "test"}'
# Expected: 401
```

#### Evidence Endpoint Tests

```bash
# Test 2.5.5: Evidence returns grounded=true for known content
curl -s -X POST https://chinvex.yourdomain.com/v1/evidence \
  -H "Authorization: Bearer $CHINVEX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": "Chinvex", "query": "hybrid retrieval scoring"}'
# Expected: grounded=true, evidence_pack.chunks not empty

# Test 2.5.6: Evidence returns grounded=false for unknown content
curl -s -X POST https://chinvex.yourdomain.com/v1/evidence \
  -H "Authorization: Bearer $CHINVEX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": "Chinvex", "query": "recipe for chocolate cake"}'
# Expected: grounded=false, message present
```

#### Search Endpoint Tests

```bash
# Test 2.5.7: Search returns ranked results
curl -s -X POST https://chinvex.yourdomain.com/v1/search \
  -H "Authorization: Bearer $CHINVEX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": "Chinvex", "query": "search", "k": 5}'
# Expected: results array with 1-5 items, scores present

# Test 2.5.8: Search respects k limit
curl -s -X POST https://chinvex.yourdomain.com/v1/search \
  -H "Authorization: Bearer $CHINVEX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": "Chinvex", "query": "test", "k": 3}'
# Expected: results.length <= 3
```

#### Security Tests

```bash
# Test 2.5.9: Unknown context returns 404
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://chinvex.yourdomain.com/v1/evidence \
  -H "Authorization: Bearer $CHINVEX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": "NonExistent", "query": "test"}'
# Expected: 404

# Test 2.5.10: k exceeding max is capped
curl -s -X POST https://chinvex.yourdomain.com/v1/search \
  -H "Authorization: Bearer $CHINVEX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": "Chinvex", "query": "test", "k": 100}'
# Expected: 200, results.length <= 20

# Test 2.5.11: Invalid context name rejected
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://chinvex.yourdomain.com/v1/evidence \
  -H "Authorization: Bearer $CHINVEX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": "../../../etc/passwd", "query": "test"}'
# Expected: 400
```

#### Optional Answer Endpoint Tests

```bash
# Test 2.5.12: Answer endpoint disabled by default
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://chinvex.yourdomain.com/v1/answer \
  -H "Authorization: Bearer $CHINVEX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": "Chinvex", "query": "How does scoring work?"}'
# Expected: 403 or specific error about endpoint being disabled

# Test 2.5.13: Answer endpoint works when enabled
# (Set GATEWAY_ENABLE_SERVER_LLM=true, restart gateway)
curl -s -X POST https://chinvex.yourdomain.com/v1/answer \
  -H "Authorization: Bearer $CHINVEX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": "Chinvex", "query": "How does scoring work?", "grounded": true}'
# Expected: 200, answer field present, citations present
```

#### ChatGPT Integration Test

Manual test:

1. Create custom GPT at chat.openai.com
2. Add Action, import from `https://chinvex.yourdomain.com/openapi.json`
3. Configure bearer auth with your token
4. Ask: "Search my Chinvex memory for hybrid retrieval"
5. Expected: GPT calls `/v1/evidence`, receives results, synthesizes answer

### Smoke Test Script

```python
#!/usr/bin/env python3
"""P2 Gateway Smoke Test"""

import os
import sys
import requests

BASE_URL = os.environ.get("CHINVEX_GATEWAY_URL", "https://chinvex.yourdomain.com")
TOKEN = os.environ["CHINVEX_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def test(name, response, expected_status, check_fn=None):
    passed = response.status_code == expected_status
    if passed and check_fn:
        passed = check_fn(response.json())
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    return passed

results = []

# Health
r = requests.get(f"{BASE_URL}/health")
results.append(test("Health endpoint", r, 200, lambda j: j.get("status") == "ok"))

# Auth required
r = requests.post(f"{BASE_URL}/v1/evidence", json={"context": "Chinvex", "query": "test"})
results.append(test("Auth required", r, 401))

# Evidence with auth
r = requests.post(f"{BASE_URL}/v1/evidence", headers=HEADERS, json={"context": "Chinvex", "query": "scoring"})
results.append(test("Evidence endpoint", r, 200, lambda j: "grounded" in j))

# Search with auth
r = requests.post(f"{BASE_URL}/v1/search", headers=HEADERS, json={"context": "Chinvex", "query": "test", "k": 5})
results.append(test("Search endpoint", r, 200, lambda j: "results" in j))

# Unknown context
r = requests.post(f"{BASE_URL}/v1/evidence", headers=HEADERS, json={"context": "FakeContext", "query": "test"})
results.append(test("Unknown context", r, 404))

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
```

---

## 6. Implementation Order

### Phase 1: Core Gateway (Days 1-2)

1. FastAPI app skeleton with `/health`
2. Bearer token auth middleware
3. `/v1/search` endpoint (wraps existing search)
4. `/v1/evidence` endpoint (wraps existing search + grounding check)
5. `/v1/chunks` endpoint (batch chunk fetch)
6. `/v1/contexts` endpoint (list contexts)

### Phase 2: Security (Day 2)

7. Context allowlist validation
8. Request parameter caps (k, query length)
9. Response size caps (chunk text truncation)
10. Input validation (context name, chunk IDs)
11. Audit log (JSONL append)

### Phase 3: Configuration & CLI (Day 3)

12. `gateway.json` config loading
13. `chinvex gateway serve` CLI command
14. `chinvex gateway token generate/rotate` commands
15. Rate limiting (in-memory token bucket)

### Phase 4: Deployment (Day 3-4)

16. Cloudflare Tunnel setup docs
17. Caddy alternative docs
18. PM2 integration
19. Health check script

### Phase 5: ChatGPT Integration (Day 4)

20. OpenAPI schema refinement
21. CORS configuration
22. GPT Action import test
23. GPT system prompt template

### Phase 6: Optional Answer Endpoint (Day 5, if needed)

24. `/v1/answer` endpoint (flag-gated)
25. `GATEWAY_ENABLE_SERVER_LLM` flag handling

---

## 7. Dependencies

### New Packages

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
```

### Existing (No Changes)

- requests
- chromadb
- sqlite3 (stdlib)
- pydantic (already used in P1)

---

## 8. Configuration Reference

### gateway.json

```json
{
  "gateway": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 7778,
    "token_env": "CHINVEX_API_TOKEN",
    "context_allowlist": ["Chinvex", "Personal"],
    "cors_origins": [
      "https://chat.openai.com",
      "https://chatgpt.com"
    ],
    "rate_limit": {
      "requests_per_minute": 60,
      "requests_per_hour": 500
    },
    "limits": {
      "max_k": 20,
      "max_chunk_ids": 20,
      "max_query_length": 1000,
      "max_chunk_text_length": 5000
    },
    "enable_server_llm": false,
    "audit_log_path": "P:\\ai_memory\\gateway_audit.jsonl"
  }
}
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CHINVEX_API_TOKEN` | Yes | Bearer token for authentication |
| `GATEWAY_ENABLE_SERVER_LLM` | No | Enable `/v1/answer` (default: false) |
| `CHINVEX_GATEWAY_CONFIG` | No | Path to gateway.json |

---

## 9. Acceptance Test Summary

| ID | Test | Pass Criteria |
|----|------|---------------|
| 2.5.1 | Health endpoint | Returns 200 with status=ok |
| 2.5.2 | No auth rejected | Returns 401 |
| 2.5.3 | Valid token accepted | Returns 200 |
| 2.5.4 | Invalid token rejected | Returns 401 |
| 2.5.5 | Evidence grounded=true | Chunks returned for known query |
| 2.5.6 | Evidence grounded=false | Message returned for unknown query |
| 2.5.7 | Search returns results | Ranked results with scores |
| 2.5.8 | Search respects k | results.length <= k |
| 2.5.9 | Unknown context | Returns 404 |
| 2.5.10 | k capped | results.length <= 20 regardless of input |
| 2.5.11 | Invalid context name | Returns 400 |
| 2.5.12 | Answer disabled | Returns error when flag off |
| 2.5.13 | Answer enabled | Returns answer when flag on |

---

## Appendix A: ChatGPT Custom GPT Setup

### Step-by-Step

1. Go to chat.openai.com
2. Click your profile → "My GPTs" → "Create a GPT"
3. Go to "Configure" tab
4. Add name: "Chinvex Memory"
5. Add description: "Query my personal knowledge base"
6. Scroll to "Actions" → "Create new action"
7. Click "Import from URL"
8. Enter: `https://chinvex.yourdomain.com/openapi.json`
9. Click "Authentication" → "API Key"
   - Auth Type: Bearer
   - Paste your `CHINVEX_API_TOKEN`
10. Save and test

### System Instructions

```
You are a helpful assistant with access to the user's personal knowledge base via the Chinvex API.

## Core Rules

1. When the user asks about their projects, decisions, code, past conversations, or anything personal:
   - ALWAYS call the getEvidence action first
   - Do NOT answer from general knowledge

2. Interpreting the response:
   - If grounded=true: Synthesize an answer using ONLY the returned chunks
   - If grounded=false: Say "I couldn't find information about that in your memory. Would you like me to search a different context or rephrase the question?"

3. NEVER make up information that isn't in the evidence pack

4. When citing sources, use format: [filename:line_start-line_end]

## Available Contexts

Ask the user which context to search. Common ones:
- Chinvex: Technical project work
- Personal: General notes and conversations

## Example Interaction

User: "How does scoring work in Chinvex?"
→ Call getEvidence with context="Chinvex", query="scoring"
→ If grounded=true, explain based on the chunks
→ Cite the source files
```

---

## Appendix B: Troubleshooting

### Gateway won't start

```bash
# Check port availability
netstat -an | grep 7778

# Check config path
echo $CHINVEX_GATEWAY_CONFIG

# Run with debug logging
chinvex gateway serve --port 7778 --debug
```

### 401 errors

```bash
# Verify token is set
echo $CHINVEX_API_TOKEN

# Test token directly
curl -H "Authorization: Bearer $CHINVEX_API_TOKEN" https://chinvex.yourdomain.com/v1/contexts
```

### Cloudflare Tunnel issues

```bash
# Check tunnel status
cloudflared tunnel info chinvex

# Check tunnel logs
cloudflared tunnel run chinvex 2>&1 | head -50

# Verify DNS
nslookup chinvex.yourdomain.com
```

### ChatGPT can't connect

1. Verify OpenAPI spec is accessible: `curl https://chinvex.yourdomain.com/openapi.json`
2. Check CORS headers are correct
3. Re-import the Action in ChatGPT
4. Check the "Test" button in GPT Action config

---

*End of P2 spec.*
