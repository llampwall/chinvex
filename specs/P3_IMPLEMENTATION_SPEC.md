# Chinvex P3 Implementation Spec

**Version:** 1.0  
**Date:** 2026-01-27  
**Status:** Draft — for future reference  
**Depends on:** P2 complete

---

## Overview

P3 is the "quality of life" release. P2 got ChatGPT connected. P3 makes everything work better: smarter chunking, proactive alerts, search across all your stuff, and fading old content gracefully.

### P3 Scope

1. **P3.1** Chunking strategy improvements (overlap, semantic boundaries, code-aware)
2. **P3.2** Watch history + webhook notifications
3. **P3.3** Cross-context search
4. **P3.4** Archive tier for old content
5. **P3.5** Gateway improvements (Redis rate limiting, metrics)

### Non-Goals (P4+)

- Multi-user auth / OAuth
- PDF/email/Slack ingestion adapters
- Web UI dashboard
- Mobile app
- Real-time sync / live updates
- Conversation memory (multi-turn context in gateway)

---

## 1. Chunking Strategy Improvements (P3.1)

### Current State (P0-P2)

- Fixed character limit (~3000 chars)
- Repo files: simple overlap (300 chars)
- Conversations: never split turns
- No semantic awareness

### Problems

1. Hard splits break context — function split mid-body
2. No preference for natural boundaries
3. Code-unaware — doesn't respect function/class structure

### Improved Strategy

#### 1.1 Overlap (All Sources)

Always include overlap to recover context at chunk boundaries:

```python
CHUNK_SIZE = 3000      # chars
OVERLAP = 300          # chars (~10%)
```

**Implementation:**
```python
def chunk_with_overlap(text: str, size: int = 3000, overlap: int = 300) -> list[tuple[int, int]]:
    """Return list of (start, end) positions for chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append((start, end))
        if end >= len(text):
            break
        start = end - overlap
    return chunks
```

#### 1.2 Semantic Boundary Detection

Try to split at natural boundaries, falling back gracefully:

```python
SPLIT_PRIORITIES = [
    (r'\n## ', 100),           # Markdown H2
    (r'\n### ', 90),           # Markdown H3
    (r'\n---\n', 85),          # Markdown horizontal rule
    (r'\nclass ', 80),         # Python class
    (r'\ndef ', 75),           # Python function
    (r'\nasync def ', 75),     # Python async function
    (r'\nfunction ', 75),      # JS function
    (r'\nconst \w+ = ', 70),   # JS const declaration
    (r'\nexport ', 70),        # JS/TS export
    (r'\n\n\n', 60),           # Multiple blank lines
    (r'\n\n', 50),             # Paragraph break
    (r'\n', 10),               # Line break (last resort)
]

def find_best_split(text: str, target_pos: int, window: int = 300) -> int:
    """
    Find best split point near target_pos.
    Searches within ±window chars for highest-priority boundary.
    Returns position of the boundary (start of the pattern match).
    """
    search_start = max(0, target_pos - window)
    search_end = min(len(text), target_pos + window)
    search_region = text[search_start:search_end]
    
    best_pos = target_pos
    best_priority = 0
    
    for pattern, priority in SPLIT_PRIORITIES:
        for match in re.finditer(pattern, search_region):
            match_pos = search_start + match.start()
            # Prefer boundaries closer to target
            distance_penalty = abs(match_pos - target_pos) / window * 10
            effective_priority = priority - distance_penalty
            if effective_priority > best_priority:
                best_priority = effective_priority
                best_pos = match_pos
    
    return best_pos
```

#### 1.3 Code-Aware Splitting

For code files, extract logical boundaries first:

```python
import ast

def extract_python_boundaries(text: str) -> list[int]:
    """Extract line numbers where top-level definitions start."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    
    boundaries = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            boundaries.append(node.lineno - 1)  # 0-indexed
    return boundaries

def line_to_char_pos(text: str, line_num: int) -> int:
    """Convert line number to character position."""
    lines = text.split('\n')
    return sum(len(line) + 1 for line in lines[:line_num])

def chunk_python_file(text: str) -> list[Chunk]:
    """
    Split Python file respecting function/class boundaries.
    """
    boundaries = extract_python_boundaries(text)
    boundary_positions = [line_to_char_pos(text, ln) for ln in boundaries]
    
    if not boundary_positions:
        # Fallback to generic chunking
        return chunk_with_overlap_and_boundaries(text)
    
    chunks = []
    current_start = 0
    current_text = ""
    
    for pos in boundary_positions:
        segment = text[current_start:pos]
        
        if len(current_text) + len(segment) > CHUNK_SIZE:
            # Flush current chunk
            if current_text:
                chunks.append(make_chunk(current_text, current_start))
            current_text = segment
            current_start = pos - len(segment)
        else:
            current_text += segment
            
    # Don't forget the last chunk
    if current_text:
        chunks.append(make_chunk(current_text, current_start))
    
    return chunks
```

#### 1.4 Language Detection

```python
LANGUAGE_MAP = {
    '.py': 'python',
    '.pyw': 'python',
    '.js': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.jsx': 'javascript',
    '.md': 'markdown',
    '.mdx': 'markdown',
}

def detect_language(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    return LANGUAGE_MAP.get(ext, 'generic')

def chunk_file(text: str, filepath: str) -> list[Chunk]:
    language = detect_language(filepath)
    
    if language == 'python':
        return chunk_python_file(text)
    elif language in ('javascript', 'typescript'):
        return chunk_js_file(text)
    elif language == 'markdown':
        return chunk_markdown_file(text)
    else:
        return chunk_generic_file(text)
```

### Configuration

```json
{
  "chunking": {
    "max_chars": 3000,
    "overlap_chars": 300,
    "semantic_boundaries": true,
    "code_aware": true,
    "skip_files_larger_than_mb": 5
  }
}
```

### Migration

Bump `chunker_version` in fingerprints. On next ingest, files will be re-chunked automatically.

```python
CHUNKER_VERSION = 2  # Was 1 in P0-P2
```

### Acceptance Tests

```bash
# Test 3.1.1: Function not split mid-body
# Ingest Python file with 50-line function
# Verify function is in single chunk OR split at nested function/class

# Test 3.1.2: Markdown splits at headers
# Ingest markdown with ## sections
# Verify chunks start at or near ## boundaries

# Test 3.1.3: Overlap present
# Get two consecutive chunks
# Verify last ~300 chars of chunk N appear in chunk N+1

# Test 3.1.4: Large files handled
# Ingest 10MB file
# Verify it's either chunked or skipped (not crashed)
```

---

## 2. Watch History + Webhook Notifications (P3.2)

### Current State (P1)

- Watch hits stored in `state.json` (ephemeral, overwritten each ingest)
- No persistent history
- No notification mechanism

### P3 Additions

#### 2.1 Watch History Log

Append-only log of all watch triggers:

**File:** `P:\ai_memory\contexts\<Context>\watch_history.jsonl`

```jsonl
{"ts": "2026-01-27T04:00:00Z", "run_id": "run_abc", "watch_id": "p2_https", "query": "P2 HTTPS", "hits": [{"chunk_id": "abc123", "score": 0.85, "snippet": "...first 200 chars..."}]}
{"ts": "2026-01-27T05:00:00Z", "run_id": "run_def", "watch_id": "bugs", "query": "bug error", "hits": [{"chunk_id": "def456", "score": 0.78, "snippet": "..."}]}
```

**Fields:**
- `ts`: ISO8601 timestamp
- `run_id`: Ingest run that triggered the watch
- `watch_id`: Watch identifier
- `query`: Watch query
- `hits`: Array of matching chunks (snippet is first 200 chars)

#### 2.2 CLI Commands

```bash
# View watch history
chinvex watch history --context Chinvex [--since 7d] [--id p2_https] [--limit 50]

# Output formats
chinvex watch history --context Chinvex --format json
chinvex watch history --context Chinvex --format table  # default

# Clear old history
chinvex watch history clear --context Chinvex --older-than 90d
```

#### 2.3 Webhook Notifications

Fire HTTP POST when watch triggers.

**Configuration:**
```json
{
  "notifications": {
    "enabled": true,
    "webhook_url": "https://your-webhook.example.com/chinvex",
    "webhook_secret": "env:CHINVEX_WEBHOOK_SECRET",
    "notify_on": ["watch_hit"],
    "min_score_for_notify": 0.75,
    "retry_count": 2,
    "retry_delay_sec": 5
  }
}
```

**Payload:**
```json
{
  "event": "watch_hit",
  "timestamp": "2026-01-27T04:00:00Z",
  "context": "Chinvex",
  "run_id": "run_abc123",
  "watch": {
    "id": "p2_https",
    "query": "P2 HTTPS endpoint"
  },
  "hits": [
    {
      "chunk_id": "abc123",
      "score": 0.85,
      "source_uri": "...",
      "snippet": "...first 200 chars..."
    }
  ],
  "signature": "sha256=..."
}
```

**Signature verification:**
```python
import hmac
import hashlib

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = 'sha256=' + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

#### 2.4 Notification Destinations (Future)

P3 ships with webhook only. Future options:
- Discord webhook (just a different URL format)
- Slack incoming webhook
- Email (requires SMTP config)
- Desktop notification (requires native integration)

### Acceptance Tests

```bash
# Test 3.2.1: History appends
chinvex ingest --context Chinvex
chinvex ingest --context Chinvex  # second run
chinvex watch history --context Chinvex
# Expected: Entries from both runs

# Test 3.2.2: History filtering
chinvex watch history --context Chinvex --since 1h --id p2_https
# Expected: Only recent, matching entries

# Test 3.2.3: Webhook fires
# Configure webhook to requestbin.com or similar
# Trigger watch, verify POST received with correct payload

# Test 3.2.4: Webhook signature valid
# Verify signature in received payload matches expected
```

---

## 3. Cross-Context Search (P3.3)

### Purpose

Search across multiple contexts at once. Useful when you don't remember which project something was in.

### CLI Interface

```bash
# Search all contexts
chinvex search --all "that authentication bug"

# Search specific contexts
chinvex search --contexts Chinvex,Personal "OAuth flow"

# Exclude contexts
chinvex search --all --exclude Work "personal project"
```

### API Endpoint

Extend `/v1/search` and `/v1/evidence`:

```json
// POST /v1/search
{
  "contexts": ["Chinvex", "Personal"],  // or "all"
  "query": "authentication",
  "k": 10  // total results, not per-context
}
```

**Response:**
```json
{
  "query": "authentication",
  "contexts_searched": ["Chinvex", "Personal"],
  "results": [
    {
      "context": "Chinvex",
      "chunk_id": "abc123",
      "text": "...",
      "score": 0.89,
      ...
    },
    {
      "context": "Personal",
      "chunk_id": "def456",
      "text": "...",
      "score": 0.85,
      ...
    }
  ],
  "total_results": 10
}
```

### Implementation

```python
def search_multi_context(
    contexts: list[str] | Literal["all"],
    query: str,
    k: int = 10
) -> list[SearchResult]:
    """
    Search across multiple contexts, merge results by score.
    """
    if contexts == "all":
        contexts = list_all_contexts()
    
    # Gather results from each context
    all_results = []
    for ctx_name in contexts:
        ctx = load_context(ctx_name)
        results = search_context(ctx, query, k=k)  # Get k from each
        for r in results:
            r.context = ctx_name  # Tag with source context
        all_results.extend(results)
    
    # Sort by score descending, take top k
    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:k]
```

### Configuration

```json
{
  "cross_context": {
    "enabled": true,
    "max_contexts_per_query": 10,
    "default_contexts": ["Chinvex", "Personal"]  // for "all" shorthand
  }
}
```

### Gateway Security

Cross-context search respects allowlist:
- If `context_allowlist` is set, `"all"` means "all allowed contexts"
- Cannot search contexts not in allowlist

### Acceptance Tests

```bash
# Test 3.3.1: --all searches all contexts
chinvex search --all "test"
# Expected: Results from multiple contexts

# Test 3.3.2: Results tagged with context
# Verify each result shows context name

# Test 3.3.3: API multi-context
curl -X POST /v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"contexts": ["Chinvex", "Personal"], "query": "test"}'
# Expected: Mixed results with context tags

# Test 3.3.4: Allowlist respected
# Set allowlist to ["Chinvex"]
# Request contexts: ["Chinvex", "Personal"]
# Expected: Only Chinvex results (Personal silently excluded or error)
```

---

## 4. Archive Tier (P3.4)

### Purpose

Old content clutters search results. Archive tier moves stale content out of default search while keeping it accessible on request.

### Mechanics

```
┌─────────────────┐
│  Active Index   │  ← Default search
│  (recent docs)  │
└────────┬────────┘
         │ age > threshold
         ▼
┌─────────────────┐
│  Archived       │  ← Search with --include-archive
│  (old docs)     │
└─────────────────┘
```

### Implementation: Flag-Based

Add `archived` column to documents table (simpler than separate indexes):

```sql
ALTER TABLE documents ADD COLUMN archived INTEGER DEFAULT 0;
ALTER TABLE documents ADD COLUMN archived_at TEXT;  -- ISO8601
CREATE INDEX idx_documents_archived ON documents(archived);
```

**Search modification:**
```python
def search_context(ctx, query, k=10, include_archive=False):
    # ... existing search logic ...
    
    if not include_archive:
        # Filter out archived docs at the SQL level
        where_clause += " AND d.archived = 0"
    
    # ... rest of search ...
```

### Configuration

```json
{
  "archive": {
    "enabled": true,
    "age_threshold_days": 180,
    "auto_archive_on_ingest": true
  }
}
```

### CLI Commands

```bash
# Manual archive (mark docs older than threshold)
chinvex archive run --context Chinvex [--older-than 180d] [--dry-run]

# Search including archived
chinvex search --context Chinvex "old decision" --include-archive

# List archived documents
chinvex archive list --context Chinvex [--limit 50]

# Restore specific document
chinvex archive restore --context Chinvex --doc-id abc123

# Restore by query (interactive)
chinvex archive restore --context Chinvex --query "project X" --interactive

# Permanently delete archived (careful!)
chinvex archive purge --context Chinvex --older-than 365d [--confirm]
```

### API Extension

```json
// POST /v1/search
{
  "context": "Chinvex",
  "query": "old project",
  "include_archive": true
}

// POST /v1/evidence
{
  "context": "Chinvex",
  "query": "that decision from last year",
  "include_archive": true
}
```

### Auto-Archive on Ingest

If `auto_archive_on_ingest` is true, archive check runs after each ingest:

```python
def post_ingest_hook(ctx, result):
    # ... existing state generation ...
    
    if ctx.config.archive.enabled and ctx.config.archive.auto_archive_on_ingest:
        archive_old_documents(ctx, ctx.config.archive.age_threshold_days)
```

### Acceptance Tests

```bash
# Test 3.4.1: Auto-archive runs
# Ingest doc with old updated_at (200 days ago)
chinvex ingest --context Chinvex
chinvex archive list --context Chinvex
# Expected: Old doc in archived list

# Test 3.4.2: Archived excluded by default
chinvex search --context Chinvex "old content"
# Expected: Archived doc not in results

# Test 3.4.3: --include-archive finds it
chinvex search --context Chinvex "old content" --include-archive
# Expected: Archived doc in results (marked as archived)

# Test 3.4.4: Restore works
chinvex archive restore --context Chinvex --doc-id abc123
chinvex search --context Chinvex "old content"
# Expected: Doc now in normal results

# Test 3.4.5: Purge removes permanently
chinvex archive purge --context Chinvex --older-than 365d --confirm
# Expected: Very old docs gone from both active and archive
```

---

## 5. Gateway Improvements (P3.5)

### 5.1 Redis-Backed Rate Limiting

P2 uses in-memory rate limiting (resets on restart). P3 adds Redis for persistence.

```json
{
  "gateway": {
    "rate_limit": {
      "backend": "redis",  // or "memory"
      "redis_url": "redis://localhost:6379/0",
      "requests_per_minute": 60,
      "requests_per_hour": 500
    }
  }
}
```

**Fallback:** If Redis unavailable, fall back to in-memory with warning.

### 5.2 Metrics Endpoint

```
GET /metrics
```

Returns Prometheus-format metrics:

```
# HELP chinvex_requests_total Total requests by endpoint and status
# TYPE chinvex_requests_total counter
chinvex_requests_total{endpoint="/v1/evidence",status="200"} 1542
chinvex_requests_total{endpoint="/v1/evidence",status="401"} 23
chinvex_requests_total{endpoint="/v1/search",status="200"} 892

# HELP chinvex_request_duration_seconds Request latency
# TYPE chinvex_request_duration_seconds histogram
chinvex_request_duration_seconds_bucket{endpoint="/v1/evidence",le="0.1"} 1200
chinvex_request_duration_seconds_bucket{endpoint="/v1/evidence",le="0.5"} 1500
chinvex_request_duration_seconds_bucket{endpoint="/v1/evidence",le="1.0"} 1540

# HELP chinvex_grounded_ratio Ratio of grounded=true responses
# TYPE chinvex_grounded_ratio gauge
chinvex_grounded_ratio 0.73
```

**Configuration:**
```json
{
  "gateway": {
    "metrics_enabled": true,
    "metrics_auth_required": false  // Usually internal only
  }
}
```

### 5.3 Request Logging Improvements

Add request ID header for tracing:

```
X-Request-ID: abc123-def456
```

If client provides `X-Request-ID`, use it. Otherwise generate UUID.

Include in all responses and audit log.

### 5.4 Health Check Improvements

```json
// GET /health?detailed=true
{
  "status": "ok",
  "version": "0.3.0",
  "uptime_seconds": 86400,
  "contexts": {
    "Chinvex": {"status": "ok", "docs": 1542, "chunks": 8923},
    "Personal": {"status": "ok", "docs": 234, "chunks": 1456}
  },
  "ollama": {"status": "ok", "model": "mxbai-embed-large"},
  "rate_limit_backend": "redis"
}
```

### Acceptance Tests

```bash
# Test 3.5.1: Redis rate limiting persists
# Hit rate limit, restart gateway, verify still limited

# Test 3.5.2: Metrics endpoint
curl http://localhost:7778/metrics
# Expected: Prometheus-format metrics

# Test 3.5.3: Request ID propagation
curl -H "X-Request-ID: test123" .../v1/search
# Expected: Response includes X-Request-ID: test123

# Test 3.5.4: Detailed health check
curl .../health?detailed=true
# Expected: Per-context status, Ollama status
```

---

## 6. Implementation Order

### Phase 1: Chunking (P3.1) — 2-3 days
1. Overlap implementation
2. Semantic boundary detection
3. Python code-aware splitting
4. JS/TS code-aware splitting
5. Markdown-aware splitting
6. Language detection
7. Bump chunker_version
8. Re-index test

### Phase 2: Watch History (P3.2) — 1-2 days
9. watch_history.jsonl append on trigger
10. `chinvex watch history` CLI
11. History filtering (--since, --id)
12. Webhook notification implementation
13. Webhook signature generation
14. Retry logic

### Phase 3: Cross-Context Search (P3.3) — 1-2 days
15. Multi-context search logic
16. Result merging by score
17. CLI --all / --contexts flags
18. API multi-context support
19. Allowlist enforcement

### Phase 4: Archive Tier (P3.4) — 2 days
20. Add archived column to schema
21. Archive run command
22. Search filtering
23. Archive list command
24. Restore command
25. Auto-archive on ingest
26. Purge command

### Phase 5: Gateway Improvements (P3.5) — 1-2 days
27. Redis rate limiting
28. Metrics endpoint
29. Request ID handling
30. Detailed health check

---

## 7. Dependencies

### New Packages

```
redis>=5.0.0  # For rate limiting (optional)
prometheus-client>=0.19.0  # For metrics (optional)
```

### Existing (No Changes)

- fastapi
- uvicorn
- chromadb
- sqlite3

---

## 8. Configuration Reference

### Complete P3 Config

```json
{
  "chunking": {
    "max_chars": 3000,
    "overlap_chars": 300,
    "semantic_boundaries": true,
    "code_aware": true,
    "skip_files_larger_than_mb": 5
  },
  "notifications": {
    "enabled": false,
    "webhook_url": "",
    "webhook_secret": "env:CHINVEX_WEBHOOK_SECRET",
    "notify_on": ["watch_hit"],
    "min_score_for_notify": 0.75,
    "retry_count": 2,
    "retry_delay_sec": 5
  },
  "cross_context": {
    "enabled": true,
    "max_contexts_per_query": 10
  },
  "archive": {
    "enabled": true,
    "age_threshold_days": 180,
    "auto_archive_on_ingest": true
  },
  "gateway": {
    "rate_limit": {
      "backend": "memory",
      "redis_url": null,
      "requests_per_minute": 60,
      "requests_per_hour": 500
    },
    "metrics_enabled": true,
    "metrics_auth_required": false
  }
}
```

---

## 9. Acceptance Test Summary

| ID | Test | Pass Criteria |
|----|------|---------------|
| 3.1.1 | Function not split | Logical unit in single chunk |
| 3.1.2 | Markdown splits at headers | Chunks start near ## |
| 3.1.3 | Overlap present | ~300 chars shared between chunks |
| 3.1.4 | Large files handled | No crash on big files |
| 3.2.1 | History appends | Multiple runs create multiple entries |
| 3.2.2 | History filtering | --since and --id work |
| 3.2.3 | Webhook fires | POST received at webhook URL |
| 3.2.4 | Webhook signature | Signature validates correctly |
| 3.3.1 | Multi-context search | Results from multiple contexts |
| 3.3.2 | Results tagged | Context name on each result |
| 3.3.3 | API multi-context | Mixed results via API |
| 3.3.4 | Allowlist respected | Cannot search unauthorized contexts |
| 3.4.1 | Auto-archive | Old docs flagged automatically |
| 3.4.2 | Archived excluded | Not in default search |
| 3.4.3 | Include-archive | Old docs found with flag |
| 3.4.4 | Restore | Doc back in active search |
| 3.4.5 | Purge | Very old docs permanently removed |
| 3.5.1 | Redis rate limit | Persists across restart |
| 3.5.2 | Metrics endpoint | Prometheus format returned |
| 3.5.3 | Request ID | Propagated through request/response |
| 3.5.4 | Detailed health | Per-context and Ollama status |

---

*End of P3 spec.*
