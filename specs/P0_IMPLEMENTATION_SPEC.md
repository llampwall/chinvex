# Chinvex P0 Implementation Spec

**Version:** 1.0  
**Date:** 2026-01-26  
**Status:** Ready for implementation

---

## Overview

Chinvex is a hybrid retrieval engine that provides grounded, citation-backed answers from personal knowledge sources (repos, chats, Codex sessions, notes). This spec defines the exact contracts for P0 implementation.

### P0 Scope

1. Context registry + profiles (`--context Chinvex` just works)
2. Codex sessions ingestion (via app-server, with JSONL fallback)
3. Grounded mode with citations as hard gate
4. `chinvex_answer` MCP tool (search → batch get_chunk → evidence_pack)
5. Incremental ingest (fingerprints, skip unchanged files)

### Non-Goals (P1+)

- STATE.md condensation / brain-stem file
- Time decay / aging
- Hosted HTTPS endpoint for ChatGPT Actions
- Proactive "tap on shoulder" behaviors

---

## 1. Context Registry

### Directory Structure

```
P:\ai_memory\
├── contexts\
│   └── <ContextName>\
│       └── context.json
└── indexes\
    └── <ContextName>\
        ├── hybrid.db      # SQLite: FTS5 + metadata + fingerprints
        └── chroma\        # Vector store
```

### `context.json` Schema (v1)

```json
{
  "schema_version": 1,
  "name": "Chinvex",
  "aliases": ["chindex", "chinvex-engine"],
  "includes": {
    "repos": ["C:\\Code\\chinvex", "C:\\Code\\godex"],
    "chat_roots": ["P:\\ai_memory\\projects\\Chinvex\\chats"],
    "codex_session_roots": ["C:\\Users\\Jordan\\.codex\\sessions"],
    "note_roots": ["P:\\ai_memory\\projects\\Chinvex\\notes"]
  },
  "index": {
    "sqlite_path": "P:\\ai_memory\\indexes\\Chinvex\\hybrid.db",
    "chroma_dir": "P:\\ai_memory\\indexes\\Chinvex\\chroma"
  },
  "weights": {
    "repo": 1.0,
    "chat": 0.8,
    "codex_session": 0.9,
    "note": 0.7
  },
  "created_at": "2026-01-26T00:00:00Z",
  "updated_at": "2026-01-26T00:00:00Z"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | int | yes | Always `1` for this spec |
| `name` | string | yes | Canonical context name (case-sensitive) |
| `aliases` | string[] | no | Alternative names for CLI convenience |
| `includes.repos` | string[] | no | Paths to git repos to index |
| `includes.chat_roots` | string[] | no | Directories containing chat exports |
| `includes.codex_session_roots` | string[] | no | Directories for Codex session discovery |
| `includes.note_roots` | string[] | no | Directories containing markdown notes |
| `index.sqlite_path` | string | yes | Path to hybrid.db |
| `index.chroma_dir` | string | yes | Path to Chroma vector store |
| `weights` | object | yes | Source-type weights for ranking (see §4) |
| `created_at` | ISO8601 | yes | Creation timestamp |
| `updated_at` | ISO8601 | yes | Last modification timestamp |

### CLI Commands

#### `chinvex context create <Name>`

1. Validate `<Name>` is not empty, contains no path separators
2. Check `P:\ai_memory\contexts\<Name>\` does not exist
3. Create directory structure:
   - `P:\ai_memory\contexts\<Name>\context.json` (template with empty includes)
   - `P:\ai_memory\indexes\<Name>\` (empty dir)
4. Initialize `hybrid.db` with schema (see §5)
5. Initialize Chroma collection
6. Print success message with path

**Exit codes:** 0 = success, 1 = already exists, 2 = invalid name

#### `chinvex context list`

1. Enumerate `P:\ai_memory\contexts\*\context.json`
2. Parse each, extract `{name, aliases, updated_at}`
3. Print table sorted by `updated_at` desc

**Output format:**
```
NAME        ALIASES                 UPDATED
Chinvex     chindex, chinvex-engine 2026-01-26T12:00:00Z
Personal    -                       2026-01-25T08:30:00Z
```

#### Context Resolution

When `--context <X>` is passed to any command:

1. Check `P:\ai_memory\contexts\<X>\context.json` exists → use it
2. Else, scan all `context.json` files for `aliases` containing `<X>` → use first match
3. Else, **error**: `Unknown context: <X>. Use 'chinvex context list' to see available contexts.`

**CRITICAL:** Unknown contexts MUST error. Never auto-create.

---

## 2. Codex Sessions Ingestion

### Primary Source: Codex App-Server

Use the official Codex app-server API (https://developers.openai.com/codex/app-server) as the canonical source.

**Endpoints used:**
- `thread/list` — enumerate threads
- `thread/resume` — get full thread content

**Adapter name:** `cx_appserver`

**⚠️ Schema Capture Requirement**

The app-server docs may be vague on exact response shapes. The adapter MUST:

1. **Log raw responses during development:**
   ```
   debug/appserver_samples/
   ├── thread_list_2026-01-26.json
   └── thread_resume_abc123.json
   ```

2. **Define strict schemas from actual responses (Zod/Pydantic):**
   ```python
   # adapters/cx_appserver/schemas.py
   from pydantic import BaseModel
   
   class AppServerThread(BaseModel):
       id: str
       title: str | None
       created_at: str
       updated_at: str
       # ... fields discovered from actual responses
   
   class AppServerMessage(BaseModel):
       # ... actual shape from samples
   ```

3. **Validate before normalizing:**
   ```python
   def fetch_thread(thread_id: str) -> ConversationDoc:
       raw = api.get(f"/thread/resume/{thread_id}")
       # Validate against discovered schema
       parsed = AppServerThread.model_validate(raw)
       # Then normalize to ConversationDoc
       return normalize_to_conversation_doc(parsed)
   ```

**Do NOT guess fields. Capture → Schema → Normalize.**

### Secondary Source: Local JSONL (Fallback)

Location: `~/.codex/sessions/<year>/<month>/<day>/rollout-*.jsonl`

Use only for:
- Offline/airgapped operation
- Backfilling historical logs not available via app-server
- Capturing debug telemetry not exposed by app-server

**Adapter name:** `cx_sessions_jsonl`

### Normalized Internal Schema: `ConversationDoc`

Both adapters produce this shape:

```json
{
  "doc_type": "conversation",
  "source": "cx_appserver|cx_sessions_jsonl",
  "thread_id": "unique-thread-identifier",
  "title": "Optional title or first user message truncated",
  "created_at": "2026-01-26T10:00:00Z",
  "updated_at": "2026-01-26T10:30:00Z",
  "turns": [
    {
      "turn_id": "unique-turn-id",
      "ts": "2026-01-26T10:00:00Z",
      "role": "user|assistant|tool|system",
      "text": "Message content...",
      "tool": {
        "name": "bash",
        "input": {},
        "output": {}
      },
      "attachments": [],
      "meta": {
        "model": "o4-mini",
        "tokens": { "in": 150, "out": 420 }
      }
    }
  ],
  "links": {
    "workspace_id": "optional",
    "repo_paths": ["C:\\Code\\project"]
  }
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `doc_type` | string | yes | Always `"conversation"` |
| `source` | string | yes | `"cx_appserver"` or `"cx_sessions_jsonl"` |
| `thread_id` | string | yes | Unique identifier for deduplication |
| `title` | string | no | Display title |
| `created_at` | ISO8601 | yes | First message timestamp |
| `updated_at` | ISO8601 | yes | Last message timestamp |
| `turns` | array | yes | Ordered list of turns |
| `turns[].turn_id` | string | yes | Unique within thread |
| `turns[].ts` | ISO8601 | yes | Turn timestamp |
| `turns[].role` | string | yes | One of: `user`, `assistant`, `tool`, `system` |
| `turns[].text` | string | no | Message text (may be empty for tool-only turns) |
| `turns[].tool` | object | no | Present if role=tool or assistant used a tool |
| `turns[].attachments` | array | no | File references |
| `turns[].meta` | object | no | Model info, token counts |
| `links.workspace_id` | string | no | Codex workspace ID if available |
| `links.repo_paths` | string[] | no | Associated repository paths |

### Chunking Strategy for Conversations

Conversations are chunked by **logical turn groups**, not arbitrary token windows:

1. Group consecutive turns into chunks of ~1500 tokens max
2. Never split a single turn across chunks
3. Include `[Turn N of M]` markers in chunk text
4. Preserve thread_id and turn_id range in chunk metadata

---

## 3. Grounded Mode

### Purpose

Grounded mode ensures answers are directly supported by retrieved chunks. If retrieval doesn't support an answer, the system refuses to invent.

### Behavior Contract

**When `grounded=true` (strict mode):**

1. Retrieve chunks using hybrid search
2. For each claim in the answer, identify supporting chunk(s)
3. If ANY claim cannot be traced to a chunk → fail grounding

**Success criteria:**
- `grounded: true` in response
- `citations.length > 0`
- Every substantive claim has a citation

**Failure criteria:**
- `grounded: false` in response
- `citations: []`
- `errors: [{ "code": "GROUNDING_FAILED", ... }]`
- `answer: "Not stated in retrieved sources."`
- `evidence_pack.chunks` still populated (shows what WAS retrieved)

### Pragmatic Citation Rule (P0)

"Every substantive claim" is hard to enforce deterministically. For P0, use this simpler rule:

**Require ≥1 citation per paragraph of output.**

- If a paragraph has no citation → grounding fails
- Exception: Paragraphs that are explicitly uncertainty/questions ("Not stated...", "I'm not sure...")
- The per-claim ideal is P1 (requires better claim extraction)

```python
def validate_grounding(answer: str, citations: list) -> bool:
    paragraphs = [p for p in answer.split('\n\n') if p.strip()]
    uncertainty_markers = ["not stated", "not found", "i'm not sure", "unclear"]
    
    for para in paragraphs:
        para_lower = para.lower()
        if any(marker in para_lower for marker in uncertainty_markers):
            continue  # uncertainty paragraphs don't need citations
        if not any(cite['chunk_id'] in para for cite in citations):
            return False  # paragraph without citation = fail
    return len(citations) > 0
```

### Prompt Enforcement (for LLM layer)

When calling an LLM with retrieved chunks in grounded mode, include this instruction:

```
You MUST cite chunk IDs for every factual claim. Use format: [chunk:abc123]

If no retrieved chunk supports your answer, respond ONLY with:
"Not stated in retrieved sources."

Do NOT invent, extrapolate, or use external knowledge. Only use the provided chunks.
```

---

## 4. Ranking and Source Weights

### Score Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FTS5      │     │   Vector    │     │   Blend     │
│   Search    │────▶│   Search    │────▶│   Scores    │
│ (BM25-ish)  │     │ (cosine)    │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Apply     │
                                        │   Source    │
                                        │   Weights   │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Final     │
                                        │   Ranking   │
                                        └─────────────┘
```

### Score Normalization (CRITICAL)

FTS5 BM25 scores are unbounded. Vector cosine similarities are [-1, 1]. You **must** normalize before blending or weights are meaningless.

**Strategy: Per-query min-max normalization within candidate set**

```python
def normalize_scores(scores: list[float]) -> list[float]:
    """
    Min-max normalize within the candidate set for this query.
    Returns values in [0, 1].
    """
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)  # all equal = all max
    return [(s - min_s) / (max_s - min_s) for s in scores]
```

**Normalization flow:**
1. Run FTS5 search → get raw BM25 scores for top N candidates
2. Run vector search → get raw cosine similarities for top N candidates
3. Union candidate sets (some chunks may appear in only one)
4. Normalize FTS scores within candidate set (missing = 0.0)
5. Normalize vector scores within candidate set (missing = 0.0)
6. Blend normalized scores

### Blended Score (Pure Relevance)

```python
def blend_scores(fts_norm: float, vec_norm: float) -> float:
    """
    Combine NORMALIZED FTS and vector scores.
    FTS weight slightly higher to preserve exact match priority.
    """
    FTS_WEIGHT = 0.6
    VEC_WEIGHT = 0.4
    return (fts_norm * FTS_WEIGHT) + (vec_norm * VEC_WEIGHT)
```

**CRITICAL:** Blended score = pure relevance. No source weights here.

### Rank Score (Policy Applied)

```python
def rank_score(blended: float, source_type: str, weights: dict) -> float:
    """
    Apply source-type weight as a post-retrieval prior.
    Multiplication is fine for v1. Log-additive is smoother but not required.
    """
    weight = weights.get(source_type, 0.5)  # default if unknown
    return blended * weight
```

### Tie-Breaking

When `rank_score` is equal:
1. Prefer more recent (`updated_at` desc)
2. Prefer `source_type` priority: `repo > codex_session > chat > note`
3. Stable sort by `chunk_id` (deterministic)

### Default Weights

```json
{
  "repo": 1.0,
  "chat": 0.8,
  "codex_session": 0.9,
  "note": 0.7
}
```

Rationale:
- `repo`: Source of truth, highest weight
- `codex_session`: High signal (debugging history), near-repo weight
- `chat`: Valuable but may contain exploratory/wrong turns
- `note`: Often summaries, may be stale

---

## 5. SQLite Schema

All tables live in `hybrid.db` for the context.

### `documents` Table

```sql
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,           -- repo|chat|codex_session|note
    source_uri TEXT NOT NULL,            -- file path or thread ID
    title TEXT,
    created_at TEXT,                     -- ISO8601
    updated_at TEXT,                     -- ISO8601
    metadata_json TEXT                   -- additional fields as JSON
);

CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at);
```

### `chunks` Table

```sql
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    chunk_index INTEGER NOT NULL,        -- position within document
    text TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    line_start INTEGER,
    line_end INTEGER,
    metadata_json TEXT,
    UNIQUE(doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
```

### Range Field Guarantees by Source Type

| Source Type | `char_start/end` | `line_start/end` |
|-------------|------------------|------------------|
| `repo` (code) | ✅ Guaranteed | ✅ Guaranteed |
| `repo` (non-code) | ✅ Guaranteed | ⚠️ Best-effort |
| `chat` (markdown) | ✅ Guaranteed | ⚠️ Best-effort |
| `codex_session` | ✅ Guaranteed | ❌ Not provided |
| `note` (markdown) | ✅ Guaranteed | ⚠️ Best-effort |

**Client code must:**
- Always check for null/undefined before using line ranges
- Prefer `char_start/end` for universal positioning
- Use line ranges only for display/IDE integration when available

### `chunks_fts` Virtual Table (FTS5)

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id,
    text,
    content='chunks',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES (new.rowid, new.chunk_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text) VALUES('delete', old.rowid, old.chunk_id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text) VALUES('delete', old.rowid, old.chunk_id, old.text);
    INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES (new.rowid, new.chunk_id, new.text);
END;
```

**⚠️ FTS5 Trigger Fragility Warning**

The `content_rowid='rowid'` coupling is fragile if you:
- Use `INSERT OR REPLACE` (rowid changes on replace)
- Do bulk upserts that delete+reinsert
- VACUUM or restore from backup

**Safer pattern for upserts:** Always delete-then-insert, never `INSERT OR REPLACE`:

```python
def upsert_chunk(chunk: Chunk):
    # Delete existing (triggers FTS cleanup)
    db.execute("DELETE FROM chunks WHERE chunk_id = ?", [chunk.chunk_id])
    # Insert fresh (triggers FTS insert)
    db.execute("INSERT INTO chunks (...) VALUES (...)", [...])
```

**P1 consideration:** Switch to chunk_id-keyed FTS (deterministic rowid from hash) for bulletproof consistency.

### `source_fingerprints` Table

```sql
CREATE TABLE IF NOT EXISTS source_fingerprints (
    source_uri TEXT NOT NULL,
    context_name TEXT NOT NULL,
    source_type TEXT NOT NULL,           -- repo|chat|codex_session|note
    doc_id TEXT NOT NULL,
    -- File-based sources (repo, chat, note):
    size_bytes INTEGER,
    mtime_unix INTEGER,
    content_sha256 TEXT,
    -- Thread-based sources (codex_session via app-server):
    thread_updated_at TEXT,              -- ISO8601 from app-server
    last_turn_id TEXT,                   -- last known turn for delta detection
    -- Common:
    parser_version TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    embedded_model TEXT,                 -- e.g. mxbai-embed-large
    last_ingested_at_unix INTEGER NOT NULL,
    last_status TEXT NOT NULL,           -- ok|skipped|error
    last_error TEXT,
    PRIMARY KEY (source_uri, context_name)
);

CREATE INDEX IF NOT EXISTS idx_fingerprints_type ON source_fingerprints(source_type);
CREATE INDEX IF NOT EXISTS idx_fingerprints_status ON source_fingerprints(last_status);
```

### Fingerprint Strategy by Source Type

| Source Type | `source_uri` | Change Detection |
|-------------|--------------|------------------|
| `repo` | File path | `mtime_unix` + `size_bytes` (+ optional `content_sha256`) |
| `chat` | File path | `mtime_unix` + `size_bytes` |
| `note` | File path | `mtime_unix` + `size_bytes` |
| `codex_session` (app-server) | `thread_id` | `thread_updated_at` or `last_turn_id` |
| `codex_session` (JSONL) | File path | `mtime_unix` + `size_bytes` |

**Thread fingerprint check:**
```python
def should_ingest_thread(thread_id: str, context: str, current_updated_at: str) -> tuple[bool, str]:
    fp = get_fingerprint(thread_id, context)
    if fp is None:
        return (True, "new_thread")
    if fp.thread_updated_at != current_updated_at:
        return (True, "thread_updated")
    # ... version checks same as files
    return (False, "unchanged")
```

### `ingest_runs` Table

```sql
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id TEXT PRIMARY KEY,
    context_name TEXT NOT NULL,
    started_at TEXT NOT NULL,            -- ISO8601
    finished_at TEXT,
    status TEXT NOT NULL,                -- running|completed|failed
    stats_json TEXT                      -- {files_scanned, files_indexed, files_skipped, errors}
);
```

---

## 6. `chinvex_answer` MCP Tool

### Tool Definition

```json
{
  "name": "chinvex_answer",
  "description": "Search personal knowledge and return a grounded answer with citations",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language question"
      },
      "context": {
        "type": "string",
        "description": "Context name (e.g., 'Chinvex')"
      },
      "k": {
        "type": "integer",
        "default": 8,
        "description": "Number of chunks to retrieve"
      },
      "grounded": {
        "type": "boolean",
        "default": true,
        "description": "Require citations; refuse if unsupported"
      },
      "source_types": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Filter to specific source types (optional)"
      }
    },
    "required": ["query", "context"]
  }
}
```

### Response Schema (v1)

```json
{
  "schema_version": 1,
  "context": "Chinvex",
  "query": "How does hybrid retrieval work?",
  "grounded": true,
  "answer": "Hybrid retrieval combines FTS5 lexical search with vector similarity...",
  "citations": [
    {
      "chunk_id": "b061c5a908d5",
      "source_uri": "C:\\Code\\chinvex\\src\\retrieval.ts",
      "range": {
        "line_start": 42,
        "line_end": 67
      }
    },
    {
      "chunk_id": "f3a2b1c4d5e6",
      "source_uri": "P:\\ai_memory\\projects\\Chinvex\\chats\\2026-01-25_design.md",
      "range": {
        "char_start": 1500,
        "char_end": 2200
      }
    }
  ],
  "evidence_pack": {
    "chunks": [
      {
        "chunk_id": "b061c5a908d5",
        "text": "export function hybridSearch(query: string, k: number)...",
        "metadata": {
          "doc_id": "abc123",
          "source_type": "repo",
          "source_uri": "C:\\Code\\chinvex\\src\\retrieval.ts",
          "char_start": 1234,
          "char_end": 2345,
          "line_start": 42,
          "line_end": 67,
          "timestamps": {
            "created_at": "2026-01-20T10:00:00Z",
            "updated_at": "2026-01-25T14:30:00Z"
          }
        },
        "scores": {
          "fts": 12.3,
          "vector": 0.81,
          "blended": 0.88,
          "rank": 0.88
        }
      }
    ]
  },
  "retrieval": {
    "k": 8,
    "filters": {
      "source_types": ["repo", "chat", "codex_session"]
    },
    "weights_used": {
      "repo": 1.0,
      "chat": 0.8,
      "codex_session": 0.9
    }
  },
  "errors": []
}
```

### Grounding Failure Response

```json
{
  "schema_version": 1,
  "context": "Chinvex",
  "query": "What is the airspeed velocity of an unladen swallow?",
  "grounded": false,
  "answer": "Not stated in retrieved sources.",
  "citations": [],
  "evidence_pack": {
    "chunks": [
      // Still populated with what WAS retrieved (may be empty)
    ]
  },
  "retrieval": {
    "k": 8,
    "filters": {}
  },
  "errors": [
    {
      "code": "GROUNDING_FAILED",
      "detail": "No retrieved chunk supports a direct answer to this query."
    }
  ]
}
```

### Error Codes

| Code | Meaning |
|------|---------|
| `GROUNDING_FAILED` | Retrieved chunks don't support an answer |
| `CONTEXT_NOT_FOUND` | Unknown context name |
| `RETRIEVAL_ERROR` | Database or embedding error |
| `EMPTY_QUERY` | Query was empty or whitespace |

---

## 7. Incremental Ingest Logic

### Fingerprint Check

```python
def should_ingest(source_uri: str, context: str, current_stat: FileStat) -> tuple[bool, str]:
    """
    Returns (should_ingest, reason).
    """
    fp = get_fingerprint(source_uri, context)
    
    if fp is None:
        return (True, "new_file")
    
    if fp.last_status == "error":
        return (True, "retry_after_error")
    
    if current_stat.mtime_unix != fp.mtime_unix:
        return (True, "mtime_changed")
    
    if current_stat.size_bytes != fp.size_bytes:
        return (True, "size_changed")
    
    if fp.parser_version != CURRENT_PARSER_VERSION:
        return (True, "parser_upgraded")
    
    if fp.chunker_version != CURRENT_CHUNKER_VERSION:
        return (True, "chunker_upgraded")
    
    if fp.embedded_model != CURRENT_EMBED_MODEL:
        return (True, "embed_model_changed")
    
    return (False, "unchanged")
```

### Ingest Flow

```
1. Start ingest run, record run_id
2. For each source in context.includes:
   a. Enumerate files
   b. For each file:
      - Check fingerprint
      - If skip: log reason, continue
      - If ingest:
        - Parse → chunks
        - Embed chunks
        - Upsert to SQLite + Chroma
        - Update fingerprint (status=ok or error)
3. Finish run, record stats
```

### Versioning

Track these in fingerprints:
- `parser_version`: e.g., `"chatgpt_export_v2"`
- `chunker_version`: e.g., `"sliding_window_v1"`
- `embedded_model`: e.g., `"mxbai-embed-large"`

When any changes, affected files re-ingest.

---

## 8. CLI Reference

### Global Flags

| Flag | Description |
|------|-------------|
| `--context <name>` | Required for most commands. Context to operate on. |
| `--verbose` | Enable debug logging |
| `--json` | Output as JSON (for scripting) |

### Commands

#### `chinvex context create <name>`
Create a new context with empty configuration.

#### `chinvex context list`
List all contexts with name, aliases, and last updated time.

#### `chinvex context show <name>`
Print full `context.json` for a context.

#### `chinvex ingest --context <name> [--full]`
Run incremental ingest. `--full` forces re-ingest of all files.

#### `chinvex search --context <name> <query> [--k N] [--source-types T1,T2]`
Run hybrid search, return ranked chunks.

#### `chinvex answer --context <name> <query> [--grounded] [--k N]`
Run search + generate grounded answer with citations.

#### `chinvex chunk get <chunk_id> --context <name>`
Retrieve a single chunk by ID.

#### `chinvex stats --context <name>`
Show index statistics (doc counts, chunk counts, last ingest run).

---

## 9. Acceptance Tests (Golden Queries)

Define these early. Each should pass before P0 is "done."

### Test 1: Context Creation
```bash
chinvex context create TestContext
# Expected: Creates dirs, initializes DB, exits 0
chinvex context create TestContext
# Expected: "Already exists" error, exits 1
```

### Test 2: Unknown Context Errors
```bash
chinvex search --context NonExistent "test query"
# Expected: Error message, exits non-zero
# MUST NOT auto-create
```

### Test 3: Incremental Ingest Skips Unchanged
```bash
chinvex ingest --context Test
# (index a file)
chinvex ingest --context Test
# Expected: File shows "skipped: unchanged" in verbose output
```

### Test 4: Grounding Failure
```bash
chinvex answer --context Test "What is the meaning of life?" --grounded
# Expected: grounded=false, answer="Not stated in retrieved sources."
```

### Test 5: Successful Grounded Answer
```bash
# (after indexing Chinvex repo)
chinvex answer --context Chinvex "How does hybrid retrieval combine scores?"
# Expected: grounded=true, citations present, answer references code
```

### Test 6: Source Weight Effect
```bash
# Query that matches both repo and chat chunks equally
# Expected: repo chunk ranks higher due to weight 1.0 vs 0.8
```

---

## 10. Implementation Order

### Phase 1: Foundation
1. SQLite schema setup (all tables)
2. Context create/list/show commands
3. Context resolution logic (name + alias lookup)

### Phase 2: Ingest
4. Fingerprint check logic
5. Repo file ingestion (code files)
6. Chat export ingestion (markdown from ChatGPT userscript)
7. Codex app-server adapter
8. Codex JSONL adapter (fallback)

### Phase 3: Retrieval
9. FTS5 search
10. Chroma vector search
11. Score blending
12. Source weight ranking
13. `chinvex search` CLI command

### Phase 4: Answer
14. `chinvex_answer` MCP tool
15. Grounded mode enforcement
16. Citation extraction
17. Evidence pack assembly

### Phase 5: Polish
18. Acceptance tests
19. Error handling audit
20. Performance testing with real data

---

## Appendix A: File Type Handling

| Source Type | File Patterns | Parser |
|-------------|---------------|--------|
| repo | `*.ts`, `*.py`, `*.md`, `*.json`, etc. | Language-aware chunker |
| chat | `*.md` (ChatGPT export format) | Markdown turn parser |
| codex_session | app-server API / `*.jsonl` | ConversationDoc normalizer |
| note | `*.md` | Markdown section chunker |

### Excluded Patterns (Repos)
- `node_modules/`
- `.git/`
- `dist/`, `build/`
- Binary files
- Files > 1MB

---

## Appendix B: Embedding Model

**Default:** `mxbai-embed-large` via Ollama

**Dimension:** 1024

**Normalization:** L2 normalize before storage

**Distance metric:** Cosine similarity (Chroma default)

---

## Appendix C: Chunk ID Generation

```python
import hashlib

def generate_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """
    Deterministic chunk ID from content.
    """
    content = f"{doc_id}:{chunk_index}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]
```

12 hex chars = 48 bits = sufficient uniqueness for millions of chunks.

---

*End of spec.*
