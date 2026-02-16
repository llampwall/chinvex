<!-- DO: Append new entries to current month. Rewrite Recent rollup. -->
<!-- DON'T: Edit or delete old entries. Don't log trivial changes. -->

# Decisions

## Recent (last 30 days)
- Implemented proper connection management for ChromaDB and SQLite (fixes Windows file lock errors)
- Fixed OpenAI as default embedding provider; search reads provider from meta.json (prevents dimension mismatch)
- Added automatic context.json backup system (30 backups per context, auto-prune)
- Created using-chinvex skill for Claude Code and Codex with full CLI workflow docs
- Documented using-chinvex skill in README with Skills for AI Agents section
- Add `chinvex context sync-metadata-from-strap` command to sync registry.json → context.json
- Implement .chinvex-status.json files with PID tracking for dashboard integration
- Exclude status files from sync daemon to prevent infinite loop
- Complete P5.3 eval suite with golden queries, metrics, and CI gate
- Complete P5.4 reranker with Cohere, Jina, and local cross-encoder providers

## 2026-02

### 2026-02-16 — Implemented proper connection management for ChromaDB and SQLite

- **Why:** Gateway restarts and file deletions were failing with Windows PermissionError [WinError 32] due to lingering database connections. ChromaDB's internal SQLite connections were not being closed, causing file locks.
- **Decision:** Use ChromaDB's internal `client._system.stop()` method to properly shut down connections. Add `VectorStore.close()` method and context manager support. Add gateway shutdown handler.
- **Impact:**
  - Added `VectorStore.close()` method that calls `client._system.stop()` to release SQLite connections
  - Added context manager support (`__enter__`, `__exit__`) to VectorStore for automatic cleanup
  - Added gateway shutdown handler (`@app.on_event("shutdown")`) that calls `Storage.force_close_global_connection()`
  - Updated gateway warmup and healthz endpoint to close VectorStore after use
  - Created comprehensive tests in `test_vector_store_cleanup.py` - all pass on Windows
  - Documented in `docs/CONNECTION_MANAGEMENT.md`
- **Alternatives Considered:** Simply setting `client = None` doesn't work because Python's GC is non-deterministic and ChromaDB's background threads keep references alive
- **Evidence:** Test failures before fix showed `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process` on `chroma.sqlite3`. After fix, all tests pass and temp directories clean up successfully.

### 2026-02-10 — Added using-chinvex skill for Claude Code and Codex

- **Why:** Agents needed comprehensive reference for all chinvex CLI workflows, gotchas, and best practices
- **Impact:** New skill at `skills/using-chinvex/SKILL.md` symlinked to `.claude/skills` and `.codex/skills`. Covers search, ingest, context management, sync daemon, provider switching, depth changes
- **Evidence:** 81aecf6, 3558ef9

### 2026-02-09 — Added automatic backup system for context.json files

- **Why:** Prevent data loss from accidental edits or schema migrations to context.json
- **Impact:** Automatic backups before every context.json write to `P:\ai_memory\backups\<name>\`. Keeps 30 most recent per context. Configurable via `CHINVEX_BACKUPS_ROOT`. Integrated into 9 write locations
- **Evidence:** 735ee72

### 2026-02-07 — Fixed OpenAI as default embedding provider in search

- **Symptom:** Dimension mismatch errors when querying contexts ingested with OpenAI
- **Root cause:** `hybrid_search_from_context` hardcoded Ollama embedder (1024 dims) for query generation, but contexts were ingested with OpenAI (1536 dims)
- **Fix:** Search now reads embedding provider from meta.json. Removed Ollama fallbacks. Errors if meta.json missing
- **Prevention:** Never hardcode embedding provider; always read from meta.json
- **Evidence:** bc68ab7

### 2026-02-05 — Added chinvex context sync-metadata-from-strap command

- **Why:** Strap manages repo metadata in registry.json, chinvex needs to sync that to context.json without triggering unnecessary reingests
- **Impact:** New command syncs status, tags, and chinvex_depth from strap registry to chinvex context. Enables metadata-only updates (fast) vs. depth changes that require rebuild-index (slow)
- **Trade-off:** Separate command vs. making ingest handle it - chose clean separation of concerns
- **Evidence:** d4681d1

### 2026-02-05 — Integrated .chinvex-status.json files for dashboard visibility

- **Why:** Allmind dashboard needs real-time visibility into which repos are being ingested
- **Impact:** Ingest process writes status files (ingesting/idle/error) with PID tracking. Dashboard can detect stale processes (PID no longer running) and show accurate status
- **Trade-off:** File-based status vs. database - chose files for simplicity and cross-process visibility
- **Evidence:** d4681d1

### 2026-02-05 — Excluded .chinvex-status.json from sync daemon file watching

- **Why:** Status files written during ingest would trigger sync daemon, creating infinite loop
- **Impact:** Added `**/.chinvex-status.json` to EXCLUDE_PATTERNS in patterns.py
- **Prevention:** Always exclude files that are written as a side effect of ingestion
- **Evidence:** d4681d1

### 2026-02-05 — Documented sync daemon behavior and status integration in README

- **Why:** Complex behavior (debouncing, force caps, status tracking) needs clear documentation
- **Impact:** README now explains how sync daemon works, when it triggers ingests, and how status files integrate with dashboard
- **Evidence:** d4681d1

### 2026-02-05 — Fixed logs directory ingestion causing index bloat

- **Symptom:** Repos like streamside with `logs/` directories containing large JSON files were being ingested, causing performance issues and index bloat
- **Root cause:** `.json` files are in `ALLOWED_EXTS`, but `logs` directories weren't in `SKIP_DIRS`
- **Fix:** Added "logs" and "log" to `SKIP_DIRS` in `src/chinvex/util.py`
- **Prevention:** Consider file size limits or file count limits per directory during ingestion
- **Evidence:** 87120fd

### 2026-02-05 — Fixed OpenAI token limit errors with token-aware batching

- **Symptom:** OpenAI embeddings API errors: "Requested 1536000 tokens, max 300000 tokens per request" when ingesting repos with many large chunks
- **Root cause:** Batching only by count (max 2048 texts) could exceed the 300K token limit (e.g., 2048 chunks × 750 tokens each = 1.5M tokens)
- **Fix:** Added `MAX_BATCH_TOKENS = 250_000` and `estimate_tokens()` helper, modified batching logic to respect both count AND token limits
- **Prevention:** Always estimate tokens before batching to external APIs with token limits
- **Evidence:** 8e23326

### 2026-02-05 — Cleaned up documentation structure

- **Why:** Removed obsolete spec files (P5_IMPLEMENTATION_PLAN_PARTIAL, PROJECT_MEMORY_SPEC_v0.3, REAL_FIX_EXPLANATION) and moved AGENTS.md to docs/
- **Impact:** Cleaner repo structure, removed duplicate/outdated docs
- **Evidence:** ba01636

### 2026-02-05 — Fixed empty string handling in OpenAI embeddings

- **Symptom:** OpenAI API errors when empty strings present in batch
- **Root cause:** Preprocessing created empty strings that weren't filtered before sending to API
- **Fix:** Filter out empty strings in `embed_many()` before batching
- **Prevention:** Validate inputs at API boundaries
- **Evidence:** 5d9bb4e

### 2026-02-05 — Transformed repo paths to dict format with metadata

- **Why:** Strap integration requires rich metadata (chinvex_depth, status, tags) for repos
- **Impact:** Context.json now stores repos as list of dicts with metadata, enables depth-based ingestion
- **Evidence:** ddc11fe

### 2026-02-05 — Made purge command completely delete context directories

- **Why:** Original purge only cleared index data, left context config behind
- **Impact:** `chinvex context purge` now fully removes context directory including config, index, chroma, meta, watch history, digests
- **Evidence:** 09cbb7a

### 2026-02-05 — Replaced Unicode characters with ASCII-safe alternatives in purge output

- **Symptom:** Windows PowerShell displays garbled characters for Unicode symbols
- **Root cause:** Unicode checkmarks/X marks not supported in default Windows console encoding
- **Fix:** Use ASCII equivalents (✓ → [OK], ✗ → [X])
- **Prevention:** Avoid Unicode symbols in CLI output for Windows compatibility
- **Evidence:** 6cda899

### 2026-02-05 — Added CLAUDE.md management to update-memory skill

- **Why:** Memory System documentation section needed in all repos using the system
- **Impact:** update-memory now ensures CLAUDE.md exists and contains Memory System section
- **Evidence:** e7eb059

## 2026-01

### 2026-01-29 — Added context purge command for batch cleanup

- **Why:** Needed ability to delete one or all contexts completely
- **Impact:** `chinvex context purge <name>` and `chinvex context purge --all` for cleanup
- **Evidence:** cb7a1cb, 26166d1

### 2026-01-29 — Implemented update-memory skill for agent-driven maintenance

- **Why:** Memory files need automated updates from git history
- **Impact:** `/update-memory` skill analyzes commits and generates STATE/CONSTRAINTS/DECISIONS
- **Evidence:** a47be3e

### 2026-01-29 — Added uninitialized memory file detection to brief

- **Why:** Need to signal when memory files are empty templates vs. populated
- **Impact:** Brief shows "ACTION REQUIRED" warning when files uninitialized, instructs user to run `/update-memory`
- **Evidence:** 4b91d50, 2a68541

### 2026-01-29 — Updated memory templates to match spec v0.3

- **Why:** Spec v0.3 finalized with DO/DON'T headers, Quick Reference section
- **Impact:** Bootstrap templates now match current spec format
- **Evidence:** 820a25f

### 2026-01-29 — Bootstrap memory files when installing SessionStart hook

- **Why:** SessionStart hook runs `chinvex brief` which expects memory files to exist
- **Impact:** Ingestion now creates bootstrap templates in `docs/memory/` if missing
- **Evidence:** 47e9a40

### 2026-01-29 — Fixed hardcoded Ollama provider in search

- **Symptom:** Search always used Ollama embeddings regardless of context config
- **Root cause:** Search code hardcoded provider instead of reading from context config
- **Fix:** Use selected embedding provider from context.json or CLI flag
- **Prevention:** Always use provider abstraction layer, never hardcode provider names
- **Evidence:** b8b6517, e3e4738

### 2026-01-28 — Implemented depth-based ingestion behavior

- **Why:** Different repos need different ingestion strategies (full, light, index-only)
- **Impact:** Added `chinvex_depth` metadata field with three levels controlling ingestion behavior
- **Evidence:** 3531618, c9a0b8c

### 2026-01-28 — Completed P5.3 eval suite implementation

- **Why:** No way to validate retrieval quality or measure impact of changes
- **Impact:** Golden query evaluation with hit rate, MRR, latency metrics, CI gate at 80% baseline
- **Evidence:** 23ef788 through 9af47fc (Tasks 11-18)

### 2026-01-28 — Completed P5.4 reranker implementation

- **Why:** Initial retrieval relevance ordering often suboptimal
- **Impact:** Two-stage retrieval with Cohere, Jina, and local cross-encoder providers
- **Evidence:** f03c1c5 through d790a55 (Tasks 19-25)
