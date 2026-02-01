# Chinvex P5a Spec (PLANNED - Tasks 1-13)

**Headline: Embedding Integrity + Brief Updates**

## Scope

This spec covers the portions of P5 that have been **fully planned** in `docs/plans/2026-01-31-p5-implementation.md`:
- **P5.1 Embedding Integrity** (Tasks 1-8)
- **P5.2.2 Brief Generation Update** (Tasks 9-10)
- **P5.2.3 Morning Brief Overhaul** (Tasks 11-13)

---

## P5.1 Embedding Integrity (CRITICAL)

### Problem

Query embedding currently falls back to Ollama when gateway starts, regardless of what embedder was used to create the index. This causes silent retrieval degradation - vectors don't match, similarity scores are meaningless.

You hit this today: MCP search failed because Ollama wasn't running, but the index was built with OpenAI embeddings.

### Solution

1. **Index metadata is source of truth**
   - `meta.json` already records: `embedding_provider`, `embedding_model`, `embedding_dimensions`
   - Gateway must read this and use the same provider/model for query embedding

2. **Hard failure on mismatch**
   - If query embedding cannot use index's embedder (missing creds, service down):
     - **Hard fail with clear error** (preferred)
     - Do NOT degrade to lexical-only (misleading - user thinks they got vector results)
   - Never silently use a different embedder

3. **Cross-context search with mixed embedding spaces**
   - Detect when contexts use different embedding providers
   - **Default behavior (P5): Refuse mixed-space search** with clear error
   - Opt-in flag `--allow-mixed-embeddings` exists but **returns error "mixed-space merge not yet supported"** in P5
   - Future (P6+): group+merge with provenance tracking

4. **Status visibility**
   - `chinvex status` shows embedding provider for each context
   - Gateway **`/health` endpoint** returns:
     ```json
     {
       "embedding_provider": "openai",
       "embedding_model": "text-embedding-3-small",
       "contexts_loaded": 8,
       "uptime_seconds": 3600
     }
     ```

5. **OpenAI as default provider**
   - New contexts default to OpenAI embeddings (text-embedding-3-small)
   - Ollama remains available via explicit `--embed-provider ollama`
   - Rationale: 45x faster, consistent quality, cost negligible for personal use

6. **Documentation**
   - Update README.md: explain embedding provider selection, default behavior, how to check/change

7. **Existing Ollama contexts**
   - Warn on query: "Context X uses Ollama embeddings. Consider migrating to OpenAI."
   - Continue to work (Ollama must be running)
   - Migration: `chinvex ingest --context X --embed-provider openai --rebuild-index`

### Acceptance

- [x] Task 1: Gateway /health endpoint returns embedding configuration
- [x] Task 2: Gateway reads embedding config from meta.json on startup
- [x] Task 3: Validate embedding provider is available before accepting queries
- [x] Task 4: Set OpenAI as default embedding provider for new contexts
- [x] Task 5: Detect mixed embedding providers in cross-context search
- [x] Task 6: Add --allow-mixed-embeddings flag (returns "not yet supported")
- [x] Task 7: Update chinvex status to show embedding provider per context
- [x] Task 8: Warn when querying Ollama contexts

---

## P5.2.2 Brief Generation Update

Update `chinvex brief` to match PROJECT_MEMORY_SPEC_DRAFT.md:

- CONSTRAINTS.md: **Exact headers only**: `## Infrastructure`, `## Rules`, `## Hazards` (no fuzzy matching)
- DECISIONS.md: Recent rollup section (rewritable summary at top) + entries from last 7 days

### Acceptance

- [x] Task 9: Update chinvex brief to use exact header matching for CONSTRAINTS.md
- [x] Task 10: Include DECISIONS.md Recent rollup + last 7 days

---

## P5.2.3 Morning Brief Overhaul

Current morning brief shows only ops metrics - useless for knowing what to work on.

**New structure:**
```markdown
# Morning Brief
Generated: YYYY-MM-DD HH:MM

## System Health (quick)
- Contexts: X (Y stale)
- Watch hits: Z

## Active Projects
### [Context Name]
**Objective:** [from STATE.md Current Objective]
**Next Actions:**
- [ ] [from STATE.md Next Actions]

## Stale Contexts (if any)
## Watch Hits (if any)
```

**Implementation:**
- **Active context = ingested within last 7 days** (uses `last_sync` from STATUS.json)
- **Stale context = >7 days since last ingest**
- Cap at **top 5 contexts by recent activity**
- For each active context with STATE.md, include objective + **next actions (max 5 bullets)**
- **ntfy integration**: Already exists in `morning_brief.ps1`, P5 updates the content

### Acceptance

- [x] Task 11: Implement active/stale context detection (7-day threshold)
- [x] Task 12: Parse STATE.md for Current Objective and Next Actions
- [x] Task 13: Format morning brief with objectives, actions, ntfy integration

---

## Implementation Plan

See: `docs/plans/2026-01-31-p5-implementation.md` (Tasks 1-13)

## Next Steps

Execute P5a using:
```bash
/batch-exec docs/plans/2026-01-31-p5-implementation.md
```

After P5a completion, continue with P5b spec.
