# Chinvex P5 Spec

**Headline: Reliability + Retrieval Quality**

## Goal

Turn Chinvex from "works if you babysit it" into a **trustworthy appliance**:
- Embedding integrity enforced (no silent mismatches)
- Retrieval quality measurable and improvable
- Memory files maintained automatically
- Operational annoyances fixed

## Non-goals (P6+)

- Smart scheduling agent
- Multi-user auth
- UI beyond CLI + `dual` alias

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

### Schema References

- `meta.json` schema: See P2_IMPLEMENTATION_SPEC.md (records embedding_provider, embedding_model, embedding_dimensions)
- `context.json` schema: See P1_IMPLEMENTATION_SPEC.md
   - Document `chinvex status` output showing embedding provider per context

### Acceptance

- [ ] Query embedding always matches index embedding space
- [ ] Missing OpenAI key with OpenAI-embedded index = clear error, not Ollama fallback
- [ ] `chinvex status` shows embedding provider per context

---

## P5.2 Memory System Completion

### Problem

The 3-file memory system exists but the chain is broken:
- `docs/memory/` files exist but are stale (template content from P4)
- `chinvex brief` exists and works, but outputs stale content
- `chinvex digest` exists and works
- No automated maintenance of memory files
- Brief generation uses wrong CONSTRAINTS rule (P4: "until first ##" vs new spec: "specific sections")
- **Morning brief is useless** - shows only ops metrics (context count, chunks, watch hits), not project state

### Solution

#### P5.2.1 Memory File Maintainer

Implement `chinvex update-memory --context X` per PROJECT_MEMORY_SPEC_DRAFT.md:

1. Read STATE.md footer for last processed commit hash
2. `git log <hash>..HEAD` for changes
3. Read specs/plans touched in those commits (files in `/specs/` and `/docs/plans/`)
4. Apply contracts:
   - STATE.md: **full regeneration** (manual edits may be lost - this file is machine-managed)
   - CONSTRAINTS.md: merge-only (add bullets, LLM reasons about contradictions to detect obsolete items and move to Superseded)
   - DECISIONS.md: append-only (new entries to monthly section, update Recent rollup - see PROJECT_MEMORY_SPEC_DRAFT.md)
5. Update coverage anchor in STATE.md footer

**Important:** If you want to preserve something, put it in CONSTRAINTS.md (merge-only), not STATE.md (rewritten).

**Automation:** Manual only for P5. User runs `chinvex update-memory`. Scheduling is P6.

**Coverage anchor format:**
```markdown
<!-- chinvex:last-commit:abc123def -->
```
Placed at end of STATE.md. Maintainer reads this to know where to start processing.

**Missing memory files:**
- If `docs/memory/` doesn't exist, create it
- If STATE.md/CONSTRAINTS.md/DECISIONS.md don't exist, create from template (see PROJECT_MEMORY_SPEC_DRAFT.md)
- First run bootstraps the structure

**Bounded inputs guardrails:**
- Max 50 commits per run (if exceeded, use only commit messages, skip file analysis)
- Max 20 files to analyze in detail
- Max 100KB total spec/plan content
- If limits exceeded: summarize from commit messages only, flag for manual review

**Modes:**
- **Review mode (default)**: write changes, print diff, don't commit
- **Commit mode (`--commit`)**: auto-commit with `docs: update memory files`

**Documentation:**
- Update README.md: explain `chinvex update-memory` command, modes, bounded inputs, when to run

#### P5.2.2 Brief Generation Update

Update `chinvex brief` to match PROJECT_MEMORY_SPEC_DRAFT.md:

- CONSTRAINTS.md: **Exact headers only**: `## Infrastructure`, `## Rules`, `## Hazards` (no fuzzy matching)
- DECISIONS.md: Recent rollup section (rewritable summary at top) + entries from last 7 days

**Documentation:**
- Update README.md: explain what `chinvex brief` includes, memory file requirements

#### P5.2.3 Morning Brief Overhaul

Current morning brief shows only ops metrics - useless for knowing what to work on.

**New morning brief structure:**

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

### [Another Context]
...

## Stale Contexts (if any)
- [context]: X hours since sync

## Watch Hits (if any)
- "query" hit in [context]: [file]
```

**Implementation:**
- **Active context = ingested within last 7 days** (uses `last_sync` from STATUS.json)
- **Stale context = >7 days since last ingest** (inverse of active)
- Cap at **top 5 contexts by recent activity** (sorted by last_sync timestamp)
- For each active context with STATE.md, include objective + **next actions (max 5 bullets)**
- Skip contexts with no STATE.md or no recent activity
- Ops metrics become a small header, not the whole brief
- **ntfy integration**: Already exists in `morning_brief.ps1`, triggered by 7am scheduled task. P5 updates the content.

**Documentation:**
- Update README.md: explain new morning brief structure, what makes a context "active", ntfy push content

#### P5.2.4 Startup Hook Installation

During `chinvex ingest`, install Claude Code startup hook:

1. Detect repo roots from context config (all repos in `includes.repos`)
2. For **each repo** in the context, create/update `<repo>/.claude/settings.json`:
   ```json
   {
     "hooks": {
       "startup": ["chinvex brief --context <context-name>"]
     }
   }
   ```
3. Merge with existing settings (deep merge - don't clobber other config)
4. `--no-claude-hook` flag to skip installation
5. Document in README.md: explain hook behavior, `--no-claude-hook` flag, and how to manually remove

**Edge cases:**
- Non-git directories: skip hook installation, warn user
- Existing hooks.startup as string: convert to array, append
- Malformed settings.json: warn and skip (don't corrupt)

**Rationale:** Ingestion is when Chinvex "claims" a repo - natural time to install the hook. Guarantees context exists when hook runs.

### Acceptance

- [ ] `chinvex update-memory --context Chinvex` produces updated memory files
- [ ] Empty commit range = early exit (no work done)
- [ ] Coverage anchor updated correctly
- [ ] Review mode shows diff without committing
- [ ] `chinvex brief` outputs correct CONSTRAINTS sections per new spec
- [ ] Morning brief includes project objectives and next actions
- [ ] Morning brief ntfy push includes top 1-2 objectives, not just counts
- [ ] Chinvex ingest installs `.claude/settings.json` hook in repo
- [ ] Existing settings.json merged, not clobbered
- [ ] `--no-claude-hook` skips installation

---

## P5.3 Retrieval Eval Suite

### Problem

No way to know if retrieval is working well. Changes to chunking, weighting, or reranking can't be validated.

### Solution

1. **Golden query set**
   - 20+ queries **per context** (each context needs its own eval set)
   - Mix of: code lookups, concept searches
   - Stored in `tests/eval/golden_queries_<context>.json`

2. **Success criteria (per query)**
   - **Pass**: At least one chunk from expected file appears in top K (file path match)
   - **Default K=5**, configurable per query via `k` field
   - **Anchor match (optional)**: Chunk contains expected anchor string (bonus metric, not pass/fail)
   - **Multiple acceptable sources**: Query specifies `expected_files` array, any match passes

3. **Golden query schema** (`tests/eval/golden_queries.json`):
   ```json
   {
     "queries": [
       {
         "query": "chromadb batch limit",
         "context": "Chinvex",
         "expected_files": ["src/chinvex/ingest.py", "docs/constraints.md"],
         "anchor": "5000 vectors",
         "k": 5
       }
     ]
   }
   ```

4. **Eval command**
   - `chinvex eval --context X` runs golden queries
   - Reports: hit rate @K, MRR, latency
   - **CI gate**: Fails if hit rate drops below 80% compared to **fixed baseline** in `tests/eval/baseline_metrics.json`

5. **Query logging (optional for P5)**
   - Log every search: query, chunks returned, scores, context
   - Location: `.chinvex/logs/queries.jsonl`
   - Retention: 30 days, auto-rotated
   - For debugging and future golden query candidates

6. **Documentation**
   - Update README.md: explain `chinvex eval` command, golden query format, how to add queries

### Acceptance

- [ ] `chinvex eval --context Chinvex` runs and reports metrics
- [ ] At least 20 golden queries defined
- [ ] Baseline metrics recorded

---

## P5.4 Reranker

### Problem

Initial retrieval returns top-K by vector similarity, but relevance ordering is often wrong. A reranker provides massive quality boost.

### Solution

1. **Two-stage retrieval**
   - Stage 1: vector search returns top N candidates (default 20, configurable via `reranker.candidates`)
   - Stage 2: cross-encoder reranks to top K (default 5, configurable via `reranker.top_k`)

2. **Provider options**
   - Cohere Rerank API (easiest, best quality)
   - Jina Reranker API
   - Local cross-encoder (ms-marco-MiniLM) - **downloads on first use** to `~/.cache/huggingface/`, slower but free

3. **Configuration**
   - `reranker` field in **context.json** (per-context, not global for P5)
   - **Default: disabled** (field absent or `"reranker": null`)
   - `--rerank` flag on search commands to enable for single query
   - Example context.json:
     ```json
     {
       "reranker": {
         "provider": "cohere",
         "model": "rerank-english-v3.0",
         "candidates": 20,
         "top_k": 5
       }
     }
     ```

5. **Budget guardrails**
   - Only runs when initial retrieval returns ≥10 candidates
   - Max 50 candidates to reranker (truncate if more)
   - Latency budget: 2s max for rerank step
   - **Fallback**: If provider unavailable (missing creds OR service down OR timeout), skip rerank with warning to stderr, return pre-rerank results

5. **Documentation**
   - Update README.md: explain `--rerank` flag, provider configuration, latency expectations

### Acceptance

- [ ] `chinvex search --context X --rerank "query"` uses reranker
- [ ] Eval suite shows improved hit rate with reranker enabled
- [ ] Reranker provider configurable

---

## Implementation Order

1. **P5.1 Embedding Integrity** - Critical fix, unblocks trust in retrieval
2. **P5.2.2 + P5.2.3 Brief/Morning Brief** - Immediate daily value, verifies freshness signals
3. **P5.2.1 + P5.2.4 Memory Maintainer + Startup Hook** - Automation, requires brief to be working
4. **P5.3 Eval Suite** - Required before reranker work
5. **P5.4 Reranker** - Quality improvement, requires eval to validate

---

## Dependencies

- P5.1 requires gateway changes
- P5.2.1 requires LLM access (Codex or API) to analyze commits
- P5.2.2 requires understanding of markdown section parsing
- P5.3 requires representative golden queries (manual curation)
- P5.4 requires P5.3 to measure improvement

---

## References

- PROJECT_MEMORY_SPEC_DRAFT.md (memory file contracts, templates)
- CHINVEX_FUTURE_IMPROVEMENTS.md (reranker, eval suite priorities)
- P1_IMPLEMENTATION_SPEC.md (context.json schema)
- P2_IMPLEMENTATION_SPEC.md (meta.json schema, embedding provider abstraction)

---

## Implementation Details (Decide During Coding)

These don't need spec-level decisions:

- Error message exact wording
- Diff output format (unified diff is fine)
- Logging verbosity (follow existing patterns)
- Edge cases for malformed files (warn and skip)
- Exit codes (0 success, 1 error, follow Unix conventions)

---

## Decision Log (Locked)

These decisions are locked for P5. Don't revisit.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default embedding provider | **OpenAI (text-embedding-3-small)** | 45x faster than Ollama, consistent quality, negligible cost |
| Embedding mismatch behavior | **Hard fail** | Lexical-only fallback is misleading - user thinks they got vector results |
| Existing Ollama contexts | **Warn on query, continue working** | Migration is manual via `--rebuild-index` |
| Mixed-space cross-context search | **Refuse by default** | Trust by default; solve merge scoring in P6+ |
| `--allow-mixed-embeddings` flag | **Exists but errors "not yet supported"** | Placeholder for P6+, no implementation in P5 |
| Active context window | **7 days** (uses last_sync from STATUS.json) | Matches brief/digest cadence |
| Stale context threshold | **>7 days since last ingest** | Inverse of active definition |
| Morning brief context cap | **Top 5 by last_sync timestamp** | Prevents noisy repos from dominating |
| Morning brief next actions | **Max 5 bullets per context** | Keep brief readable |
| Missing memory files | **Create from template** | First run bootstraps the structure |
| STATE.md update contract | **Full regeneration** | Machine-managed; preserve info in CONSTRAINTS.md instead |
| CONSTRAINTS.md obsolete detection | **LLM reasoning about contradictions** | Not automatic |
| Memory maintainer trigger | **Manual only for P5** | Scheduling is P6 |
| Coverage anchor format | `<!-- chinvex:last-commit:abc123 -->` | Simple, parseable, hidden in rendered markdown |
| Brief section header matching | **Exact headers only** | `## Infrastructure`, `## Rules`, `## Hazards` |
| Hook installation scope | **All repos in context** | Each repo gets the hook when context is ingested |
| Eval queries organization | **Per-context** | `tests/eval/golden_queries_<context>.json` |
| Eval default K | **K=5** | Configurable per query |
| Eval CI baseline | **Fixed file** `tests/eval/baseline_metrics.json` | Not compared to previous run (flaky) |
| Query logging (optional) | `.chinvex/logs/queries.jsonl`, 30-day retention | For debugging |
| Reranker config location | **context.json** | Per-context, no global config for P5 |
| Reranker default state | **Disabled** | Opt-in via `--rerank` flag or context.json |
| Reranker candidates/top_k | **Configurable** (defaults: 20, 5) | In context.json reranker config |
| Reranker latency budget | **2s max** | Skip with warning if exceeded, return pre-rerank results |
| Local cross-encoder | **Downloads on first use** | sentence-transformers caches to ~/.cache/huggingface/ |
| Memory maintainer commit cap | **50 commits** | Summarize from messages only if exceeded |
| Review mode | **Default for update-memory** | No auto-commit until trust established |
