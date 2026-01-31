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
     - Fail loudly with clear error
     - OR degrade to lexical-only with explicit warning
   - Never silently use a different embedder

3. **Cross-context search with mixed embedding spaces**
   - Detect when contexts use different embedding providers
   - **Default behavior (P5): Refuse mixed-space search** with clear error
   - Opt-in flag `--allow-mixed-embeddings` for explicit override
   - Future (P6+): group+merge with provenance tracking

4. **Status visibility**
   - `chinvex status` shows embedding provider for each context
   - Gateway health endpoint reports which embedding provider is active

5. **OpenAI as default provider**
   - New contexts default to OpenAI embeddings (text-embedding-3-small)
   - Ollama remains available via explicit `--embed-provider ollama`
   - Rationale: 45x faster, consistent quality, cost negligible for personal use

6. **Documentation**
   - Update README.md: explain embedding provider selection, default behavior, how to check/change
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
3. Read specs/plans touched in those commits
4. Apply contracts:
   - STATE.md: rewrite (current objective, active work, blockers, next actions, out of scope)
   - CONSTRAINTS.md: merge-only (add bullets, move obsolete to Superseded)
   - DECISIONS.md: append-only (new entries to monthly section, update Recent rollup)
5. Update coverage anchor in STATE.md footer

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

- CONSTRAINTS.md: Infrastructure + Rules + Hazards sections only (not "until first ##")
- DECISIONS.md: Recent rollup + entries from last 7 days (not just date-parsed entries)

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
- **Active context = ingested/touched within 7 days**
- Cap at **top 5 contexts by recent activity** (prevents noisy repos from dominating)
- For each active context with STATE.md, include objective + next actions
- Skip contexts with no STATE.md or no recent activity
- Ops metrics become a small header, not the whole brief

**Documentation:**
- Update README.md: explain new morning brief structure, what makes a context "active", ntfy push content

#### P5.2.4 Startup Hook Installation

During `chinvex ingest`, install Claude Code startup hook:

1. Detect repo root from context config
2. Create/update `<repo>/.claude/settings.json`:
   ```json
   {
     "hooks": {
       "startup": ["chinvex brief --context <context-name>"]
     }
   }
   ```
3. Merge with existing settings (don't clobber other config)
4. `--no-claude-hook` flag to skip installation
5. Document in README.md: explain hook behavior, `--no-claude-hook` flag, and how to manually remove

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
   - 20+ queries with expected source files/chunks
   - Mix of: code lookups, concept searches, cross-context queries
   - Stored in `tests/eval/golden_queries.json`

2. **Success criteria (per query)**
   - **Pass**: At least one chunk from expected file appears in top K
   - **Anchor match (optional)**: Chunk contains expected anchor string
   - **Multiple acceptable sources**: Query can specify alternatives

3. **Eval command**
   - `chinvex eval --context X` runs golden queries
   - Reports: hit rate @K, MRR, latency
   - **CI gate**: Fails if hit rate drops below 80% (configurable threshold)

3. **Query logging (optional)**
   - Log every search: query, chunks returned, scores, context
   - For debugging and future golden query candidates

4. **Documentation**
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
   - Stage 1: vector search returns top 20 candidates
   - Stage 2: cross-encoder reranks to top 5

2. **Provider options**
   - Cohere Rerank API (easiest, best quality)
   - Jina Reranker API
   - Local cross-encoder (ms-marco-MiniLM) - slower but free

3. **Configuration**
   - `reranker` field in context.json or global config
   - **Default: disabled** (preserves current behavior)
   - `--rerank` flag on search commands to enable

4. **Budget guardrails**
   - Only runs when initial retrieval returns ≥10 candidates
   - Max 50 candidates to reranker (truncate if more)
   - Latency budget: 2s max for rerank step
   - **Fallback**: If provider unavailable, skip rerank with warning (don't fail search)

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

- PROJECT_MEMORY_SPEC_DRAFT.md (memory file contracts)
- CHINVEX_FUTURE_IMPROVEMENTS.md (reranker, eval suite priorities)
- P4 spec (embedding provider abstraction)

---

## Decision Log (Locked)

These decisions are locked for P5. Don't revisit.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default embedding provider | **OpenAI (text-embedding-3-small)** | 45x faster than Ollama, consistent quality, negligible cost |
| Mixed-space cross-context search | **Refuse by default** | Trust by default; solve merge scoring in P6+ |
| Active context window | **7 days** | Matches brief/digest cadence |
| Morning brief context cap | **Top 5 by activity** | Prevents noisy repos from dominating |
| Eval pass criteria | **≥1 chunk from expected file in top K** | Simple, measurable, not subjective |
| Eval CI threshold | **80% hit rate** | Configurable, but this is the baseline |
| Reranker default state | **Disabled** | Opt-in via `--rerank` flag |
| Reranker latency budget | **2s max** | Skip with warning if exceeded |
| Memory maintainer commit cap | **50 commits** | Summarize from messages only if exceeded |
| Review mode | **Default for update-memory** | No auto-commit until trust established |
