# Chinvex P5b Spec (NOT YET PLANNED)

**Headline: Memory Automation + Retrieval Quality**

## Scope

This spec covers the portions of P5 that are **NOT yet planned**:
- **P5.2.1 Memory File Maintainer** (estimated ~5 tasks)
- **P5.2.4 Startup Hook Installation** (estimated ~3 tasks)
- **P5.3 Retrieval Eval Suite** (estimated ~8 tasks)
- **P5.4 Reranker** (estimated ~9 tasks)

Total estimated: **~25 tasks** across **batches 4-10**

---

## P5.2.1 Memory File Maintainer

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
- If STATE.md/CONSTRAINTS.MD/DECISIONS.md don't exist, create from template (see PROJECT_MEMORY_SPEC_DRAFT.md)
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

### Acceptance

- [ ] `chinvex update-memory --context Chinvex` produces updated memory files
- [ ] Empty commit range = early exit (no work done)
- [ ] Coverage anchor updated correctly
- [ ] Review mode shows diff without committing

---

## P5.2.4 Startup Hook Installation

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

4. **Budget guardrails**
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

1. **P5.2.1 + P5.2.4 Memory Maintainer + Startup Hook** (batches 4-6)
2. **P5.3 Eval Suite** (batches 7-8)
3. **P5.4 Reranker** (batches 9-10)

---

## Dependencies

- P5.2.1 requires LLM access (Codex or API) to analyze commits
- P5.3 requires representative golden queries (manual curation)
- P5.4 requires P5.3 to measure improvement

---

## References

- PROJECT_MEMORY_SPEC_DRAFT.md (memory file contracts, templates)
- CHINVEX_FUTURE_IMPROVEMENTS.md (reranker, eval suite priorities)
- P1_IMPLEMENTATION_SPEC.md (context.json schema)
- P2_IMPLEMENTATION_SPEC.md (meta.json schema, embedding provider abstraction)

---

## Decision Log (Locked)

All decisions from P5_IMPLEMENTATION_SPEC.md apply.

---

## Next Steps

After completing P5a execution, run:
```bash
/batch-plan specs/P5b_IMPLEMENTATION_SPEC.md
```

This will generate Tasks 14+ for the remaining P5 features.
