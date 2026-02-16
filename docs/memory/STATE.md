<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
Skills, backup infrastructure, and embedding provider hardening

## Active Work
- Implemented proper connection management for ChromaDB and SQLite
  - Added VectorStore.close() method using ChromaDB's _system.stop()
  - Added context manager support to VectorStore
  - Added gateway shutdown handler to close connections cleanly
  - Fixed Windows file lock issues (PermissionError on database deletion)
- Added using-chinvex skill for Claude Code and Codex with comprehensive CLI workflow docs
- Implemented automatic context.json backup system (30 backups, auto-prune)
- Fixed OpenAI as default embedding provider; search reads provider from meta.json

## Blockers
None

## Next Actions
- [ ] Test dashboard status integration end-to-end
- [ ] Validate depth change workflow (sync metadata + rebuild-index)
- [ ] Complete P5b planning and implementation (memory maintainer, startup hooks)
- [ ] Validate eval suite with >=80% hit rate baseline

## Quick Reference
- Install: `pip install -e .` (requires Python 3.12, venv)
- Ingest: `chinvex ingest --context <name> --repo <path>`
- Search: `chinvex search --context <name> "query"`
- Sync metadata: `chinvex context sync-metadata-from-strap --context <name>`
- Test: `pytest`
- Entry point: `src/chinvex/cli.py`

## Out of Scope (for now)
- Scheduled memory maintenance (deferred to P6)
- Cross-context search UI improvements
- Automated golden query generation

---
Last memory update: 2026-02-16
Commits covered through: d09574488014d1b67235b56b5586bb0f3466c38b

<!-- chinvex:last-commit:d09574488014d1b67235b56b5586bb0f3466c38b -->
