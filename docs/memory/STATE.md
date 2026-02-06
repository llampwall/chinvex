<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
Dashboard integration and metadata sync infrastructure

## Active Work
- Implemented chinvex sync daemon ingestion status visibility for allmind dashboard
- Added `chinvex context sync-metadata-from-strap` command for registry→context metadata sync
- Integrated .chinvex-status.json files with PID tracking for staleness detection

## Blockers
None

## Next Actions
- [ ] Test dashboard status integration end-to-end
- [ ] Validate depth change workflow (sync metadata + rebuild-index)
- [ ] Complete P5b planning and implementation (memory maintainer, startup hooks)
- [ ] Validate eval suite with ≥80% hit rate baseline

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
Last memory update: 2026-02-05
Commits covered through: d4681d179e391e1a97af8165853730d4ed6efe16

<!-- chinvex:last-commit:d4681d179e391e1a97af8165853730d4ed6efe16 -->
