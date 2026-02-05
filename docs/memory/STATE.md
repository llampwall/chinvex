<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
Bug fixes and hardening after P5 evaluation suite and reranker implementation

## Active Work
- Fixing edge cases in embedding providers (OpenAI token limits, empty string handling)
- Improving ingestion reliability (path normalization, skip directories)
- Cleaning up documentation and removing obsolete specs

## Blockers
None

## Next Actions
- [ ] Complete P5b planning and implementation (memory maintainer, startup hooks)
- [ ] Validate eval suite with ≥80% hit rate baseline
- [ ] Test reranker integration across all providers
- [ ] Document strap integration workflow

## Quick Reference
- Install: `pip install -e .` (requires Python 3.12, venv)
- Ingest: `chinvex ingest --context <name> --repo <path>`
- Search: `chinvex search --context <name> "query"`
- Test: `pytest`
- Entry point: `src/chinvex/cli.py`

## Out of Scope (for now)
- Scheduled memory maintenance (deferred to P6)
- Cross-context search UI improvements
- Automated golden query generation

---
Last memory update: 2026-02-05
Commits covered through: 87120fd5063f5d4a1a648132caa7b2321a2893f1

<!-- chinvex:last-commit:87120fd5063f5d4a1a648132caa7b2321a2893f1 -->
