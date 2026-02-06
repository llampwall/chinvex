<!-- DO: Add bullets. Edit existing bullets in place with (updated YYYY-MM-DD). -->
<!-- DON'T: Delete bullets. Don't write prose. Don't duplicate — search first. -->

# Constraints

## Infrastructure
- ChromaDB batch limit: 5000 vectors per batch
- OpenAI embeddings API: 300K token limit per request, 2048 texts per batch
- OpenAI token estimation: ~4 chars = 1 token (conservative limit: 250K tokens)
- SQLite FTS5 required for lexical search
- Index metadata stored in `meta.json` tracks provider/model/dimensions
- Embedding dimensions must match across ingestion runs (switching providers requires `--rebuild-index`)
- Repo ingestion status tracked in `.chinvex-status.json` files with PID for staleness detection
- Strap registry.json is source of truth for repo metadata (status, tags, chinvex_depth)

## Rules
- Always use `--rebuild-index` when switching embedding providers (prevents dimension mismatch)
- Delete-then-insert for chunk upserts to avoid duplicates
- Filter empty strings before sending to OpenAI embeddings API
- Skip ingestion of `logs/` and `log/` directories to avoid indexing large JSON log files
- Path normalization: absolute, forward slashes, lowercase on Windows
- Only one ingest process per context at a time (uses lock file)
- Exclude `.chinvex-status.json` from sync daemon file watching (prevents infinite loop)
- Status/tags changes: sync metadata only (no ingest), depth changes: sync metadata + `--rebuild-index`

## Key Facts
- Default contexts root: `P:\ai_memory\contexts`
- Default indexes root: `P:\ai_memory\indexes`
- Watcher PID file: `~/.chinvex/watcher.pid`
- Watcher log: `~/.chinvex/watcher.log`
- Query log: `.chinvex/logs/queries.jsonl` (30-day retention)
- SessionStart hook path: `<repo>/.claude/settings.json`
- Repo status file: `<repo>/.chinvex-status.json` (states: ingesting, idle, error, stale)

## Hazards
- JSON log files get ingested if `.json` is in `ALLOWED_EXTS` and logs directories aren't skipped (causes large index bloat)
- Large chunks can exceed OpenAI token limits when batched (must batch by both count AND token estimate)
- Empty strings in embedding batches cause provider errors
- Changing embedding providers without rebuild creates dimension mismatch errors
- Status files trigger sync daemon if not excluded (infinite loop)
- Going from deeper to shallower depth without rebuild leaves stale data in index (will show up in queries)

## Superseded
