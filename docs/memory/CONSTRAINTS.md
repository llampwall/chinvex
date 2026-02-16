<!-- DO: Add bullets. Edit existing bullets in place with (updated YYYY-MM-DD). -->
<!-- DON'T: Delete bullets. Don't write prose. Don't duplicate — search first. -->

# Constraints

## Infrastructure
- ChromaDB batch limit: 5000 vectors per batch
- ChromaDB uses internal SQLite database that must be explicitly closed via `client._system.stop()` before file deletion (added 2026-02-16)
- VectorStore has close() method and context manager support for proper cleanup (added 2026-02-16)
- Storage uses global SQLite connection pattern; must call `Storage.force_close_global_connection()` before file deletion
- Gateway has shutdown handler that closes SQLite/ChromaDB connections on graceful shutdown (added 2026-02-16)
- OpenAI embeddings API: 300K token limit per request, 2048 texts per batch
- OpenAI token estimation: ~4 chars = 1 token (conservative limit: 250K tokens)
- SQLite FTS5 required for lexical search
- Index metadata stored in `meta.json` tracks provider/model/dimensions
- Embedding dimensions must match across ingestion runs (switching providers requires `--rebuild-index`)
- Repo ingestion status tracked in `.chinvex-status.json` files with PID for staleness detection
- Strap registry.json is source of truth for repo metadata (status, tags, chinvex_depth)
- Context.json backups stored in `P:\ai_memory\backups\<name>\` (30 most recent kept, auto-pruned)
- Backup root configurable via `CHINVEX_BACKUPS_ROOT` environment variable

## Rules
- Always use `--rebuild-index` when switching embedding providers (prevents dimension mismatch)
- Delete-then-insert for chunk upserts to avoid duplicates
- Filter empty strings before sending to OpenAI embeddings API
- Skip ingestion of `logs/` and `log/` directories to avoid indexing large JSON log files
- Path normalization: absolute, forward slashes, lowercase on Windows
- Only one ingest process per context at a time (uses lock file)
- Exclude `.chinvex-status.json` from sync daemon file watching (prevents infinite loop)
- Status/tags changes: sync metadata only (no ingest), depth changes: sync metadata + `--rebuild-index`
- Search must read embedding provider from meta.json (never hardcode provider; errors if meta.json missing)

## Key Facts
- Default contexts root: `P:\ai_memory\contexts`
- Default indexes root: `P:\ai_memory\indexes`
- Watcher PID file: `~/.chinvex/watcher.pid`
- Watcher log: `~/.chinvex/watcher.log`
- Query log: `.chinvex/logs/queries.jsonl` (30-day retention)
- SessionStart hook path: `<repo>/.claude/settings.json`
- Repo status file: `<repo>/.chinvex-status.json` (states: ingesting, idle, error, stale)

## Hazards
- ChromaDB/SQLite connections not explicitly closed before file deletion cause Windows PermissionError [WinError 32] (fixed 2026-02-16)
- Long-running processes (gateway, daemon) accumulate VectorStore connections if not closed; use context manager or explicit close() (fixed 2026-02-16)
- Simply setting `client = None` doesn't release ChromaDB file locks; must call `client._system.stop()` (documented 2026-02-16)
- JSON log files get ingested if `.json` is in `ALLOWED_EXTS` and logs directories aren't skipped (causes large index bloat)
- Large chunks can exceed OpenAI token limits when batched (must batch by both count AND token estimate)
- Empty strings in embedding batches cause provider errors
- Changing embedding providers without rebuild creates dimension mismatch errors
- Status files trigger sync daemon if not excluded (infinite loop)
- Going from deeper to shallower depth without rebuild leaves stale data in index (will show up in queries)
- Hardcoding embedding provider in search causes dimension mismatch when context was ingested with a different provider

## Superseded
