# chinvex - chunked vector indexer

Hybrid retrieval index CLI (SQLite FTS5 + Chroma) powered by Ollama embeddings.

## Prereqs
- Python 3.12
- Ollama installed and running
- `ollama pull mxbai-embed-large`

## Install (venv required)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

## Context Registry (Recommended)

Chinvex now uses a context registry system for managing multiple projects.

### Create a context

```powershell
chinvex context create MyProject
```

This creates:
- `P:\ai_memory\contexts\MyProject\context.json`
- `P:\ai_memory\indexes\MyProject\hybrid.db`
- `P:\ai_memory\indexes\MyProject\chroma\`

### Edit context configuration

Edit `P:\ai_memory\contexts\MyProject\context.json`:

```json
{
  "schema_version": 1,
  "name": "MyProject",
  "aliases": ["myproj"],
  "includes": {
    "repos": ["C:\\Code\\myproject"],
    "chat_roots": ["P:\\ai_memory\\chats\\myproject"],
    "codex_session_roots": [],
    "note_roots": []
  },
  "index": {
    "sqlite_path": "P:\\ai_memory\\indexes\\MyProject\\hybrid.db",
    "chroma_dir": "P:\\ai_memory\\indexes\\MyProject\\chroma"
  },
  "weights": {
    "repo": 1.0,
    "chat": 0.8,
    "codex_session": 0.9,
    "note": 0.7
  }
}
```

### List contexts

```powershell
chinvex context list
```

### Ingest with context

```powershell
chinvex ingest --context MyProject
```

### Search with context

```powershell
chinvex search --context MyProject "your query"
```

### Environment Variables

- `CHINVEX_CONTEXTS_ROOT`: Override default contexts directory (default: `P:\ai_memory\contexts`)
- `CHINVEX_INDEXES_ROOT`: Override default indexes directory (default: `P:\ai_memory\indexes`)
- `CHINVEX_APPSERVER_URL`: App-server URL for Codex session ingestion (default: `http://localhost:8080`)

## Legacy Config (Deprecated)

The old config file format is still supported but deprecated. Use `--context` instead.

Create a JSON config file:
```json
{
  "index_dir": "P:\\ai_memory\\indexes\\streamside",
  "ollama_host": "http://127.0.0.1:11434",
  "embedding_model": "mxbai-embed-large",
  "sources": [
    {"type": "repo", "name": "streamside", "path": "C:\\Code\\streamside"},
    {"type": "chat", "project": "Twitch", "path": "P:\\ai_memory\\projects\\Twitch\\chats"}
  ]
}
```

Run with legacy config:
```powershell
chinvex ingest --config .\config.json
chinvex search --config .\config.json "your query"
```

### Migration from Old Config

On first use with `--config`, chinvex will auto-migrate to a new context and suggest using `--context` going forward.

## MCP Server

Run the local MCP server over stdio (legacy config):
```powershell
chinvex-mcp --config .\config.json
```

Optional overrides:
```powershell
chinvex-mcp --config .\config.json --ollama-host http://skynet:11434 --k 8 --min-score 0.30
```

### Codex config.toml (Legacy)
```
[mcp_servers.chinvex]
command = "[path_to_chinvex]\\.venv\\Scripts\\chinvex-mcp.exe"
args = ["--config", "[path_to_chinvex]\\config.json"]
cwd = "[path_to_chinvex]"
startup_timeout_sec = 30
tool_timeout_sec = 120
```

### Cursor / Claude Desktop MCP config (Legacy)
```json
{
  "mcpServers": {
    "chinvex": {
      "command": "chinvex-mcp",
      "args": ["--config", "C:\\\\path\\\\to\\\\config.json"]
    }
  }
}
```

Note: Context-based MCP server integration is planned for future updates.

### Available Tools

- `chinvex_search`: Search for relevant chunks (returns metadata and snippets)
- `chinvex_get_chunk`: Get full chunk text by ID
- `chinvex_answer`: Grounded search returning evidence pack (chunks + citations) - no LLM synthesis

## Digest & Brief

Generate daily digests and session briefs from your indexed content.

### Digest

```powershell
# Generate digest for last 24 hours
chinvex digest generate --context MyProject --since 24h

# Generate for specific date
chinvex digest generate --context MyProject --date 2026-01-28

# Push notification to ntfy
chinvex digest generate --context MyProject --push ntfy
```

### Brief

```powershell
# Generate session brief
chinvex brief --context MyProject

# Save to file
chinvex brief --context MyProject --output SESSION_BRIEF.md
```

### Memory Files

Create `docs/memory/` in your repo:
- `STATE.md`: Current objective, active work, blockers
- `CONSTRAINTS.md`: Infrastructure limits, rules, key facts
- `DECISIONS.md`: Append-only decision log

See [Memory File Format](specs/P4_IMPLEMENTATION_SPEC.md#appendix-memory-file-format) for details.

## Embedding Providers

Chinvex supports multiple embedding providers:

```powershell
# Use Ollama (default)
chinvex ingest --context MyProject

# Use OpenAI
chinvex ingest --context MyProject --embed-provider openai

# Switch providers (requires rebuild)
chinvex ingest --context MyProject --embed-provider openai --rebuild-index
```

Set `OPENAI_API_KEY` environment variable for OpenAI provider.

### Example tool calls

`chinvex_search`:
```json
{
  "query": "search text",
  "source": "repo",
  "k": 5,
  "min_score": 0.35,
  "include_text": false
}
```

Example output (truncated):
```json
[
  {
    "score": 0.72,
    "source_type": "repo",
    "title": "example.py",
    "path": "C:\\\\Code\\\\streamside\\\\example.py",
    "chunk_id": "abc123",
    "doc_id": "def456",
    "ordinal": 0,
    "snippet": "def main(): ...",
    "meta": {"repo": "streamside", "char_start": 0, "char_end": 3000}
  }
]
```

`chinvex_get_chunk`:
```json
{
  "chunk_id": "abc123"
}
```

`chinvex_answer` (Context-based, requires future MCP server update):
```json
{
  "query": "how does authentication work?",
  "context_name": "MyProject",
  "k": 8,
  "min_score": 0.35
}
```

Returns evidence pack with chunks and citations for grounded answering.

## Gateway (HTTP API)

The gateway provides an HTTP API for search, evidence retrieval, and context management.

### Manual Startup (Development)

Requires 2 terminals:

**Terminal 1 - Cloudflare Tunnel:**
```powershell
cloudflared tunnel run chinvex-gateway
```

**Terminal 2 - Gateway Server:**
```powershell
cd C:\Code\chinvex
pwsh -ExecutionPolicy Bypass -File .\start_gateway.ps1
```

Or directly:
```powershell
python -m chinvex.cli gateway serve --port 7778
```

### Verification

```powershell
curl http://localhost:7778/health
```

Or via tunnel:
```powershell
curl https://chinvex.yourdomain.com/health
```

### Production Setup

For running as a service (PM2, systemd, etc.), see:
- [Cloudflare Tunnel Setup](docs/deployment/cloudflare-tunnel.md)
- [Caddy Reverse Proxy](docs/deployment/caddy.md)

### API Documentation

See [ChatGPT Integration](docs/chatgpt-integration.md) for API examples and OpenAPI spec.

## Troubleshooting
- FTS5 missing: install a Python build with SQLite FTS5 enabled.
- Ollama connection/model missing: ensure Ollama is running and `ollama pull mxbai-embed-large` completed.
- Windows path issues: use escaped backslashes in JSON or forward slashes.
- Concurrency: only one ingest should run at a time (a lock file `hybrid.db.lock` is used). If you see lock errors, wait for the other ingest to finish.

## Smoke Test
```powershell
chinvex ingest --config path\to\config.json
chinvex search --config path\to\config.json "known token"
```
Expected: ingest creates `<index_dir>\hybrid.db` and `<index_dir>\chroma`, and search returns results.
