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

**Basic ingestion** (reads paths from context.json):
```powershell
chinvex ingest --context MyProject
```

**Inline context creation** (auto-creates context if missing):
```powershell
# Add a repo - creates context if needed
chinvex ingest --context MyProject --repo C:\Code\myproject

# Add a chat root - creates context if needed
chinvex ingest --context MyProject --chat-root P:\ai_memory\chats\myproject

# Combine multiple sources
chinvex ingest --context MyProject --repo C:\Code\myproject --chat-root P:\ai_memory\chats\myproject
```

**Path normalization**: All paths are automatically normalized (absolute, forward slashes, lowercase on Windows) and deduplicated. Running the same command twice won't add duplicate paths.

**Additional ingest flags**:
- `--embed-provider {ollama|openai}`: Choose embedding provider (default: ollama)
- `--rebuild-index`: Force full rebuild instead of incremental update
- `--no-write-context`: Prevent auto-creation of missing contexts (fail instead)

**Provider switching**: If you switch embedding providers, you MUST use `--rebuild-index`:
```powershell
# Switch from Ollama to OpenAI
chinvex ingest --context MyProject --embed-provider openai --rebuild-index
```

**Ingest run logging**: Every ingest run is logged to `{index_dir}/ingest_runs.jsonl` with:
- Run ID (UUID)
- Start/end timestamps
- Sources ingested
- Success/failure status
- Error details (if failed)

**Index metadata**: A `meta.json` file tracks provider/model/dimensions to prevent mixing incompatible embeddings:
```json
{
  "schema_version": 1,
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-small",
  "embedding_dimensions": 1536,
  "created_at": "2026-01-29T12:34:56Z"
}
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

The Chinvex MCP server is an **HTTP client** that connects to the gateway API, exposing Chinvex memory search to Claude Desktop/Code via the MCP protocol.

### Configuration

Set environment variables:
- `CHINVEX_URL`: Gateway URL (default: `https://chinvex.unkndlabs.com`)
- `CHINVEX_API_TOKEN`: Bearer token for authentication (required)

**Using the runbook script** (recommended):
```powershell
.\scripts\start_mcp.ps1
```

Loads token from `P:\secrets\chinvex_mcp_token.txt` and starts the server.

**Manual startup**:
```powershell
$env:CHINVEX_API_TOKEN = "your-token-here"
$env:CHINVEX_URL = "https://chinvex.unkndlabs.com"
python -m chinvex_mcp.server
```

### Claude Desktop / Claude Code Config

Add to `claude_desktop_config.json` or MCP settings:
```json
{
  "mcpServers": {
    "chinvex": {
      "command": "python",
      "args": ["-m", "chinvex_mcp.server"],
      "env": {
        "CHINVEX_URL": "https://chinvex.unkndlabs.com",
        "CHINVEX_API_TOKEN": "your-token-here"
      }
    }
  }
}
```

### Available Tools

- `chinvex_search`: Search personal knowledge base with grounded retrieval
  - Returns evidence chunks with citations from indexed sources
  - Supports single context or cross-context search (`contexts="all"`)
  - Parameters: `query` (required), `contexts` (default: "all"), `k` (default: 8)

- `chinvex_list_contexts`: List all available contexts in the knowledge base
  - Returns context names, aliases, and last update times
  - No parameters required

- `chinvex_get_chunks`: Retrieve full chunk content by ID
  - Use after searching to get complete text of specific chunks
  - Parameters: `context` (required), `chunk_ids` (required, list)

### Tool Examples

**Search across all contexts**:
```json
{
  "query": "how does authentication work?",
  "contexts": "all",
  "k": 8
}
```

**Search specific context**:
```json
{
  "query": "stripe integration",
  "contexts": "MyProject",
  "k": 5
}
```

**List contexts**:
```json
{}
```

**Get full chunks**:
```json
{
  "context": "MyProject",
  "chunk_ids": ["abc123def456", "789ghi012jkl"]
}
```

### Legacy MCP Server (Deprecated)

The old stdio-based MCP server with `--config` is deprecated. Use the HTTP client version above instead.

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

**Provider precedence**:
1. CLI flag (`--embed-provider`)
2. Context config (`embedding_provider` field in context.json)
3. Environment variable (`CHINVEX_EMBED_PROVIDER`)
4. Default (ollama with mxbai-embed-large)

**OpenAI provider features**:
- Batching (up to 2048 texts per request)
- Automatic retry with exponential backoff (3 attempts)
- Model: `text-embedding-3-small` (1536 dimensions)

**Dimension safety**: Chinvex prevents mixing embeddings with different dimensions. If you switch providers or models, use `--rebuild-index` to recreate the index with the new dimensions.

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

## Metrics

Chinvex exports Prometheus metrics for monitoring embeddings, digest generation, and brief generation.

**Embedding metrics**:
- `chinvex_embeddings_total{provider, model, operation}`: Counter of embedding requests
- `chinvex_embeddings_latency_seconds{provider, model, operation}`: Histogram of embedding latency
- `chinvex_embeddings_retries_total{provider, model}`: Counter of retry attempts (OpenAI only)

**Digest metrics**:
- `chinvex_digest_generated_total{context}`: Counter of digest generations

**Brief metrics**:
- `chinvex_brief_generated_total{context}`: Counter of brief generations

**Export metrics**:
```powershell
# Expose metrics on port 9090
python -c "from prometheus_client import start_http_server; from chinvex.embedding_providers import EMBEDDINGS_TOTAL; start_http_server(9090); import time; time.sleep(3600)"
```

## Operations

### Runbook Scripts

**Start MCP server** (`scripts/start_mcp.ps1`):
```powershell
.\scripts\start_mcp.ps1
```

Loads `CHINVEX_API_TOKEN` from `P:\secrets\chinvex_mcp_token.txt` and starts the MCP server with environment variables set.

**Backup contexts and indexes** (`scripts/backup.ps1`):
```powershell
.\scripts\backup.ps1
```

Snapshots all contexts and indexes to `P:\backups\chinvex\YYYY-MM-DD_HHMMSS\`.

### E2E Smoke Test

**Run comprehensive smoke test** (`scripts/e2e_smoke_p4.py`):
```powershell
python scripts/e2e_smoke_p4.py
```

Validates:
- Context creation
- Inline repo ingestion
- Digest generation (markdown + JSON)
- Brief generation
- Cleanup

## Troubleshooting

**FTS5 missing**: Install a Python build with SQLite FTS5 enabled.

**Ollama connection/model missing**: Ensure Ollama is running and `ollama pull mxbai-embed-large` completed.

**Windows path issues**: Use escaped backslashes in JSON or forward slashes. Path normalization handles this automatically for `--repo` and `--chat-root` flags.

**Concurrency**: Only one ingest should run at a time (a lock file `hybrid.db.lock` is used). If you see lock errors, wait for the other ingest to finish.

**Provider switching errors**: If you switch embedding providers without `--rebuild-index`, you'll get a dimension mismatch error:
```
ValueError: Provider mismatch: existing index has 768 dimensions (ollama/mxbai-embed-large),
new provider has 1536 dimensions (openai/text-embedding-3-small)
```
**Solution**: Use `--rebuild-index` when switching providers.

**OpenAI API errors**: If you get `401 Unauthorized`, ensure `OPENAI_API_KEY` environment variable is set. For rate limit errors, the provider will automatically retry with exponential backoff (3 attempts).

**Duplicate paths**: If you accidentally add the same path twice with different casing or slashes, path normalization prevents duplicates. Remove duplicates by editing `context.json` directly.

## Smoke Test

**Legacy config test**:
```powershell
chinvex ingest --config path\to\config.json
chinvex search --config path\to\config.json "known token"
```
Expected: ingest creates `<index_dir>\hybrid.db` and `<index_dir>\chroma`, and search returns results.

**Context-based test**:
```powershell
chinvex ingest --context TestProject --repo C:\Code\test-repo
chinvex search --context TestProject "known token"
```

**Comprehensive E2E test** (recommended):
```powershell
python scripts/e2e_smoke_p4.py
```
See [Operations](#operations) section for details.
