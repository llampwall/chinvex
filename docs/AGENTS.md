# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Chinvex is a hybrid retrieval engine (SQLite FTS5 + ChromaDB) for personal knowledge management. It indexes code repositories, chat transcripts, Codex sessions, and notes into a searchable context-based knowledge base with grounded retrieval.

**Language**: Python 3.12  
**Key Technologies**: SQLite FTS5, ChromaDB, FastAPI, Typer, OpenAI/Ollama embeddings

## Development Setup

### Install
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

### Run Tests
```powershell
pytest                           # Run all tests
pytest tests/test_search.py      # Run specific test file
pytest -k test_search_chunks     # Run specific test by pattern
```

### Prerequisites
- Python 3.12 with SQLite FTS5 support
- Ollama installed and running (if using Ollama embeddings)
- For Ollama: `ollama pull mxbai-embed-large`
- For OpenAI: Set `OPENAI_API_KEY` environment variable

## Core Commands

### Context Management
```powershell
chinvex context create <name>                    # Create new context
chinvex context list                            # List all contexts
chinvex ingest --context <name>                 # Ingest sources into context
chinvex search --context <name> "query"         # Search a context
```

### Memory Operations
```powershell
chinvex brief --context <name>                  # Generate session brief
chinvex digest generate --context <name>        # Generate daily digest
chinvex update-memory --context <name>          # Update memory files
chinvex status                                  # Show all contexts status
```

### Gateway (HTTP API)
```powershell
python -m chinvex.cli gateway serve --port 7778  # Start gateway server
curl http://localhost:7778/health                # Check gateway health
```

### Sync & Watch
```powershell
chinvex sync start                              # Start file watcher daemon
chinvex sync stop                               # Stop daemon
```

## Architecture

### High-Level Structure
- **Context Registry**: Each project is a "context" with its own index and configuration stored in `P:\ai_memory\contexts\<name>\context.json`
- **Hybrid Index**: Lexical search (SQLite FTS5) + vector search (ChromaDB) combined with score blending
- **Embedding Providers**: OpenAI (default, text-embedding-3-small) or Ollama (mxbai-embed-large)
- **Gateway**: FastAPI HTTP server exposing search/evidence endpoints with rate limiting, auth, audit logging
- **MCP Server**: HTTP client connecting to gateway, exposing memory search to Claude Desktop/Code

### Source Code Layout
```
src/chinvex/
  cli.py              # Main CLI entry point (Typer app)
  context.py          # Context config dataclasses
  context_cli.py      # Context management commands
  ingest.py           # Ingestion orchestrator (repos, chats, Codex sessions)
  search.py           # Hybrid search implementation
  storage.py          # SQLite FTS5 wrapper
  vectors.py          # ChromaDB wrapper
  embedding_providers.py  # OpenAI & Ollama embedding adapters
  
  chunking.py         # Document chunking (repo, chat, conversation)
  brief.py            # Session brief generation
  digest.py           # Daily digest generation
  archive.py          # Archive tier management
  
  hooks/              # Git/Claude Code hook installation
  gateway/            # FastAPI HTTP gateway
    app.py            # Main FastAPI app
    endpoints/        # API endpoints (health, search, evidence, etc.)
    auth.py           # Bearer token auth
    rate_limit.py     # Rate limiting
    audit.py          # Audit logging
  
  rerankers/          # Reranking providers (Cohere, Jina, local)
  adapters/           # External system adapters (Codex app-server)
  sync/               # File watcher and sync daemon
  state/              # Memory file management (STATE.md, CONSTRAINTS.md, DECISIONS.md)
  
src/chinvex_mcp/
  server.py           # MCP server (HTTP client to gateway)

scripts/              # PowerShell automation
  start_mcp.ps1       # Start MCP server with token from P:\secrets
  backup.ps1          # Backup contexts and indexes
  morning_brief.ps1   # Generate morning brief and push to ntfy
  scheduled_sweep.ps1 # Scheduled sweep for auto-archiving
```

### Data Flow
1. **Ingestion**: Documents chunked → embeddings generated → stored in SQLite + ChromaDB
2. **Search**: Query embedded → FTS5 lexical search + ChromaDB vector search → scores normalized and blended → reranked (optional) → results
3. **Memory Files**: `docs/memory/` (STATE.md, CONSTRAINTS.md, DECISIONS.md) maintained by `chinvex update-memory`

### Key Design Patterns
- **Delete-then-insert for chunk updates**: When re-ingesting a document, old chunks are deleted from both SQLite and ChromaDB, then new chunks inserted
- **Shared connection per process**: `storage.py` uses a global singleton connection for SQLite to avoid contention
- **Lock file for ingestion**: `hybrid.db.lock` ensures only one ingest runs at a time
- **Path normalization**: All paths normalized (absolute, forward slashes, lowercase on Windows) to prevent duplicates
- **Index metadata**: `meta.json` tracks embedding provider/model/dimensions to prevent mixing incompatible embeddings

### Embedding Provider Selection
- **Default**: OpenAI (text-embedding-3-small, 1536 dimensions)
- **Precedence**: CLI flag → context config → environment variable → default
- **Switching providers**: Requires `--rebuild-index` to recreate index with new dimensions
- **Mixed providers across contexts**: P5 implementation refuses mixed-space search with clear error

### Context Configuration Schema
Each context has `context.json`:
```json
{
  "schema_version": 1,
  "name": "MyProject",
  "aliases": ["myproj"],
  "includes": {
    "repos": ["C:/Code/myproject"],
    "chat_roots": ["P:/ai_memory/chats/myproject"],
    "codex_session_roots": [],
    "note_roots": []
  },
  "index": {
    "sqlite_path": "P:/ai_memory/indexes/MyProject/hybrid.db",
    "chroma_dir": "P:/ai_memory/indexes/MyProject/chroma"
  },
  "weights": {
    "repo": 1.0,
    "chat": 0.8,
    "codex_session": 0.9,
    "note": 0.7
  },
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small"
  },
  "reranker": {
    "provider": "cohere",
    "model": "rerank-english-v3.0",
    "candidates": 20,
    "top_k": 5
  }
}
```

## Implementation Phases

The project follows a phased implementation spec:
- **P0-P2**: Core indexing, hybrid search, embedding providers
- **P3**: Archive tier, webhook notifications
- **P4**: Memory system (STATE.md, CONSTRAINTS.md, DECISIONS.md), OpenAI embeddings
- **P5**: Embedding integrity enforcement, memory file maintainer, morning brief overhaul, reranking
- **P5a-P5b**: Evaluation framework, strap integration
- **Specs location**: `/specs/P{0-5}*.md`

Refer to the highest phase spec for current work. P5 is the current active phase.

## Memory Files

Contexts can have a `docs/memory/` directory in each repository with:
- **STATE.md**: Current objective, active work, blockers (machine-managed, full regeneration)
- **CONSTRAINTS.md**: Infrastructure limits, rules, key facts (merge-only, LLM reasons about contradictions)
- **DECISIONS.md**: Append-only decision log with monthly sections + Recent rollup

**Maintainer**: `chinvex update-memory --context <name>` reads git log since last processed commit, analyzes specs/plans, updates memory files per contracts.

**Brief generation**: `chinvex brief --context <name>` includes STATE.md, specific CONSTRAINTS.md sections (## Infrastructure, ## Rules, ## Hazards), and DECISIONS.md Recent rollup + last 7 days.

## Testing Conventions

- Test files in `tests/` mirror `src/` structure
- Use pytest fixtures for temp directories and SQLite databases
- Prefer integration tests over mocking (test with real SQLite + ChromaDB when possible)
- E2E smoke test: `python scripts/e2e_smoke_p4.py`

## Common Pitfalls

- **FTS5 not available**: Ensure Python build has SQLite compiled with FTS5
- **Concurrency**: Only one ingest can run at a time (lock file enforced)
- **Provider switching**: Must use `--rebuild-index` when changing embedding providers
- **Path handling**: Use `normalized_path()` utility for cross-platform path normalization
- **ChromaDB batch limits**: OpenAI embedder batches up to 2048 texts per request; avoid exceeding this in single chunk set
- **Missing credentials**: OpenAI requires `OPENAI_API_KEY`; Ollama requires running service at configured host

## External Integrations

- **Claude Code hooks**: Installed during ingestion to `.claude/settings.json` in each repo
- **Gateway + Cloudflare Tunnel**: Production deployment uses `cloudflared tunnel` + FastAPI gateway
- **MCP**: Claude Desktop/Code integration via MCP server (HTTP client to gateway)
- **ntfy**: Morning brief push notifications
- **Strap**: External tool for context registration (integration in P5b)
