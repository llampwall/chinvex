# chinvex - chunked vector indexer

Hybrid retrieval index CLI (SQLite FTS5 + Chroma) powered by configurable embeddings. A personal knowledge base with grounded retrieval, automatic ingestion, and scheduled automation.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Context Management](#context-management)
- [Ingestion & Search](#ingestion--search)
- [Bootstrap & Automation](#bootstrap--automation)
- [Advanced Features](#advanced-features)
- [MCP Server](#mcp-server)
- [Gateway API](#gateway-api)
- [Technical Deep Dive](#technical-deep-dive)
- [Operations](#operations)
- [Troubleshooting](#troubleshooting)

## Quick Start

```powershell
# Install
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -e .

# Create a context and ingest a repo
chinvex ingest --context MyProject --repo C:\Code\myproject

# Search
chinvex search --context MyProject "your query"

# Install automation (scheduled sync, morning briefs)
chinvex bootstrap install
```

## Installation

### Prerequisites
- Python 3.12
- Ollama installed and running (for local embeddings)
- `ollama pull mxbai-embed-large` (or use OpenAI with `--embed-provider openai`)

### Install (venv required)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

### Verify Installation
```powershell
chinvex --help
chinvex status
```

## Core Concepts

### Architecture Overview

Chinvex is a hybrid retrieval system with multiple components:

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
├─────────────────────────────────────────────────────────┤
│  CLI Commands  │  MCP Server  │  Gateway API  │  Hooks  │
└────────┬────────┴──────┬───────┴───────┬───────┴────────┘
         │               │               │
         ├───────────────┴───────────────┤
         │      Context Registry          │
         │   (contexts/, indexes/)        │
         └───────────────┬────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │                                 │
    ┌────▼────┐                      ┌────▼─────┐
    │ SQLite  │ ◄────Hybrid────────► │  Chroma  │
    │  FTS5   │      Search          │  Vectors │
    │ (BM25)  │                      │ (Cosine) │
    └─────────┘                      └──────────┘

Background Services:
├─ Sync Daemon: File watcher for auto-ingestion
├─ Scheduled Sweep: Ensures services running (30 min)
└─ Morning Brief: Daily digest generation
```

### Components

**Context Registry**: Multi-project organization system
- Each context has its own index (SQLite + Chroma)
- Tracks repos, chats, codex sessions, notes
- Configurable weights and embedding providers

**Hybrid Search**: Combines two retrieval methods
- **FTS5 (60%)**: Keyword/lexical search using SQLite BM25
- **Vector (40%)**: Semantic search using embeddings
- **Optional Reranking**: Cross-encoder for relevance refinement

**Sync Daemon**: Background file watcher
- Monitors repos for changes
- Auto-ingests on git commits
- Reconciles sources from context registry

**Bootstrap System**: Production automation
- Scheduled sweep (every 30 min)
- Morning brief (daily digest)
- PowerShell profile integration
- Environment variable setup

### Data Flow

```
1. Ingest:    Repo/Chat → Chunking → FTS Index + Embeddings → Storage
2. Search:    Query → FTS + Vector → Score Blending → Rerank → Results
3. Retrieval: chunk_id → Full Text + Metadata → Citations
```

## Context Management

Chinvex uses a context registry for managing multiple projects.

### Create a Context

```powershell
chinvex context create MyProject
```

**Idempotent creation** (no-op if context exists):
```powershell
chinvex context create MyProject --idempotent
```

This creates:
- `P:\ai_memory\contexts\MyProject\context.json`
- `P:\ai_memory\indexes\MyProject\hybrid.db`
- `P:\ai_memory\indexes\MyProject\chroma\`

### Context Configuration

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
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small"
  },
  "weights": {
    "repo": 1.0,
    "chat": 0.8,
    "codex_session": 0.9,
    "note": 0.7
  },
  "ranking": {
    "recency_enabled": true,
    "recency_half_life_days": 90
  },
  "reranker": {
    "provider": "cohere",
    "model": "rerank-english-v3.0",
    "candidates": 20,
    "top_k": 5
  }
}
```

### Context Commands

**List contexts**:
```powershell
chinvex context list
chinvex context list --json  # JSON output for scripting
```

**Check if context exists**:
```powershell
chinvex context exists MyProject
# Exit code: 0 if exists, 1 if not
```

**Rename a context**:
```powershell
chinvex context rename OldName --to NewName
```
Renames directory, index, and updates all internal paths. No re-ingestion required.

**Remove a repo from context**:
```powershell
chinvex context remove-repo MyProject --repo C:\Code\old-repo
```
Removes path from `context.json`. Indexed chunks remain until `--rebuild-index`.

**Purge contexts completely**:
```powershell
# Purge one context (completely deletes the context)
chinvex context purge MyProject

# Purge all contexts (single confirmation prompt)
chinvex context purge --all
```

Completely deletes the entire context directory including:
- Context configuration (`context.json`)
- SQLite FTS5 index (`hybrid.db`)
- ChromaDB vector embeddings (`chroma/` directory)
- Index metadata (`meta.json`)
- Watch history log (`watch_history.jsonl`)
- Digest cache (`.digests/` directory)

Use this when you want to completely remove a context. You'll need to re-register the context (e.g., with `strap adopt`) to use it again.

### Archiving Contexts

Archive an existing context to the `_archive` table of contents:

```powershell
chinvex context archive MyOldProject
```

This extracts a description, adds an entry to `_archive`, then deletes the full context and index. The `_archive` context acts as a lightweight catalog - just name and description, making the system aware of what exists without holding full index data.

**Description extraction fallback chain**:
1. `docs/memory/STATE.md` → "Current Objective" line
2. `README.md` → first non-empty paragraph

**Archive an unmanaged directory**:
```powershell
# With explicit description
chinvex archive-unmanaged --name oldrepo --dir P:\software\oldrepo --desc "An old experiment"

# Auto-detect description from README
chinvex archive-unmanaged --name oldrepo --dir P:\software\oldrepo
```

## Ingestion & Search

### Ingest with Context

**Basic ingestion** (reads paths from context.json):
```powershell
chinvex ingest --context MyProject
```

**Inline context creation** (auto-creates context if missing):
```powershell
# Add a repo - creates context if needed
chinvex ingest --context MyProject --repo C:\Code\myproject

# Add a chat root
chinvex ingest --context MyProject --chat-root P:\ai_memory\chats\myproject

# Combine multiple sources
chinvex ingest --context MyProject --repo C:\Code\myproject --chat-root P:\ai_memory\chats\myproject
```

**Path normalization**: All paths are automatically normalized (absolute, forward slashes, lowercase on Windows) and deduplicated.

### Ingestion Flags

- `--embed-provider {ollama|openai}`: Choose embedding provider (default: ollama)
- `--rebuild-index`: Force full rebuild instead of incremental update
- `--no-write-context`: Prevent auto-creation of missing contexts (fail instead)
- `--no-claude-hook`: Skip automatic Claude Code hook installation
- `--register-only`: Add paths to context.json without running ingestion

**Register-only mode** (add paths without ingesting):
```powershell
chinvex ingest --context MyProject --repo C:\Code\myproject \
  --register-only \
  --chinvex-depth full \
  --status active \
  --tags python,web
```
Creates context if missing, adds paths with metadata, but skips embedding/indexing. Useful for external tooling or deferred ingestion.

**Metadata fields:**
- `--chinvex-depth`: Ingestion depth (default: `full`)
  - `full`: Deep ingestion - all files, detailed parsing
  - `light`: Lightweight - skip large files, basic parsing
  - `index`: Minimal - file tree only, no content
- `--status`: Lifecycle state (default: `active`)
  - `active`: Under active development
  - `stable`: Stable, infrequent changes
  - `dormant`: Archived, rarely accessed
- `--tags`: Comma-separated tags for grouping (default: none)
  - Example: `python,ml,third-party`

### Strap Integration

**If you use [strap](https://github.com/llampwall/_strap) for dev environment management, you get automatic chinvex integration.**

Strap's registry becomes the source of truth - when you manage repos with strap, chinvex contexts are created and synchronized automatically:

- `strap clone <url>` → Registers repo in chinvex with metadata
- `strap adopt <path>` → Adds to chinvex context with metadata
- `strap move <name>` → Updates chinvex path
- `strap rename <name>` → Updates chinvex context name
- `strap uninstall <name>` → Archives chinvex context

**Metadata-based classification:**

All repos live in `P:\software` with rich metadata for flexible classification:

- **chinvex_depth**: Controls how deeply each repo is ingested
  - Software projects typically use `full`
  - Third-party libraries might use `light`
  - Archived projects might use `index`
- **status**: Tracks lifecycle state (`active`, `stable`, `dormant`)
- **tags**: Freeform grouping (e.g., `["python", "ml", "third-party"]`)

Strap passes these fields when registering repos with chinvex using `--register-only` mode.

**Strap verification commands:**
```powershell
# View chinvex contexts and sync status
strap contexts

# Preview registry/chinvex drift
strap sync-chinvex

# Fix any drift between registry and chinvex
strap sync-chinvex --reconcile
```

This integration uses the `--register-only` flag internally to manage context registration without triggering ingestion. Actual embedding/indexing happens separately (via scheduled sync daemon, manual ingest, or bootstrap automation).

For full details, see the [Chinvex Integration section](../../../_strap/README.md#chinvex-integration) in strap's README.

**Claude Code integration**: By default, ingestion installs a `SessionStart` hook in `.claude/settings.json` that runs `chinvex brief --context <name>` when you open the repo in Claude Code.

**Provider switching**: If you switch embedding providers, you MUST use `--rebuild-index`:
```powershell
chinvex ingest --context MyProject --embed-provider openai --rebuild-index
```

### Embedding Providers

**Ollama (default)**:
```powershell
chinvex ingest --context MyProject
```
- Model: `mxbai-embed-large` (1024 dimensions)
- Free, runs locally
- Requires Ollama running

**OpenAI**:
```powershell
chinvex ingest --context MyProject --embed-provider openai
```
- Model: `text-embedding-3-small` (1536 dimensions)
- Fast, reliable, requires API key
- Batching (up to 2048 texts per request)
- Automatic retry with exponential backoff

Set `OPENAI_API_KEY` environment variable for OpenAI.

**Provider precedence**:
1. CLI flag (`--embed-provider`)
2. Context config (`embedding.provider` in context.json)
3. Environment variable (`CHINVEX_EMBED_PROVIDER`)
4. Default (ollama with mxbai-embed-large)

**Dimension safety**: Chinvex prevents mixing embeddings with different dimensions. Switching providers requires `--rebuild-index`.

### Search

**Basic search**:
```powershell
chinvex search --context MyProject "your query"
```

**Search with reranking** (see Reranking section):
```powershell
chinvex search --context MyProject --rerank "your query"
```

**Cross-context search**:
```powershell
chinvex search --context all "your query"
```

**Search parameters**:
- `--k`: Number of results (default: 8)
- `--min-score`: Minimum relevance score (default: 0.35)
- `--rerank`: Enable reranking (if configured)

### Ingestion Tracking

**Ingest run logging**: Every run is logged to `{index_dir}/ingest_runs.jsonl`:
```json
{
  "run_id": "uuid",
  "started_at": "2026-01-29T12:34:56Z",
  "completed_at": "2026-01-29T12:35:12Z",
  "sources": ["C:/Code/myproject"],
  "status": "success",
  "error": null
}
```

**Index metadata**: `meta.json` tracks provider/model/dimensions:
```json
{
  "schema_version": 1,
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-small",
  "embedding_dimensions": 1536,
  "created_at": "2026-01-29T12:34:56Z"
}
```

## Bootstrap & Automation

The bootstrap system sets up production automation for hands-free operation.

### Install Bootstrap

```powershell
chinvex bootstrap install
```

This configures:
1. **ChinvexSweep scheduled task** (every 30 min)
   - Ensures sync daemon is running
   - Reconciles sources from contexts
   - Logs to sync daemon output

2. **ChinvexMorningBrief scheduled task** (daily at configured time)
   - Generates digest for last 24 hours
   - Pushes to ntfy (if configured)

3. **PowerShell profile integration**
   - Adds chinvex to PATH
   - Sets environment variables

4. **Environment variables**
   - `CHINVEX_CONTEXTS_ROOT`: Contexts directory
   - `CHINVEX_INDEXES_ROOT`: Indexes directory
   - `CHINVEX_APPSERVER_URL`: App-server URL for Codex sessions

### Check Bootstrap Status

```powershell
chinvex bootstrap status
```

Shows:
- Watcher daemon status
- Scheduled task installation
- Environment variable configuration
- Last sweep/brief run times

### Uninstall Bootstrap

```powershell
chinvex bootstrap uninstall
```

Removes:
- Scheduled tasks
- PowerShell profile entries
- Does NOT delete contexts or indexes

### Sync Daemon (File Watcher)

The sync daemon monitors repos for changes and auto-ingests.

**Start daemon**:
```powershell
chinvex sync start
```

**Stop daemon**:
```powershell
chinvex sync stop
```

**Check status**:
```powershell
chinvex sync status
```

**Ensure running** (idempotent):
```powershell
chinvex sync ensure-running
```
Starts daemon if not running, no-op if already running. Used by scheduled sweep.

**Reconcile sources**:
```powershell
chinvex sync reconcile-sources
```
Updates watcher to monitor all repos from all contexts. Restarts daemon to apply changes.

**How it works**:
- Monitors git repos for `.git/refs/heads/` changes
- Triggers incremental ingest on commit
- Logs to `~/.chinvex/watcher.log`
- PID file: `~/.chinvex/watcher.pid`

## Advanced Features

### Reranking

Chinvex supports optional two-stage retrieval:

1. **Stage 1**: Hybrid search returns top N candidates (default: 20)
2. **Stage 2**: Cross-encoder reranks to top K (default: 5)

#### Enable Reranking

**Per-query (CLI flag)**:
```powershell
chinvex search --context MyProject --rerank "query"
```

**Always-on (context.json)**:
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

#### Reranker Providers

**Cohere (Recommended)**:
- Provider: `cohere`
- Model: `rerank-english-v3.0`
- Setup: Set `COHERE_API_KEY` environment variable
- Get API key: https://cohere.com/
- Latency: ~200-500ms for 20 candidates
- Quality: Excellent

**Jina**:
- Provider: `jina`
- Model: `jina-reranker-v1-base-en`
- Setup: Set `JINA_API_KEY` environment variable
- Get API key: https://jina.ai/
- Latency: ~300-600ms for 20 candidates
- Quality: Good

**Local Cross-Encoder**:
- Provider: `local`
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Setup: No API key required (downloads to `~/.cache/huggingface/`)
- Latency: ~1-2s for 20 candidates (slower but free)
- Quality: Good
- Model size: ~400MB

#### Budget Guardrails

- **Minimum candidates**: Reranker only runs if ≥10 candidates
- **Maximum candidates**: Truncates to 50 (prevents excessive latency/cost)
- **Latency budget**: 2s max for rerank step (warning if exceeded)
- **Fallback**: Returns pre-rerank results with warning if reranker fails

### Watch Queries

Watch queries provide ongoing monitoring of search results for specific queries.

**Add a watch**:
```powershell
chinvex watch add --context MyProject --query "bug fix" --notify-on change
```

**List watches**:
```powershell
chinvex watch list --context MyProject
```

**Remove a watch**:
```powershell
chinvex watch remove --context MyProject --id <watch-id>
```

**View watch history**:
```powershell
chinvex watch history --context MyProject --id <watch-id>
```

Watches track result changes over time and can trigger notifications when results change.

### State Management

Chinvex can generate and manage `STATE.md` files for project context.

**Generate state files**:
```powershell
chinvex state generate --context MyProject
```

Generates:
- `state.json`: Machine-readable state
- `STATE.md`: Human-readable state document

**Show current state**:
```powershell
chinvex state show --context MyProject
```

**Add state note**:
```powershell
chinvex state note --context MyProject "Working on authentication bug"
```

### Project Memory System

Chinvex provides an automated project memory system that gives Claude Code rich context about your repos. When you open a repo in Claude Code, chinvex automatically generates a session brief from structured memory files.

#### How It Works

1. **SessionStart Hook**: When you run `chinvex ingest`, it installs a hook in `.claude/settings.json`:
   ```json
   {
     "hooks": {
       "SessionStart": {
         "command": "chinvex brief --context MyProject"
       }
     }
   }
   ```

2. **Brief Generation**: When Claude Code opens the repo, it runs `chinvex brief` which reads:
   - `docs/memory/STATE.md` - Current objective, active work, blockers, next actions
   - `docs/memory/CONSTRAINTS.md` - Infrastructure facts, rules, hazards
   - `docs/memory/DECISIONS.md` - Recent decisions (last 7 days)
   - Latest digest and watch history

3. **Uninitialized Detection**: If memory files don't exist or contain bootstrap templates, the brief includes an "ACTION REQUIRED" warning instructing Claude to run `/update-memory`.

4. **Memory Population**: The `/update-memory` skill (available in Claude Code) analyzes git history and generates meaningful STATE/CONSTRAINTS/DECISIONS content from commits.

5. **Subsequent Sessions**: Future sessions get real project context instead of empty templates.

#### Memory File Structure

Create `docs/memory/` in your repos with these files:

**STATE.md** (rewritten on each update):
- Current Objective
- Active Work
- Blockers
- Next Actions
- Quick Reference (run/test commands)
- Out of Scope (deferred items)

**CONSTRAINTS.md** (merge-only, never deleted):
- Infrastructure (ports, paths, technical limits)
- Rules (invariants, "don't do X because Y")
- Key Facts (URLs, env var names, commands)
- Hazards (things that bite you)
- Superseded (obsolete constraints)

**DECISIONS.md** (append-only decision log):
- Recent rollup (last 30 days summary)
- Dated entries (### YYYY-MM-DD — Title)
- Bug fix format (Symptom/Root cause/Fix/Prevention)
- Evidence (commit hashes)

#### Using the Memory System

**Let Claude populate memory files** (recommended):
1. Open repo in Claude Code
2. See "ACTION REQUIRED" warning in brief
3. Tell Claude: "run /update-memory"
4. Claude analyzes git history and generates memory files
5. Future sessions get rich context automatically

**Manual creation**:
```powershell
# Bootstrap empty templates (not recommended - let Claude populate them)
chinvex ingest --context MyProject --repo C:\Code\myproject
# This creates bootstrap templates in docs/memory/
```

**Skip hook installation** (if you don't want SessionStart integration):
```powershell
chinvex ingest --context MyProject --no-claude-hook
```

For implementation details, see [PROJECT_MEMORY_SPEC_v0.3.md](docs/PROJECT_MEMORY_SPEC_v0.3.md) and the `/update-memory` skill in `skills/update-memory/SKILL.md`.

### Git Hooks

Chinvex can install git hooks to automate ingestion and state updates.

**Install hooks in current repo**:
```powershell
cd C:\Code\myproject
chinvex hook install
```

Installs:
- `post-commit`: Triggers incremental ingest after commits
- Updates state tracking

**Check hook status**:
```powershell
chinvex hook status
```

**Uninstall hooks**:
```powershell
chinvex hook uninstall
```

Hooks work alongside the sync daemon for comprehensive automation.

### Digest & Brief

Generate daily digests and session briefs from indexed content.

#### Digest

```powershell
# Generate digest for last 24 hours
chinvex digest generate --context MyProject --since 24h

# Generate for specific date
chinvex digest generate --context MyProject --date 2026-01-28

# Push notification to ntfy
chinvex digest generate --context MyProject --push ntfy
```

Digest output includes:
- Recent document changes
- New commits and their summaries
- Chat activity
- Markdown and JSON formats

#### Brief

```powershell
# Generate session brief (printed to stdout)
chinvex brief --context MyProject

# Save to file
chinvex brief --context MyProject --output SESSION_BRIEF.md
```

Brief includes:
- Current state from STATE.md
- Recent changes since last session
- Active work and blockers
- Relevant context for resuming work

Claude Code integration automatically runs `chinvex brief` when opening repos via `SessionStart` hook.

### Evaluation

Run evaluation suite against golden queries:

```powershell
chinvex eval --context MyProject --queries queries.jsonl
```

Queries file format:
```jsonl
{"query": "how does auth work?", "expected_docs": ["auth.py"], "min_score": 0.6}
{"query": "database schema", "expected_docs": ["schema.sql", "models.py"], "min_score": 0.5}
```

Outputs metrics:
- Precision, recall, F1 score
- Mean reciprocal rank (MRR)
- NDCG (normalized discounted cumulative gain)
- Latency statistics

## MCP Server

The Chinvex MCP server is an **HTTP client** that connects to the gateway API, exposing Chinvex memory search to Claude Desktop/Code via the MCP protocol.

### Configuration

Set environment variables:
- `CHINVEX_URL`: Gateway URL (default: `https://chinvex.yourdomain.com`)
- `CHINVEX_API_TOKEN`: Bearer token for authentication (required)

**Using the runbook script** (recommended):
```powershell
.\scripts\start_mcp.ps1
```
Loads token from `P:\secrets\chinvex_mcp_token.txt` and starts the server.

**Manual startup**:
```powershell
$env:CHINVEX_API_TOKEN = "your-token-here"
$env:CHINVEX_URL = "https://chinvex.yourdomain.com"
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
        "CHINVEX_URL": "https://chinvex.yourdomain.com",
        "CHINVEX_API_TOKEN": "your-token-here"
      }
    }
  }
}
```

### Available Tools

**`chinvex_search`**: Search personal knowledge base with grounded retrieval
- Returns evidence chunks with citations from indexed sources
- Supports single context or cross-context search (`contexts="all"`)
- Parameters: `query` (required), `contexts` (default: "all"), `k` (default: 8)

**`chinvex_list_contexts`**: List all available contexts
- Returns context names, aliases, and last update times
- No parameters required

**`chinvex_get_chunks`**: Retrieve full chunk content by ID
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

**Get full chunks**:
```json
{
  "context": "MyProject",
  "chunk_ids": ["abc123def456", "789ghi012jkl"]
}
```

## Gateway API

The gateway provides an HTTP API for search, evidence retrieval, and context management.

### Starting the Gateway

**Development (manual)**:

Terminal 1 - Cloudflare Tunnel:
```powershell
cloudflared tunnel --protocol http2 run chinvex-gateway
```

Terminal 2 - Gateway Server:
```powershell
python -m chinvex.cli gateway serve --port 7778
```

**Production**: Use PM2 or systemd. See [Cloudflare Tunnel Setup](docs/deployment/cloudflare-tunnel.md) and [Caddy Reverse Proxy](docs/deployment/caddy.md).

### Gateway Commands

**Start server**:
```powershell
chinvex gateway serve --port 7778 --host 0.0.0.0
```

**Generate API token**:
```powershell
chinvex gateway token-generate
```

**Rotate API token**:
```powershell
chinvex gateway token-rotate
```
Generates new token, displays both old and new for graceful migration.

**Check gateway status**:
```powershell
chinvex gateway status
```

### API Endpoints

**Health check**:
```bash
curl http://localhost:7778/health
```

**Search**:
```bash
curl -X POST http://localhost:7778/search \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"query": "your query", "contexts": ["MyProject"], "k": 8}'
```

**List contexts**:
```bash
curl http://localhost:7778/contexts \
  -H "Authorization: Bearer your-token"
```

**Get chunks**:
```bash
curl -X POST http://localhost:7778/chunks \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"context": "MyProject", "chunk_ids": ["abc123"]}'
```

For full API documentation, see [ChatGPT Integration](docs/chatgpt-integration.md) and OpenAPI spec at `docs/chinvex_openapi_v0.3.0.json`.

### Authentication

The gateway requires bearer token authentication. Generate tokens with:
```powershell
chinvex gateway token-generate
```

Tokens are stored in gateway configuration and can be rotated without downtime.

## Technical Deep Dive

### Hybrid Search Architecture

Chinvex combines two complementary search methods:

```
User Query
    ↓
[FTS Search] ────→ Top 30 results (exact keyword matches, BM25)
    ↓
[Vector Search] ─→ Top 30 results (semantic meaning, cosine similarity)
    ↓
[Merge & Normalize] ─→ Combine candidates by chunk_id
    ↓
[Score Blending] ───→ 60% FTS + 40% Vector (weighted average)
    ↓
[Source Weighting] ─→ Apply context priorities (repo: 1.0, chat: 0.8)
    ↓
[Optional Reranking] → Cross-encoder for final ordering
    ↓
[Optional Recency] ──→ Exponential decay (90-day half-life)
    ↓
Top K Results
```

### Score Blending

**Normalization**: Both FTS and vector scores are min-max normalized to [0,1] within the candidate set.

**Blending formula**:
```
final_score = 0.6 * fts_normalized + 0.4 * vector_normalized
```

**Smart fallback**:
- If only FTS has results: Use 100% of FTS signal
- If only vector has results: Use 100% of vector signal
- If both: Apply 60/40 weighting

**Why 60/40?**
- FTS excels at: Exact terms, technical identifiers, acronyms
- Vector excels at: Conceptual similarity, paraphrasing, intent
- 60/40 favors precision (FTS) while benefiting from semantic understanding

### Source Type Weighting

Configurable per-context in `context.json`:
```json
{
  "weights": {
    "repo": 1.0,      // Code/docs get full weight
    "chat": 0.8,      // Chat history slightly downweighted
    "codex_session": 0.9,
    "note": 0.7       // Personal notes least weighted
  }
}
```

Applied after blending: `rank_score = blended_score * source_weight`

### Recency Decay

Optional temporal relevance boosting based on document age:

```json
{
  "ranking": {
    "recency_enabled": true,
    "recency_half_life_days": 90
  }
}
```

**Formula**: `decay_factor = 0.5 ^ (age_days / half_life_days)`

**Applied as**: `final_score = rank_score * decay_factor`

Downranks stale content without eliminating it. 90-day half-life means documents lose 50% of their score every 3 months.

### Storage Architecture

**SQLite (hybrid.db)**:
- `documents` table: Metadata about sources
- `chunks` table: Text chunks with metadata, chunk_key for deduplication
- `chunks_fts` virtual table: FTS5 index for full-text search
- `source_fingerprints`: Ingestion tracking for change detection
- `ingestion_runs`: Audit trail

**Chroma (vector store)**:
- Persistent vector database in `chroma/` directory
- Embeddings stored with metadata for filtering
- Batch operations (5000 items max per batch)

**Index metadata** (`meta.json`):
- Tracks embedding provider, model, dimensions
- Prevents mixing incompatible embeddings
- Created on first ingest, validated on subsequent runs

### Chunking Strategy

**Code/Docs**:
- Max chunk size: 3000 characters
- Overlap: 200 characters
- Preserves function/class boundaries when possible

**Chat/Conversations**:
- Chunked by message groups
- Preserves conversation context
- Metadata includes speaker, timestamps

**Chunk reuse**: Identical chunks (same text, same source type) are deduplicated via `chunk_key` hash. Embeddings are computed once and reused.

### Performance Characteristics

**Search latency**:
- FTS query: ~10-50ms
- Vector query: ~50-200ms (depends on collection size)
- Score blending: <5ms
- Reranking (if enabled): +200-2000ms

**Ingestion throughput**:
- Local embeddings (Ollama): ~100-500 docs/min
- OpenAI embeddings: ~1000-5000 docs/min (batching)
- SQLite writes: ~5000-10000 chunks/sec

**Scalability limits**:
- Tested with: 100k+ chunks, 50+ contexts
- Chroma batch size: 5000 items (internal chunking)
- OpenAI batch size: 2048 texts per request

## Operations

### Environment Variables

- `CHINVEX_CONTEXTS_ROOT`: Override contexts directory (default: `P:\ai_memory\contexts`)
- `CHINVEX_INDEXES_ROOT`: Override indexes directory (default: `P:\ai_memory\indexes`)
- `CHINVEX_APPSERVER_URL`: App-server URL for Codex session ingestion (default: `http://localhost:8080`)
- `CHINVEX_EMBED_PROVIDER`: Default embedding provider (ollama or openai)
- `OPENAI_API_KEY`: OpenAI API key for embeddings
- `COHERE_API_KEY`: Cohere API key for reranking
- `JINA_API_KEY`: Jina API key for reranking

### Runbook Scripts

Located in `scripts/`:

**Start MCP server** (`start_mcp.ps1`):
```powershell
.\scripts\start_mcp.ps1
```
Loads token from `P:\secrets\chinvex_mcp_token.txt` and starts MCP server.

**Backup contexts and indexes** (`backup.ps1`):
```powershell
.\scripts\backup.ps1
```
Snapshots all contexts and indexes to `P:\backups\chinvex\YYYY-MM-DD_HHMMSS\`.

**Morning brief** (`morning_brief.ps1`):
```powershell
.\scripts\morning_brief.ps1
```
Generates digest for last 24 hours across all contexts, pushes to ntfy.

**Scheduled sweep** (`scheduled_sweep.ps1`):
```powershell
.\scripts\scheduled_sweep.ps1
```
Ensures sync daemon running, reconciles sources. Called by scheduled task every 30 min.

### System Status

**Global status**:
```powershell
chinvex status
```

Shows:
- Contexts count and last update times
- Sync daemon status
- Scheduled task status
- Index sizes and document counts

### Metrics

Chinvex exports Prometheus metrics for monitoring:

**Embedding metrics**:
- `chinvex_embeddings_total{provider, model, operation}`: Counter of embedding requests
- `chinvex_embeddings_latency_seconds{provider, model, operation}`: Histogram of latency
- `chinvex_embeddings_retries_total{provider, model}`: Counter of retry attempts

**Digest metrics**:
- `chinvex_digest_generated_total{context}`: Counter of digest generations

**Brief metrics**:
- `chinvex_brief_generated_total{context}`: Counter of brief generations

**Export metrics** (example):
```powershell
python -c "from prometheus_client import start_http_server; from chinvex.embedding_providers import EMBEDDINGS_TOTAL; start_http_server(9090); import time; time.sleep(3600)"
```

### E2E Smoke Tests

Comprehensive smoke tests validate the full system:

```powershell
# Run all smoke tests
python scripts/e2e_smoke_p0.py  # Basic ingest/search
python scripts/e2e_smoke_p1.py  # Context management
python scripts/e2e_smoke_p2.py  # Gateway API
python scripts/e2e_smoke_p3.py  # MCP server
python scripts/e2e_smoke_p4.py  # Digest/brief
```

Or run the simple wrapper:
```powershell
.\tests\run_smoke_test.ps1
```

Tests validate:
- Context creation and configuration
- Inline repo ingestion
- Hybrid search accuracy
- Digest generation (markdown + JSON)
- Brief generation
- Gateway endpoints
- MCP tool calls
- Cleanup and teardown

## Troubleshooting

### Common Issues

**FTS5 missing**:
- Install a Python build with SQLite FTS5 enabled
- Check: `python -c "import sqlite3; print(sqlite3.connect(':memory:').execute('pragma compile_options').fetchall())"`
- Look for `ENABLE_FTS5` in output

**Ollama connection/model missing**:
- Ensure Ollama is running: `ollama list`
- Pull model: `ollama pull mxbai-embed-large`
- Check host: `$env:OLLAMA_HOST = "http://127.0.0.1:11434"`

**Provider switching errors**:
```
ValueError: Provider mismatch: existing index has 1024 dimensions (ollama/mxbai-embed-large),
new provider has 1536 dimensions (openai/text-embedding-3-small)
```
**Solution**: Use `--rebuild-index` when switching providers:
```powershell
chinvex ingest --context MyProject --embed-provider openai --rebuild-index
```

**OpenAI API errors**:
- `401 Unauthorized`: Set `OPENAI_API_KEY` environment variable
- Rate limit errors: Provider automatically retries with exponential backoff (3 attempts)
- Context length errors: Text is automatically split into smaller chunks

**Windows path issues**:
- Use escaped backslashes in JSON: `"C:\\Code\\repo"`
- Or use forward slashes: `"C:/Code/repo"`
- Path normalization handles this automatically for CLI flags

**Concurrency / Lock errors**:
- Only one ingest should run at a time per context
- Lock file: `hybrid.db.lock`
- If stuck: Stop all ingest processes, delete lock file manually

**Duplicate paths**:
- Path normalization prevents duplicates
- If duplicates exist: Edit `context.json` directly and remove

**Sync daemon not starting**:
- Check logs: `~/.chinvex/watcher.log`
- Check PID file: `~/.chinvex/watcher.pid`
- Kill stale process: `Stop-Process -Id (Get-Content ~/.chinvex/watcher.pid)`
- Restart: `chinvex sync start`

**Scheduled tasks not running**:
- Check task exists: `Get-ScheduledTask -TaskName "ChinvexSweep"`
- Check last run: `Get-ScheduledTaskInfo -TaskName "ChinvexSweep"`
- Run manually: `Start-ScheduledTask -TaskName "ChinvexSweep"`
- View logs: Check sync daemon and task scheduler logs

**Gateway authentication errors**:
- Ensure token is valid: `chinvex gateway token-generate`
- Check token in request: `Authorization: Bearer <token>`
- Verify gateway is running: `curl http://localhost:7778/health`

**Reranker failures**:
- Check API keys are set: `COHERE_API_KEY`, `JINA_API_KEY`
- For local reranker: Check disk space for model cache (~400MB)
- System falls back to pre-rerank results with warning

### Debug Mode

Enable verbose logging:
```powershell
$env:CHINVEX_DEBUG = "1"
chinvex ingest --context MyProject
```

Logs include:
- Embedding batch sizes and timings
- Search query execution details
- Score blending calculations
- Reranker decisions

### Getting Help

- Check logs: `~/.chinvex/watcher.log`, task scheduler logs
- Run system status: `chinvex status`, `chinvex bootstrap status`
- Test components: Run E2E smoke tests
- File issues: https://github.com/yourusername/chinvex/issues

---

## Legacy Config (Deprecated)

The old config file format is still supported but deprecated. Use `--context` instead.

<details>
<summary>Click to expand legacy config documentation</summary>

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

On first use with `--config`, chinvex will auto-migrate to a new context and suggest using `--context` going forward.

</details>
