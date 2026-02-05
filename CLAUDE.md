# CLAUDE.md

## Project

Chinvex - hybrid retrieval engine for personal knowledge management.

## Language

Python 3.12

## Structure

- `/src/chinvex` - core library and CLI
- `/src/chinvex_mcp` - MCP server for Claude integration
- `/specs` - implementation specs (P0-P5)
- `/scripts` - PowerShell automation (bootstrap, scheduled tasks)
- `/tests` - pytest test suite

## Commands

```bash
pip install -e .          # Install in dev mode
pytest                    # Run tests
chinvex --help            # CLI help
```

## Key CLI Commands

```bash
chinvex ingest --context <name>              # Ingest sources into context
chinvex search --context <name> "query"      # Search a context
chinvex brief --context <name>               # Generate session brief
chinvex status                               # Show all contexts status
chinvex sync start                           # Start file watcher daemon
chinvex context purge <name>                 # Purge index/embedding data for one context
chinvex context purge --all                  # Purge ALL contexts (single confirmation)
```

## Current Sprint

See `/specs/` for implementation specs. Look at the highest phase number (P0, P1, etc.) for current work.

## Architecture

- **Hybrid index**: SQLite FTS5 (lexical) + ChromaDB (vector)
- **Embeddings**: OpenAI text-embedding-3-small (default)
- **Gateway**: HTTP API for search, served via PM2 + cloudflared tunnel
- **Automation**: File watcher daemon + scheduled sweep + git hooks

## Memory System

Chinvex repos use structured memory files in `docs/memory/`:

- **STATE.md**: Current objective, active work, blockers, next actions
- **CONSTRAINTS.md**: Infrastructure facts, rules, hazards (merge-only)
- **DECISIONS.md**: Append-only decision log with dated entries

**SessionStart Integration**: When you open a chinvex-managed repo, a hook runs `chinvex brief --context <name>` to load project context.

**If memory files are uninitialized** (empty or bootstrap templates), the brief will show "ACTION REQUIRED" instructing you to run `/update-memory`.

**The /update-memory skill** analyzes git history and populates memory files with:
- Current state from recent commits
- Constraints learned from bugs/infrastructure
- Decisions with evidence (commit hashes)

See `skills/update-memory/SKILL.md` and `docs/PROJECT_MEMORY_SPEC_v0.3.md` for details.

## Rules

- Follow the spec exactly
- Ask before adding dependencies
- Delete-then-insert for chunk upserts
- OpenAI embeddings by default (Ollama available via `--embed-provider ollama`)
- When opening a repo, check if brief shows "ACTION REQUIRED" - if so, offer to run `/update-memory`
