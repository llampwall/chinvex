# CLAUDE.md

## Project
Chinvex - hybrid retrieval engine for personal knowledge

## Implementation Spec
**Read README.md** for an overview of the how this currrently runs
**P0_IMPLEMENTATION_SPEC.md** Is the bible for this sprint. It defines all contracts, schemas, and behavior. Do not invent beyond it.

## Language
Python

## Structure
everything is in /src/chinvex besides the mcp server, which is in /src/chinvex_mcp

## Commands
- `npm run build` / `pip install -e .` (whatever applies)
- `npm test` / `pytest`

## Current State

**Artifacts**

* **Hybrid index** per “project”:

  * **SQLite FTS5** for lexical search (`hybrid.db`)
  * **Chroma** for vector search (`chroma/`)
* **Sources ingested**

  * `repo` (e.g. `C:\Code\streamside`)
  * `chat` exports (e.g. `P:\ai_memory\projects\Twitch\chats\*.json/.md`)
* **Embedding**

  * Ollama embedding model: `mxbai-embed-large`
  * Remote Ollama host supported (e.g. `http://skynet:11434`)
  * Chunk-size fallback exists for “input length exceeds context length” style failures.
* **CLI**

  * `chinvex ingest --config … [--ollama-host …]`
  * `chinvex search --config … "query" [--source repo|chat]`
* **Local MCP server wrapper**

  * Tools exposed (names may vary depending on the wrapper):

    * `chinvex_search`
    * `chinvex_get_chunk`
  * Verified: can do search → select a chunk_id → get_chunk → answer “using only that chunk”.

**Robustness patches already applied**

* SQLite WAL + retry/backoff on transient “disk I/O error”
* “single writer” ingest lock (`hybrid.db.lock`) to prevent concurrent ingest collisions (and to cover Chroma writes)

## What still hurts / why we care

  * “make it feel like memory,” not like “manual grep with extra steps”
  * reduced configuration friction
  * quality controls (grounding guarantees, citations, reranking)
  * time-awareness (chat exports often have missing timestamps)
  * **Codex session chats** are not ingested yet (and they matter a lot)

## Rules
- Follow the spec exactly
- Ask before adding dependencies
- Delete-then-insert for chunk upserts (see spec §5)