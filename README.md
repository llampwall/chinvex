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

## Config
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

## Run
```powershell
chinvex ingest --config .\config.json
chinvex search --config .\config.json "your query"
```

## MCP Server
Run the local MCP server over stdio:
```powershell
chinvex-mcp --config .\config.json
```

Optional overrides:
```powershell
chinvex-mcp --config .\config.json --ollama-host http://skynet:11434 --k 8 --min-score 0.30
```

### Cursor / Claude Desktop MCP config
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

## Troubleshooting
- FTS5 missing: install a Python build with SQLite FTS5 enabled.
- Ollama connection/model missing: ensure Ollama is running and `ollama pull mxbai-embed-large` completed.
- Windows path issues: use escaped backslashes in JSON or forward slashes.

## Smoke Test
```powershell
chinvex ingest --config path\to\config.json
chinvex search --config path\to\config.json "known token"
```
Expected: ingest creates `<index_dir>\hybrid.db` and `<index_dir>\chroma`, and search returns results.
