from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chinvex.config import AppConfig, ConfigError, load_config
from chinvex.embed import OllamaEmbedder
from chinvex.search import ScoredChunk, make_snippet, search_chunks
from chinvex.storage import Storage
from chinvex.vectors import VectorStore

try:
    from mcp.server.fastmcp import FastMCP, MCPError
except ImportError:  # pragma: no cover - fallback if MCPError name differs
    from mcp.server.fastmcp import FastMCP

    class MCPError(RuntimeError):
        pass


class ToolError(RuntimeError):
    pass


@dataclass
class IndexHandle:
    config_path: Path
    config: AppConfig
    storage: Storage
    vectors: VectorStore


@dataclass
class ServerState:
    handle: IndexHandle
    default_k: int
    default_min_score: float
    ollama_host_override: str | None


_INDEX_CACHE: dict[str, IndexHandle] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="chinvex MCP server (stdio)")
    parser.add_argument("--config", required=True, help="Path to chinvex config JSON")
    parser.add_argument("--ollama-host", help="Override Ollama host for query embeddings")
    parser.add_argument("--k", type=int, default=8, help="Default top-k for search")
    parser.add_argument("--min-score", type=float, default=0.30, help="Default minimum score filter")
    args = parser.parse_args()

    state = build_state(
        Path(args.config),
        default_k=args.k,
        default_min_score=args.min_score,
        ollama_host_override=args.ollama_host,
    )

    mcp = FastMCP("chinvex")

    @mcp.tool(name="chinvex_search")
    def chinvex_search(
        query: str,
        source: str | None = None,
        k: int | None = None,
        min_score: float | None = None,
        include_text: bool = False,
    ) -> list[dict]:
        try:
            return handle_search(
                state,
                query=query,
                source=source,
                k=k,
                min_score=min_score,
                include_text=include_text,
            )
        except ToolError as exc:
            raise MCPError(str(exc)) from exc

    @mcp.tool(name="chinvex_get_chunk")
    def chinvex_get_chunk(chunk_id: str) -> dict:
        try:
            return handle_get_chunk(state, chunk_id=chunk_id)
        except ToolError as exc:
            raise MCPError(str(exc)) from exc

    mcp.run()


def build_state(
    config_path: Path,
    *,
    default_k: int,
    default_min_score: float,
    ollama_host_override: str | None,
) -> ServerState:
    handle = _get_index_handle(config_path)
    return ServerState(
        handle=handle,
        default_k=default_k,
        default_min_score=default_min_score,
        ollama_host_override=ollama_host_override,
    )


def handle_search(
    state: ServerState,
    *,
    query: str,
    source: str | None,
    k: int | None,
    min_score: float | None,
    include_text: bool,
) -> list[dict]:
    source_value = source or "all"
    if source_value not in {"all", "repo", "chat"}:
        raise ToolError("source must be one of: repo, chat")
    effective_k = k if k is not None else state.default_k
    effective_min = min_score if min_score is not None else state.default_min_score

    embedder = OllamaEmbedder(
        state.ollama_host_override or state.handle.config.ollama_host,
        state.handle.config.embedding_model,
    )

    scored = search_chunks(
        state.handle.storage,
        state.handle.vectors,
        embedder,
        query,
        k=effective_k,
        min_score=effective_min,
        source=source_value,
    )
    return [format_result(state.handle.storage, item, include_text=include_text) for item in scored]


def handle_get_chunk(state: ServerState, *, chunk_id: str) -> dict:
    row = state.handle.storage.conn.execute(
        "SELECT * FROM chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        raise ToolError(f"Chunk not found: {chunk_id}")
    meta = json.loads(row["meta_json"]) if row["meta_json"] else {}
    doc = state.handle.storage.get_document(row["doc_id"])
    path = meta.get("path") or (doc["source_uri"] if doc else "")
    title = _title_from_row(row, path)
    return {
        "chunk_id": row["chunk_id"],
        "source_type": row["source_type"],
        "path": path,
        "title": title,
        "text": row["text"],
        "meta": meta,
    }


def format_result(storage: Storage, item: ScoredChunk, *, include_text: bool) -> dict:
    row = item.row
    meta = json.loads(row["meta_json"]) if row["meta_json"] else {}
    doc = storage.get_document(row["doc_id"])
    path = meta.get("path") or (doc["source_uri"] if doc else "")
    title = _title_from_row(row, path)
    result = {
        "score": item.score,
        "source_type": row["source_type"],
        "title": title,
        "path": path,
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "ordinal": row["ordinal"],
        "snippet": make_snippet(row["text"], limit=300),
        "meta": meta,
    }
    if include_text:
        result["text"] = row["text"]
    return result


def _title_from_row(row: Any, path: str) -> str:
    if row["source_type"] == "repo":
        return Path(path).name if path else row["doc_id"]
    meta = json.loads(row["meta_json"]) if row["meta_json"] else {}
    return meta.get("title") or row["doc_id"]


def _get_index_handle(config_path: Path) -> IndexHandle:
    key = str(config_path.resolve())
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    try:
        config = load_config(config_path)
    except (ConfigError, OSError) as exc:
        raise ToolError(f"Invalid config: {exc}") from exc

    db_path = config.index_dir / "hybrid.db"
    chroma_dir = config.index_dir / "chroma"
    if not db_path.exists() or not chroma_dir.exists():
        raise ToolError(
            "Index not found. Run `chinvex ingest --config <path>` before using the MCP server."
        )

    storage = Storage(db_path)
    storage.ensure_schema()
    vectors = VectorStore(chroma_dir)
    handle = IndexHandle(
        config_path=config_path,
        config=config,
        storage=storage,
        vectors=vectors,
    )
    _INDEX_CACHE[key] = handle
    return handle

