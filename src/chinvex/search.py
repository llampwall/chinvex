from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .embed import OllamaEmbedder
from .scoring import normalize_scores, blend_scores
from .storage import Storage
from .vectors import VectorStore


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    source_type: str
    title: str
    citation: str
    snippet: str


@dataclass
class ScoredChunk:
    chunk_id: str
    score: float
    row: Any


def search(
    config: AppConfig,
    query: str,
    *,
    k: int = 8,
    min_score: float = 0.35,
    source: str = "all",
    project: str | None = None,
    repo: str | None = None,
    ollama_host_override: str | None = None,
    weights: dict[str, float] | None = None,
) -> list[SearchResult]:
    db_path = config.index_dir / "hybrid.db"
    chroma_dir = config.index_dir / "chroma"
    storage = Storage(db_path)
    storage.ensure_schema()

    filters = {}
    if source in {"repo", "chat"}:
        filters["source_type"] = source
    if project:
        filters["project"] = project
    if repo:
        filters["repo"] = repo

    ollama_host = ollama_host_override or config.ollama_host
    embedder = OllamaEmbedder(ollama_host, config.embedding_model)
    vectors = VectorStore(chroma_dir)
    scored = search_chunks(
        storage,
        vectors,
        embedder,
        query,
        k=k,
        min_score=min_score,
        source=source,
        project=project,
        repo=repo,
        weights=weights,
    )
    results = [
        SearchResult(
            chunk_id=item.chunk_id,
            score=item.score,
            source_type=item.row["source_type"],
            title=_title_from_row(item.row),
            citation=_citation_from_row(item.row),
            snippet=make_snippet(item.row["text"], limit=200),
        )
        for item in scored
    ]
    storage.close()
    return results[:k]


def search_chunks(
    storage: Storage,
    vectors: VectorStore,
    embedder: OllamaEmbedder,
    query: str,
    *,
    k: int = 8,
    min_score: float = 0.35,
    source: str = "all",
    project: str | None = None,
    repo: str | None = None,
    weights: dict[str, float] | None = None,
) -> list[ScoredChunk]:
    """
    Hybrid search with score normalization and weight renormalization.

    Args:
        weights: Optional source-type weights dict (e.g., {"repo": 1.0, "chat": 0.8})
                 If None, no weight adjustment applied.
    """
    filters = {}
    if source in {"repo", "chat", "codex_session"}:
        filters["source_type"] = source
    if project:
        filters["project"] = project
    if repo:
        filters["repo"] = repo

    # Perform FTS search
    lex_rows = storage.search_fts(query, limit=30, filters=filters)

    # Perform vector search
    where = {}
    if source in {"repo", "chat", "codex_session"}:
        where["source_type"] = source
    if project:
        where["project"] = project
    if repo:
        where["repo"] = repo

    query_embedding = embedder.embed([query])
    vec_result = vectors.query(query_embedding, n_results=30, where=where)

    # Build candidate set
    candidates: dict[str, dict] = {}

    for row in lex_rows:
        chunk_id = row["chunk_id"]
        candidates[chunk_id] = {
            "row": row,
            "fts_raw": float(row["rank"]),
            "vec_raw": None,
        }

    vec_ids = vec_result.get("ids", [[]])[0]
    vec_distances = vec_result.get("distances", [[]])[0]
    for chunk_id, dist in zip(vec_ids, vec_distances):
        if dist is None:
            continue
        if chunk_id not in candidates:
            # Fetch row from storage
            row = _fetch_chunk(storage, chunk_id)
            if row is None:
                continue
            candidates[chunk_id] = {
                "row": row,
                "fts_raw": None,
                "vec_raw": float(dist),
            }
        else:
            candidates[chunk_id]["vec_raw"] = float(dist)

    # Normalize FTS scores (BM25 ranks - lower is better, invert)
    fts_raws = [c["fts_raw"] for c in candidates.values() if c["fts_raw"] is not None]
    if fts_raws:
        fts_normalized = normalize_scores([-r for r in fts_raws])  # invert ranks
        fts_map = dict(zip([cid for cid, c in candidates.items() if c["fts_raw"] is not None], fts_normalized))
    else:
        fts_map = {}

    # Normalize vector scores (cosine distance - lower is better, invert)
    vec_raws = [c["vec_raw"] for c in candidates.values() if c["vec_raw"] is not None]
    if vec_raws:
        vec_normalized = normalize_scores([1.0 / (1.0 + d) for d in vec_raws])
        vec_map = dict(zip([cid for cid, c in candidates.items() if c["vec_raw"] is not None], vec_normalized))
    else:
        vec_map = {}

    # Blend scores with weight renormalization
    results: list[ScoredChunk] = []
    for chunk_id, data in candidates.items():
        row = data["row"]
        fts_norm = fts_map.get(chunk_id)
        vec_norm = vec_map.get(chunk_id)

        blended = blend_scores(fts_norm, vec_norm)

        # Apply source-type weight if provided
        if weights:
            source_type = row["source_type"]
            weight = weights.get(source_type, 0.5)
            score = blended * weight
        else:
            score = blended

        if score < min_score:
            continue

        results.append(ScoredChunk(chunk_id=chunk_id, score=score, row=row))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]


def _fetch_chunk(storage: Storage, chunk_id: str):
    cur = storage.conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
    return cur.fetchone()


def _title_from_row(row) -> str:
    meta = json.loads(row["meta_json"]) if row["meta_json"] else {}
    if row["source_type"] == "repo":
        return meta.get("path") or row["doc_id"]
    return meta.get("title") or row["doc_id"]


def _citation_from_row(row) -> str:
    meta = json.loads(row["meta_json"]) if row["meta_json"] else {}
    if row["source_type"] == "repo":
        path = meta.get("path", "")
        ordinal = meta.get("ordinal", 0)
        start = meta.get("char_start", 0)
        end = meta.get("char_end", 0)
        return f"{path} (chunk {ordinal}, chars {start}-{end})"
    conversation_id = meta.get("doc_id") or row["doc_id"]
    title = meta.get("title", "")
    msg_start = meta.get("msg_start", 0)
    msg_end = meta.get("msg_end", 0)
    return f"{conversation_id} [{title}] (msgs {msg_start}-{msg_end})"


def make_snippet(text: str, limit: int = 200) -> str:
    snippet = " ".join(text.split())
    return snippet[:limit]
