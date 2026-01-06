from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .config import AppConfig
from .embed import OllamaEmbedder
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
) -> list[ScoredChunk]:
    filters = {}
    if source in {"repo", "chat"}:
        filters["source_type"] = source
    if project:
        filters["project"] = project
    if repo:
        filters["repo"] = repo

    lex_rows = storage.search_fts(query, limit=30, filters=filters)
    lex_scores = _normalize_lex(lex_rows)

    where = {}
    if source in {"repo", "chat"}:
        where["source_type"] = source
    if project:
        where["project"] = project
    if repo:
        where["repo"] = repo

    query_embedding = embedder.embed([query])
    vec_result = vectors.query(query_embedding, n_results=30, where=where)
    vec_scores = _normalize_vec(vec_result)

    merged: dict[str, dict[str, Any]] = {}
    for row in lex_rows:
        merged[row["chunk_id"]] = {"lex": lex_scores.get(row["chunk_id"], 0.0), "row": row}
    for chunk_id, vec_score in vec_scores.items():
        merged.setdefault(chunk_id, {})["vec"] = vec_score

    recency_map = _recency_norm(lex_rows)
    results: list[ScoredChunk] = []
    for chunk_id, data in merged.items():
        row = data.get("row")
        if row is None:
            row = _fetch_chunk(storage, chunk_id)
        if row is None:
            continue
        lex = data.get("lex", 0.0)
        vec = data.get("vec", 0.0)
        score = 0.55 * lex + 0.45 * vec

        meta = json.loads(row["meta_json"]) if row["meta_json"] else {}
        if project and meta.get("project") == project:
            score += 0.05
        if repo and meta.get("repo") == repo:
            score += 0.05
        score += 0.05 * recency_map.get(chunk_id, 0.0)
        score = min(1.0, max(0.0, score))
        if score < min_score:
            continue
        results.append(ScoredChunk(chunk_id=chunk_id, score=score, row=row))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]


def _normalize_lex(rows) -> dict[str, float]:
    if not rows:
        return {}
    ranks = [row["rank"] for row in rows]
    min_rank = min(ranks)
    max_rank = max(ranks) if max(ranks) != min_rank else min_rank + 1.0
    scores = {}
    for row in rows:
        norm = (row["rank"] - min_rank) / (max_rank - min_rank)
        scores[row["chunk_id"]] = 1.0 - max(0.0, min(1.0, norm))
    return scores


def _normalize_vec(result: dict) -> dict[str, float]:
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    scores: dict[str, float] = {}
    for chunk_id, dist in zip(ids, distances):
        if dist is None:
            continue
        score = 1.0 / (1.0 + float(dist))
        scores[chunk_id] = max(0.0, min(1.0, score))
    return scores


def _recency_norm(rows) -> dict[str, float]:
    dates = []
    for row in rows:
        updated = row["updated_at"]
        if not updated:
            continue
        try:
            dates.append(datetime.fromisoformat(updated.replace("Z", "+00:00")))
        except ValueError:
            continue
    if not dates:
        return {}
    min_dt = min(dates)
    max_dt = max(dates)
    span = (max_dt - min_dt).total_seconds() or 1.0
    scores = {}
    for row in rows:
        updated = row["updated_at"]
        if not updated:
            continue
        try:
            dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        except ValueError:
            continue
        scores[row["chunk_id"]] = max(0.0, min(1.0, (dt - min_dt).total_seconds() / span))
    return scores


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
