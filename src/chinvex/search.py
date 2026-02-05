from __future__ import annotations

import json
import os
import sys
import time
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
    context: str | None = None  # Source context for multi-context search


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
    include_archive: bool = False,
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
        include_archive=include_archive,
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
    include_archive: bool = False,
) -> list[ScoredChunk]:
    """
    Hybrid search with score normalization and weight renormalization.

    Args:
        weights: Optional source-type weights dict (e.g., {"repo": 1.0, "chat": 0.8})
                 If None, no weight adjustment applied.
        include_archive: If False (default), exclude archived documents from results.
    """
    filters = {}
    if source in {"repo", "chat", "codex_session"}:
        filters["source_type"] = source
    if project:
        filters["project"] = project
    if repo:
        filters["repo"] = repo
    if not include_archive:
        filters["archived"] = 0

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
    if not include_archive:
        where["archived"] = 0

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


def _get_reranker(config):
    """Factory function to instantiate reranker based on config.

    Args:
        config: RerankerConfig object

    Returns:
        Reranker instance (Cohere, Jina, or Local)

    Raises:
        ValueError: If provider is unknown
        ValueError: If required API key is missing
    """
    if config.provider == "cohere":
        from .rerankers.cohere import CohereReranker, CohereRerankerConfig
        api_key = os.getenv("COHERE_API_KEY")
        reranker_config = CohereRerankerConfig(
            provider=config.provider,
            model=config.model,
            candidates=config.candidates,
            top_k=config.top_k,
        )
        return CohereReranker(reranker_config, api_key=api_key)
    elif config.provider == "jina":
        from .rerankers.jina import JinaReranker, JinaRerankerConfig
        api_key = os.getenv("JINA_API_KEY")
        reranker_config = JinaRerankerConfig(
            provider=config.provider,
            model=config.model,
            candidates=config.candidates,
            top_k=config.top_k,
        )
        return JinaReranker(reranker_config, api_key=api_key)
    elif config.provider == "local":
        from .rerankers.local import LocalReranker, LocalRerankerConfig
        reranker_config = LocalRerankerConfig(
            provider=config.provider,
            model=config.model,
            candidates=config.candidates,
            top_k=config.top_k,
        )
        return LocalReranker(reranker_config)
    else:
        raise ValueError(f"Unknown reranker provider: {config.provider}")


def search_context(
    ctx,
    query: str,
    *,
    k: int = 8,
    min_score: float = 0.35,
    source: str = "all",
    project: str | None = None,
    repo: str | None = None,
    ollama_host_override: str | None = None,
    recency_enabled: bool = True,
    rerank: bool = False,
) -> list[SearchResult]:
    """
    Search within a context using context-aware weights.

    Args:
        rerank: If True, enable reranking for this query (overrides context config)
    """
    from .context import ContextConfig

    # NEW: Warn if context uses Ollama embeddings
    if ctx.embedding and ctx.embedding.provider == "ollama":
        print(
            f"Warning: Context '{ctx.name}' uses Ollama embeddings. "
            f"Consider migrating to OpenAI for faster and more consistent results. "
            f"Run: chinvex ingest --context {ctx.name} --embed-provider openai --rebuild-index",
            file=sys.stderr
        )
    elif not ctx.embedding:
        # Legacy context without embedding field defaults to Ollama
        print(
            f"Warning: Context '{ctx.name}' uses Ollama embeddings (legacy default). "
            f"Consider migrating to OpenAI for faster and more consistent results. "
            f"Run: chinvex ingest --context {ctx.name} --embed-provider openai --rebuild-index",
            file=sys.stderr
        )

    db_path = ctx.index.sqlite_path
    chroma_dir = ctx.index.chroma_dir

    storage = Storage(db_path)
    storage.ensure_schema()

    # Get embedding provider (matches ingest behavior)
    from .embedding_providers import get_provider
    import os

    ollama_host = ollama_host_override or ctx.ollama.base_url
    env_provider = os.getenv("CHINVEX_EMBED_PROVIDER")

    # Build context config for provider selection
    context_config = None
    if ctx.embedding:
        context_config = {
            "embedding": {
                "provider": ctx.embedding.provider,
                "model": ctx.embedding.model,
            }
        }

    embedder = get_provider(
        cli_provider=None,  # Search doesn't have CLI override
        context_config=context_config,
        env_provider=env_provider,
        ollama_host=ollama_host
    )
    vectors = VectorStore(chroma_dir)

    # Determine if reranking should be used
    use_reranker = (ctx.reranker is not None) or rerank

    # If reranking enabled, fetch more candidates
    if use_reranker and ctx.reranker:
        candidates_k = ctx.reranker.candidates
    else:
        candidates_k = k

    # Use context weights for source-type prioritization
    scored = search_chunks(
        storage,
        vectors,
        embedder,
        query,
        k=candidates_k,
        min_score=min_score,
        source=source,
        project=project,
        repo=repo,
        weights=ctx.weights,
    )

    # Apply reranking if enabled and conditions met
    if use_reranker and ctx.reranker and len(scored) >= 10:
        try:
            # Prepare candidates for reranking
            candidates = [
                {"chunk_id": item.chunk_id, "text": item.row["text"]}
                for item in scored
            ]

            # Truncate to max 50 candidates
            if len(candidates) > 50:
                candidates = candidates[:50]
                scored = scored[:50]

            # Get reranker and rerank with 2s timeout
            start_time = time.time()
            reranker = _get_reranker(ctx.reranker)
            reranked = reranker.rerank(query, candidates, top_k=k)
            elapsed = time.time() - start_time

            if elapsed > 2.0:
                print(
                    f"Warning: Reranking took {elapsed:.2f}s (exceeded 2s budget)",
                    file=sys.stderr
                )

            # Map reranked results back to ScoredChunk objects
            reranked_map = {r["chunk_id"]: r["rerank_score"] for r in reranked}
            scored = [
                item for item in scored
                if item.chunk_id in reranked_map
            ]
            # Update scores with rerank scores
            for item in scored:
                item.score = reranked_map[item.chunk_id]

            # Sort by rerank score
            scored.sort(key=lambda r: r.score, reverse=True)

        except Exception as e:
            # Fallback: use pre-rerank results
            print(
                f"Warning: Reranker failed ({type(e).__name__}: {e}). "
                f"Falling back to pre-rerank results.",
                file=sys.stderr
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


def hybrid_search_from_context(
    context,
    query: str,
    k: int = 8,
    source_types: list[str] | None = None,
    no_recency: bool = False
):
    """
    Wrapper for context-based search (used by gateway).

    Args:
        context: Context object from chinvex.context.load_context()
        query: Search query
        k: Number of results
        source_types: Filter by source types
        no_recency: Disable recency decay

    Returns:
        List of search results with scores
    """
    from .storage import Storage
    from .vectors import VectorStore
    from .embed import OllamaEmbedder
    from .ranking import apply_recency_decay

    db_path = context.index.sqlite_path
    chroma_dir = context.index.chroma_dir

    storage = Storage(db_path)
    storage.ensure_schema()

    # FTS search
    fts_results = storage.search_fts(query, limit=k * 2)

    # Vector search
    embedder = OllamaEmbedder(
        model=context.ollama.embed_model,
        host=context.ollama.base_url
    )
    vec_store = VectorStore(chroma_dir)
    query_embedding = embedder.embed([query])
    vec_results = vec_store.query(query_embedding, n_results=k * 2)

    # Merge and score
    from .scoring import merge_and_rank
    results = merge_and_rank(
        fts_results=fts_results,
        vec_results=vec_results,
        storage=storage,
        weights=context.weights,
        k=k
    )

    # Apply recency if enabled
    if not no_recency and hasattr(context, 'ranking') and context.ranking.recency_enabled:
        results = apply_recency_decay(
            results,
            half_life_days=context.ranking.recency_half_life_days
        )

    return results


def _detect_mixed_embedding_providers(contexts: list[str], contexts_root) -> None:
    """
    Detect if contexts use different embedding providers and raise error if so.

    Args:
        contexts: List of context names
        contexts_root: Root directory for contexts (Path object)

    Raises:
        ValueError: If contexts use different embedding providers
    """
    from .context import load_context

    providers = {}
    for ctx_name in contexts:
        try:
            ctx = load_context(ctx_name, contexts_root)
            # Determine provider from embedding config or default to ollama
            if ctx.embedding:
                provider = ctx.embedding.provider
                model = ctx.embedding.model or "unknown"
            else:
                # Legacy contexts default to ollama
                provider = "ollama"
                model = ctx.ollama.embed_model

            providers[ctx_name] = (provider, model)
        except Exception:
            # Skip contexts that fail to load
            continue

    # Check if all providers are the same
    if not providers:
        return  # No contexts loaded successfully

    unique_providers = set(p[0] for p in providers.values())
    if len(unique_providers) > 1:
        # Build detailed error message
        provider_details = ", ".join(
            f"{ctx}={prov}" for ctx, (prov, _) in sorted(providers.items())
        )
        raise ValueError(
            f"Cross-context search with mixed embedding providers is not allowed. "
            f"Contexts have different providers: {provider_details}. "
            f"Use --context to search a single context, or ensure all contexts use the same embedding provider."
        )


def search_multi_context(
    contexts: list[str] | str,
    query: str,
    k: int = 10,
    min_score: float = 0.35,
    source: str = "all",
    ollama_host: str | None = None,
    recency_enabled: bool = True,
    allow_mixed_embeddings: bool = False,
    rerank: bool = False,
) -> list[SearchResult]:
    """
    Search across multiple contexts, merge results by score.

    Args:
        contexts: List of context names, or "all" for all contexts
        query: Search query
        k: Total number of results to return (not per-context)
        min_score: Minimum score threshold
        source: Filter by source type (all/repo/chat/codex_session)
        ollama_host: Ollama host override
        recency_enabled: Enable recency decay
        rerank: Enable reranking for this query
        allow_mixed_embeddings: Allow mixed embedding providers (not yet supported in P5)

    Returns:
        List of SearchResult objects sorted by score descending

    Raises:
        ValueError: If contexts use mixed embedding providers and allow_mixed_embeddings=False
        NotImplementedError: If allow_mixed_embeddings=True (not supported in P5)
    """
    from pathlib import Path
    import os

    # Check for mixed embeddings flag
    if allow_mixed_embeddings:
        raise NotImplementedError(
            "Mixed-space embedding merge is not yet supported. "
            "This feature is planned for P6+. "
            "For now, ensure all contexts use the same embedding provider."
        )

    # Get contexts root
    contexts_root_str = os.getenv("CHINVEX_CONTEXTS_ROOT", "P:/ai_memory/contexts")
    contexts_root = Path(contexts_root_str)

    # Expand "all" to all available contexts
    if contexts == "all":
        from .context import list_contexts
        contexts = [c.name for c in list_contexts(contexts_root)]

    # Cap contexts to prevent slowdown
    max_contexts = 10  # TODO: Make configurable
    if len(contexts) > max_contexts:
        contexts = contexts[:max_contexts]

    # Detect mixed embedding providers before search
    _detect_mixed_embedding_providers(contexts, contexts_root)

    # Per-context cap: fetch more than k to ensure good merged results
    k_per_context = min(k * 2, 20)

    # Gather results from each context
    all_results = []
    for ctx_name in contexts:
        try:
            from .context import load_context
            ctx = load_context(ctx_name, contexts_root)
            results = search_context(
                ctx=ctx,
                query=query,
                k=k_per_context,
                min_score=min_score,
                source=source,
                ollama_host_override=ollama_host,
                recency_enabled=recency_enabled,
                rerank=rerank,
            )
            # Tag each result with source context
            for r in results:
                r.context = ctx_name
            all_results.extend(results)
        except Exception as e:
            # Log warning but continue with other contexts
            print(f"Warning: Failed to search context {ctx_name}: {e}")
            continue

    # Sort by score descending, take top k
    all_results.sort(key=lambda r: r.score, reverse=True)
    final_results = all_results[:k]

    # Debug logging: score distribution across contexts
    if final_results:
        score_min = min(r.score for r in final_results)
        score_max = max(r.score for r in final_results)
        context_counts = {}
        for r in final_results:
            context_counts[r.context] = context_counts.get(r.context, 0) + 1
        print(f"[DEBUG] Cross-context scores: min={score_min:.3f}, max={score_max:.3f}, by_context={context_counts}")

    return final_results
