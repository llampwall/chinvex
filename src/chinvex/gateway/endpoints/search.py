"""Search endpoint - raw hybrid search."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from chinvex.context import load_context
from chinvex.search import hybrid_search_from_context
from chinvex.gateway.validation import SearchRequest
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class SearchResponse(BaseModel):
    """Response from search endpoint."""
    context: str
    query: str
    results: list[dict]
    total_results: int


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request):
    """
    Raw hybrid search. Returns ranked chunks without grounding check.
    """
    request.state.context = req.context

    try:
        context = load_context(req.context)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Context not found")

    config = load_gateway_config()
    if config.context_allowlist and req.context not in config.context_allowlist:
        raise HTTPException(status_code=404, detail="Context not found")

    results = hybrid_search_from_context(
        context=context,
        query=req.query,
        k=req.k,
        source_types=req.source_types,
        no_recency=req.no_recency
    )

    return SearchResponse(
        context=req.context,
        query=req.query,
        results=[
            {
                "chunk_id": r.chunk_id,
                "text": r.text[:5000] + (" [truncated]" if len(r.text) > 5000 else ""),
                "source_uri": r.source_uri,
                "source_type": r.source_type,
                "scores": {
                    "fts": r.fts_score,
                    "vector": r.vector_score,
                    "blended": r.blended_score,
                    "rank": r.rank_score
                },
                "metadata": {
                    "line_start": getattr(r, 'line_start', None),
                    "line_end": getattr(r, 'line_end', None),
                    "updated_at": getattr(r, 'updated_at', None)
                }
            }
            for r in results
        ],
        total_results=len(results)
    )
