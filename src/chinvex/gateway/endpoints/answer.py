"""Answer endpoint - full synthesis with LLM (optional, flag-gated)."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from chinvex.context import load_context, ContextNotFoundError
from chinvex.context_cli import get_contexts_root
from chinvex.search import hybrid_search_from_context
from chinvex.gateway.validation import AnswerRequest
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class AnswerResponse(BaseModel):
    """Response from answer endpoint."""
    schema_version: int
    context: str
    query: str
    grounded: bool
    answer: str
    citations: list[dict]
    evidence_pack: dict
    errors: list[dict]


@router.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest, request: Request):
    """
    Full synthesis with LLM. Disabled by default.
    Enable with GATEWAY_ENABLE_SERVER_LLM=true.

    Args:
        req: Answer request with context, query, k, grounded
        request: FastAPI request (for audit logging)

    Returns:
        Answer with LLM-synthesized response and citations

    Raises:
        HTTPException: 403 if endpoint disabled, 404 if context not found
    """
    config = load_gateway_config()

    # Check if endpoint is enabled
    if not config.enable_server_llm:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "answer_endpoint_disabled",
                "message": "Server-side synthesis is disabled. Use /v1/evidence instead."
            }
        )

    request.state.context = req.context

    # Load and verify context
    try:
        contexts_root = get_contexts_root()
        context = load_context(req.context, contexts_root)
    except ContextNotFoundError:
        raise HTTPException(status_code=404, detail="Context not found")

    if config.context_allowlist and req.context not in config.context_allowlist:
        raise HTTPException(status_code=404, detail="Context not found")

    # Perform search
    try:
        results = hybrid_search_from_context(
            context=context,
            query=req.query,
            k=req.k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Generate answer with LLM
    # NOTE: This is a placeholder - actual LLM synthesis would integrate
    # with the existing answer logic from MCP server or similar
    # For P2, we implement the endpoint structure but leave synthesis
    # as a future enhancement

    GROUNDING_THRESHOLD = 0.35
    grounded_chunks = [r for r in results if r.rank_score >= GROUNDING_THRESHOLD]
    grounded = len(grounded_chunks) >= 1

    if grounded and req.grounded:
        # TODO: Actual LLM synthesis here
        # For now, return a structured response indicating synthesis would happen
        answer_text = "LLM synthesis not yet implemented. Use /v1/evidence for retrieval."
        citations = [
            {
                "chunk_id": r.chunk_id,
                "source_uri": r.source_uri,
                "range": build_range(r)
            }
            for r in grounded_chunks[:3]
        ]
        evidence_pack = {
            "chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "text": truncate_text(r.text, 5000),
                    "source_uri": r.source_uri,
                    "source_type": r.source_type,
                    "range": build_range(r),
                    "score": r.rank_score
                }
                for r in grounded_chunks
            ]
        }
        errors = []
    else:
        answer_text = "Not stated in retrieved sources."
        citations = []
        evidence_pack = {"chunks": []}
        errors = [
            {
                "code": "GROUNDING_FAILED",
                "detail": "No retrieved chunk supports a direct answer to this query."
            }
        ]
        grounded = False

    return AnswerResponse(
        schema_version=1,
        context=req.context,
        query=req.query,
        grounded=grounded,
        answer=answer_text,
        citations=citations,
        evidence_pack=evidence_pack,
        errors=errors
    )


def truncate_text(text: str, max_len: int) -> str:
    """Truncate text with marker."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + " [truncated]"


def build_range(result) -> dict:
    """Build range object from search result."""
    if hasattr(result, 'line_start') and result.line_start:
        return {
            "line_start": result.line_start,
            "line_end": result.line_end
        }
    elif hasattr(result, 'char_start'):
        return {
            "char_start": result.char_start,
            "char_end": result.char_end
        }
    return {}
