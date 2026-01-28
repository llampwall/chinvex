"""Evidence endpoint - search with grounding check."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from chinvex.context import load_context
from chinvex.context_cli import get_contexts_root
from chinvex.search import hybrid_search_from_context
from chinvex.gateway.validation import EvidenceRequest
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class EvidenceResponse(BaseModel):
    """Response from evidence endpoint."""
    context: str
    query: str
    grounded: bool
    evidence_pack: dict
    retrieval_debug: dict
    message: Optional[str] = None


@router.post("/evidence", response_model=EvidenceResponse)
async def get_evidence(req: EvidenceRequest, request: Request):
    """
    Search with grounding check. Primary endpoint for ChatGPT Actions.

    Args:
        req: Evidence request with context, query, k
        request: FastAPI request (for audit logging)

    Returns:
        Evidence response with grounded status and chunks
    """
    # Store context in request state for audit logging
    request.state.context = req.context

    # Load context and verify it exists
    try:
        contexts_root = get_contexts_root()
        context = load_context(req.context, contexts_root)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Context not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load context: {str(e)}")

    # Check context allowlist
    config = load_gateway_config()
    if config.context_allowlist and req.context not in config.context_allowlist:
        raise HTTPException(status_code=404, detail="Context not found")

    # Perform search
    try:
        results = hybrid_search_from_context(
            context=context,
            query=req.query,
            k=req.k,
            source_types=req.source_types
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Grounding check
    GROUNDING_THRESHOLD = 0.35
    grounded_chunks = [r for r in results if r.rank_score >= GROUNDING_THRESHOLD]

    grounded = len(grounded_chunks) >= 1

    if grounded:
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
        message = None
    else:
        evidence_pack = {"chunks": []}
        message = "No retrieved content supports a direct answer to this query."

    return EvidenceResponse(
        context=req.context,
        query=req.query,
        grounded=grounded,
        evidence_pack=evidence_pack,
        retrieval_debug={
            "k": req.k,
            "chunks_retrieved": len(results),
            "chunks_above_threshold": len(grounded_chunks)
        },
        message=message
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
