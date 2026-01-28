"""Chunks endpoint - fetch specific chunks by ID."""

import json
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from chinvex.context import load_context, ContextNotFoundError
from chinvex.context_cli import get_contexts_root
from chinvex.storage import Storage
from chinvex.gateway.validation import ChunksRequest
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class ChunksResponse(BaseModel):
    """Response from chunks endpoint."""
    context: str
    chunks: list[dict]


@router.post("/chunks", response_model=ChunksResponse)
async def get_chunks(req: ChunksRequest, request: Request):
    """
    Fetch specific chunks by ID.
    """
    request.state.context = req.context

    try:
        contexts_root = get_contexts_root()
        context = load_context(req.context, contexts_root)
    except ContextNotFoundError:
        raise HTTPException(status_code=404, detail="Context not found")

    config = load_gateway_config()
    if config.context_allowlist and req.context not in config.context_allowlist:
        raise HTTPException(status_code=404, detail="Context not found")

    storage = Storage(context.index.sqlite_path)
    chunks = storage.get_chunks_by_ids(req.chunk_ids)

    return ChunksResponse(
        context=req.context,
        chunks=[
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source_uri": c["source_uri"],
                "source_type": c["source_type"],
                "metadata": json.loads(c["meta_json"]) if c["meta_json"] else {}
            }
            for c in chunks
        ]
    )
