"""Contexts endpoint - list available contexts."""

from fastapi import APIRouter
from pydantic import BaseModel

from chinvex.context import list_contexts
from chinvex.context_cli import get_contexts_root
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class ContextInfo(BaseModel):
    """Context information."""
    name: str
    aliases: list[str]
    updated_at: str


class ContextsResponse(BaseModel):
    """Response from contexts endpoint."""
    contexts: list[ContextInfo]


@router.get("/contexts", response_model=ContextsResponse)
async def list_available_contexts():
    """
    List available contexts. Respects allowlist.
    """
    contexts_root = get_contexts_root()
    all_contexts = list_contexts(contexts_root)
    config = load_gateway_config()

    # Filter by allowlist if configured
    if config.context_allowlist:
        filtered = [c for c in all_contexts if c.name in config.context_allowlist]
    else:
        filtered = all_contexts

    return ContextsResponse(
        contexts=[
            ContextInfo(
                name=c.name,
                aliases=c.aliases,
                updated_at=c.updated_at
            )
            for c in filtered
        ]
    )
