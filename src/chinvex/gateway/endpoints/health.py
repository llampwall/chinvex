"""Health check endpoint."""

from fastapi import APIRouter
from chinvex.gateway import __version__
from chinvex.context import list_contexts
from chinvex.context_cli import get_contexts_root


router = APIRouter()


@router.get("/health")
async def health():
    """
    Health check endpoint. No authentication required.

    Returns:
        Status information including version and context count
    """
    try:
        contexts_root = get_contexts_root()
        contexts = list_contexts(contexts_root)
        contexts_available = len(contexts)
    except Exception:
        # If context listing fails, don't fail health check
        contexts_available = 0

    return {
        "status": "ok",
        "version": __version__,
        "contexts_available": contexts_available
    }
