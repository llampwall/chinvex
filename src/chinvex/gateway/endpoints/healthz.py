"""Healthz endpoint - checks DB and Chroma readiness."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from chinvex.context import load_context
from chinvex.context_cli import get_contexts_root, list_contexts
from chinvex.storage import Storage

router = APIRouter()


class HealthzResponse(BaseModel):
    """Response from healthz endpoint."""
    status: str
    checks: dict


@router.get("/healthz", response_model=HealthzResponse)
async def healthz():
    """
    Deep health check - verifies DB and Chroma readiness.

    Returns:
        Status with individual check results
    """
    checks = {}
    all_ok = True

    # Check context registry
    try:
        contexts_root = get_contexts_root()
        contexts = list_contexts(contexts_root)
        checks["context_registry"] = {
            "status": "ok",
            "count": len(contexts)
        }
    except Exception as e:
        checks["context_registry"] = {
            "status": "error",
            "error": str(e)
        }
        all_ok = False

    # Check SQLite readiness (try to load a context)
    try:
        contexts_root = get_contexts_root()
        contexts = list_contexts(contexts_root)
        if contexts:
            test_context = load_context(contexts[0].name, contexts_root)
            storage = Storage(test_context.index.sqlite_path)
            # Try a simple query
            storage._execute("SELECT 1")
            checks["sqlite"] = {"status": "ok"}
        else:
            checks["sqlite"] = {"status": "skip", "reason": "no contexts"}
    except Exception as e:
        checks["sqlite"] = {
            "status": "error",
            "error": str(e)
        }
        all_ok = False

    # Check Chroma readiness
    try:
        contexts_root = get_contexts_root()
        contexts = list_contexts(contexts_root)
        if contexts:
            test_context = load_context(contexts[0].name, contexts_root)
            # Try to access vector store
            from chinvex.vectors import VectorStore
            vec_store = VectorStore(test_context.index.chroma_path)
            # Try to get collection metadata
            vec_store.collection.count()
            checks["chroma"] = {"status": "ok"}
        else:
            checks["chroma"] = {"status": "skip", "reason": "no contexts"}
    except Exception as e:
        checks["chroma"] = {
            "status": "error",
            "error": str(e)
        }
        all_ok = False

    if not all_ok:
        raise HTTPException(status_code=503, detail={"status": "degraded", "checks": checks})

    return HealthzResponse(
        status="ok",
        checks=checks
    )
