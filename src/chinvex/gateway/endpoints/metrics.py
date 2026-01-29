"""Prometheus metrics endpoint."""
from fastapi import APIRouter, Response

from ..metrics import generate_metrics

router = APIRouter()


@router.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus exposition format.
    """
    from ..app import metrics_collector

    metrics_text = generate_metrics(metrics_collector)
    return Response(content=metrics_text, media_type="text/plain")
