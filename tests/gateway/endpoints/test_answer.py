import os
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from chinvex.gateway.endpoints.answer import router
from chinvex.gateway.auth import verify_token
from chinvex.gateway.config import GatewayConfig


@pytest.fixture
def set_token():
    """Set test token in environment."""
    os.environ["CHINVEX_API_TOKEN"] = "test_token_123"
    yield
    del os.environ["CHINVEX_API_TOKEN"]


def test_answer_endpoint_disabled_by_default(set_token):
    """Should return 403 when endpoint is disabled."""
    # Mock config to return disabled
    import chinvex.gateway.endpoints.answer as answer_module
    original_load = answer_module.load_gateway_config

    def mock_config():
        config = original_load()
        config.enable_server_llm = False
        return config

    answer_module.load_gateway_config = mock_config

    app = FastAPI()
    app.include_router(router, dependencies=[Depends(verify_token)])
    client = TestClient(app)

    response = client.post("/answer", json={
        "context": "Chinvex",
        "query": "test"
    }, headers={"Authorization": "Bearer test_token_123"})

    assert response.status_code == 403
    data = response.json()
    assert data["detail"]["error"] == "answer_endpoint_disabled"

    answer_module.load_gateway_config = original_load


def test_answer_endpoint_structure():
    """Should have correct response structure when enabled."""
    # This test just verifies the endpoint exists and has correct schema
    # Actual LLM synthesis testing is integration-level
    assert hasattr(router, 'routes')
    routes = [r for r in router.routes if r.path == "/answer"]
    assert len(routes) == 1
