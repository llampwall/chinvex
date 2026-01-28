from fastapi.testclient import TestClient
from fastapi import FastAPI
from chinvex.gateway.endpoints.health import router


app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_health_endpoint_returns_ok():
    """Should return status ok without auth."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "contexts_available" in data
