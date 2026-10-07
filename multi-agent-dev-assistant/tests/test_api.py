from uuid import UUID

from fastapi.testclient import TestClient

import app.main as main
from app.main import app


def test_root_endpoint_returns_api_links() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "Multi-Agent Dev Assistant"
    assert response.json()["docs"] == "/docs"
    assert response.json()["endpoints"]["run_assistant"] == {
        "method": "POST",
        "path": "/api/assistant/run",
    }


def test_assistant_endpoint_returns_structured_pipeline_result_without_provider() -> None:
    response = TestClient(app).post(
        "/api/assistant/run",
        json={
            "requirement": "Create a Python calculator API with addition and division.",
            "run_tests": True,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["development_plan"]["summary"]
    assert result["development_plan"]["files_to_create"]
    assert result["code_result"]["files_created"] == []
    assert result["code_result"]["warnings"]
    assert result["test_result"] is None
    assert result["status"] == "failed"
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["attempt_number"] == 1
    assert result["attempts"][0]["passed"] is False


def test_health_endpoint_and_cors_preflight() -> None:
    client = TestClient(app)

    health = client.get("/health")
    preflight = client.options(
        "/projects",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_request_id_is_echoed_and_added_to_request_logs(caplog) -> None:
    request_id = "client-request-123"

    response = TestClient(app).get(
        "/health",
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == request_id
    assert f"request_id={request_id}" in caplog.text


def test_missing_or_invalid_request_id_is_replaced_with_uuid() -> None:
    client = TestClient(app)

    for headers in ({}, {"X-Request-ID": "untrusted id\ninjected"}):
        response = client.get("/health", headers=headers)
        assert response.status_code == 200
        assert str(UUID(response.headers["x-request-id"]))


def test_unhandled_errors_return_generic_500_without_leaking_details(
    monkeypatch,
) -> None:
    def fail_request(_request):
        raise RuntimeError("private implementation details")

    monkeypatch.setattr(main, "run_assistant", fail_request)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/assistant/run",
        json={"requirement": "Build a todo API", "run_tests": True},
        headers={"X-Request-ID": "failure-request"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal server error.",
        "request_id": "failure-request",
    }
    assert response.headers["x-request-id"] == "failure-request"
    assert "private implementation details" not in response.text
