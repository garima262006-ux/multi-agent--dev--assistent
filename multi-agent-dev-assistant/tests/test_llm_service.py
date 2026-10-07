import json

import httpx
import pytest

from app.models import DevelopmentPlan, GeneratedFileContents
from app.services.llm_service import LLMService, LLMServiceError


def _mock_chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


def test_llm_service_parses_openai_compatible_structured_json() -> None:
    content = json.dumps({"files": {"app.py": "VALUE = 1\n"}})
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers.get("authorization")
        observed["body"] = json.loads(request.content)
        return _mock_chat_response(content)

    service = LLMService(
        api_key="test-secret",
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(handler),
    )

    result = service.generate_structured("Make a file.", GeneratedFileContents)

    assert result.files == {"app.py": "VALUE = 1\n"}
    assert observed["url"] == "https://llm.example/v1/chat/completions"
    assert observed["authorization"] == "Bearer test-secret"
    assert observed["body"]["model"] == "test-model"
    assert observed["body"]["response_format"] == {"type": "json_object"}


def test_llm_service_rejects_provider_output_that_is_not_valid_json() -> None:
    service = LLMService(
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(lambda _: _mock_chat_response("not json")),
    )

    with pytest.raises(LLMServiceError, match="expected JSON schema"):
        service.generate_structured("Make a file.", GeneratedFileContents)


def test_llm_service_requires_a_model_and_base_url() -> None:
    service = LLMService(api_key="", model="", base_url="")

    with pytest.raises(LLMServiceError, match="LLM_MODEL"):
        service.generate_structured("prompt", GeneratedFileContents)


def test_llm_service_retries_transient_statuses_with_exponential_backoff(
    monkeypatch,
) -> None:
    responses = iter(
        [
            httpx.Response(500, json={"error": "temporary"}),
            httpx.Response(429, json={"error": "rate limited"}),
            _mock_chat_response(json.dumps({"files": {"app.py": "VALUE = 1\n"}})),
        ]
    )
    calls = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr("app.services.llm_service.time.sleep", delays.append)
    service = LLMService(
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(handler),
    )

    result = service.generate_structured("Make a file.", GeneratedFileContents)

    assert result.files == {"app.py": "VALUE = 1\n"}
    assert calls == 3
    assert delays == [0.5, 1.0]


def test_llm_service_retries_timeout_and_does_not_retry_non_retryable_status(
    monkeypatch,
) -> None:
    timeout_calls = 0
    timeout_delays: list[float] = []

    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal timeout_calls
        timeout_calls += 1
        if timeout_calls == 1:
            raise httpx.ReadTimeout("temporary timeout")
        return _mock_chat_response('{"files": {}}')

    monkeypatch.setattr("app.services.llm_service.time.sleep", timeout_delays.append)
    timeout_service = LLMService(
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(timeout_handler),
    )
    assert timeout_service.generate_structured(
        "prompt",
        GeneratedFileContents,
    ).files == {}
    assert timeout_calls == 2
    assert timeout_delays == [0.5]

    client_error_calls = 0

    def client_error_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal client_error_calls
        client_error_calls += 1
        return httpx.Response(400, json={"error": "invalid request"})

    client_error_service = LLMService(
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(client_error_handler),
    )
    with pytest.raises(LLMServiceError, match="HTTP 400"):
        client_error_service.generate_structured("prompt", GeneratedFileContents)
    assert client_error_calls == 1


def test_llm_service_stops_after_three_timeouts_without_leaking_key(
    monkeypatch,
) -> None:
    api_key = "private-timeout-key"
    calls = 0
    delays: list[float] = []

    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("provider timeout")

    monkeypatch.setattr("app.services.llm_service.time.sleep", delays.append)
    service = LLMService(
        api_key=api_key,
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(timeout_handler),
    )

    with pytest.raises(LLMServiceError, match="timed out") as exc_info:
        service.generate_structured("prompt", GeneratedFileContents)

    assert calls == 3
    assert delays == [0.5, 1.0]
    assert api_key not in str(exc_info.value)


def test_llm_service_rejects_provider_connection_errors_without_retry(
    monkeypatch,
) -> None:
    calls = 0
    delays: list[float] = []

    def connection_error(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("app.services.llm_service.time.sleep", delays.append)
    service = LLMService(
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(connection_error),
    )

    with pytest.raises(LLMServiceError, match="Could not connect"):
        service.generate_structured("prompt", GeneratedFileContents)

    assert calls == 1
    assert delays == []


def test_llm_service_validates_endpoint_and_exact_generated_paths() -> None:
    invalid_url = LLMService(model="test-model", base_url="ftp://llm.example/v1")
    with pytest.raises(LLMServiceError, match="HTTP or HTTPS"):
        invalid_url.generate_structured("prompt", GeneratedFileContents)

    empty_plan = DevelopmentPlan(summary="Nothing to generate.")
    valid_service = LLMService(
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(
            lambda _: _mock_chat_response('{"files":{"unexpected.py":"x=1"}}')
        ),
    )
    assert valid_service.generate_files(empty_plan) == {}

    plan = DevelopmentPlan(
        summary="Generate one file.",
        files_to_create=["expected.py"],
    )
    with pytest.raises(LLMServiceError, match="exactly the planned file paths"):
        valid_service.generate_files(plan)


def test_llm_service_rejects_malformed_provider_envelopes() -> None:
    service = LLMService(
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"unexpected": "shape"})
        ),
    )
    with pytest.raises(LLMServiceError, match="expected JSON schema"):
        service.generate_structured("prompt", GeneratedFileContents)


def test_llm_service_mock_mode_never_calls_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK", "true")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("Mock mode must not issue a network request.")

    service = LLMService(transport=httpx.MockTransport(handler))
    plan = DevelopmentPlan(
        summary="A mock-generated module.",
        files_to_create=["app/generated.py"],
    )

    structured = service.generate_structured("prompt", GeneratedFileContents)
    generated_files = service.generate_files(plan)
    generated_tests = service.generate_tests(
        plan,
        ["app/generated.py"],
        "tests/test_generated.py",
    )

    assert structured.files == {}
    assert generated_files == {
        "app/generated.py": '"""Deterministic mock-generated module."""\n'
    }
    assert "def test_mock_generated_output_is_deterministic" in generated_tests
    assert calls == 0


def test_llm_service_reads_timeout_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")

    service = LLMService()

    assert service._timeout_seconds == 12.5
