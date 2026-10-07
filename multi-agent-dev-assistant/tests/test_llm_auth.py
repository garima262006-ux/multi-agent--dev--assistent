import httpx
import pytest

from app.models import GeneratedFileContents
from app.services.llm_service import LLMService, LLMServiceError


def test_llm_service_sends_api_key_without_logging_it(caplog) -> None:
    api_key = "unit-test-provider-key"
    observed_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_headers.append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"files": {}}'}}]},
        )

    service = LLMService(
        api_key=api_key,
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(handler),
    )

    service.generate_structured("prompt", GeneratedFileContents)
    assert observed_headers == ["Bearer " + api_key]
    assert api_key not in caplog.text


def test_llm_service_never_includes_api_key_in_provider_error_or_logs(
    caplog,
) -> None:
    api_key = "do-not-leak-this-provider-key"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": f"invalid authorization: {api_key}"},
        )

    service = LLMService(
        api_key=api_key,
        model="test-model",
        base_url="https://llm.example/v1",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMServiceError) as exc_info:
        service.generate_structured("prompt", GeneratedFileContents)

    assert api_key not in str(exc_info.value)
    assert api_key not in caplog.text
