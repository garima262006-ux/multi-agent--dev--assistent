"""OpenAI-compatible chat-completions adapter, isolated from agent logic."""

import json
import re
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.models import DevelopmentPlan, GeneratedFileContents


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
_JSON_FENCE = re.compile(r"\A```(?:json)?\s*|\s*```\Z", re.IGNORECASE)
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 0.5


class LLMServiceError(RuntimeError):
    """Raised when provider configuration, transport, or response is invalid."""


class LLMService:
    """Make structured requests to a configurable OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        mock_mode: bool | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.llm_timeout_seconds
        )
        if not 1 <= effective_timeout <= 300:
            raise ValueError("LLM timeout must be between 1 and 300 seconds.")
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._model = model if model is not None else settings.llm_model
        self._base_url = (
            base_url if base_url is not None else settings.llm_base_url
        ).rstrip("/")
        self._timeout_seconds = effective_timeout
        self._mock_mode = mock_mode if mock_mode is not None else settings.llm_mock
        self._transport = transport

    def generate_structured(
        self,
        prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        """Call chat completions and validate the JSON response as a Pydantic model."""
        if self._mock_mode:
            return self._mock_structured_response(response_model)

        endpoint = self._completion_endpoint()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only a JSON object matching the requested schema. "
                        "Do not use markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        response: httpx.Response | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                with httpx.Client(
                    timeout=self._timeout_seconds,
                    transport=self._transport,
                    trust_env=False,
                ) as client:
                    response = client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                break
            except httpx.TimeoutException as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise LLMServiceError("LLM provider request timed out.") from exc
                time.sleep(_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code != 429 and status_code < 500:
                    raise LLMServiceError(
                        f"LLM provider returned HTTP {status_code}."
                    ) from exc
                if attempt == _MAX_ATTEMPTS:
                    raise LLMServiceError(
                        f"LLM provider returned HTTP {status_code} after {_MAX_ATTEMPTS} attempts."
                    ) from exc
                time.sleep(_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            except httpx.RequestError as exc:
                raise LLMServiceError(
                    "Could not connect to the configured LLM provider."
                ) from exc

        if response is None:
            raise LLMServiceError("LLM provider request did not return a response.")

        try:
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("Chat completion content must be text.")
            decoded = json.loads(_JSON_FENCE.sub("", content.strip()))
            return response_model.model_validate(decoded)
        except (ValueError, KeyError, IndexError, TypeError, ValidationError) as exc:
            raise LLMServiceError(
                "LLM provider returned a response that did not match the expected JSON schema."
            ) from exc

    def generate_files(self, development_plan: DevelopmentPlan) -> dict[str, str]:
        """Generate file contents for exact plan-declared relative paths."""
        return self.generate_files_with_feedback(development_plan, [])

    def generate_files_with_feedback(
        self,
        development_plan: DevelopmentPlan,
        previous_test_errors: list[str],
    ) -> dict[str, str]:
        """Generate planned file contents, incorporating prior test failures."""
        planned_paths = [
            *development_plan.files_to_create,
            *development_plan.files_to_modify,
        ]
        if not planned_paths:
            return {}
        if self._mock_mode:
            return {
                path: self._mock_file_contents(path)
                for path in planned_paths
            }
        prompt = (
            "Generate source file contents for the development plan below. Return "
            "a JSON object with a single 'files' object keyed only by the exact "
            "relative paths listed below. Each value must be source text, never a "
            "shell command. Do not add files that are not listed.\n\n"
            f"Development plan:\n{development_plan.model_dump_json(indent=2)}\n\n"
            f"Required file paths:\n{json.dumps(planned_paths)}"
        )
        if previous_test_errors:
            prompt += (
                "\n\nThe previous generated implementation failed its tests. "
                "Use these test errors to correct the planned files:\n"
                f"{json.dumps(previous_test_errors)}"
            )
        response = self.generate_structured(prompt, GeneratedFileContents)
        if set(response.files) != set(planned_paths):
            raise LLMServiceError(
                "LLM provider response did not contain exactly the planned file paths."
            )
        return response.files

    def generate_tests(
        self,
        development_plan: DevelopmentPlan,
        generated_files: list[str],
        test_path: str,
    ) -> str:
        """Generate one test module to inspect behavior described by a plan."""
        if self._mock_mode:
            return "def test_mock_generated_output_is_deterministic():\n    assert True\n"
        response = self.generate_structured(
            "Write a pytest test module for the project described below. Inspect "
            "the generated file paths and infer the appropriate public behavior. "
            "Return JSON with a 'files' object containing exactly one key, "
            f"{json.dumps(test_path)}, whose value is the complete test module source. "
            "Do not use subprocesses, networking, filesystem access outside the test "
            "directory, or destructive operations in the tests.\n\n"
            f"Development plan:\n{development_plan.model_dump_json(indent=2)}\n\n"
            f"Generated project files:\n{json.dumps(generated_files)}",
            GeneratedFileContents,
        )
        if set(response.files) != {test_path}:
            raise LLMServiceError("LLM provider did not return the requested test file.")
        return response.files[test_path]

    @staticmethod
    def _mock_structured_response(
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        """Return the deterministic empty instance used by explicit mock calls."""
        if response_model is GeneratedFileContents:
            return response_model.model_validate({"files": {}})
        return response_model.model_validate({})

    @staticmethod
    def _mock_file_contents(path: str) -> str:
        """Produce deterministic, harmless placeholder content for a planned file."""
        if path.endswith(".py"):
            return '"""Deterministic mock-generated module."""\n'
        return ""

    def _completion_endpoint(self) -> str:
        if not self._model:
            raise LLMServiceError("Set LLM_MODEL to configure the chat-completions model.")
        if not self._base_url:
            raise LLMServiceError(
                "Set LLM_BASE_URL to the OpenAI-compatible API base URL."
            )
        if not self._base_url.startswith(("https://", "http://")):
            raise LLMServiceError("LLM_BASE_URL must use HTTP or HTTPS.")
        if self._base_url.endswith("/chat/completions"):
            return self._base_url
        return f"{self._base_url}/chat/completions"
