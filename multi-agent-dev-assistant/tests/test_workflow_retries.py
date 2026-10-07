from pathlib import Path

from app.agents.code_agent import CodeAgent
from app.agents.requirement_agent import RequirementAgent
from app.agents.test_agent import TestAgent
from app.agents.workflow import run_assistant
from app.models import (
    AssistantRunRequest,
    CodeAgentInput,
    CodeAgentResult,
    DevelopmentPlan,
    RequirementInput,
    TestAgentResult as _TestAgentResult,
)


class StubRequirementAgent(RequirementAgent):
    def create_plan(self, _request: RequirementInput) -> DevelopmentPlan:
        return DevelopmentPlan(
            summary="Create one module.",
            files_to_create=["app/generated.py"],
        )


def test_workflow_retries_with_test_feedback_and_succeeds_on_second_attempt(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "3")
    code_inputs: list[CodeAgentInput] = []
    test_results = iter(
        [
            _TestAgentResult(
                failed=1,
                errors=["AssertionError: expected 2, got 1"],
                exit_code=1,
            ),
            _TestAgentResult(passed=1, exit_code=0),
        ]
    )

    def generate_files(
        _agent: CodeAgent,
        agent_input: CodeAgentInput,
    ) -> CodeAgentResult:
        code_inputs.append(agent_input)
        return CodeAgentResult(
            files_created=["app/generated.py"] if len(code_inputs) == 1 else [],
            files_modified=["app/generated.py"] if len(code_inputs) > 1 else [],
            generated_files=["app/generated.py"],
        )

    def run_tests(_agent: TestAgent, _agent_input) -> _TestAgentResult:
        return next(test_results)

    monkeypatch.setattr(CodeAgent, "generate_files", generate_files)
    monkeypatch.setattr(TestAgent, "run_tests", run_tests)

    result = run_assistant(
        AssistantRunRequest(requirement="Create a module.", run_tests=True),
        workspace_root=tmp_path / "workspace",
        requirement_agent=StubRequirementAgent(),
    )

    assert result["status"] == "success"
    assert len(result["attempts"]) == 2
    assert [attempt.attempt_number for attempt in result["attempts"]] == [1, 2]
    assert [attempt.passed for attempt in result["attempts"]] == [False, True]
    assert code_inputs[1].previous_test_errors == [
        "AssertionError: expected 2, got 1",
        "pytest reported 1 failed test(s).",
        "pytest exited with status 1.",
    ]
    assert code_inputs[1].files_to_overwrite == ["app/generated.py"]
    assert result["test_result"].passed == 1
    assert "Workflow stage=requirement" in caplog.text
    assert "Workflow stage=code attempt=1" in caplog.text
    assert "Workflow stage=test attempt=1" in caplog.text
    assert "Workflow stage=retry attempt=1" in caplog.text


def test_workflow_returns_failure_after_configured_retries_are_exhausted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "2")
    generated_attempts = 0

    def generate_files(
        _agent: CodeAgent,
        _agent_input: CodeAgentInput,
    ) -> CodeAgentResult:
        nonlocal generated_attempts
        generated_attempts += 1
        return CodeAgentResult(
            files_created=["app/generated.py"] if generated_attempts == 1 else [],
            files_modified=["app/generated.py"] if generated_attempts > 1 else [],
            generated_files=["app/generated.py"],
        )

    def run_tests(_agent: TestAgent, _agent_input) -> _TestAgentResult:
        return _TestAgentResult(
            failed=1,
            errors=["AssertionError: still failing"],
            exit_code=1,
        )

    monkeypatch.setattr(CodeAgent, "generate_files", generate_files)
    monkeypatch.setattr(TestAgent, "run_tests", run_tests)

    result = run_assistant(
        AssistantRunRequest(requirement="Create a module.", run_tests=True),
        workspace_root=tmp_path / "workspace",
        requirement_agent=StubRequirementAgent(),
    )

    assert result["status"] == "failed"
    assert generated_attempts == 3
    assert len(result["attempts"]) == 3
    assert all(not attempt.passed for attempt in result["attempts"])
    assert result["test_result"].errors == ["AssertionError: still failing"]
