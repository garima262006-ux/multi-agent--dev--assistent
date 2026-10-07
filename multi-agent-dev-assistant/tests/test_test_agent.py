from pathlib import Path

from app.agents.test_agent import TestAgent
from app.models import DevelopmentPlan
from app.models import TestAgentInput as _TestAgentInput
from app.services.code_executor import CodeExecutor
from app.services.llm_service import LLMServiceError
from app.services.workspace_file_service import WorkspaceFileService


class StubTestGenerator:
    def generate_tests(
        self,
        development_plan: DevelopmentPlan,
        generated_files: list[str],
        test_path: str,
    ) -> str:
        assert development_plan.summary == "A small calculator."
        assert generated_files == ["app/calculator.py"]
        assert test_path.startswith("tests/test_generated_")
        return "def test_generated_value():\n    assert 2 + 2 == 4\n"


def test_test_agent_generates_test_module_and_returns_execution_report(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    test_agent = TestAgent(
        llm_service=StubTestGenerator(),
        file_service=WorkspaceFileService(workspace),
        code_executor=CodeExecutor(workspace),
        workspace_root=workspace,
    )

    result = test_agent.run_tests(
        _TestAgentInput(
            development_plan=DevelopmentPlan(summary="A small calculator."),
            generated_files=["app/calculator.py"],
            timeout_seconds=20,
        )
    )

    assert result.exit_code == 0
    assert result.passed == 1
    assert result.failed == 0
    assert result.output is not None and "passed" in result.output
    assert result.generated_tests
    assert (workspace / result.generated_tests[0]).is_file()
    assert any("not an OS-level sandbox" in suggestion for suggestion in result.suggestions)


def test_test_agent_reports_provider_errors_without_running_subprocess(
    tmp_path: Path,
) -> None:
    class FailingGenerator:
        def generate_tests(self, *_args) -> str:
            raise LLMServiceError("provider unavailable")

    workspace = tmp_path / "workspace"
    agent = TestAgent(
        llm_service=FailingGenerator(),
        file_service=WorkspaceFileService(workspace),
        code_executor=CodeExecutor(workspace),
        workspace_root=workspace,
    )

    result = agent.run_tests(
        _TestAgentInput(
            development_plan=DevelopmentPlan(summary="A module."),
            generated_files=["app/module.py"],
        )
    )

    assert result.errors == ["provider unavailable"]
    assert result.generated_tests == []
    assert result.exit_code is None
