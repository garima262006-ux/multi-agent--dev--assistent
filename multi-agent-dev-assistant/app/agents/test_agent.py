"""Test Agent that generates planned tests and runs them with bounded pytest."""

from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.config import get_settings
from app.models import DevelopmentPlan
from app.models import TestAgentInput as _TestAgentInput
from app.models import TestAgentResult as _TestAgentResult
from app.services.code_executor import CodeExecutor
from app.services.llm_service import LLMServiceError
from app.services.workspace_file_service import WorkspaceFileService


class TestGenerator(Protocol):
    """Provider-independent interface for creating generated project tests."""

    __test__ = False

    def generate_tests(
        self,
        development_plan: DevelopmentPlan,
        generated_files: list[str],
        test_path: str,
    ) -> str:
        """Return test module source for the exact requested path."""


class TestAgent:
    """Inspect a plan, generate a pytest module, and return an execution report."""

    __test__ = False

    def __init__(
        self,
        llm_service: TestGenerator | None = None,
        file_service: WorkspaceFileService | None = None,
        code_executor: CodeExecutor | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        root = workspace_root or get_settings().workspace_dir
        self._workspace_root = root.resolve()
        if llm_service is None:
            from app.services.llm_service import LLMService

            llm_service = LLMService()
        self._llm_service = llm_service
        self._file_service = file_service or WorkspaceFileService(self._workspace_root)
        self._code_executor = code_executor or CodeExecutor(self._workspace_root)

    def run_tests(self, agent_input: _TestAgentInput) -> _TestAgentResult:
        """Generate a test module, then run that explicit file with a timeout."""
        test_path = f"tests/test_generated_{uuid4().hex}.py"
        try:
            test_source = self._llm_service.generate_tests(
                agent_input.development_plan,
                agent_input.generated_files,
                test_path,
            )
            self._file_service.write_planned_files(
                files_to_create=[test_path],
                files_to_modify=[],
                generated_files={test_path: test_source},
            )
            report = self._code_executor.run_pytest(
                timeout_seconds=agent_input.timeout_seconds,
                test_file_paths=[test_path],
            )
        except LLMServiceError as exc:
            return _TestAgentResult(
                errors=[str(exc)],
                suggestions=["Configure the OpenAI-compatible provider and retry."],
            )
        except (OSError, ValueError) as exc:
            return _TestAgentResult(
                errors=[str(exc)],
                suggestions=["Inspect the workspace path and generated test output."],
            )

        report.generated_tests = [test_path]
        report.suggestions.insert(
            0,
            "Tests execute in a bounded subprocess without a shell; this is not an "
            "OS-level sandbox. Run generated code only from a trusted provider.",
        )
        return report
