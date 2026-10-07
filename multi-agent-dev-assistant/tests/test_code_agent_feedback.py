from pathlib import Path

from app.agents.code_agent import CodeAgent
from app.models import CodeAgentInput, CodeAgentResult, DevelopmentPlan
from app.services.llm_service import LLMServiceError
from app.services.workspace_file_service import WorkspaceFileService


class FeedbackCodeGenerator:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def generate_files(self, development_plan: DevelopmentPlan) -> dict[str, str]:
        return {"app.py": "VALUE = 1\n"}

    def generate_files_with_feedback(
        self,
        development_plan: DevelopmentPlan,
        previous_test_errors: list[str],
    ) -> dict[str, str]:
        self.errors = previous_test_errors
        return {"app.py": "VALUE = 2\n"}


def test_code_agent_supplies_test_errors_to_generator_and_overwrites_its_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    generator = FeedbackCodeGenerator()
    agent = CodeAgent(
        llm_service=generator,
        file_service=WorkspaceFileService(workspace),
    )
    plan = DevelopmentPlan(
        summary="Create a module.",
        files_to_create=["app.py"],
    )

    first = agent.generate_files(CodeAgentInput(development_plan=plan))
    second = agent.generate_files(
        CodeAgentInput(
            development_plan=plan,
            previous_test_errors=["AssertionError: expected 2"],
            files_to_overwrite=first.files_created,
        )
    )

    assert generator.errors == ["AssertionError: expected 2"]
    assert (workspace / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert second.files_modified == ["app.py"]


def test_code_agent_handles_empty_plan_and_provider_error(tmp_path: Path) -> None:
    class FailingCodeGenerator:
        def generate_files(self, _development_plan: DevelopmentPlan) -> dict[str, str]:
            raise LLMServiceError("provider unavailable")

    agent = CodeAgent(
        llm_service=FailingCodeGenerator(),
        file_service=WorkspaceFileService(tmp_path / "workspace"),
    )
    empty_plan = DevelopmentPlan(summary="No files.")
    empty_result = agent.generate_files(CodeAgentInput(development_plan=empty_plan))
    assert empty_result.warnings == [
        "Development plan does not declare any files to generate."
    ]

    planned = DevelopmentPlan(
        summary="Create a module.",
        files_to_create=["app.py"],
    )
    failed_result = agent.generate_files(CodeAgentInput(development_plan=planned))
    assert failed_result == CodeAgentResult(warnings=["provider unavailable"])


def test_code_agent_reports_filesystem_errors(tmp_path: Path) -> None:
    class CodeGenerator:
        def generate_files(self, _development_plan: DevelopmentPlan) -> dict[str, str]:
            return {"app.py": "VALUE = 1\n"}

    class FailingFileService:
        def write_planned_files(self, **_kwargs):
            raise OSError("workspace is read-only")

    agent = CodeAgent(
        llm_service=CodeGenerator(),
        file_service=FailingFileService(),
    )
    result = agent.generate_files(
        CodeAgentInput(
            development_plan=DevelopmentPlan(
                summary="Create a module.",
                files_to_create=["app.py"],
            )
        )
    )

    assert result.warnings == ["workspace is read-only"]
