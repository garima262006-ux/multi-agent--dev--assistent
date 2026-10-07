from pathlib import Path

import pytest

from app.agents.code_agent import CodeAgent
from app.models import CodeAgentInput, DevelopmentPlan
from app.services.workspace_file_service import WorkspaceFileService


class StubCodeGenerator:
    def __init__(self, generated_files: dict[str, str]) -> None:
        self.generated_files = generated_files

    def generate_files(self, development_plan: DevelopmentPlan) -> dict[str, str]:
        assert development_plan.files_to_create
        return self.generated_files


def test_code_agent_creates_file_in_workspace_and_reports_generated_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    plan = DevelopmentPlan(
        summary="Create a calculator API.",
        files_to_create=["app/main.py"],
    )
    agent = CodeAgent(
        llm_service=StubCodeGenerator({"app/main.py": "app = 'ready'\n"}),
        file_service=WorkspaceFileService(workspace),
    )

    result = agent.generate_files(CodeAgentInput(development_plan=plan))

    assert (workspace / "app" / "main.py").read_text(encoding="utf-8") == "app = 'ready'\n"
    assert result.files_created == ["app/main.py"]
    assert result.files_modified == []
    assert result.generated_files == ["app/main.py"]
    assert result.warnings == []


def test_code_agent_rejects_path_traversal_without_writing_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside_file = tmp_path / "outside.py"
    plan = DevelopmentPlan(
        summary="Create an unsafe file.",
        files_to_create=["../outside.py"],
    )
    agent = CodeAgent(
        llm_service=StubCodeGenerator({"../outside.py": "unexpected = True\n"}),
        file_service=WorkspaceFileService(workspace),
    )

    with pytest.raises(ValueError, match="inside workspace"):
        agent.generate_files(CodeAgentInput(development_plan=plan))

    assert not outside_file.exists()


def test_workspace_writer_refuses_to_overwrite_unplanned_existing_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    existing_file = workspace / "app.py"
    existing_file.parent.mkdir(parents=True)
    existing_file.write_text("existing = True\n", encoding="utf-8")
    service = WorkspaceFileService(workspace)

    with pytest.raises(FileExistsError):
        service.write_planned_files(
            files_to_create=["app.py"],
            files_to_modify=[],
            generated_files={"app.py": "replacement = True\n"},
        )

    assert existing_file.read_text(encoding="utf-8") == "existing = True\n"
