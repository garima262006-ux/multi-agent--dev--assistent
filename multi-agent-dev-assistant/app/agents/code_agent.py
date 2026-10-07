"""Code Agent: generate planned files and delegate safe writes to a service."""

from pathlib import Path
from typing import Protocol

from app.models import CodeAgentInput, CodeAgentResult, DevelopmentPlan
from app.services.llm_service import LLMServiceError
from app.services.workspace_file_service import WorkspaceFileService


class CodeGenerator(Protocol):
    """Provider-independent interface for generating planned source contents."""

    def generate_files(self, development_plan: DevelopmentPlan) -> dict[str, str]:
        """Return file contents keyed by the exact planned relative paths."""

    def generate_files_with_feedback(
        self,
        development_plan: DevelopmentPlan,
        previous_test_errors: list[str],
    ) -> dict[str, str]:
        """Return updated contents using earlier test failures as guidance."""


class CodeAgent:
    """Generate file contents for a development plan without executing them."""

    def __init__(
        self,
        llm_service: CodeGenerator | None = None,
        file_service: WorkspaceFileService | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        if llm_service is None:
            from app.services.llm_service import LLMService

            llm_service = LLMService()
        self._llm_service = llm_service
        self._file_service = file_service or WorkspaceFileService(workspace_root)

    def generate_files(self, agent_input: CodeAgentInput) -> CodeAgentResult:
        """Generate and write only files listed in the supplied development plan."""
        plan = agent_input.development_plan
        planned_paths = [*plan.files_to_create, *plan.files_to_modify]
        if not planned_paths:
            return CodeAgentResult(
                warnings=["Development plan does not declare any files to generate."]
            )

        try:
            if agent_input.previous_test_errors and hasattr(
                self._llm_service, "generate_files_with_feedback"
            ):
                generated_contents = self._llm_service.generate_files_with_feedback(
                    plan,
                    agent_input.previous_test_errors,
                )
            else:
                generated_contents = self._llm_service.generate_files(plan)
        except LLMServiceError as exc:
            return CodeAgentResult(warnings=[str(exc)])

        try:
            overwrite_paths = set(agent_input.files_to_overwrite)
            if not overwrite_paths.issubset(plan.files_to_create):
                raise ValueError(
                    "Only files created by an earlier attempt may be overwritten."
                )
            files_to_create = [
                path for path in plan.files_to_create if path not in overwrite_paths
            ]
            files_to_modify = [
                *plan.files_to_modify,
                *sorted(overwrite_paths),
            ]
            files_created, files_modified = self._file_service.write_planned_files(
                files_to_create=files_to_create,
                files_to_modify=files_to_modify,
                generated_files=generated_contents,
            )
        except OSError as exc:
            return CodeAgentResult(warnings=[str(exc)])
        return CodeAgentResult(
            files_created=files_created,
            files_modified=files_modified,
            generated_files=[*files_created, *files_modified],
        )
