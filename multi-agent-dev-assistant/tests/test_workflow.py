from pathlib import Path

from app.agents.requirement_agent import RequirementAgent
from app.agents.workflow import run_assistant
from app.models import AssistantRunRequest, DevelopmentPlan, RequirementInput
from app.services.llm_service import LLMService


class StubRequirementAgent(RequirementAgent):
    def create_plan(self, _request: RequirementInput) -> DevelopmentPlan:
        return DevelopmentPlan(
            summary="A calculator module.",
            files_to_create=["app/__init__.py", "app/calculator.py"],
        )


def test_workflow_passes_plan_through_code_and_test_agents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        LLMService,
        "generate_files",
        lambda _service, _plan: {
            "app/__init__.py": "",
            "app/calculator.py": "def add(a, b):\n    return a + b\n",
        },
    )
    monkeypatch.setattr(
        LLMService,
        "generate_tests",
        lambda _service, _plan, _paths, _test_path: (
            "def test_calculator_addition():\n"
            "    from app.calculator import add\n"
            "    assert add(2, 3) == 5\n"
        ),
    )

    result = run_assistant(
        AssistantRunRequest(requirement="Create a calculator API.", run_tests=True),
        workspace_root=tmp_path / "workspace",
        requirement_agent=StubRequirementAgent(),
    )

    assert isinstance(result["plan"], DevelopmentPlan)
    assert result["code_result"].files_created == ["app/__init__.py", "app/calculator.py"]
    assert result["test_result"] is not None
    assert result["test_result"].passed == 1, result["test_result"].errors[0][:1200]
    assert result["test_result"].failed == 0
