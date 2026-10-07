import json

import pytest

from app.agents.requirement_agent import RequirementAgent
from app.models import DevelopmentPlan, RequirementInput


def test_calculator_requirement_produces_structured_development_plan() -> None:
    request = RequirementInput(
        requirement=(
            "Create a Python calculator API with addition, subtraction, "
            "multiplication and division endpoints."
        )
    )

    result = RequirementAgent().create_plan(request)

    assert isinstance(result, DevelopmentPlan)
    payload = json.loads(result.model_dump_json())
    assert "Python calculator API" in payload["summary"]
    assert "app/__init__.py" in payload["files_to_create"]
    assert "app/main.py" in payload["files_to_create"]
    assert "app/routers/calculator.py" in payload["files_to_create"]
    assert payload["files_to_modify"] == []
    assert "Python" in payload["technologies"]
    assert any("FastAPI" in assumption for assumption in payload["assumptions"])
    assert any("addition" in step for step in payload["implementation_steps"])
    assert any("division" in item for item in payload["testing_requirements"])
    assert all("def " not in step for step in payload["implementation_steps"])
    assert set(payload) == {
        "summary",
        "assumptions",
        "technologies",
        "files_to_create",
        "files_to_modify",
        "implementation_steps",
        "testing_requirements",
    }
    assert payload == json.loads(RequirementAgent().create_plan(request).model_dump_json())


def test_requirement_input_rejects_blank_text() -> None:
    with pytest.raises(ValueError):
        RequirementInput(requirement="   ")
