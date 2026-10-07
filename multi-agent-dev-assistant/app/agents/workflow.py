"""LangGraph wiring for Requirement Agent -> Code Agent -> Test Agent handoffs."""

import logging
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.code_agent import CodeAgent
from app.agents.requirement_agent import RequirementAgent
from app.agents.test_agent import TestAgent
from app.config import get_settings
from app.models import (
    AssistantRunRequest,
    CodeAgentInput,
    CodeAgentResult,
    DevelopmentPlan,
    RequirementRequest,
    TestAgentInput,
    TestAgentResult,
    WorkflowAttempt,
)

logger = logging.getLogger(__name__)


class AssistantState(TypedDict, total=False):
    """Values exchanged by workflow nodes; agent payloads remain Pydantic models."""

    request: RequirementRequest
    plan: DevelopmentPlan
    code_result: CodeAgentResult
    test_result: TestAgentResult
    run_tests: bool
    max_retries: int
    attempt_number: int
    attempts: list[WorkflowAttempt]
    previous_test_errors: list[str]
    files_to_overwrite: list[str]
    status: str


def build_agent_workflow(
    workspace_root: Path | None = None,
    requirement_agent: RequirementAgent | None = None,
) -> CompiledStateGraph:
    """Build the typed, sequential agent graph.

    The Code Agent generates plan-declared files through the isolated LLM service.
    The Test Agent runs only after the Code Agent reports successful writes.
    """
    requirements = requirement_agent or RequirementAgent()
    code = CodeAgent(workspace_root=workspace_root)
    tests = TestAgent(workspace_root=workspace_root)

    def requirement_node(state: AssistantState) -> AssistantState:
        logger.info("Workflow stage=requirement")
        plan = requirements.create_plan(state["request"])
        return {"plan": plan}

    def code_node(state: AssistantState) -> AssistantState:
        attempt_number = state.get("attempt_number", 0) + 1
        logger.info("Workflow stage=code attempt=%d", attempt_number)
        result = code.generate_files(
            CodeAgentInput(
                development_plan=state["plan"],
                previous_test_errors=state.get("previous_test_errors", []),
                files_to_overwrite=state.get("files_to_overwrite", []),
            )
        )
        return {
            "code_result": result,
            "attempt_number": attempt_number,
            "files_to_overwrite": list(
                dict.fromkeys(
                    [
                        *state.get("files_to_overwrite", []),
                        *result.files_created,
                    ]
                )
            ),
        }

    def test_node(state: AssistantState) -> AssistantState:
        logger.info("Workflow stage=test attempt=%d", state["attempt_number"])
        return {
            "test_result": tests.run_tests(
                TestAgentInput(
                    development_plan=state["plan"],
                    generated_files=state["code_result"].generated_files,
                )
            )
        }

    def record_test_attempt_node(state: AssistantState) -> AssistantState:
        result = state["test_result"]
        passed = (
            result.exit_code == 0
            and result.failed == 0
            and not result.errors
        )
        attempt = WorkflowAttempt(
            attempt_number=state["attempt_number"],
            files=state["code_result"].generated_files,
            test_output=result.output or "\n".join(result.errors) or None,
            test_result=result,
            passed=passed,
        )
        feedback = list(result.errors)
        if result.failed:
            feedback.append(f"pytest reported {result.failed} failed test(s).")
        if result.exit_code not in (None, 0):
            feedback.append(f"pytest exited with status {result.exit_code}.")
        return {
            "attempts": [*state.get("attempts", []), attempt],
            "previous_test_errors": feedback,
            "status": "success" if passed else "failed",
        }

    def record_code_attempt_node(state: AssistantState) -> AssistantState:
        result = state["code_result"]
        passed = bool(result.generated_files) and not result.warnings
        attempt = WorkflowAttempt(
            attempt_number=state["attempt_number"],
            files=result.generated_files,
            test_output="\n".join(result.warnings) or None,
            passed=passed,
        )
        return {
            "attempts": [*state.get("attempts", []), attempt],
            "status": "success" if passed else "failed",
        }

    def route_after_code(state: AssistantState) -> str:
        code_result = state["code_result"]
        return (
            "test"
            if code_result.generated_files
            and not code_result.warnings
            and state.get("run_tests", True)
            else "record"
        )

    def route_after_test(state: AssistantState) -> str:
        if state["attempts"][-1].passed:
            return "end"
        if state["attempt_number"] <= state["max_retries"]:
            logger.info(
                "Workflow stage=retry attempt=%d next_attempt=%d max_retries=%d",
                state["attempt_number"],
                state["attempt_number"] + 1,
                state["max_retries"],
            )
            return "code"
        logger.info(
            "Workflow stage=retry status=exhausted attempts=%d",
            state["attempt_number"],
        )
        return "end"

    graph = StateGraph(AssistantState)
    graph.add_node("requirement", requirement_node)
    graph.add_node("code", code_node)
    graph.add_node("test", test_node)
    graph.add_node("record_test_attempt", record_test_attempt_node)
    graph.add_node("record_code_attempt", record_code_attempt_node)
    graph.add_edge(START, "requirement")
    graph.add_edge("requirement", "code")
    graph.add_conditional_edges(
        "code",
        route_after_code,
        {"test": "test", "record": "record_code_attempt"},
    )
    graph.add_edge("record_code_attempt", END)
    graph.add_edge("test", "record_test_attempt")
    graph.add_conditional_edges(
        "record_test_attempt",
        route_after_test,
        {"code": "code", "end": END},
    )
    return graph.compile()


def run_assistant(
    request: AssistantRunRequest,
    workspace_root: Path | None = None,
    requirement_agent: RequirementAgent | None = None,
) -> AssistantState:
    """Execute the full graph and return each completed agent's structured output."""
    workflow = build_agent_workflow(workspace_root, requirement_agent)
    max_retries = get_settings().max_retries
    initial_state: AssistantState = {
        "request": RequirementRequest(requirement=request.requirement),
        "run_tests": request.run_tests,
        "max_retries": max_retries,
        "attempt_number": 0,
        "attempts": [],
        "previous_test_errors": [],
        "files_to_overwrite": [],
    }
    return workflow.invoke(
        initial_state,
        config={"recursion_limit": max(25, 3 * (max_retries + 1) + 1)},
    )
