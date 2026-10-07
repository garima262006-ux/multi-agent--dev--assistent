"""FastAPI entry point for the Multi-Agent Dev Assistant foundation."""

import logging
from pathlib import Path, PureWindowsPath
import re
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agents.requirement_agent import RequirementAgent
from app.agents.workflow import run_assistant
from app.config import get_settings
from app.models import (
    AssistantRunRequest,
    AssistantRunResult,
    DevelopmentPlan,
    ProjectCreateRequest,
    ProjectCreateResponse,
    ProjectFilesResponse,
    ProjectResponse,
    RequirementInput,
)
from app.services.project_repository import ProjectRepository
from app.request_context import RequestIdLogFilter, request_id_context

settings = get_settings()
logging.basicConfig(level=settings.log_level)
for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIdLogFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"
        )
    )

app = FastAPI(
    title="Multi-Agent Dev Assistant",
    version="0.1.0",
    description="Requirement, code, and test agents with asynchronous project runs.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
requirement_agent = RequirementAgent()
project_repository = ProjectRepository()
project_workspace_root = settings.workspace_dir
logger = logging.getLogger(__name__)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next,
) -> Response:
    """Attach a validated incoming or newly generated ID to request logs/responses."""
    supplied_id = request.headers.get("X-Request-ID", "")
    request_id = (
        supplied_id
        if _REQUEST_ID_PATTERN.fullmatch(supplied_id)
        else str(uuid4())
    )
    request.state.request_id = request_id
    token = request_id_context.set(request_id)
    try:
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "Unhandled exception while handling %s %s exception_type=%s",
                request.method,
                request.url.path,
                type(exc).__name__,
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error.",
                    "request_id": request_id,
                },
            )
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "HTTP request completed request_id=%s method=%s path=%s status=%d",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
        )
        return response
    finally:
        request_id_context.reset(token)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Log unexpected failures and return a generic, non-sensitive response."""
    request_id = getattr(request.state, "request_id", None) or str(uuid4())
    logger.error(
        "Unhandled exception while handling %s %s exception_type=%s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return validation details without echoing unsafe or unencodable input."""
    safe_errors = [
        {
            "loc": error.get("loc", ()),
            "msg": error.get("msg", "Invalid request."),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": safe_errors},
        headers={
            "X-Request-ID": getattr(request.state, "request_id", None) or str(uuid4())
        },
    )


def _run_project_workflow(project_id: str) -> None:
    """Run a queued project and persist its status, outputs, and attempt logs."""
    project = project_repository.get(project_id)
    if project is None:
        return
    project_repository.update(project_id, status="running")
    project_repository.append_log(project_id, "Workflow started.")
    try:
        result = run_assistant(
            AssistantRunRequest(requirement=project.request, run_tests=True),
            workspace_root=project_workspace_root,
        )
    except Exception as exc:
        failure_message = f"Workflow failed with {type(exc).__name__}."
        logger.error("Project %s %s", project_id, failure_message)
        project_repository.append_log(project_id, failure_message)
        project_repository.update(project_id, status="failed")
        return

    attempts = result.get("attempts", [])
    for attempt in attempts:
        test_result = attempt.test_result
        if test_result is None:
            summary = "completed without test execution"
        elif attempt.passed:
            summary = f"tests passed ({test_result.passed} passed)"
        else:
            summary = f"tests failed ({test_result.failed} failed)"
        project_repository.append_log(
            project_id,
            f"Attempt {attempt.attempt_number}: {summary}.",
        )

    succeeded = result.get("status") == "success"
    project_repository.append_log(
        project_id,
        "Workflow completed successfully." if succeeded else "Workflow failed.",
    )
    project_repository.update(
        project_id,
        status="completed" if succeeded else "failed",
        development_plan=result.get("plan"),
        code_result=result.get("code_result"),
        test_result=result.get("test_result"),
        attempts=attempts,
    )


def _workspace_file_exists(relative_path: str) -> bool:
    """Check a generated relative path safely beneath the project workspace."""
    candidate = Path(relative_path.replace("\\", "/"))
    windows_path = PureWindowsPath(relative_path)
    if (
        not relative_path
        or candidate.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in candidate.parts
    ):
        return False
    workspace_root = project_workspace_root.resolve()
    resolved = (workspace_root / candidate).resolve()
    return resolved.is_relative_to(workspace_root) and resolved.is_file()


@app.get("/")
def index() -> dict[str, object]:
    """Show a helpful entry point when the API root is opened in a browser."""
    return {
        "name": "Multi-Agent Dev Assistant",
        "status": "running",
        "message": "Use the API documentation or one of the endpoints below.",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "plan_requirement": {
                "method": "POST",
                "path": "/api/requirements/plan",
            },
            "run_assistant": {
                "method": "POST",
                "path": "/api/assistant/run",
            },
            "create_project": {
                "method": "POST",
                "path": "/projects",
            },
        },
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.post(
    "/projects",
    response_model=ProjectCreateResponse,
    status_code=202,
)
def create_project(
    request: ProjectCreateRequest,
    background_tasks: BackgroundTasks,
) -> ProjectCreateResponse:
    """Queue a project workflow and return its identifier immediately."""
    project = project_repository.create(request.request)
    background_tasks.add_task(_run_project_workflow, project.id)
    return ProjectCreateResponse(id=project.id, status="queued")


@app.get("/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str) -> ProjectResponse:
    """Return the current status, logs, attempts, and results for a project."""
    project = project_repository.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


@app.get("/projects/{project_id}/files", response_model=ProjectFilesResponse)
def get_project_files(project_id: str) -> ProjectFilesResponse:
    """List generated files that currently exist inside workspace/."""
    project = project_repository.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    paths: list[str] = []
    if project.code_result is not None:
        paths.extend(project.code_result.generated_files)
    if project.test_result is not None:
        paths.extend(project.test_result.generated_tests)
    safe_existing_paths = [
        path for path in dict.fromkeys(paths) if _workspace_file_exists(path)
    ]
    return ProjectFilesResponse(project_id=project_id, files=safe_existing_paths)


@app.post("/api/requirements/plan", response_model=DevelopmentPlan)
def create_development_plan(request: RequirementInput) -> DevelopmentPlan:
    """Create a structured plan without writing or modifying source code."""
    try:
        return requirement_agent.create_plan(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/assistant/run", response_model=AssistantRunResult)
def run_development_assistant(request: AssistantRunRequest) -> AssistantRunResult:
    """Run the Requirement and Code Agents, then optionally generate and run tests."""
    try:
        result = run_assistant(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AssistantRunResult(
        development_plan=result["plan"],
        code_result=result["code_result"],
        test_result=result.get("test_result"),
        status=result.get("status", "success"),
        attempts=result.get("attempts", []),
    )
