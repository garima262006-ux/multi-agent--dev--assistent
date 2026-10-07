"""Thread-safe in-memory repository for asynchronous project runs."""

from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from app.models import ProjectResponse


class ProjectRepository:
    """Store project workflow state behind a replaceable repository interface."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._projects: dict[str, ProjectResponse] = {}

    def create(self, request: str) -> ProjectResponse:
        """Create and return a new project in the queued state."""
        now = datetime.now(UTC)
        project = ProjectResponse(
            id=str(uuid4()),
            request=request,
            status="queued",
            logs=["Project queued for processing."],
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._projects[project.id] = project
        return project.model_copy(deep=True)

    def get(self, project_id: str) -> ProjectResponse | None:
        """Return a detached project snapshot, or ``None`` if it does not exist."""
        with self._lock:
            project = self._projects.get(project_id)
            return project.model_copy(deep=True) if project is not None else None

    def update(self, project_id: str, **changes: object) -> ProjectResponse | None:
        """Atomically apply validated project fields and refresh its timestamp."""
        with self._lock:
            project = self._projects.get(project_id)
            if project is None:
                return None
            updated = ProjectResponse.model_validate(
                {
                    **project.model_dump(),
                    **changes,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._projects[project_id] = updated
            return updated.model_copy(deep=True)

    def append_log(self, project_id: str, message: str) -> ProjectResponse | None:
        """Append one log message without exposing mutable stored state."""
        with self._lock:
            project = self._projects.get(project_id)
            if project is None:
                return None
            updated = ProjectResponse.model_validate(
                {
                    **project.model_dump(),
                    "logs": [*project.logs, message],
                    "updated_at": datetime.now(UTC),
                }
            )
            self._projects[project_id] = updated
            return updated.model_copy(deep=True)
