"""Safe file writing for generated content inside the configured workspace."""

from pathlib import Path, PureWindowsPath

from app.config import get_settings


class WorkspaceFileService:
    """Create and modify only explicitly planned files within one workspace root."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        requested_root = workspace_root or get_settings().workspace_dir
        is_junction = getattr(requested_root, "is_junction", lambda: False)
        if requested_root.is_symlink() or is_junction():
            raise ValueError("The workspace root cannot be a symbolic link.")
        if requested_root.exists() and not requested_root.is_dir():
            raise ValueError("The workspace root must be a directory.")
        self._workspace_root = requested_root.resolve()
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        if not self._workspace_root.is_dir():
            raise ValueError("The workspace root must be a directory.")

    def write_planned_files(
        self,
        files_to_create: list[str],
        files_to_modify: list[str],
        generated_files: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        """Write provided contents, rejecting unplanned or unsafe paths first."""
        self._validate_plan(files_to_create, files_to_modify, generated_files)

        create_targets = {
            relative_path: self._resolve_path(relative_path)
            for relative_path in files_to_create
        }
        modify_targets = {
            relative_path: self._resolve_path(relative_path)
            for relative_path in files_to_modify
        }

        for relative_path, destination in create_targets.items():
            if destination.exists():
                raise FileExistsError(
                    f"Cannot create '{relative_path}': the workspace file already exists."
                )
        for relative_path, destination in modify_targets.items():
            if not destination.is_file():
                raise FileNotFoundError(
                    f"Cannot modify '{relative_path}': the workspace file does not exist."
                )

        resolved_targets = [
            *create_targets.values(),
            *modify_targets.values(),
        ]
        if len(resolved_targets) != len(set(resolved_targets)):
            raise ValueError("Multiple planned paths resolve to the same workspace file.")

        for targets in (create_targets, modify_targets):
            for relative_path, destination in targets.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Re-resolve after directory creation to catch pre-existing symlinks.
                if not destination.resolve().is_relative_to(self._workspace_root):
                    raise ValueError(
                        f"File path '{relative_path}' resolves outside workspace/."
                    )
                mode = "x" if targets is create_targets else "w"
                with destination.open(mode, encoding="utf-8", newline="") as file:
                    file.write(generated_files[relative_path])

        return list(create_targets), list(modify_targets)

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path and reject traversal and absolute path syntax."""
        if not relative_path or "\0" in relative_path:
            raise ValueError("Workspace file paths must be non-empty relative paths.")
        normalized_path = relative_path.replace("\\", "/")
        windows_path = PureWindowsPath(relative_path)
        candidate = Path(normalized_path)
        if (
            candidate.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in candidate.parts
            or candidate == Path(".")
        ):
            raise ValueError("Code Agent file paths must remain inside workspace/.")

        destination = (self._workspace_root / candidate).resolve()
        if not destination.is_relative_to(self._workspace_root):
            raise ValueError("Code Agent file paths must remain inside workspace/.")
        return destination

    @staticmethod
    def _validate_plan(
        files_to_create: list[str],
        files_to_modify: list[str],
        generated_files: dict[str, str],
    ) -> None:
        """Require one generated body per declared path and no duplicate targets."""
        create_set = set(files_to_create)
        modify_set = set(files_to_modify)
        planned_set = create_set | modify_set
        if len(create_set) != len(files_to_create):
            raise ValueError("The development plan contains duplicate create paths.")
        if len(modify_set) != len(files_to_modify):
            raise ValueError("The development plan contains duplicate modify paths.")
        if create_set & modify_set:
            raise ValueError("A file cannot be planned for both creation and modification.")
        if set(generated_files) != planned_set:
            missing = sorted(planned_set - set(generated_files))
            unexpected = sorted(set(generated_files) - planned_set)
            details = []
            if missing:
                details.append(f"missing generated contents for: {', '.join(missing)}")
            if unexpected:
                details.append(f"unplanned generated files: {', '.join(unexpected)}")
            raise ValueError("; ".join(details))
