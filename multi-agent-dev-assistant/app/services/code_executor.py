"""Bounded pytest execution; never executes a command supplied by a user."""

import os
import re
import subprocess
import sys
from pathlib import Path, PureWindowsPath

from app.config import get_settings
from app.models import TestAgentResult


_PASSED_COUNT = re.compile(r"\b(\d+) passed\b")
_FAILED_COUNT = re.compile(r"\b(\d+) failed\b")


class CodeExecutor:
    """Execute explicitly selected tests using a fixed Python/pytest invocation."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        requested_root = workspace_root or get_settings().workspace_dir
        is_junction = getattr(requested_root, "is_junction", lambda: False)
        if requested_root.is_symlink() or is_junction():
            raise ValueError("The test workspace root cannot be a symbolic link.")
        if requested_root.exists() and not requested_root.is_dir():
            raise ValueError("The test workspace root must be a directory.")
        self._workspace_root = requested_root.resolve()
        self._workspace_root.mkdir(parents=True, exist_ok=True)

    def run_pytest(
        self,
        timeout_seconds: int = 60,
        test_file_paths: list[str] | None = None,
    ) -> TestAgentResult:
        """Run pytest on selected workspace test modules, never a user command."""
        if not 1 <= timeout_seconds <= 300:
            raise ValueError("Test timeout must be between 1 and 300 seconds.")
        if not test_file_paths:
            return TestAgentResult(
                errors=["No generated workspace test files were selected."],
                suggestions=["Generate tests before requesting test execution."],
            )

        resolved_paths = [self._resolve_test_path(path) for path in test_file_paths]
        missing = [
            path for path, resolved in zip(test_file_paths, resolved_paths)
            if not resolved.is_file()
        ]
        if missing:
            return TestAgentResult(
                errors=[f"Generated test files do not exist: {', '.join(missing)}"],
                suggestions=["Ensure the Test Agent writes the planned test module first."],
            )

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    *[str(path) for path in resolved_paths],
                ],
                cwd=self._workspace_root,
                capture_output=True,
                check=False,
                shell=False,
                text=True,
                timeout=timeout_seconds,
                env=self._test_environment(),
            )
        except subprocess.TimeoutExpired:
            error = f"pytest exceeded the {timeout_seconds}-second timeout."
            return TestAgentResult(
                errors=[error],
                output=error,
                suggestions=["Inspect slow or blocked tests before increasing the timeout."],
            )
        except OSError as exc:
            error = f"Could not start pytest: {exc}"
            return TestAgentResult(
                errors=[error],
                output=error,
                suggestions=["Install project dependencies and verify Python is available."],
            )

        output = f"{completed.stdout}\n{completed.stderr}".strip()
        summary = self._parse_summary(output)
        errors: list[str] = []
        suggestions: list[str] = []
        if completed.returncode == 1:
            errors.append(
                output[-2_000:] or f"pytest exited with status {completed.returncode}."
            )
            suggestions.append("Review the failure details and add or update focused tests.")
        elif completed.returncode not in (0, 5):
            errors.append(output[-2_000:] or "pytest failed before reporting test results.")

        if completed.returncode == 5:
            suggestions.append("Add tests to tests/; pytest did not discover any tests.")

        return TestAgentResult(
            passed=summary["passed"],
            failed=summary["failed"],
            errors=errors,
            suggestions=suggestions,
            output=output[-10_000:],
            exit_code=completed.returncode,
        )

    def _resolve_test_path(self, relative_path: str) -> Path:
        """Allow only relative test-file paths resolved beneath workspace/."""
        candidate = Path(relative_path.replace("\\", "/"))
        windows_path = PureWindowsPath(relative_path)
        if (
            candidate.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in candidate.parts
            or candidate == Path(".")
        ):
            raise ValueError("Test file paths must remain inside workspace/.")
        resolved = (self._workspace_root / candidate).resolve()
        if not resolved.is_relative_to(self._workspace_root):
            raise ValueError("Test file paths must remain inside workspace/.")
        return resolved

    def _test_environment(self) -> dict[str, str]:
        """Keep Python runnable without passing application secrets to test code."""
        safe_environment = {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "PROCESSOR_ARCHITECTURE",
        }
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in safe_environment
        }
        environment["PYTHONPATH"] = str(self._workspace_root)
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return environment

    @staticmethod
    def _parse_summary(output: str) -> dict[str, int]:
        """Extract pass/fail counts from pytest's short summary when available."""
        for line in reversed(output.splitlines()):
            if " passed" in line or " failed" in line:
                passed = _PASSED_COUNT.search(line)
                failed = _FAILED_COUNT.search(line)
                return {
                    "passed": int(passed.group(1)) if passed else 0,
                    "failed": int(failed.group(1)) if failed else 0,
                }
        return {"passed": 0, "failed": 0}
