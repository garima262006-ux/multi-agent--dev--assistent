"""Shared isolated fixtures for API and workspace tests."""

from collections.abc import Iterator
import os
from pathlib import Path
import subprocess

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.services.project_repository import ProjectRepository


@pytest.fixture
def isolated_workspace(tmp_path: Path) -> Path:
    """Provide a per-test workspace that cannot leave files in the project."""
    return tmp_path / "workspace"


@pytest.fixture
def make_directory_link():
    """Create a directory symlink, using a Windows junction if needed."""

    def create(link: Path, target: Path) -> None:
        try:
            link.symlink_to(target, target_is_directory=True)
            return
        except OSError:
            if os.name != "nt":
                pytest.skip("Directory symlink creation is unavailable.")
        result = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            pytest.skip("Directory symlink or junction creation is unavailable.")

    return create


@pytest.fixture
def project_client(
    isolated_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """Provide an API client with fresh in-memory project storage."""
    monkeypatch.setattr(main, "project_repository", ProjectRepository())
    monkeypatch.setattr(main, "project_workspace_root", isolated_workspace)
    with TestClient(main.app) as client:
        yield client
