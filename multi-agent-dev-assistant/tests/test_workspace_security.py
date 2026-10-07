from pathlib import Path

import pytest

from app.services.code_executor import CodeExecutor
from app.services.workspace_file_service import WorkspaceFileService


def test_workspace_writer_rejects_parent_and_absolute_paths(
    tmp_path: Path,
    isolated_workspace: Path,
) -> None:
    service = WorkspaceFileService(isolated_workspace)
    outside_file = tmp_path / "outside.py"

    for path in ("../outside.py", str(outside_file), "C:\\outside.py"):
        with pytest.raises(ValueError, match="inside workspace"):
            service.write_planned_files(
                files_to_create=[path],
                files_to_modify=[],
                generated_files={path: "unsafe = True\n"},
            )
    assert not outside_file.exists()


def test_workspace_writer_rejects_symlinked_parent_directory(
    tmp_path: Path,
    isolated_workspace: Path,
    make_directory_link,
) -> None:
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    isolated_workspace.mkdir()
    linked_directory = isolated_workspace / "linked"
    make_directory_link(linked_directory, outside_directory)

    service = WorkspaceFileService(isolated_workspace)
    with pytest.raises(ValueError, match="inside workspace"):
        service.write_planned_files(
            files_to_create=["linked/escaped.py"],
            files_to_modify=[],
            generated_files={"linked/escaped.py": "unsafe = True\n"},
        )
    assert not (outside_directory / "escaped.py").exists()


def test_workspace_writer_refuses_symlinked_workspace_root(
    tmp_path: Path,
    make_directory_link,
) -> None:
    actual_workspace = tmp_path / "actual"
    actual_workspace.mkdir()
    linked_workspace = tmp_path / "linked-workspace"
    make_directory_link(linked_workspace, actual_workspace)

    with pytest.raises(ValueError, match="symbolic link"):
        WorkspaceFileService(linked_workspace)


def test_workspace_writer_validates_root_and_plan_consistency(
    tmp_path: Path,
    isolated_workspace: Path,
) -> None:
    file_root = tmp_path / "not-a-directory"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        WorkspaceFileService(file_root)

    service = WorkspaceFileService(isolated_workspace)
    invalid_plans = [
        (["same.py", "same.py"], [], {"same.py": "x = 1\n"}),
        ([], ["same.py", "same.py"], {"same.py": "x = 1\n"}),
        (["same.py"], ["same.py"], {"same.py": "x = 1\n"}),
        (["expected.py"], [], {}),
        ([], [], {"unexpected.py": "x = 1\n"}),
        (["expected.py"], [], {
            "expected.py": "x = 1\n",
            "unexpected.py": "y = 2\n",
        }),
    ]
    for creates, modifies, generated in invalid_plans:
        with pytest.raises(ValueError):
            service.write_planned_files(creates, modifies, generated)


def test_workspace_writer_updates_only_declared_existing_files(
    isolated_workspace: Path,
) -> None:
    existing_file = isolated_workspace / "app.py"
    existing_file.parent.mkdir(parents=True)
    existing_file.write_text("VALUE = 1\n", encoding="utf-8")
    service = WorkspaceFileService(isolated_workspace)

    created, modified = service.write_planned_files(
        files_to_create=[],
        files_to_modify=["app.py"],
        generated_files={"app.py": "VALUE = 2\n"},
    )

    assert created == []
    assert modified == ["app.py"]
    assert existing_file.read_text(encoding="utf-8") == "VALUE = 2\n"
    with pytest.raises(FileNotFoundError):
        service.write_planned_files(
            files_to_create=[],
            files_to_modify=["missing.py"],
            generated_files={"missing.py": "VALUE = 3\n"},
        )


def test_workspace_writer_rejects_empty_dot_and_null_paths(
    isolated_workspace: Path,
) -> None:
    service = WorkspaceFileService(isolated_workspace)
    for unsafe_path in ("", ".", "app\x00.py"):
        with pytest.raises(ValueError):
            service.write_planned_files(
                files_to_create=[unsafe_path],
                files_to_modify=[],
                generated_files={unsafe_path: "unsafe = True\n"},
            )


def test_code_executor_rejects_traversal_absolute_and_symlink_paths(
    tmp_path: Path,
    isolated_workspace: Path,
    make_directory_link,
) -> None:
    executor = CodeExecutor(isolated_workspace)
    outside_test = tmp_path / "outside_test.py"
    with pytest.raises(ValueError, match="inside workspace"):
        executor.run_pytest(test_file_paths=["../outside_test.py"])
    with pytest.raises(ValueError, match="inside workspace"):
        executor.run_pytest(test_file_paths=[str(outside_test)])

    isolated_workspace.mkdir(exist_ok=True)
    outside_test.write_text("def test_unexpected(): pass\n", encoding="utf-8")
    linked_directory = isolated_workspace / "linked"
    make_directory_link(linked_directory, tmp_path)
    with pytest.raises(ValueError, match="inside workspace"):
        executor.run_pytest(test_file_paths=["linked/outside_test.py"])


def test_code_executor_enforces_timeout_and_terminates_infinite_test(
    isolated_workspace: Path,
) -> None:
    test_file = isolated_workspace / "tests" / "test_hang.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "def test_infinite_loop():\n"
        "    while True:\n"
        "        pass\n",
        encoding="utf-8",
    )
    report = CodeExecutor(isolated_workspace).run_pytest(
        timeout_seconds=1,
        test_file_paths=["tests/test_hang.py"],
    )

    assert report.exit_code is None
    assert report.failed == 0
    assert report.errors == ["pytest exceeded the 1-second timeout."]


def test_code_executor_reports_invalid_timeout_missing_and_empty_tests(
    isolated_workspace: Path,
) -> None:
    executor = CodeExecutor(isolated_workspace)
    with pytest.raises(ValueError, match="between 1 and 300"):
        executor.run_pytest(timeout_seconds=0, test_file_paths=["test.py"])

    empty_report = executor.run_pytest(test_file_paths=[])
    assert "No generated workspace test files" in empty_report.errors[0]

    missing_report = executor.run_pytest(test_file_paths=["tests/missing.py"])
    assert "do not exist" in missing_report.errors[0]
