from threading import Thread

import app.main as main
from app.models import (
    CodeAgentResult,
    DevelopmentPlan,
    TestAgentResult as _TestAgentResult,
    WorkflowAttempt,
)
from app.services.project_repository import ProjectRepository


def _successful_workflow_result() -> dict[str, object]:
    plan = DevelopmentPlan(
        summary="Create a greeting module.",
        files_to_create=["app/greeting.py"],
    )
    code_result = CodeAgentResult(
        files_created=["app/greeting.py"],
        generated_files=["app/greeting.py"],
    )
    test_result = _TestAgentResult(
        passed=1,
        generated_tests=["tests/test_greeting.py"],
        output="1 passed",
        exit_code=0,
    )
    attempt = WorkflowAttempt(
        attempt_number=1,
        files=["app/greeting.py"],
        test_output="1 passed",
        test_result=test_result,
        passed=True,
    )
    return {
        "plan": plan,
        "code_result": code_result,
        "test_result": test_result,
        "attempts": [attempt],
        "status": "success",
    }


def test_create_project_queues_background_work_and_gets_status_and_outputs(
    monkeypatch,
    project_client,
) -> None:
    client = project_client
    workspace = main.project_workspace_root
    generated_file = workspace / "app" / "greeting.py"
    generated_file.parent.mkdir(parents=True)
    generated_file.write_text("def greet(): return 'hello'\n", encoding="utf-8")
    generated_test = workspace / "tests" / "test_greeting.py"
    generated_test.parent.mkdir(parents=True)
    generated_test.write_text("def test_greet(): pass\n", encoding="utf-8")
    monkeypatch.setattr(
        main,
        "run_assistant",
        lambda *_args, **_kwargs: _successful_workflow_result(),
    )

    response = client.post(
        "/projects",
        json={"request": "Create a greeting module."},
    )
    assert response.status_code == 202
    created = response.json()
    assert created["id"]
    assert created["status"] == "queued"

    detail_response = client.get(f"/projects/{created['id']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["status"] == "completed"
    assert detail["request"] == "Create a greeting module."
    assert detail["logs"][0] == "Project queued for processing."
    assert "Workflow started." in detail["logs"]
    assert "Attempt 1: tests passed (1 passed)." in detail["logs"]
    assert detail["attempts"][0]["passed"] is True
    assert detail["test_result"]["passed"] == 1

    files_response = client.get(f"/projects/{created['id']}/files")
    assert files_response.status_code == 200
    assert files_response.json() == {
        "project_id": created["id"],
        "files": ["app/greeting.py", "tests/test_greeting.py"],
    }


def test_create_project_rejects_blank_request_and_unknown_projects_return_404(
    project_client,
) -> None:
    invalid = project_client.post("/projects", json={"request": "  \n "})
    assert invalid.status_code == 422

    missing_detail = project_client.get("/projects/unknown")
    missing_files = project_client.get("/projects/unknown/files")
    assert missing_detail.status_code == 404
    assert missing_files.status_code == 404


def test_project_request_rejects_empty_oversized_and_invalid_unicode_inputs(
    project_client,
) -> None:
    empty = project_client.post("/projects", json={"request": ""})
    oversized = project_client.post("/projects", json={"request": "x" * 10_001})
    invalid_unicode = project_client.post(
        "/projects",
        content=b'{"request":"\\ud800"}',
        headers={"content-type": "application/json"},
    )

    assert empty.status_code == 422
    assert oversized.status_code == 422
    assert invalid_unicode.status_code == 422


def test_project_request_accepts_valid_non_ascii_unicode(project_client) -> None:
    created = project_client.post(
        "/projects",
        json={"request": "Crée un module de calcul ✓"},
    )

    assert created.status_code == 202
    project = project_client.get(f"/projects/{created.json()['id']}").json()
    assert project["request"] == "Crée un module de calcul ✓"


def test_project_workflow_failure_is_saved_without_crashing_background_task(
    monkeypatch,
    project_client,
) -> None:
    def fail_workflow(*_args, **_kwargs):
        raise RuntimeError("provider failure")

    monkeypatch.setattr(main, "run_assistant", fail_workflow)
    created = project_client.post(
        "/projects",
        json={"request": "Create a greeting module."},
    )
    assert created.status_code == 202

    detail = project_client.get(f"/projects/{created.json()['id']}").json()
    assert detail["status"] == "failed"
    assert "Workflow failed with RuntimeError." in detail["logs"]
    assert "provider failure" not in " ".join(detail["logs"])


def test_project_file_listing_omits_paths_outside_workspace(
    monkeypatch,
    project_client,
    make_directory_link,
) -> None:
    outside_directory = main.project_workspace_root.parent / "outside-files"
    outside_directory.mkdir()
    outside_file = outside_directory / "leak.py"
    outside_file.write_text("secret = True\n", encoding="utf-8")
    linked_directory = main.project_workspace_root / "linked"
    main.project_workspace_root.mkdir(parents=True)
    make_directory_link(linked_directory, outside_directory)

    project = main.project_repository.create("Create a greeting module.")
    main.project_repository.update(
        project.id,
        code_result=CodeAgentResult(
            generated_files=[
                "../outside.py",
                "C:\\outside.py",
                "app/missing.py",
                "linked/leak.py",
            ]
        ),
        status="completed",
    )

    response = project_client.get(f"/projects/{project.id}/files")

    assert response.status_code == 200
    assert response.json()["files"] == []


def test_project_repository_updates_are_thread_safe() -> None:
    repository = ProjectRepository()
    project = repository.create("Create a module.")

    workers = [
        Thread(target=repository.append_log, args=(project.id, f"log-{index}"))
        for index in range(40)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    stored = repository.get(project.id)
    assert stored is not None
    assert len(stored.logs) == 41
    assert len(set(stored.logs[1:])) == 40
