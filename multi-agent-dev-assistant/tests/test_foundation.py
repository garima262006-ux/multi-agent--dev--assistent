from app.services.code_executor import CodeExecutor


def test_test_summary_parser_reports_passes_and_failures() -> None:
    assert CodeExecutor._parse_summary("1 failed, 2 passed in 0.05s") == {
        "passed": 2,
        "failed": 1,
    }
