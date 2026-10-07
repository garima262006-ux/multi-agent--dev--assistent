"""Deterministic requirement analysis; it never writes or modifies files."""

from app.models import DevelopmentPlan, RequirementInput


class RequirementAgent:
    """Convert natural language into a repeatable, Code-Agent-friendly plan."""

    def create_plan(self, request: RequirementInput) -> DevelopmentPlan:
        """Analyze a requirement and return a JSON-compatible plan.

        All listed paths are relative to ``workspace/``. This agent performs no
        filesystem operations and does not generate source code.
        """
        text = request.requirement
        normalized = text.casefold()
        is_api = self._contains_any(normalized, ("api", "endpoint", "http", "rest"))
        is_python = "python" in normalized
        is_calculator = "calculator" in normalized or "arithmetic" in normalized
        is_auth = self._contains_any(
            normalized,
            ("auth", "authentication", "login", "sign in", "sign-in"),
        )

        technologies: list[str] = []
        assumptions: list[str] = []
        if is_python:
            technologies.append("Python")
        else:
            technologies.append("Python")
            assumptions.append("Use Python because no programming language was specified.")
        if is_api:
            technologies.extend(["FastAPI", "Pydantic"])
            assumptions.extend(
                [
                    "Use REST-style HTTP endpoints because the requirement asks for an API.",
                    "Use FastAPI and Pydantic because no web framework or schema library was specified.",
                ]
            )
        if "pytest" not in normalized:
            technologies.append("pytest")
            assumptions.append("Use pytest for automated tests because no test framework was specified.")
        else:
            technologies.append("pytest")

        files_to_create = self._planned_files(
            is_api=is_api,
            is_calculator=is_calculator,
            is_auth=is_auth,
        )

        implementation_steps = self._implementation_steps(
            is_api=is_api,
            is_calculator=is_calculator,
            is_auth=is_auth,
        )
        testing_requirements = self._testing_requirements(
            is_api=is_api,
            is_calculator=is_calculator,
            is_auth=is_auth,
        )

        if is_calculator and self._contains_any(normalized, ("division", "divide")):
            assumptions.append(
                "The response or error behavior for division by zero is unspecified; "
                "reject it with a clear client error."
            )
        if is_api and not self._contains_any(normalized, ("database", "persist", "storage")):
            assumptions.append(
                "No persistence requirement was given; avoid adding a database unless "
                "implementation details require one."
            )
        if is_auth and not self._contains_any(normalized, ("jwt", "oauth", "session")):
            assumptions.append(
                "The authentication mechanism is unspecified; choose a standard, "
                "maintainable approach during implementation."
            )

        return DevelopmentPlan(
            summary=f"Build the requested software: {text}",
            assumptions=assumptions,
            technologies=self._unique(technologies),
            files_to_create=files_to_create,
            files_to_modify=[],
            implementation_steps=implementation_steps,
            testing_requirements=testing_requirements,
        )

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _planned_files(*, is_api: bool, is_calculator: bool, is_auth: bool) -> list[str]:
        paths = [
            "app/__init__.py",
            "app/main.py",
            "tests/test_api.py",
            "requirements.txt",
            "README.md",
        ]
        if is_api:
            paths.insert(1, "app/models.py")
        if is_calculator:
            paths.extend(["app/routers/__init__.py", "app/routers/calculator.py"])
        if is_auth:
            paths.extend(["app/services/__init__.py", "app/services/auth.py"])
        return paths

    @staticmethod
    def _implementation_steps(
        *,
        is_api: bool,
        is_calculator: bool,
        is_auth: bool,
    ) -> list[str]:
        steps: list[str] = []
        if is_api:
            steps.extend(
                [
                    "Define validated request and response schemas for each endpoint.",
                    "Create the FastAPI application and register the requested routes.",
                ]
            )
        if is_calculator:
            steps.extend(
                [
                    "Implement separate handlers for addition, subtraction, multiplication, and division.",
                    "Validate numeric inputs and return a clear client error for division by zero.",
                ]
            )
        if is_auth:
            steps.append("Implement the agreed authentication mechanism and protect required routes.")
        steps.extend(
            [
                "Add the required runtime and test dependencies.",
                "Document how to run the application and its tests.",
            ]
        )
        return steps

    @staticmethod
    def _testing_requirements(
        *,
        is_api: bool,
        is_calculator: bool,
        is_auth: bool,
    ) -> list[str]:
        requirements: list[str] = []
        if is_calculator:
            requirements.extend(
                [
                    "Test the addition, subtraction, multiplication, and division endpoints with representative numeric inputs and response data.",
                    "Test division by zero and verify the documented client-error response.",
                ]
            )
        elif is_api:
            requirements.append("Test each requested endpoint for valid input and expected response data.")
        if is_api:
            requirements.append("Test malformed or missing request fields and verify validation errors.")
        if is_auth:
            requirements.extend(
                [
                    "Test successful authentication with valid credentials.",
                    "Test rejection of invalid credentials and access to protected routes without authentication.",
                ]
            )
        requirements.append("Run the complete automated test suite with pytest.")
        return requirements
