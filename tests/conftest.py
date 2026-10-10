import pytest
import functools
from pathlib import Path


def _load_excluded_projects():
    """Load excluded project names from excluded_kicad_projects.txt"""
    excluded_file = Path(__file__).absolute().parent / "excluded_kicad_projects.txt"
    excluded = set()

    if not excluded_file.exists():
        return excluded

    with open(excluded_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                excluded.add(line)

    return excluded


def _kicad_test_projects():
    # Imported lazily: conftest.py is loaded before the typeguard pytest plugin
    # installs its import hook, so a module-level import would leave padne
    # uninstrumented.
    from padne.kicad import KiCadProject

    kicad_dir = Path(__file__).parent / "kicad"
    excluded_projects = _load_excluded_projects()

    projects = {}
    for project_dir in kicad_dir.iterdir():
        if any(excluded in project_dir.name for excluded in excluded_projects):
            continue

        pro_files = list(project_dir.glob("*.kicad_pro"))
        if not pro_files:
            continue

        projects[project_dir.name] = KiCadProject.from_pro_file(pro_files[0])

    return projects


def for_all_kicad_projects(_func=None, *, include=None, exclude=None):
    """Decorator parametrizing a test over the KiCad test projects as `project`."""
    if include is not None and exclude is not None:
        raise ValueError("Cannot specify both include and exclude.")

    def decorator(func):
        filtered_projects = [
            project
            for project in _kicad_test_projects().values()
            if (include is None or project.name in include)
                and (exclude is None or project.name not in exclude)
        ]
        filtered_projects.sort(key=lambda x: x.name)

        @pytest.mark.parametrize(
            "project",
            filtered_projects,
            ids=[project.name for project in filtered_projects],
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)


@pytest.fixture
def kicad_test_projects():
    """
    Fixture that provides a dictionary of KiCad test projects.

    Returns:
        dict: A dictionary where keys are project names and values are KiCadProject objects.
    """
    return _kicad_test_projects()


def pytest_generate_tests(metafunc):
    """Run solver_backend tests on every backend, skipping the ones not built."""
    if "solver_backend" not in metafunc.fixturenames:
        return
    # Imported lazily for typeguard, see _kicad_test_projects
    from padne import solver

    metafunc.parametrize("solver_backend", [
        pytest.param(backend, id=backend.value, marks=pytest.mark.skipif(
            backend not in solver.solver_backends(),
            reason=f"{backend.value} backend not available in this build"))
        for backend in solver.SolverBackend
    ])
