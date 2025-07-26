import warnings
# This is to suppress pcbnew deprecation warning. Unfortunately the RPC API
# is not yet cooked enough for us
warnings.simplefilter("ignore", DeprecationWarning)

import pytest
import functools
from pathlib import Path

from padne.kicad import KiCadProject


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
    kicad_dir = Path(__file__).parent / "kicad"
    excluded_projects = _load_excluded_projects()

    # Dictionary to store all discovered projects
    projects = {}

    # Check if the kicad test directory exists
    if not kicad_dir.exists() or not kicad_dir.is_dir():
        return projects

    # Scan through each subdirectory in the kicad test directory
    for project_dir in kicad_dir.iterdir():
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name

        # Check if project is in the excluded list
        if any(excluded in project_name for excluded in excluded_projects):
            continue

        # Find .kicad_pro file
        pro_files = list(project_dir.glob("*.kicad_pro"))
        if not pro_files:
            continue  # Skip directories without project files

        pro_path = pro_files[0]

        try:
            # Create KiCadProject using the from_pro_file classmethod
            project = KiCadProject.from_pro_file(pro_path)
            projects[project_name] = project
        except FileNotFoundError:
            # Skip projects with missing files
            continue

    return projects


def for_all_kicad_projects(_func=None, *, include=None, exclude=None):
    """
    Decorator that provides a list of all KiCad test projects.

    Returns:
        list: A list of KiCadProject objects.
    """
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
