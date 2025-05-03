import warnings
# This is to suppress pcbnew deprecation warning. Unfortunately the RPC API
# is not yet cooked enough for us
warnings.simplefilter("ignore", DeprecationWarning)

import pytest
import functools
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class KicadTestProject:
    """Represents a KiCad test project with paths to its files."""
    name: str
    pro_path: Optional[Path] = None  # Path to .kicad_pro file
    pcb_path: Optional[Path] = None  # Path to .kicad_pcb file
    sch_path: Optional[Path] = None  # Path to .kicad_sch file


def _kicad_test_projects():
    kicad_dir = Path(__file__).parent / "kicad"
    
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
        
        # Create a new KicadTestProject
        project = KicadTestProject(name=project_name)
        
        # Find .kicad_pro file
        pro_files = list(project_dir.glob("*.kicad_pro"))
        if pro_files:
            project.pro_path = pro_files[0]
        
        # Use base filename for finding related files
        # If we have a project file, use its stem, otherwise use directory name
        base_name = project.pro_path.stem if project.pro_path else project_name
        base_path = project_dir / base_name
        
        # Find .kicad_pcb file using with_suffix
        pcb_path = base_path.with_suffix('.kicad_pcb')
        if pcb_path.exists():
            project.pcb_path = pcb_path
        
        # Find .kicad_sch file using with_suffix
        sch_path = base_path.with_suffix('.kicad_sch')
        if sch_path.exists():
            project.sch_path = sch_path
        
        # Add project to dictionary
        projects[project_name] = project
    
    return projects


def for_all_kicad_projects(_func=None, *, include=None, exclude=None):
    """
    Fixture that provides a list of all KiCad test projects.
    
    Returns:
        list: A list of KicadTestProject objects.
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
        dict: A dictionary where keys are project names and values are KicadTestProject objects.
    """
    return _kicad_test_projects()
