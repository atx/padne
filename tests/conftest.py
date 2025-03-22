import warnings
# This is to suppress pcbnew deprecation warning. Unfortunately the RPC API
# is not yet cooked enough for us
warnings.simplefilter("ignore", DeprecationWarning)

import pytest
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


@pytest.fixture
def kicad_test_projects():
    """
    Fixture that provides a dictionary of KiCad test projects.
    
    Returns a dictionary where:
    - Keys are the project names (folder names in tests/kicad/)
    - Values are KicadTestProject objects containing paths to project files
    """
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
