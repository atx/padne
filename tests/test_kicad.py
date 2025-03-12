import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from padne import kicad


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


def test_fixture_files_exist(kicad_test_projects):
    """Test that all the projects in the test fixture have existing files."""
    # Skip if no projects were found
    if not kicad_test_projects:
        pytest.skip("No KiCad test projects found")
    
    # Check that at least one project was found
    assert len(kicad_test_projects) > 0, "No KiCad test projects were found"
    
    # Check that all project files that were identified actually exist
    for project_name, project in kicad_test_projects.items():
        # Check project name is not empty
        assert project.name, f"Project has empty name"
        
        # Check that project files exist if they were found
        assert project.pro_path.exists(), f"Project file does not exist: {project.pro_path}"
        assert project.pcb_path.exists(), f"PCB file does not exist: {project.pcb_path}"
        assert project.sch_path.exists(), f"Schematic file does not exist: {project.sch_path}"


def test_gerber_render_outputs_something(kicad_test_projects):
    """Test that the gerber rendering process outputs valid layer data."""
    
    project = kicad_test_projects["simple_geometry"]
    
    # Skip if the PCB file doesn't exist
    # Render gerbers from the PCB file
    layers = kicad.render_gerbers_from_kicad(project.pcb_path)
    
    # Check that we got some layers back
    assert len(layers) > 0, "No layers were rendered from the PCB file"
    
    # Check that each layer has valid geometry
    for layer in layers:
        assert layer.name, "Layer has no name"
        assert layer.layer_id >= 0, "Layer has invalid ID"
        assert not layer.geometry.is_empty, "Layer geometry is empty"

    # The simple_geometry project does not have a B.Cu layer
    assert sorted([layer.name for layer in layers]) == ["F.Cu"]
