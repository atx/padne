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


class TestDirectiveParser:

    def test_valid_voltage_directive(self):
        directive = "!padne VOLTAGE 5V R1.1 R2.1"
        spec = kicad.parse_padne_eeschema_directive(directive)
        assert spec.type == kicad.LumpedSpec.Type.VOLTAGE
        assert spec.value == 5.0
        assert spec.endpoint_a.designator == "R1"
        assert spec.endpoint_a.pad == "1"
        assert spec.endpoint_b.designator == "R2"
        assert spec.endpoint_b.pad == "1"

    def test_valid_resistor_directive(self):
        directive = "!padne RESISTOR 1k R3.2 R4.3"
        spec = kicad.parse_padne_eeschema_directive(directive)
        assert spec.type == kicad.LumpedSpec.Type.RESISTOR
        assert spec.value == 1000.0  # "1k" becomes 1000.0
        assert spec.endpoint_a.designator == "R3"
        assert spec.endpoint_a.pad == "2"
        assert spec.endpoint_b.designator == "R4"
        assert spec.endpoint_b.pad == "3"

    def test_valid_current_directive(self):
        directive = "!padne CURRENT 1A R5.1 R6.1"
        spec = kicad.parse_padne_eeschema_directive(directive)
        assert spec.type == kicad.LumpedSpec.Type.CURRENT
        assert spec.value == 1.0
        assert spec.endpoint_a.designator == "R5"
        assert spec.endpoint_a.pad == "1"
        assert spec.endpoint_b.designator == "R6"
        assert spec.endpoint_b.pad == "1"

    def test_invalid_token_count(self):
        directive = "!padne VOLTAGE 5V R1.1"  # Missing endpoint token
        with pytest.raises(ValueError, match="Directive must have 5 tokens"):
            kicad.parse_padne_eeschema_directive(directive)

    def test_unknown_directive_type(self):
        directive = "!padne UNKNOWN 5V R1.1 R2.1"
        with pytest.raises(ValueError, match="Unknown directive type"):
            kicad.parse_padne_eeschema_directive(directive)

    def test_parse_directives_from_simple_geometry(self, kicad_test_projects):
        # Get the simple_geometry project's schematic file
        project = kicad_test_projects["simple_geometry"]
        assert project.sch_path.exists(), "Schematic file of simple_geometry project does not exist"
        
        # Extract the raw directive strings from the schematic file
        directives = kicad.extract_lumped_from_eeschema(project.sch_path)
        # Expecting exactly two directives based on our simple_geometry project
        assert len(directives) == 2, f"Expected 2 directives, got {len(directives)}"
        
        # Parse each directive, then assign by type
        specs = [kicad.parse_padne_eeschema_directive(d) for d in directives]
        voltage_spec = next(spec for spec in specs if spec.type == kicad.LumpedSpec.Type.VOLTAGE)
        resistor_spec = next(spec for spec in specs if spec.type == kicad.LumpedSpec.Type.RESISTOR)
        
        # Validate the voltage directive
        assert voltage_spec.value == 1.0, "Voltage value should be 1.0"
        assert voltage_spec.endpoint_a.designator == "R2", "Voltage directive endpoint A designator should be R2"
        assert voltage_spec.endpoint_a.pad == "1", "Voltage directive endpoint A pad should be 1"
        assert voltage_spec.endpoint_b.designator == "R2", "Voltage directive endpoint B designator should be R2"
        assert voltage_spec.endpoint_b.pad == "2", "Voltage directive endpoint B pad should be 2"
        
        # Validate the resistor directive
        assert resistor_spec.value == 1000.0, "Resistor value should be 1000.0"
        assert resistor_spec.endpoint_a.designator == "R3", "Resistor directive endpoint A designator should be R3"
        assert resistor_spec.endpoint_a.pad == "1", "Resistor directive endpoint A pad should be 1"
        assert resistor_spec.endpoint_b.designator == "R3", "Resistor directive endpoint B designator should be R3"
        assert resistor_spec.endpoint_b.pad == "2", "Resistor directive endpoint B pad should be 2"
        # Endpoint missing the dot separator.
        directive = "!padne VOLTAGE 5V R11 R2.1"
        with pytest.raises(ValueError, match="Invalid endpoint format"):
            kicad.parse_padne_eeschema_directive(directive)
