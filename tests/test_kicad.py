import warnings
# This is to suppress pcbnew deprecation warning. Unfortunately the RPC API
# is not yet cooked enough for us
warnings.simplefilter("ignore", DeprecationWarning)

import pytest
import pcbnew
import shapely.geometry

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from padne import kicad, problem


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
        directive = kicad.ParsedDirective.from_string("!padne VOLTAGE 5V R1.1 R2.1")
        spec = kicad.parse_lumped_spec_directive(directive)
        assert spec.type == problem.Lumped.Type.VOLTAGE
        assert spec.value == 5.0
        assert spec.endpoint_a.designator == "R1"
        assert spec.endpoint_a.pad == "1"
        assert spec.endpoint_b.designator == "R2"
        assert spec.endpoint_b.pad == "1"

    def test_valid_resistor_directive(self):
        directive = kicad.ParsedDirective.from_string("!padne RESISTANCE 1k R3.2 R4.3")
        spec = kicad.parse_lumped_spec_directive(directive)
        assert spec.type == problem.Lumped.Type.RESISTANCE
        assert spec.value == 1000.0  # "1k" becomes 1000.0
        assert spec.endpoint_a.designator == "R3"
        assert spec.endpoint_a.pad == "2"
        assert spec.endpoint_b.designator == "R4"
        assert spec.endpoint_b.pad == "3"

    def test_valid_current_directive(self):
        directive = kicad.ParsedDirective.from_string("!padne CURRENT 1A R5.1 R6.1")
        spec = kicad.parse_lumped_spec_directive(directive)
        assert spec.type == problem.Lumped.Type.CURRENT
        assert spec.value == 1.0
        assert spec.endpoint_a.designator == "R5"
        assert spec.endpoint_a.pad == "1"
        assert spec.endpoint_b.designator == "R6"
        assert spec.endpoint_b.pad == "1"

    def test_invalid_token_count(self):
        with pytest.raises(ValueError, match="Invalid directive format"):
            d = kicad.ParsedDirective.from_string("!padne VOLTAGE 5V R1.1")  # Missing endpoint token
            kicad.parse_lumped_spec_directive(d)

    def test_unknown_directive_type(self):
        directive = kicad.ParsedDirective.from_string("!padne UNKNOWN 5V R1.1 R2.1")
        with pytest.raises(ValueError, match="Unknown directive type"):
            kicad.parse_lumped_spec_directive(directive)

    def test_parse_directives_from_simple_geometry(self, kicad_test_projects):
        # Get the simple_geometry project's schematic file
        project = kicad_test_projects["simple_geometry"]
        assert project.sch_path.exists(), "Schematic file of simple_geometry project does not exist"
        
        # Extract the raw directive strings from the schematic file
        directives = kicad.process_directives(
            kicad.extract_directives_from_eeschema(project.sch_path)
        )
        # Expecting exactly two directives based on our simple_geometry project
        assert len(directives.lumpeds) == 2, f"Expected 2 lumped elements, got {len(directives)}"

        print(directives.lumpeds[0].type.__class__)
        print(directives.lumpeds)
        
        # Parse each directive, then assign by type
        voltage_spec = next(
            spec for spec in directives.lumpeds
            if spec.type == problem.Lumped.Type.VOLTAGE
        )
        resistor_spec = next(
            spec for spec in directives.lumpeds
            if spec.type == problem.Lumped.Type.RESISTANCE
        )
        
        # Validate the voltage directive
        assert voltage_spec.value == 1.0, "Voltage value should be 1.0"
        assert voltage_spec.endpoint_a.designator == "R2", \
            "Voltage directive endpoint A designator should be R2"
        assert voltage_spec.endpoint_a.pad == "1", \
            "Voltage directive endpoint A pad should be 1"
        assert voltage_spec.endpoint_b.designator == "R2", \
            "Voltage directive endpoint B designator should be R2"
        assert voltage_spec.endpoint_b.pad == "2", \
            "Voltage directive endpoint B pad should be 2"
        
        # Validate the resistor directive
        assert resistor_spec.value == 0.1
        assert resistor_spec.endpoint_a.designator == "R3", \
            "Resistor directive endpoint A designator should be R3"
        assert resistor_spec.endpoint_a.pad == "1", \
            "Resistor directive endpoint A pad should be 1"
        assert resistor_spec.endpoint_b.designator == "R3", \
            "Resistor directive endpoint B designator should be R3"
        assert resistor_spec.endpoint_b.pad == "2", \
            "Resistor directive endpoint B pad should be 2"


class TestPadFinder:

    def test_simple_geometry_pad(self, kicad_test_projects):
        # Get the simple_geometry project's PCB file
        project = kicad_test_projects["simple_geometry"]
        assert project.pcb_path is not None, "simple_geometry project must have a PCB file"
        
        # Load the KiCad board from the PCB file
        board = pcbnew.LoadBoard(str(project.pcb_path))
        
        # Try to find the pad R2.1
        layer_name, point = kicad.find_pad_location(board, "R3", "1")

        assert layer_name == "F.Cu", "Pad should be on the F.Cu layer"

        assert abs(point.x - 129) < 1e-3, "Pad X coordinate should be 129"
        assert abs(point.y - 101.375) < 1e-3, "Pad Y coordinate should be 129"


class TestLoadKicadProject:
    """Tests for the load_kicad_project function."""

    def test_basic_loading(self, kicad_test_projects):
        """Test that the function loads a project successfully."""
        project = kicad_test_projects["simple_geometry"]
        result = kicad.load_kicad_project(project.pro_path)
        
        # Check that we got a Problem object back
        assert isinstance(result, problem.Problem)
        # Should have at least one layer (F.Cu)
        assert len(result.layers) >= 1
        # Should have our two lumped elements
        assert len(result.lumpeds) == 2

    def test_file_not_found_handling(self):
        """Test that appropriate exceptions are raised for missing files."""
        with pytest.raises(FileNotFoundError, match="Project file not found"):
            kicad.load_kicad_project(Path("/nonexistent/file.kicad_pro"))

    def test_layer_properties(self, kicad_test_projects):
        """Test that the loaded layers have expected properties."""
        project = kicad_test_projects["simple_geometry"]
        result = kicad.load_kicad_project(project.pro_path)
        
        # Check the F.Cu layer specifically
        f_cu_layer = next(layer for layer in result.layers if layer.name == "F.Cu")
        assert f_cu_layer is not None
        assert isinstance(f_cu_layer.shape, shapely.geometry.MultiPolygon)
        assert not f_cu_layer.shape.is_empty

    def test_custom_resistivity(self, kicad_test_projects, monkeypatch):
        """Test that custom resistivity is applied correctly."""
        project = kicad_test_projects["simple_geometry"]

        result = kicad.load_kicad_project(project.pro_path)
        
        # F.Cu layer should have the custom resistivity
        f_cu_layer = next(layer for layer in result.layers if layer.name == "F.Cu")
        assert f_cu_layer.conductance == 212.0

    def test_lumped_elements(self, kicad_test_projects):
        """Test that lumped elements are loaded correctly."""
        project = kicad_test_projects["simple_geometry"]
        result = kicad.load_kicad_project(project.pro_path)
        
        # Find the voltage source and resistor
        voltage_source = next(l for l in result.lumpeds if l.type == problem.Lumped.Type.VOLTAGE)
        resistor = next(l for l in result.lumpeds if l.type == problem.Lumped.Type.RESISTANCE)
        
        # Check voltage source properties
        assert voltage_source.value == 1.0
        # Check that it's connected to component R2, pads 1 and 2
        r2_1_point = kicad.find_pad_location(pcbnew.LoadBoard(str(project.pcb_path)), "R2", "1")[1]
        r2_2_point = kicad.find_pad_location(pcbnew.LoadBoard(str(project.pcb_path)), "R2", "2")[1]
        
        assert (voltage_source.a_point.x == r2_1_point.x and 
                voltage_source.a_point.y == r2_1_point.y)
        assert (voltage_source.b_point.x == r2_2_point.x and 
                voltage_source.b_point.y == r2_2_point.y)
        
        # Check resistor properties
        assert resistor.value == 0.1
        # Check that it's connected to component R3, pads 1 and 2
        r3_1_point = kicad.find_pad_location(pcbnew.LoadBoard(str(project.pcb_path)), "R3", "1")[1]
        r3_2_point = kicad.find_pad_location(pcbnew.LoadBoard(str(project.pcb_path)), "R3", "2")[1]
        
        assert (resistor.a_point.x == r3_1_point.x and 
                resistor.a_point.y == r3_1_point.y)
        assert (resistor.b_point.x == r3_2_point.x and 
                resistor.b_point.y == r3_2_point.y)
                
    def test_lumped_points_inside_layers(self, kicad_test_projects):
        """
        Test that for all test projects, the start and end points of lumped elements 
        are located inside their respective layer shapes.
        """
        # Skip if no projects were found
        if not kicad_test_projects:
            pytest.skip("No KiCad test projects found")
        
        for project_name, project in kicad_test_projects.items():
            # Load the KiCad project
            try:
                kicad_problem = kicad.load_kicad_project(project.pro_path)
            except Exception as e:
                pytest.fail(f"Failed to load project {project_name}: {e}")
            
            # Skip if no lumped elements
            if not kicad_problem.lumpeds:
                continue
                
            # For each lumped element, verify that its endpoints are inside the layers
            for i, lumped in enumerate(kicad_problem.lumpeds):
                a_point_inside = lumped.a_layer.shape.contains(lumped.a_point)
                b_point_inside = lumped.b_layer.shape.contains(lumped.b_point)
                
                assert a_point_inside, (
                    f"Project {project_name}, lumped element {i} ({lumped.type.name}): "
                    f"a_point {lumped.a_point} is not inside its layer shape {lumped.a_layer.name}"
                )
                    
                assert b_point_inside, (
                    f"Project {project_name}, lumped element {i} ({lumped.type.name}): "
                    f"b_point {lumped.b_point} is not inside its layer shape {lumped.b_layer.name}"
                )

    def test_all_layer_shapes_are_multipolygons(self, kicad_test_projects):
        """
        Test that for all test projects, the shapes of all layers are MultiPolygons.
        This is regression testing for a bug where a layer with a single connected
        component would be loaded as a Polygon instead of a MultiPolygon
        (this originates in pygerber).
        """
        
        for project_name, project in kicad_test_projects.items():
            # Load the KiCad project
            try:
                kicad_problem = kicad.load_kicad_project(project.pro_path)
            except Exception as e:
                pytest.fail(f"Failed to load project {project_name}: {e}")
                
            # For each layer, verify that its shape is a MultiPolygon
            for i, layer in enumerate(kicad_problem.layers):
                assert layer.shape.geom_type == "MultiPolygon", (
                    f"Project {project_name}, layer {i} ({layer.name}): "
                    f"shape is not a MultiPolygon"
                )
