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

from conftest import for_all_kicad_projects


class TestFixture:

    def test_fixture_files_exist(self, kicad_test_projects):
        """Test that all the projects in the test fixture have existing files."""
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

    @for_all_kicad_projects(exclude=["simple_geometry"])
    def test_fixture_exclude(self, project):
        assert project.name != "simple_geometry"

    @for_all_kicad_projects(include=["simple_geometry"])
    def test_fixture_include(self, project):
        assert project.name == "simple_geometry"


def test_gerber_render_outputs_something(kicad_test_projects):
    """Test that the gerber rendering process outputs valid layer data."""
    
    project = kicad_test_projects["simple_geometry"]
    
    # Skip if the PCB file doesn't exist
    # Render gerbers from the PCB file
    board = pcbnew.LoadBoard(str(project.pcb_path))
    layers = kicad.render_gerbers_from_kicad(board)
    
    # Check that we got some layers back
    assert len(layers) > 0, "No layers were rendered from the PCB file"
    
    # Check that each layer has valid geometry
    for layer in layers:
        assert layer.name, "Layer has no name"
        assert layer.layer_id >= 0, "Layer has invalid ID"
        assert not layer.geometry.is_empty, "Layer geometry is empty"

    # The simple_geometry project does not have a B.Cu layer
    assert sorted([layer.name for layer in layers]) == ["F.Cu"]


class TestPadFinder:

    def test_simple_geometry_pad(self, kicad_test_projects):
        # Get the simple_geometry project's PCB file
        project = kicad_test_projects["simple_geometry"]
        assert project.pcb_path is not None, "simple_geometry project must have a PCB file"
        
        # Load the KiCad board from the PCB file
        board = pcbnew.LoadBoard(str(project.pcb_path))
        
        # Create PadIndex and load SMD pads
        pad_index = kicad.PadIndex()
        pad_index.load_smd_pads(board)
        
        # Try to find the pad R3.1
        endpoint = kicad.Endpoint(designator="R3", pad="1")
        layer_points = pad_index.find_by_endpoint(endpoint)
        
        assert len(layer_points) == 1, "Expected exactly one LayerPoint for R3.1"
        layer_point = layer_points[0]

        assert layer_point.layer == "F.Cu", "Pad should be on the F.Cu layer"

        assert abs(layer_point.point.x - 129) < 1e-3, "Pad X coordinate should be 129"
        assert abs(layer_point.point.y - 101.375) < 1e-3, "Pad Y coordinate should be 129"


class TestViaSpecs:

    def test_extract_tht_component_pad_specs(self, kicad_test_projects):
        project = kicad_test_projects["tht_component"]

        board = pcbnew.LoadBoard(str(project.pcb_path))

        via_specs = kicad.extract_tht_pad_specs_from_pcb(board)
        assert len(via_specs) == 10

        # Check that there is a pad located on x=139 y=103.46
        pad = next((pad for pad in via_specs if pad.point.x == 139 and pad.point.y == 103.46), None)
        assert pad is not None, "Pad not found at expected location"

    def test_extract_via_specs(self, kicad_test_projects):
        """Test that via specifications are correctly extracted from a PCB."""
        # Get the simple_via project
        project = kicad_test_projects["simple_via"]
        assert project.pcb_path.exists(), "PCB file of simple_via project does not exist"

        # Load the KiCad board
        board = pcbnew.LoadBoard(str(project.pcb_path))

        # Extract via specifications
        via_specs = kicad.extract_via_specs_from_pcb(board)

        # We expect exactly one via in the simple_via project
        assert len(via_specs) == 1, f"Expected 1 via, got {len(via_specs)}"

        # Get the via specification
        via_spec = via_specs[0]

        # Verify drill diameter (0.3mm)
        assert abs(via_spec.drill_diameter - 0.3) < 1e-6, f"Expected drill diameter 0.3mm, got {via_spec.drill_diameter}mm"

        # Verify position (x=132, y=100)
        assert abs(via_spec.point.x - 132) < 1e-3, f"Expected x=132, got {via_spec.point.x}"
        assert abs(via_spec.point.y - 100) < 1e-3, f"Expected y=100, got {via_spec.point.y}"

        # Verify that the via connects F.Cu and B.Cu layers
        expected_layers = ["F.Cu", "B.Cu"]
        assert set(via_spec.layer_names) == set(expected_layers), \
            f"Expected layers {expected_layers}, got {via_spec.layer_names}"

    def test_simple_via_gets_converted_to_a_resistor(self, kicad_test_projects):
        project = kicad_test_projects["simple_via"]

        # Check for a resistor connecting F.Cu and B.Cu at (132, 100)
        result = kicad.load_kicad_project(project.pro_path)
        found = False
        for network in result.networks:
            for element in network.elements:
                if isinstance(element, problem.Resistor):
                    # Find the connections associated with this resistor within this network
                    conn_a = next((c for c in network.connections if c.node_id == element.a), None)
                    conn_b = next((c for c in network.connections if c.node_id == element.b), None)

                    if not conn_a or not conn_b:
                        continue # Should not happen in a valid network

                    # Check if this resistor connects the expected layers at the expected point
                    cond1 = (
                        conn_a.layer.name == "F.Cu" and
                        conn_b.layer.name == "B.Cu" and
                        abs(conn_a.point.x - 132) < 1e-3 and
                        abs(conn_a.point.y - 100) < 1e-3 and
                        abs(conn_b.point.x - 132) < 1e-3 and
                        abs(conn_b.point.y - 100) < 1e-3
                    )
                    cond2 = (
                        conn_b.layer.name == "F.Cu" and
                        conn_a.layer.name == "B.Cu" and
                        abs(conn_b.point.x - 132) < 1e-3 and
                        abs(conn_b.point.y - 100) < 1e-3 and
                        abs(conn_a.point.x - 132) < 1e-3 and
                        abs(conn_a.point.y - 100) < 1e-3
                    )
                    if cond1 or cond2:
                        found = True
                        break
            if found:
                break
        assert found, "No resistor found connecting F.Cu and B.Cu at (132, 100)"

    def test_4layer_via_gets_converted_to_resistor_stack(self, kicad_test_projects):
        project = kicad_test_projects["via_tht_4layer"]

        # This project contains a via on x=118.8 y=105.9. Check that resistors
        # connecting the layers F.Cu - In1.Cu - In2.Cu - B.Cu have been created
        result = kicad.load_kicad_project(project.pro_path)
        # The via should be at (118.8, 105.9) and connect F.Cu, In1.Cu, In2.Cu, B.Cu in sequence
        expected_layers = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
        expected_points = [(118.8, 105.9)] * 4

        # Find all resistors at this location and on these layers
        found_layers = set()
        for network in result.networks:
            for element in network.elements:
                if isinstance(element, problem.Resistor):
                    # Find the connections associated with this resistor within this network
                    conn_a = next((c for c in network.connections if c.node_id == element.a), None)
                    conn_b = next((c for c in network.connections if c.node_id == element.b), None)

                    if not conn_a or not conn_b:
                        continue

                    # Check if both endpoints are at the via location
                    if (
                        abs(conn_a.point.x - 118.8) < 1e-3 and abs(conn_a.point.y - 105.9) < 1e-3 and
                        abs(conn_b.point.x - 118.8) < 1e-3 and abs(conn_b.point.y - 105.9) < 1e-3
                    ):
                        # Record the pair of layers this resistor connects
                        found_layers.add(tuple(sorted([conn_a.layer.name, conn_b.layer.name])))

        # The expected resistor stack is between each adjacent pair of layers
        expected_pairs = [
            tuple(sorted([expected_layers[i], expected_layers[i+1]]))
            for i in range(len(expected_layers)-1)
        ]
        for pair in expected_pairs:
            assert pair in found_layers, f"Missing resistor between layers {pair} at (118.8, 105.9)"


class TestDirectiveParse:

    def test_basic_directive_parsing(self):
        """Test parsing a simple directive with key-value pairs."""
        directive_str = "!padne VOLTAGE v=12.0V p=R1.4 n=R13.1"
        directive = kicad.Directive.parse(directive_str)
        
        assert directive.name == "VOLTAGE"
        assert directive.params == {"v": "12.0V", "p": "R1.4", "n": "R13.1"}

    def test_directive_with_numeric_values(self):
        """Test parsing a directive with numeric values."""
        directive_str = "!padne RESISTANCE r=4.7k from=R5.1 to=R5.2"
        directive = kicad.Directive.parse(directive_str)
        
        assert directive.name == "RESISTANCE"
        assert directive.params == {"r": "4.7k", "from": "R5.1", "to": "R5.2"}

    def test_directive_with_special_characters(self):
        """Test parsing a directive with special characters in values."""
        directive_str = "!padne CURRENT i=500mA source=U1.OUT+ sink=GND.1"
        directive = kicad.Directive.parse(directive_str)
        
        assert directive.name == "CURRENT"
        assert directive.params == {"i": "500mA", "source": "U1.OUT+", "sink": "GND.1"}

    def test_directive_with_empty_params(self):
        """Test parsing a directive with no parameters."""
        directive_str = "!padne DEBUG"
        directive = kicad.Directive.parse(directive_str)
        
        assert directive.name == "DEBUG"
        assert directive.params == {}

    def test_directive_with_duplicate_keys(self):
        """Test parsing a directive with duplicate keys (last one should win)."""
        directive_str = "!padne TEST key=value1 key=value2"
        directive = kicad.Directive.parse(directive_str)
        
        assert directive.name == "TEST"
        assert directive.params == {"key": "value2"}

    # Error case tests
    
    def test_missing_padne_prefix(self):
        """Test that a ValueError is raised when the !padne prefix is missing."""
        with pytest.raises(ValueError, match="Directive must start with '!padne'"):
            kicad.Directive.parse("VOLTAGE v=12V p=R1.1 n=R1.2")

    def test_missing_directive_name(self):
        """Test that a ValueError is raised when the directive name is missing."""
        with pytest.raises(ValueError, match="Directive must have a name"):
            kicad.Directive.parse("!padne")

    def test_invalid_key_value_format(self):
        """Test that a ValueError is raised when the key-value format is invalid."""
        with pytest.raises(ValueError, match="Invalid parameter format"):
            kicad.Directive.parse("!padne VOLTAGE v12V p=R1.1 n=R1.2")

    def test_empty_key(self):
        """Test that a ValueError is raised when a parameter has an empty key."""
        with pytest.raises(ValueError, match="Empty parameter key"):
            kicad.Directive.parse("!padne VOLTAGE =12V p=R1.1 n=R1.2")

    def test_parse_directives_from_simple_geometry(self, kicad_test_projects):
        # Get the simple_geometry project's schematic file
        project = kicad_test_projects["simple_geometry"]
        assert project.sch_path.exists(), "Schematic file of simple_geometry project does not exist"
        
        # Extract the raw directive strings from the schematic file
        directives = kicad.process_directives(
            kicad.extract_directives_from_eeschema(project.sch_path)
        )
        # Expecting exactly two directives based on our simple_geometry project
        assert len(directives.lumped_specs) == 2, f"Expected 2 lumped elements, got {len(directives.lumped_specs)}"
        
        # Parse each directive, then assign by type
        voltage_spec = next(
            spec for spec in directives.lumped_specs
            if isinstance(spec, kicad.VoltageSourceSpec)
        )
        resistor_spec = next(
            spec for spec in directives.lumped_specs
            if isinstance(spec, kicad.ResistorSpec)
        )
        
        # Validate the voltage directive
        assert voltage_spec.values["v"] == 1.0, "Voltage value should be 1.0"
        assert voltage_spec.endpoints["p"][0].designator == "R2", \
            "Voltage directive endpoint A designator should be R2"
        assert voltage_spec.endpoints["p"][0].pad == "1", \
            "Voltage directive endpoint A pad should be 1"
        assert voltage_spec.endpoints["n"][0].designator == "R2", \
            "Voltage directive endpoint B designator should be R2"
        assert voltage_spec.endpoints["n"][0].pad == "2", \
            "Voltage directive endpoint B pad should be 2"
        
        # Validate the resistor directive
        assert resistor_spec.values["r"] == 0.01
        assert resistor_spec.endpoints["a"][0].designator == "R3", \
            "Resistor directive endpoint A designator should be R3"
        assert resistor_spec.endpoints["a"][0].pad == "1", \
            "Resistor directive endpoint A pad should be 1"
        assert resistor_spec.endpoints["b"][0].designator == "R3", \
            "Resistor directive endpoint B designator should be R3"
        assert resistor_spec.endpoints["b"][0].pad == "2", \
            "Resistor directive endpoint B pad should be 2"


class TestStackup:

    def test_extract_stackup(self, kicad_test_projects):
        """Test that the stackup is correctly extracted from a KiCad PCB file."""
        # Get the simple_via project
        project = kicad_test_projects["simple_via"]
        assert project.pcb_path.exists(), "PCB file of simple_via project does not exist"

        # Load the KiCad board
        board = pcbnew.LoadBoard(str(project.pcb_path))

        # Extract stackup
        stackup = kicad.extract_stackup_from_kicad_pcb(board)

        # Check that we got a valid Stackup object
        assert isinstance(stackup, kicad.Stackup)
        assert len(stackup.items) == 3, f"Expected 3 stackup items, got {len(stackup.items)}"

        # Check F.Cu layer
        f_cu = next((item for item in stackup.items if item.name == "F.Cu"), None)
        assert f_cu is not None, "F.Cu layer not found in stackup"
        assert f_cu.thickness == 0.035, f"Expected F.Cu thickness 0.035mm, got {f_cu.thickness}mm"
        assert f_cu.conductivity == 5.95e4, "Expected F.Cu conductivity to be 5.95e7 S/m"

        # Check dielectric layer
        dielectric = next((item for item in stackup.items if item.conductivity is None), None)

        assert dielectric is not None, "Dielectric layer not found in stackup"
        assert dielectric.thickness == 1.51, f"Expected dielectric thickness 1.51mm, got {dielectric.thickness}mm"

        # Check B.Cu layer
        b_cu = next((item for item in stackup.items if item.name == "B.Cu"), None)
        assert b_cu is not None, "B.Cu layer not found in stackup"
        assert b_cu.thickness == 0.035, f"Expected B.Cu thickness 0.035mm, got {b_cu.thickness}mm"
        assert b_cu.conductivity == 5.95e4, "Expected B.Cu conductivity to be 5.95e7 S/m"

    @for_all_kicad_projects
    def test_extract_stackup_extracts_every_project(self, project):
        # Load the KiCad board
        board = pcbnew.LoadBoard(str(project.pcb_path))

        # Extract stackup
        stackup = kicad.extract_stackup_from_kicad_pcb(board)

        # Check that we got a valid Stackup object
        assert isinstance(stackup, kicad.Stackup), f"Stackup extraction failed for {project.name}"
        assert len(stackup.items) > 0, f"No stackup items found for {project.name}"


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
        # Should have our two lumped elements, each in its own network
        assert len(result.networks) == 2

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

    def test_conductance_vaguely_makes_sense(self, kicad_test_projects, monkeypatch):
        """Test that custom resistivity is applied correctly."""
        project = kicad_test_projects["simple_geometry"]

        result = kicad.load_kicad_project(project.pro_path)
        
        # F.Cu layer should have the custom resistivity
        f_cu_layer = next(layer for layer in result.layers if layer.name == "F.Cu")
        assert 1900 < f_cu_layer.conductance < 2300

    def test_lumped_elements(self, kicad_test_projects):
        """Test that lumped elements are loaded correctly."""
        project = kicad_test_projects["simple_geometry"]
        result = kicad.load_kicad_project(project.pro_path)
        
        # Find the voltage source and resistor by searching through networks
        voltage_source_element = None
        voltage_source_connections = []
        resistor_element = None
        resistor_connections = []

        for network in result.networks:
            for element in network.elements:
                if isinstance(element, problem.VoltageSource):
                    voltage_source_element = element
                    voltage_source_connections = network.connections
                    break
            if voltage_source_element:
                break
        
        for network in result.networks:
            for element in network.elements:
                if isinstance(element, problem.Resistor):
                    resistor_element = element
                    resistor_connections = network.connections
                    break
            if resistor_element:
                break

        assert voltage_source_element is not None, "Voltage source element not found"
        assert resistor_element is not None, "Resistor element not found"

        # Check voltage source properties
        assert voltage_source_element.voltage == 1.0
        # Check that it's connected to component R2, pads 1 and 2
        board = pcbnew.LoadBoard(str(project.pcb_path))

        # Create PadIndex and load SMD pads
        pad_index = kicad.PadIndex()
        pad_index.load_smd_pads(board)

        r2_1_point = pad_index.find_by_endpoint(kicad.Endpoint("R2", "1"))[0].point
        r2_2_point = pad_index.find_by_endpoint(kicad.Endpoint("R2", "2"))[0].point

        # Find the connections corresponding to the voltage source terminals
        conn_p = next(c for c in voltage_source_connections if c.node_id == voltage_source_element.p)
        conn_n = next(c for c in voltage_source_connections if c.node_id == voltage_source_element.n)

        assert (conn_p.point.x == r2_1_point.x and
                conn_p.point.y == r2_1_point.y)
        assert (conn_n.point.x == r2_2_point.x and
                conn_n.point.y == r2_2_point.y)

        # Check resistor properties
        assert resistor_element.resistance == 0.01
        # Check that it's connected to component R3, pads 1 and 2
        r3_1_point = pad_index.find_by_endpoint(kicad.Endpoint("R3", "1"))[0].point
        r3_2_point = pad_index.find_by_endpoint(kicad.Endpoint("R3", "2"))[0].point

        # Find the connections corresponding to the resistor terminals
        conn_a = next(c for c in resistor_connections if c.node_id == resistor_element.a)
        conn_b = next(c for c in resistor_connections if c.node_id == resistor_element.b)
        
        assert (conn_a.point.x == r3_1_point.x and 
                conn_a.point.y == r3_1_point.y)
        assert (conn_b.point.x == r3_2_point.x and 
                conn_b.point.y == r3_2_point.y)
                
    @for_all_kicad_projects
    def test_lumped_points_inside_layers(self, project):
        """
        Test that for all test projects, the start and end points of lumped elements 
        are located inside their respective layer shapes.
        """
        
        # Load the KiCad project
        kicad_problem = kicad.load_kicad_project(project.pro_path)
            
        # For each lumped element network, verify that its connection points are inside the layers
        for i, network in enumerate(kicad_problem.networks):
            for j, connection in enumerate(network.connections):
                point_inside = connection.layer.shape.contains(connection.point)

                assert point_inside, (
                    f"Project {project.name}, network {i}, connection {j} "
                    f"point {connection.point} is not inside its layer shape {connection.layer.name}"
                )

    @for_all_kicad_projects
    def test_all_layer_shapes_are_multipolygons(self, project):
        """
        Test that for all test projects, the shapes of all layers are MultiPolygons.
        This is regression testing for a bug where a layer with a single connected
        component would be loaded as a Polygon instead of a MultiPolygon
        (this originates in pygerber).
        """
        # Load the KiCad project
        kicad_problem = kicad.load_kicad_project(project.pro_path)
            
        # For each layer, verify that its shape is a MultiPolygon
        for i, layer in enumerate(kicad_problem.layers):
            assert layer.shape.geom_type == "MultiPolygon", (
                f"Project {project.name}, layer {i} ({layer.name}): "
                f"shape is not a MultiPolygon"
            )

    def test_flipped_pads_work(self, kicad_test_projects):
        """Test that flipped pads are handled correctly."""
        project = kicad_test_projects["simple_via"]
        
        # Load the project
        result = kicad.load_kicad_project(project.pro_path)
        
        # Find the voltage source lumped element by searching networks
        voltage_source_element = None
        voltage_source_connections = []
        for network in result.networks:
            for element in network.elements:
                if isinstance(element, problem.VoltageSource):
                    voltage_source_element = element
                    voltage_source_connections = network.connections
                    break
            if voltage_source_element:
                break
        
        # Check that we found a voltage source
        assert voltage_source_element is not None, "No voltage source found in the simple_via project"

        # Find the connections corresponding to the voltage source terminals
        conn_p = next(c for c in voltage_source_connections if c.node_id == voltage_source_element.p)
        conn_n = next(c for c in voltage_source_connections if c.node_id == voltage_source_element.n)
        
        # Check that one endpoint is on F.Cu at position (122, 100)
        # and the other is on B.Cu at (142, 100)
        if conn_p.layer.name == "F.Cu":
            f_cu_conn = conn_p
            b_cu_conn = conn_n
        elif conn_n.layer.name == "F.Cu":
            f_cu_conn = conn_n
            b_cu_conn = conn_p
        else:
            pytest.fail("Neither connection point p nor n was on F.Cu")

        f_cu_point = f_cu_conn.point
        b_cu_point = b_cu_conn.point
        f_cu_layer = f_cu_conn.layer
        b_cu_layer = b_cu_conn.layer
        
        # Verify F.Cu point is at expected coordinates (122, 100)
        assert abs(f_cu_point.x - 122) < 1e-3, f"F.Cu point X should be 122, got {f_cu_point.x}"
        assert abs(f_cu_point.y - 100) < 1e-3, f"F.Cu point Y should be 100, got {f_cu_point.y}"
        
        # Verify B.Cu point is at expected coordinates (142, 100)
        assert abs(b_cu_point.x - 142) < 1e-3, f"B.Cu point X should be 142, got {b_cu_point.x}"
        assert abs(b_cu_point.y - 100) < 1e-3, f"B.Cu point Y should be 100, got {b_cu_point.y}"
        
        # Verify the layer names
        assert f_cu_layer.name == "F.Cu", f"Expected F.Cu layer, got {f_cu_layer.name}"
        assert b_cu_layer.name == "B.Cu", f"Expected B.Cu layer, got {b_cu_layer.name}"

    def test_layer_order(self, kicad_test_projects):
        project = kicad_test_projects["via_tht_4layer"]
        
        # Load the project
        result = kicad.load_kicad_project(project.pro_path)
        
        # Check the layer order
        expected_order = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
        actual_order = [layer.name for layer in result.layers]
        
        assert actual_order == expected_order, (
            f"Layer order mismatch: expected {expected_order}, got {actual_order}"
        )
