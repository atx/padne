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


class Utils:
    """Utility functions for KiCad test operations."""

    @staticmethod
    def setup_layer_dict_and_pad_index(board):
        """
        Set up layer dictionary and pad index from a KiCad board.

        This utility method encapsulates the common pattern of:
        1. Rendering gerbers from KiCad
        2. Extracting and applying board outline clipping
        3. Extracting via specs and punching holes
        4. Creating stackup and layer dictionary
        5. Creating and loading PadIndex with SMD pads

        Args:
            board: A KiCad board object loaded via pcbnew.LoadBoard()

        Returns:
            tuple: (layer_dict, pad_index) where layer_dict contains processed
                   layer geometry and pad_index contains SMD pad locations
        """
        # TODO: I am not particularly fond of this method being here,
        # since it mostly replicates the code in kicad.py
        # Render gerbers from the KiCad board
        plotted_layers = kicad.render_gerbers_from_kicad(board, kicad.copper_layers(board))

        # Extract board outline and clip layers if outline exists
        outline = kicad.extract_board_outline(board)
        if outline is not None:
            plotted_layers = [
                kicad.clip_layer_with_outline(plotted_layer, outline)
                for plotted_layer in plotted_layers
            ]

        # Get via specs (both via and THT pad specs) and punch holes
        via_specs = kicad.extract_via_specs_from_pcb(board) + kicad.extract_tht_pad_specs_from_pcb(board)
        plotted_layers = kicad.punch_via_holes(plotted_layers, via_specs)

        # Create stackup and layer dictionary
        stackup = kicad.extract_stackup_from_kicad_pcb(board)
        layer_dict = kicad.construct_layer_dict(plotted_layers, stackup)

        # Create PadIndex and load SMD pads
        pad_index = kicad.PadIndex()
        pad_index.load_smd_pads(board, layer_dict)

        return layer_dict, pad_index

    @staticmethod
    def find_first_network_with_element_type(kicad_problem, element_type, assert_only_one=False):
        """
        Find the first network containing an element of the specified type.

        Args:
            kicad_problem: A Problem object containing networks and elements
            element_type: The type class to search for (e.g., problem.VoltageSource)
            assert_only_one: If True, assert that exactly one element of this type exists

        Returns:
            tuple: (element, network) where element is the found element and network is its containing network

        Raises:
            AssertionError: If no element is found or if assert_only_one=True and multiple elements exist
        """
        found_elements = []

        for network in kicad_problem.networks:
            for element in network.elements:
                if isinstance(element, element_type):
                    found_elements.append((element, network))

        assert found_elements, f"No element of type {element_type.__name__} found in any network"

        if assert_only_one:
            assert len(found_elements) == 1, f"Expected exactly one element of type {element_type.__name__}, found {len(found_elements)}"

        return found_elements[0]


class TestKiCadProject:

    def test_from_pro_file_success(self):
        """Test that from_pro_file() successfully creates KiCadProject instances."""
        # Get the simple_geometry project path manually
        kicad_dir = Path(__file__).parent / "kicad"
        pro_path = kicad_dir / "simple_geometry" / "simple_geometry.kicad_pro"

        # Create a KiCadProject using from_pro_file
        project = kicad.KiCadProject.from_pro_file(pro_path)

        # Verify the project was created correctly
        assert project.name == "simple_geometry"
        assert project.pro_path == pro_path
        assert project.pcb_path == pro_path.parent / "simple_geometry.kicad_pcb"
        assert project.sch_path == pro_path.parent / "simple_geometry.kicad_sch"

        # Verify all files exist
        assert project.pro_path.exists()
        assert project.pcb_path.exists()
        assert project.sch_path.exists()

    def test_simple_geometry_paths(self):
        """Test that paths are correctly resolved for the simple_geometry project."""
        kicad_dir = Path(__file__).parent / "kicad"
        pro_path = kicad_dir / "simple_geometry" / "simple_geometry.kicad_pro"

        project = kicad.KiCadProject.from_pro_file(pro_path)

        # Test that all paths point to the expected locations
        expected_dir = kicad_dir / "simple_geometry"
        assert project.pro_path == expected_dir / "simple_geometry.kicad_pro"
        assert project.pcb_path == expected_dir / "simple_geometry.kicad_pcb"
        assert project.sch_path == expected_dir / "simple_geometry.kicad_sch"

        # Test that all paths are absolute
        assert project.pro_path.is_absolute()
        assert project.pcb_path.is_absolute()
        assert project.sch_path.is_absolute()

        # Test that all files have correct extensions
        assert project.pro_path.suffix == ".kicad_pro"
        assert project.pcb_path.suffix == ".kicad_pcb"
        assert project.sch_path.suffix == ".kicad_sch"

    def test_from_pro_file_missing_project_file(self):
        """Test that from_pro_file() raises FileNotFoundError for missing project file."""
        nonexistent_path = Path("/nonexistent/project.kicad_pro")

        with pytest.raises(FileNotFoundError, match="Project file not found"):
            kicad.KiCadProject.from_pro_file(nonexistent_path)

    def test_from_pro_file_missing_pcb_file(self, tmp_path):
        """Test that from_pro_file() raises FileNotFoundError for missing PCB file."""
        # Create only the project file, but not the PCB file
        pro_path = tmp_path / "test.kicad_pro"
        pro_path.write_text("dummy content")

        with pytest.raises(FileNotFoundError, match="PCB file not found"):
            kicad.KiCadProject.from_pro_file(pro_path)

    def test_from_pro_file_missing_sch_file(self, tmp_path):
        """Test that from_pro_file() raises FileNotFoundError for missing schematic file."""
        # Create project and PCB files, but not the schematic file
        pro_path = tmp_path / "test.kicad_pro"
        pcb_path = tmp_path / "test.kicad_pcb"
        pro_path.write_text("dummy content")
        pcb_path.write_text("dummy content")

        with pytest.raises(FileNotFoundError, match="Schematic file not found"):
            kicad.KiCadProject.from_pro_file(pro_path)


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
    layers = kicad.render_gerbers_from_kicad(board, kicad.copper_layers(board))

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

        # Set up layer dictionary and pad index using utility function
        layer_dict, pad_index = Utils.setup_layer_dict_and_pad_index(board)

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

        # Check for multiple resistors connecting F.Cu and B.Cu around via boundary at (132, 100)
        result = kicad.load_kicad_project(project.pro_path)
        via_center = shapely.geometry.Point(132, 100)
        drill_diameter = 0.3  # mm
        expected_radius = drill_diameter / 2
        tolerance = expected_radius * 0.1  # 10% tolerance

        resistors_found = 0
        for network in result.networks:
            for element in network.elements:
                if isinstance(element, problem.Resistor):
                    # Find the connections associated with this resistor within this network
                    conn_a = next((c for c in network.connections if c.node_id == element.a), None)
                    conn_b = next((c for c in network.connections if c.node_id == element.b), None)

                    if not conn_a or not conn_b:
                        continue # Should not happen in a valid network

                    # Check if this resistor connects F.Cu and B.Cu layers
                    layers_match = (
                        (conn_a.layer.name == "F.Cu" and conn_b.layer.name == "B.Cu") or
                        (conn_b.layer.name == "F.Cu" and conn_a.layer.name == "B.Cu")
                    )

                    # Check if points are on the via boundary (within 10% of expected radius)
                    dist_a = via_center.distance(conn_a.point)
                    dist_b = via_center.distance(conn_b.point)
                    points_on_boundary = (
                        abs(dist_a - expected_radius) < tolerance and
                        abs(dist_b - expected_radius) < tolerance
                    )

                    if layers_match and points_on_boundary:
                        resistors_found += 1

        # We should have multiple resistors (boundary ring stitching) instead of just one
        assert resistors_found > 0, f"No resistors found connecting F.Cu and B.Cu on via boundary at {via_center}"
        assert resistors_found >= 4, f"Expected at least 4 boundary resistors, found {resistors_found} (quad_segs=4 creates ~16 points)"

    def test_4layer_via_gets_converted_to_resistor_stack(self, kicad_test_projects):
        project = kicad_test_projects["via_tht_4layer"]

        # This project contains a via on x=118.8 y=105.9. Check that resistors
        # connecting the layers F.Cu - In1.Cu - In2.Cu - B.Cu have been created
        result = kicad.load_kicad_project(project.pro_path)
        via_center = shapely.geometry.Point(118.8, 105.9)
        drill_diameter = 0.3  # mm (assuming same as simple via)
        expected_radius = drill_diameter / 2
        tolerance = expected_radius * 0.1  # 10% tolerance

        expected_layers = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]

        # Find all resistors at this via boundary and on these layers
        found_layers = set()
        for network in result.networks:
            for element in network.elements:
                if isinstance(element, problem.Resistor):
                    # Find the connections associated with this resistor within this network
                    conn_a = next((c for c in network.connections if c.node_id == element.a), None)
                    conn_b = next((c for c in network.connections if c.node_id == element.b), None)

                    if not conn_a or not conn_b:
                        continue

                    # Check if both endpoints are on the via boundary (within 10% of expected radius)
                    dist_a = via_center.distance(conn_a.point)
                    dist_b = via_center.distance(conn_b.point)
                    points_on_boundary = (
                        abs(dist_a - expected_radius) < tolerance and
                        abs(dist_b - expected_radius) < tolerance
                    )

                    if points_on_boundary:
                        # Record the pair of layers this resistor connects
                        found_layers.add(tuple(sorted([conn_a.layer.name, conn_b.layer.name])))

        # The expected resistor stack is between each adjacent pair of layers
        expected_pairs = [
            tuple(sorted([expected_layers[i], expected_layers[i+1]]))
            for i in range(len(expected_layers)-1)
        ]
        for pair in expected_pairs:
            assert pair in found_layers, f"Missing resistor between layers {pair} at via {via_center}"


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

    def test_directive_with_simple_quotes(self):
        """Test that simple quotes get eliminated. We do not yet support spaces, so that is undefined behavior for now."""
        directive_str = '!padne LABEL text="HelloWorld" position=R1.1'

        directive = kicad.Directive.parse(directive_str)

        assert directive.name == "LABEL"
        assert directive.params == {"text": "HelloWorld", "position": "R1.1"}

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

    def test_multiline_directive_parsing(self):
        """Test parsing multiple directives from a single text block with newlines."""
        text = '!padne VOLTAGE v=1.0V p=R2.1 n=R2.2\n!padne RESISTANCE r=0.01 a=R3.1 b=R3.2'

        directives = kicad.extract_directives_from_text(text)

        assert len(directives) == 2
        assert directives[0].name == 'VOLTAGE'
        assert directives[0].params == {'v': '1.0V', 'p': 'R2.1', 'n': 'R2.2'}
        assert directives[1].name == 'RESISTANCE'
        assert directives[1].params == {'r': '0.01', 'a': 'R3.1', 'b': 'R3.2'}

    def test_multiline_directive_with_whitespace(self):
        """Test that directives with leading/trailing whitespace are properly stripped."""
        text = '  !padne VOLTAGE v=3.3V p=U1.VCC n=U1.GND  \n\t!padne CURRENT i=1.0A f=R1.1 t=R1.2\t'

        directives = kicad.extract_directives_from_text(text)

        assert len(directives) == 2
        assert directives[0].name == 'VOLTAGE'
        assert directives[0].params == {'v': '3.3V', 'p': 'U1.VCC', 'n': 'U1.GND'}
        assert directives[1].name == 'CURRENT'
        assert directives[1].params == {'i': '1.0A', 'f': 'R1.1', 't': 'R1.2'}

    def test_multiline_directive_ignore_non_padne_lines(self):
        """Test that non-!padne lines in multiline text blocks are ignored."""
        text = 'This is a comment\n!padne VOLTAGE v=5V p=VCC n=GND\nAnother comment\n!padne RESISTANCE r=10 a=R1.1 b=R1.2\n'

        directives = kicad.extract_directives_from_text(text)

        assert len(directives) == 2
        assert directives[0].name == 'VOLTAGE'
        assert directives[0].params == {'v': '5V', 'p': 'VCC', 'n': 'GND'}
        assert directives[1].name == 'RESISTANCE'
        assert directives[1].params == {'r': '10', 'a': 'R1.1', 'b': 'R1.2'}

    def test_multiline_directive_empty_lines(self):
        """Test that empty lines in multiline text blocks are ignored."""
        text = '\n\n!padne VOLTAGE v=12V p=PWR n=GND\n\n\n!padne CURRENT i=2A f=J1.1 t=J1.2\n\n'

        directives = kicad.extract_directives_from_text(text)

        assert len(directives) == 2
        assert directives[0].name == 'VOLTAGE'
        assert directives[0].params == {'v': '12V', 'p': 'PWR', 'n': 'GND'}
        assert directives[1].name == 'CURRENT'
        assert directives[1].params == {'i': '2A', 'f': 'J1.1', 't': 'J1.2'}

    def test_parse_directives_from_simple_geometry(self, kicad_test_projects):
        # Get the simple_geometry project's schematic file
        project = kicad_test_projects["simple_geometry"]
        assert project.sch_path.exists(), "Schematic file of simple_geometry project does not exist"

        # Extract directives using the hierarchy-based approach
        schema_hierarchy = kicad.build_schema_hierarchy(project.sch_path)
        directives = kicad.process_directives(
            kicad.extract_directives_from_hierarchy(schema_hierarchy)
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

    def test_nested_schematic_directives(self, kicad_test_projects):
        """Test that directives are loaded from both root and nested schematics."""
        # Get the nested_schematic project
        project = kicad_test_projects["nested_schematic"]
        assert project.sch_path.exists(), "Schematic file of nested_schematic project does not exist"

        # Load the project and extract directives
        kicad_problem = kicad.load_kicad_project(project.pro_path)

        # Should have exactly 2 lumped elements: 1 from root + 1 from nested schematic
        assert len(kicad_problem.networks) == 2, f"Expected 2 networks, got {len(kicad_problem.networks)}"

        # Extract the voltage source and resistor by type
        voltage_source_element, _ = Utils.find_first_network_with_element_type(kicad_problem, problem.VoltageSource)
        resistor_element, _ = Utils.find_first_network_with_element_type(kicad_problem, problem.Resistor)

        # Verify the voltage source properties (from root schematic)
        assert voltage_source_element.voltage == 1.0, "Voltage value should be 1.0V"

        # Verify the resistor properties (from nested schematic)
        assert resistor_element.resistance == 0.01, "Resistance value should be 0.01 ohms"

    def test_multiline_directives_from_project(self, kicad_test_projects):
        """Test that the multiline_directive project loads correctly with multiple directives."""
        project = kicad_test_projects["multiline_directive"]

        # Load the entire project - this tests the full integration
        problem = kicad.load_kicad_project(project.pro_path)

        # Should have both a voltage source and a resistor from the multiline directive
        from padne.problem import VoltageSource, Resistor

        voltage_sources = [e for network in problem.networks for e in network.elements if isinstance(e, VoltageSource)]
        resistors = [e for network in problem.networks for e in network.elements if isinstance(e, Resistor)]

        assert len(voltage_sources) == 1
        assert len(resistors) == 1

        # Check the voltage source parameters
        assert voltage_sources[0].voltage == 1.0

        # Check the resistor parameters
        assert resistors[0].resistance == 0.01

    def test_nested_schematic_twoinstances_directive_deduplication(self, kicad_test_projects):
        """Test that directives from multiple instances of the same file are deduplicated."""
        # Get the nested_schematic_twoinstances project
        project = kicad_test_projects["nested_schematic_twoinstances"]
        assert project.sch_path.exists(), "Schematic file of nested_schematic_twoinstances project does not exist"

        # Load the project and extract directives
        with pytest.warns(UserWarning, match="Schematic files with multiple instances are not supported"):
            kicad_problem = kicad.load_kicad_project(project.pro_path)

        # Should have exactly 2 lumped elements: 1 from root + 1 from nested schematic
        # Even though nested schematic is referenced twice, directive should only be extracted once
        assert len(kicad_problem.networks) == 2, f"Expected 2 networks, got {len(kicad_problem.networks)}"

        # Extract the voltage source and resistor by type
        voltage_source_element, _ = Utils.find_first_network_with_element_type(kicad_problem, problem.VoltageSource)
        resistor_element, _ = Utils.find_first_network_with_element_type(kicad_problem, problem.Resistor)

        # Verify the voltage source properties (from root schematic)
        assert voltage_source_element.voltage == 1.0, "Voltage value should be 1.0V"

        # Verify the resistor properties (from nested schematic)
        assert resistor_element.resistance == 0.01, "Resistance value should be 0.01 ohms"

    def test_nested_schematic_twoinstances_hierarchy_structure(self, kicad_test_projects):
        """Test that hierarchy correctly preserves multiple instances with proper names."""
        # Get the nested_schematic_twoinstances project
        project = kicad_test_projects["nested_schematic_twoinstances"]
        assert project.sch_path.exists(), "Schematic file of nested_schematic_twoinstances project does not exist"

        # Build the schema hierarchy
        schema_hierarchy = kicad.build_schema_hierarchy(project.sch_path)

        # Root should have exactly 2 children
        assert len(schema_hierarchy.child_instances) == 2, f"Expected 2 child instances, got {len(schema_hierarchy.child_instances)}"

        # Extract child instances
        child_a = None
        child_b = None

        for child in schema_hierarchy.child_instances:
            if child.sheet_name == "Nested A":
                child_a = child
            elif child.sheet_name == "Nested B":
                child_b = child

        # Verify both children exist with correct names
        assert child_a is not None, "Child instance 'Nested A' not found"
        assert child_b is not None, "Child instance 'Nested B' not found"

        # Verify both children reference the same nested.kicad_sch file
        assert child_a.file_path.name == "nested.kicad_sch", f"Child A should reference nested.kicad_sch, got {child_a.file_path.name}"
        assert child_b.file_path.name == "nested.kicad_sch", f"Child B should reference nested.kicad_sch, got {child_b.file_path.name}"

        # Verify both children have the same file path (since they reference the same file)
        assert child_a.file_path == child_b.file_path, "Both children should reference the same file path"

        # Verify both children have no further children (nested.kicad_sch has no sheet references)
        assert len(child_a.child_instances) == 0, "Child A should have no further children"
        assert len(child_b.child_instances) == 0, "Child B should have no further children"

        # Verify both children have parsed content
        assert child_a.parsed_sexp is not None, "Child A should have parsed S-expression content"
        assert child_b.parsed_sexp is not None, "Child B should have parsed S-expression content"


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
        voltage_source_element, voltage_source_network = Utils.find_first_network_with_element_type(result, problem.VoltageSource)
        resistor_element, resistor_network = Utils.find_first_network_with_element_type(result, problem.Resistor)

        voltage_source_connections = voltage_source_network.connections
        resistor_connections = resistor_network.connections

        # Check voltage source properties
        assert voltage_source_element.voltage == 1.0
        # Check that it's connected to component R2, pads 1 and 2
        board = pcbnew.LoadBoard(str(project.pcb_path))

        # Set up layer dictionary and pad index using utility function
        layer_dict, pad_index = Utils.setup_layer_dict_and_pad_index(board)

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

    @for_all_kicad_projects(exclude=["nested_schematic_twoinstances",
                                     "many_meshes_many_vias"])
    def test_lumped_points_inside_layers(self, project):
        """
        Test that for all test projects, the start and end points of lumped elements
        are located inside their respective layer shapes.
        """
        # Some projects of note:
        # via_in_pad:
        #    has a THT pad inside a via. This test should check that the connection point
        #    of that particular pad gets eliminated

        # Load the KiCad project
        kicad_problem = kicad.load_kicad_project(project.pro_path)

        # For each lumped element network, verify that its connection points are inside the layers
        for i, network in enumerate(kicad_problem.networks):
            for j, connection in enumerate(network.connections):
                point_inside = connection.layer.shape.intersects(connection.point)

                assert point_inside, (
                    f"Project {project.name}, network {i}, connection {j} "
                    f"point {connection.point} is not inside its layer shape {connection.layer.name}"
                )

    def test_via_in_pad_no_floating_connections(self, kicad_test_projects):
        """
        Test that when a THT pad is placed inside an SMD pad,
        the SMD pad's connection point that would fall in the hole is eliminated.
        Specific test for the via_in_pad project where THT TP1 at (150, 100)
        overlaps with SMD TP3 at the same location.
        """
        project = kicad_test_projects["via_in_pad"]

        # Load the KiCad project with our fix applied
        kicad_problem = kicad.load_kicad_project(project.pro_path)

        # Check that no connection point exists at exactly (150, 100)
        # This is where the SMD pad center would be, but it should be eliminated
        # because it falls inside the THT pad's drill hole
        target_point = shapely.geometry.Point(150, 100)
        exclusion_zone = target_point.buffer(0.5)

        connections_at_target = []
        for network in kicad_problem.networks:
            for connection in network.connections:
                # Check if this connection is at the problematic location
                if exclusion_zone.intersects(connection.point):
                    connections_at_target.append({
                        'layer': connection.layer.name,
                        'point': connection.point,
                        'in_geometry': connection.layer.shape.intersects(connection.point)
                    })

        # There should be NO connections at exactly (150, 100) since it's in a hole
        assert len(connections_at_target) == 0, (
            f"Found {len(connections_at_target)} connection(s) at (150, 100) which should be in a hole. "
            f"Connections: {connections_at_target}"
        )

    @for_all_kicad_projects(exclude=["nested_schematic_twoinstances",
                                     "many_meshes_many_vias"])
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

    def test_via_hole_punching(self, kicad_test_projects):
        """Test that holes are punched in layers for THT pads and vias."""
        project = kicad_test_projects["via_tht_4layer"]

        # Load the project
        result = kicad.load_kicad_project(project.pro_path)

        # Test points and hole sizes
        test_cases = [
            # (x, y, hole_diameter, description)
            (113.39, 104.25, 1.0, "THT pad hole"),
            (118.8, 105.9, 0.3, "via hole")
        ]

        for x, y, hole_diameter, description in test_cases:
            hole_center = shapely.geometry.Point(x, y)

            # Test 1: Make hole 5% smaller - intersection should be empty (hole is punched)
            smaller_hole_circle = hole_center.buffer(hole_diameter / 2 * 0.95)

            for layer in result.layers:
                intersection = layer.shape.intersection(smaller_hole_circle)

                assert intersection.is_empty, (
                    f"{description} at ({x}, {y}) not properly punched in layer {layer.name}. "
                    f"Expected empty intersection but found geometry"
                )

            # Test 2: Make hole 5% larger - intersection should NOT be empty (copper around hole)
            larger_hole_circle = hole_center.buffer(hole_diameter / 2 * 1.05)

            for layer in result.layers:
                intersection = layer.shape.intersection(larger_hole_circle)

                assert not intersection.is_empty, (
                    f"{description} at ({x}, {y}) should have copper around the hole in layer {layer.name}. "
                    f"Expected non-empty intersection but found empty geometry"
                )

    def test_via_hole_punching_overlapping(self, kicad_test_projects):
        """Test that overlapping vias have proper hole punching with 1.8mm radius circles."""
        project = kicad_test_projects["overlapping_vias"]

        # Load the project
        result = kicad.load_kicad_project(project.pro_path)

        centers = [
            (101.2, 100, 1.8),
            (122.5, 100, 1.3),
            (148.7, 100, 1.8),
        ]

        # Test that a 1.8mm radius circle doesn't intersect any layer shapes
        for x, y, r in centers:
            circle_center = shapely.geometry.Point(x, y)
            test_circle = circle_center.buffer(r)

            for layer in result.layers:
                intersection = layer.shape.intersection(test_circle)
                assert intersection.is_empty, (
                    f"{r}mm radius circle at ({x}, {y}) should not intersect layer {layer.name}. "
                    f"Expected empty intersection but found geometry"
                )

    def test_copper_directive_custom_conductivity(self, kicad_test_projects):
        """Test that COPPER directive overrides default conductivity."""
        project = kicad_test_projects["long_trace_current_custom_conductivity"]
        #project = kicad_test_projects["long_trace_current"]

        # Load the project
        result = kicad.load_kicad_project(project.pro_path)

        # Expected custom conductivity from the schematic
        expected_conductivity = 29.75e3  # S/mm

        # All copper layers should have the custom conductivity
        copper_layers_found = 0
        for layer in result.layers:
            # Copper layers should have names like "F.Cu", "B.Cu", etc.
            if layer.name.endswith(".Cu"):
                copper_layers_found += 1
                # Calculate expected conductance (conductivity * thickness)
                # Default thickness is 0.035mm for copper layers
                expected_conductance = 0.035 * expected_conductivity

                assert abs(layer.conductance - expected_conductance) < 1e-6, (
                    f"Layer {layer.name} should have conductance {expected_conductance} "
                    f"(from custom conductivity {expected_conductivity} S/mm), "
                    f"but found {layer.conductance}"
                )

        # Ensure we found at least one copper layer
        assert copper_layers_found > 0, "No copper layers found in the project"

    def test_degenerate_hole_geometry(self, kicad_test_projects):
        """Test that polygons with degenerate holes are loaded successfully."""
        project = kicad_test_projects["degenerate_hole_geometry"]

        # Load the project
        result = kicad.load_kicad_project(project.pro_path)

        # Find the In2.Cu layer
        in2_cu_layer = next((layer for layer in result.layers if layer.name == "In2.Cu"), None)
        assert in2_cu_layer is not None, "In2.Cu layer should exist in the degenerate_hole_geometry project"

        # Verify the layer shape is a MultiPolygon
        assert isinstance(in2_cu_layer.shape, shapely.geometry.MultiPolygon), \
            f"In2.Cu layer shape should be MultiPolygon, got {type(in2_cu_layer.shape)}"

        # Verify we have a single polygon
        assert len(in2_cu_layer.geoms) == 1

        # Next, check its area is at least 20 mm²
        assert in2_cu_layer.shape.area >= 20.0, "In2.Cu layer area should be at least 20 mm²"


class TestCopperDirective:
    """Tests for COPPER directive parsing and validation."""

    def test_copper_directive_parsing(self):
        """Test that COPPER directives are parsed correctly."""
        directive_text = "!padne COPPER conductivity=29.75e6"
        directive = kicad.Directive.parse(directive_text)

        copper_spec = kicad.CopperSpec.from_directive(directive)
        assert copper_spec.conductivity == 29750.0

    def test_copper_directive_missing_conductivity(self):
        """Test error when conductivity parameter is missing."""
        directive_text = "!padne COPPER"
        directive = kicad.Directive.parse(directive_text)

        with pytest.raises(KeyError,
                           match="The parameter `conductivity` not specified for the COPPER directive"):
            kicad.CopperSpec.from_directive(directive)

    def test_copper_directive_negative_conductivity(self):
        """Test error when conductivity is negative."""
        directive_text = "!padne COPPER conductivity=-1000"
        directive = kicad.Directive.parse(directive_text)

        with pytest.raises(ValueError, match="Conductivity must be positive"):
            kicad.CopperSpec.from_directive(directive)

    def test_copper_directive_zero_conductivity(self):
        """Test error when conductivity is zero."""
        directive_text = "!padne COPPER conductivity=0"
        directive = kicad.Directive.parse(directive_text)

        with pytest.raises(ValueError, match="Conductivity must be positive"):
            kicad.CopperSpec.from_directive(directive)


class TestExtractBoardOutline:

    def test_extract_board_outline_castellated_vias_internal_cutout(self, kicad_test_projects):
        """Test that extract_board_outline correctly identifies inside/outside points."""
        project = kicad_test_projects["castellated_vias_internal_cutout"]

        # Load the KiCad board
        board = pcbnew.LoadBoard(str(project.pcb_path))

        # Extract the board outline
        outline = kicad.extract_board_outline(board)

        # Points that should be inside the board outline
        inside_points = [
            (100.2, 90.2),
            (100.2, 109.2),
            (101, 100),
            (117.8, 93.8),
            (149.4, 109.4),
            (141.5, 107.2)
        ]

        # Points that should be outside the board outline
        outside_points = [
            (98, 110),
            (124, 89),
            (118.5, 94.4),
            (129.1, 93.8),
            (129, 106.3),
            (119.2, 100.3),
            (166.5, 101.7),
            (126.7, 100.0)
        ]

        # Test inside points
        for x, y in inside_points:
            point = shapely.geometry.Point(x, y)
            assert outline.contains(point), f"Point ({x}, {y}) should be inside the board outline but is not"

        # Test outside points
        for x, y in outside_points:
            point = shapely.geometry.Point(x, y)
            assert not outline.contains(point), f"Point ({x}, {y}) should be outside the board outline but is inside"

    def test_extract_board_outline_simple_geometry_returns_none(self, kicad_test_projects):
        """Test that extract_board_outline returns None for boards with no outline defined."""
        project = kicad_test_projects["simple_geometry"]

        # Load the KiCad board
        board = pcbnew.LoadBoard(str(project.pcb_path))

        # Extract the board outline - should return None since no outline is defined
        outline = kicad.extract_board_outline(board)

        assert outline is None, "simple_geometry project should have no board outline defined"


class TestClipLayerWithOutline:

    @for_all_kicad_projects(include=["castellated_vias_internal_cutout", "castellated_vias_internal_cutout_aux_origin"])
    def test_layer_clipping_castellated_vias_internal_cutout(self, project):
        """Test that layer geometry is properly clipped by board outline in load_kicad_project."""
        # Load the KiCad project - this will apply layer clipping
        problem = kicad.load_kicad_project(project.pro_path)

        # Points that should be inside the board outline (from TestExtractBoardOutline)
        inside_points = [
            (100.2, 90.2),
            (100.2, 109.2),
            (101, 100),
            (117.8, 93.8),
            (149.4, 109.4),
            (141.5, 107.2)
        ]

        # Points that should be outside the board outline (from TestExtractBoardOutline)
        outside_points = [
            (98, 110),
            (124, 89),
            (118.5, 94.4),
            (129.1, 93.8),
            (129, 106.3),
            (119.2, 100.3),
            (166.5, 101.7),
            (126.7, 100.0)
        ]

        # Verify that we have layers in the problem
        assert len(problem.layers) > 0, "Problem should contain layers"

        # Test each layer's clipped geometry
        for layer in problem.layers:
            # Verify the layer has the expected shape type after clipping
            assert isinstance(layer.shape, shapely.geometry.MultiPolygon), \
                f"Layer {layer.name} shape should be MultiPolygon after clipping"

            # Test outside points - none should be contained in any layer geometry
            # since they are outside the board outline
            for x, y in outside_points:
                point = shapely.geometry.Point(x, y)
                assert not layer.shape.contains(point), \
                    f"Point ({x}, {y}) should not be contained in layer {layer.name} geometry after clipping (outside board outline)"

            # For inside points, they may or may not be contained depending on whether
            # there's actual copper geometry at those locations, but if they are contained,
            # it means the clipping is working (geometry is present and within board bounds)
            for x, y in inside_points:
                point = shapely.geometry.Point(x, y)
                # We don't assert anything here since copper may or may not be present
                # at these specific points, but the key test is that outside points
                # are never contained (tested above)

    def test_layer_clipping_simple_geometry_no_outline(self, kicad_test_projects):
        """Test layer clipping behavior when board has no outline defined."""
        project = kicad_test_projects["simple_geometry"]

        # Load the KiCad project - should work even without board outline
        problem = kicad.load_kicad_project(project.pro_path)

        # Verify that we have layers in the problem
        assert len(problem.layers) > 0, "Problem should contain layers even without board outline"

        # Test each layer's geometry (should be unclipped)
        for layer in problem.layers:
            # Verify the layer has the expected shape type
            assert isinstance(layer.shape, shapely.geometry.MultiPolygon), \
                f"Layer {layer.name} shape should be MultiPolygon"

            # Layer should have some non-empty geometry
            assert not layer.shape.is_empty, \
                f"Layer {layer.name} should have non-empty geometry"
