"""
Tests for ParaView VTK XML export functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

import lxml.etree

from padne import mesh, solver, problem, paraview


class TestFilenameSanitization:
    @pytest.mark.parametrize("name, expected", [
        ("F.Cu", "F.Cu"),
        ("Power Plane", "Power_Plane"),
        ("Layer/Name?!@", "Layer_Name"),
        ("", "layer"),
        ("   ", "layer"),
        ("Test___Name", "Test_Name"),
        ("In1-Cu.2", "In1-Cu.2"),
    ])
    def test_sanitize_filename(self, name, expected):
        used_names = set()
        assert paraview._sanitize_filename(name, used_names) == expected
        assert used_names == {expected}

    def test_sanitize_filename_duplicates(self):
        used_names = set()
        result1 = paraview._sanitize_filename("F.Cu", used_names)
        result2 = paraview._sanitize_filename("F.Cu", used_names)

        assert result1 == "F.Cu"
        assert result2 == "F.Cu_2"
        assert {"F.Cu", "F.Cu_2"} == used_names


class TestDataArrayCreation:
    def test_create_data_array_basic(self):
        parent = lxml.etree.Element("TestParent")
        values = [1.0, 2.0, 3.0]

        result = paraview.create_data_array(parent, "Float64", values)

        assert result.tag == "DataArray"
        assert result.get("type") == "Float64"
        assert result.get("format") == "ascii"
        assert result.get("Name") is None
        assert result.get("NumberOfComponents") is None
        assert result.text == "1.0 2.0 3.0"

        # Verify it was added to parent
        assert len(parent) == 1
        assert parent[0] == result

    def test_create_data_array_with_name(self):
        parent = lxml.etree.Element("TestParent")
        values = [10, 20, 30]

        result = paraview.create_data_array(parent, "Int32", values, name="test_array")

        assert result.get("type") == "Int32"
        assert result.get("Name") == "test_array"
        assert result.text == "10 20 30"

    def test_create_data_array_with_components(self):
        parent = lxml.etree.Element("TestParent")
        values = [1.0, 2.0, 0.0, 4.0, 5.0, 0.0]

        result = paraview.create_data_array(parent, "Float64", values, number_of_components=3)

        assert result.get("NumberOfComponents") == "3"
        assert result.text == "1.0 2.0 0.0 4.0 5.0 0.0"

    def test_create_data_array_with_name_and_components(self):
        parent = lxml.etree.Element("TestParent")
        values = [1, 2, 3]

        result = paraview.create_data_array(
            parent, "UInt8", values,
            name="triangle_types",
            number_of_components=1
        )

        assert result.get("type") == "UInt8"
        assert result.get("Name") == "triangle_types"
        assert result.get("NumberOfComponents") == "1"
        assert result.text == "1 2 3"

    def test_create_data_array_mixed_numeric_types(self):
        parent = lxml.etree.Element("TestParent")
        values = [1, 2.5, 3, 4.0]  # Mix of int and float

        result = paraview.create_data_array(parent, "Float64", values)

        assert result.text == "1 2.5 3 4.0"

    def test_create_data_array_empty_values(self):
        parent = lxml.etree.Element("TestParent")
        values = []

        result = paraview.create_data_array(parent, "Float64", values)

        assert result.text == ""


class TestVTKRootCreation:
    def test_create_vtk_root(self):
        root = paraview.create_vtk_root()

        assert root.tag == "VTKFile"
        assert root.get("type") == "UnstructuredGrid"
        assert root.get("version") == "0.1"
        assert root.get("byte_order") == "LittleEndian"


class TestPointDataCreation:
    def test_create_point_data_basic(self):
        # Create a minimal mesh with one vertex
        test_mesh = mesh.Mesh()
        vertex = test_mesh.make_vertex(mesh.Point(1.0, 2.0))

        # Create ZeroForm with one value
        potentials = mesh.ZeroForm(test_mesh)
        potentials[vertex] = 3.5

        point_data = paraview.create_point_data(potentials)

        assert point_data.tag == "PointData"
        assert point_data.get("Scalars") == "voltage"

        data_array = point_data.find("DataArray")
        assert data_array is not None
        assert data_array.get("type") == "Float64"
        assert data_array.get("Name") == "voltage"
        assert data_array.get("format") == "ascii"
        assert data_array.text == "3.5"

    def test_create_point_data_multiple_vertices(self):
        test_mesh = mesh.Mesh()
        vertices = [
            test_mesh.make_vertex(mesh.Point(0.0, 0.0)),
            test_mesh.make_vertex(mesh.Point(1.0, 0.0)),
            test_mesh.make_vertex(mesh.Point(0.0, 1.0))
        ]

        potentials = mesh.ZeroForm(test_mesh)
        potentials[vertices[0]] = 1.0
        potentials[vertices[1]] = 2.0
        potentials[vertices[2]] = 3.0

        point_data = paraview.create_point_data(potentials)
        data_array = point_data.find("DataArray")

        # Values should be in vertex iteration order
        expected_values = "1.0 2.0 3.0"
        assert data_array.text == expected_values


class TestPointsCreation:
    def test_create_points_basic(self):
        test_mesh = mesh.Mesh()
        vertex = test_mesh.make_vertex(mesh.Point(1.5, 2.5))

        points = paraview.create_points(test_mesh)

        assert points.tag == "Points"
        data_array = points.find("DataArray")
        assert data_array is not None
        assert data_array.get("type") == "Float64"
        assert data_array.get("NumberOfComponents") == "3"
        assert data_array.get("format") == "ascii"
        # Y coordinate is negated: -2.5
        assert data_array.text == "1.5 -2.5 0.0"

    def test_create_points_multiple_vertices(self):
        test_mesh = mesh.Mesh()
        vertices = [
            test_mesh.make_vertex(mesh.Point(0.0, 0.0)),
            test_mesh.make_vertex(mesh.Point(1.0, 0.0)),
            test_mesh.make_vertex(mesh.Point(0.0, 1.0))
        ]

        points = paraview.create_points(test_mesh)
        data_array = points.find("DataArray")

        # Y coordinates are negated: -(0,0,1) = (0,0,-1)
        # Original (0,0) -> (0, -0.0) = (0, -0.0)
        # Original (1,0) -> (1, -0.0) = (1, -0.0)
        # Original (0,1) -> (0, -1.0) = (0, -1.0)
        expected_coords = "0.0 -0.0 0.0 1.0 -0.0 0.0 0.0 -1.0 0.0"
        assert data_array.text == expected_coords


class TestCellsCreation:
    def test_create_cells_empty_mesh(self):
        test_mesh = mesh.Mesh()

        cells = paraview.create_cells(test_mesh)

        assert cells.tag == "Cells"

        # Check connectivity array
        connectivity = cells.find("DataArray[@Name='connectivity']")
        assert connectivity is not None
        assert connectivity.get("type") == "Int32"
        assert connectivity.text == ""

        # Check offsets array
        offsets = cells.find("DataArray[@Name='offsets']")
        assert offsets is not None
        assert offsets.get("type") == "Int32"
        assert offsets.text == ""

        # Check types array
        types = cells.find("DataArray[@Name='types']")
        assert types is not None
        assert types.get("type") == "UInt8"
        assert types.text == ""

    def test_create_cells_two_triangles(self):
        test_mesh = mesh.Mesh.from_triangle_soup(
            [mesh.Point(0.0, 0.0), mesh.Point(1.0, 0.0),
             mesh.Point(0.0, 1.0), mesh.Point(1.0, 1.0)],
            [(0, 1, 2), (1, 3, 2)])

        cells = paraview.create_cells(test_mesh)

        connectivity = [int(v) for v in cells.find("DataArray[@Name='connectivity']").text.split()]
        triangles = [set(connectivity[i:i + 3]) for i in range(0, len(connectivity), 3)]
        assert sorted(triangles, key=sorted) == [{0, 1, 2}, {1, 2, 3}]

        offsets = cells.find("DataArray[@Name='offsets']")
        assert offsets.text == "3 6"

        types = cells.find("DataArray[@Name='types']")
        assert types.text == "5 5"


class TestPieceCreation:
    def test_create_piece_basic_structure(self):
        # Create a minimal mesh with proper triangle structure
        test_mesh = mesh.Mesh()

        # Add three vertices for a triangle
        v0 = test_mesh.make_vertex(mesh.Point(0.0, 0.0))
        v1 = test_mesh.make_vertex(mesh.Point(1.0, 0.0))
        v2 = test_mesh.make_vertex(mesh.Point(0.0, 1.0))

        # Create face and half-edges forming a triangle
        face = test_mesh.make_face()
        e0 = test_mesh.connect_vertices(v0, v1)
        e1 = test_mesh.connect_vertices(v1, v2)
        e2 = test_mesh.connect_vertices(v2, v0)

        # Connect the edges in a loop
        mesh.HalfEdge.connect(e0, e1)
        mesh.HalfEdge.connect(e1, e2)
        mesh.HalfEdge.connect(e2, e0)

        # Associate edges with face
        e0.face = face
        e1.face = face
        e2.face = face
        face.edge = e0

        potentials = mesh.ZeroForm(test_mesh)
        potentials[v0] = 1.0
        potentials[v1] = 2.0
        potentials[v2] = 3.0

        piece = paraview.create_piece(test_mesh, potentials)

        assert piece.tag == "Piece"
        assert piece.get("NumberOfPoints") == "3"
        assert piece.get("NumberOfCells") == "1"

        # Check that all required sub-elements are present
        point_data = piece.find("PointData")
        assert point_data is not None

        points = piece.find("Points")
        assert points is not None

        cells = piece.find("Cells")
        assert cells is not None


class TestSolutionExport:
    def test_export_solution_file_creation(self):
        # Create mock problem with layer names
        mock_layer = Mock(spec=problem.Layer)
        mock_layer.name = "F.Cu"
        mock_problem = Mock(spec=problem.Problem)
        mock_problem.layers = [mock_layer]

        # Create minimal mesh and potentials
        test_mesh = mesh.Mesh()
        vertex = test_mesh.make_vertex(mesh.Point(1.0, 2.0))

        potentials = mesh.ZeroForm(test_mesh)
        potentials[vertex] = 3.3

        layer_solution = solver.LayerSolution(
            meshes=[test_mesh],
            potentials=[potentials]
        )

        solution = solver.Solution(
            problem=mock_problem,
            layer_solutions=[layer_solution],
            solver_info=solver.SolverInfo(ground_node_current=0.0, residual_norm=0.0, relative_residual=0.0)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            paraview.export_solution(solution, output_dir)

            # Verify directory was created and file exists
            assert output_dir.exists()
            vtu_files = list(output_dir.glob("*.vtu"))
            assert len(vtu_files) == 1

            output_file = vtu_files[0]
            assert output_file.name == "F.Cu.vtu"
            assert output_file.read_text(encoding="utf-8").startswith(
                "<?xml version='1.0' encoding='UTF-8'?>")

            # Parse and validate XML structure
            tree = lxml.etree.parse(str(output_file))
            root = tree.getroot()

            assert root.tag == "VTKFile"
            assert root.get("type") == "UnstructuredGrid"

            unstructured_grid = root.find("UnstructuredGrid")
            assert unstructured_grid is not None

            pieces = unstructured_grid.findall("Piece")
            assert len(pieces) == 1

            piece = pieces[0]
            assert piece.get("NumberOfPoints") == "1"

    def test_export_solution_multiple_layers(self):
        # Create mock problem with multiple layer names
        layer_names = ["F.Cu", "B.Cu"]
        mock_layers = []
        for name in layer_names:
            mock_layer = Mock(spec=problem.Layer)
            mock_layer.name = name
            mock_layers.append(mock_layer)

        mock_problem = Mock(spec=problem.Problem)
        mock_problem.layers = mock_layers

        # Create two layers with different meshes
        layer_solutions = []
        for layer_idx in range(2):
            test_mesh = mesh.Mesh()
            vertex = test_mesh.make_vertex(mesh.Point(float(layer_idx), 0.0))

            potentials = mesh.ZeroForm(test_mesh)
            potentials[vertex] = float(layer_idx + 1)

            layer_solution = solver.LayerSolution(
                meshes=[test_mesh],
                potentials=[potentials]
            )
            layer_solutions.append(layer_solution)

        solution = solver.Solution(
            problem=mock_problem,
            layer_solutions=layer_solutions,
            solver_info=solver.SolverInfo(ground_node_current=0.0, residual_norm=0.0, relative_residual=0.0)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            paraview.export_solution(solution, output_dir)

            # Verify two separate files were created
            vtu_files = list(output_dir.glob("*.vtu"))
            assert len(vtu_files) == 2

            filenames = {f.name for f in vtu_files}
            assert filenames == {"F.Cu.vtu", "B.Cu.vtu"}

            # Verify each file contains one piece
            for vtu_file in vtu_files:
                tree = lxml.etree.parse(str(vtu_file))
                root = tree.getroot()
                pieces = root.findall(".//Piece")
                assert len(pieces) == 1  # One piece per file
