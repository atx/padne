"""
ParaView VTK XML export functionality for FEM simulation results.

This module provides functions to export padne's FEM simulation results to the
VTK XML UnstructuredGrid format, compatible with ParaView and other VTK-based
visualization tools.
"""

import logging
import re
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import lxml.etree
from lxml.etree import Element, SubElement

from . import mesh, solver

log = logging.getLogger(__name__)


def _sanitize_filename(name: str, used_names: Set[str], fallback_prefix: str = "layer") -> str:
    """Sanitize a layer name for use as a filename.

    Args:
        name: Original layer name
        used_names: Set of already used filenames to avoid duplicates
        fallback_prefix: Prefix to use if name is empty or invalid

    Returns:
        Sanitized filename (without extension)
    """
    # Handle empty or whitespace-only names
    if not name or not name.strip():
        base = fallback_prefix
    else:
        # Replace spaces with underscores, keep only alphanumeric, underscore, hyphen, dots
        base = re.sub(r'[^a-zA-Z0-9_.-]', '_', name.strip())
        # Remove multiple consecutive underscores
        base = re.sub(r'_+', '_', base)
        # Remove leading/trailing underscores (but keep dots)
        base = base.strip('_')
        # If nothing left after sanitization, use fallback
        if not base:
            base = fallback_prefix

    # Handle duplicates by appending counter
    if base not in used_names:
        used_names.add(base)
        return base

    counter = 2
    while f"{base}_{counter}" in used_names:
        counter += 1

    result = f"{base}_{counter}"
    used_names.add(result)
    return result


def create_data_array(
    parent: Element,
    data_type: str,
    values: Iterable[int | float],
    name: str | None = None,
    number_of_components: int | None = None
) -> Element:
    """Create a DataArray element with specified type and values.

    Args:
        parent: Parent element to attach the DataArray to
        data_type: VTK data type (e.g., "Float64", "Int32", "UInt8")
        values: Numeric values to store in the array
        name: Optional name attribute for the DataArray
        number_of_components: Optional NumberOfComponents attribute

    Returns:
        Created DataArray element
    """
    data_array = SubElement(parent, "DataArray")
    data_array.set("type", data_type)
    data_array.set("format", "ascii")

    if name is not None:
        data_array.set("Name", name)

    if number_of_components is not None:
        data_array.set("NumberOfComponents", str(number_of_components))

    # Convert all values to strings and join with spaces
    data_array.text = " ".join(str(value) for value in values)

    return data_array


def create_vtk_root() -> Element:
    """Create the root VTKFile element with standard attributes.

    Returns:
        Root VTKFile element configured for UnstructuredGrid format
    """
    root = Element("VTKFile")
    root.set("type", "UnstructuredGrid")
    root.set("version", "0.1")
    root.set("byte_order", "LittleEndian")
    return root


def create_point_data(potentials: mesh.ZeroForm) -> Element:
    """Create PointData element with voltage scalar field values.

    Args:
        potentials: ZeroForm containing scalar values at mesh vertices

    Returns:
        PointData element containing the voltage field data
    """
    point_data = Element("PointData")
    point_data.set("Scalars", "voltage")

    # Extract values in vertex index order
    vertex_values = [potentials[vertex] for vertex in potentials.mesh.vertices]

    create_data_array(point_data, "Float64", vertex_values, name="voltage")
    return point_data


def create_points(mesh_obj: mesh.Mesh) -> Element:
    """Create Points element with vertex coordinates.

    Args:
        mesh_obj: Mesh object containing vertices

    Returns:
        Points element containing 3D coordinates (z=0 for 2D meshes)
        Note: Y coordinates are negated for ParaView orientation
    """
    points = Element("Points")

    # Extract coordinates in vertex index order with Y-axis negated
    coordinates = []
    for vertex in mesh_obj.vertices:
        coordinates.extend([vertex.p.x, -vertex.p.y, 0.0])

    create_data_array(points, "Float64", coordinates, number_of_components=3)
    return points


def _extract_triangle_connectivity(mesh_obj: mesh.Mesh) -> List[Tuple[int, int, int]]:
    """Extract triangle connectivity from mesh face structure.

    Args:
        mesh_obj: Mesh object with half-edge topology

    Returns:
        List of triangles as (v0, v1, v2) vertex index tuples
    """
    triangles = []
    vertex_to_index = {vertex: i for i, vertex in enumerate(mesh_obj.vertices)}

    for face in mesh_obj.faces:
        if face.is_boundary:
            continue

        # Extract vertices from face edges
        face_vertices = []
        for edge in face.edges:
            vertex_idx = vertex_to_index[edge.origin]
            face_vertices.append(vertex_idx)

        # Ensure we have exactly 3 vertices for a triangle
        if len(face_vertices) == 3:
            triangles.append(tuple(face_vertices))
        else:
            log.warning(f"Non-triangular face with {len(face_vertices)} vertices, skipping")

    return triangles


def create_cells(mesh_obj: mesh.Mesh) -> Element:
    """Create Cells element with triangle connectivity, offsets, and types.

    Args:
        mesh_obj: Mesh object containing triangular faces

    Returns:
        Cells element with connectivity, offsets, and types arrays
    """
    cells = Element("Cells")
    triangles = _extract_triangle_connectivity(mesh_obj)

    # Connectivity array
    connectivity_values = []
    for tri in triangles:
        connectivity_values.extend([tri[0], tri[1], tri[2]])
    create_data_array(cells, "Int32", connectivity_values, name="connectivity")

    # Offsets array
    offset_values = [3 * (i + 1) for i in range(len(triangles))]
    create_data_array(cells, "Int32", offset_values, name="offsets")

    # Types array (all triangles = type 5)
    type_values = [5] * len(triangles)
    create_data_array(cells, "UInt8", type_values, name="types")

    return cells


def create_piece(mesh_obj: mesh.Mesh, potentials: mesh.ZeroForm) -> Element:
    """Create a Piece element representing one triangular mesh with voltage data.

    Args:
        mesh_obj: Triangular mesh object
        potentials: Scalar field values at mesh vertices

    Returns:
        Piece element containing mesh geometry and voltage field
    """
    num_points = len(mesh_obj.vertices)
    num_cells = len([f for f in mesh_obj.faces if not f.is_boundary])

    piece = Element("Piece")
    piece.set("NumberOfPoints", str(num_points))
    piece.set("NumberOfCells", str(num_cells))

    # Add sub-elements
    piece.append(create_point_data(potentials))
    piece.append(create_points(mesh_obj))
    piece.append(create_cells(mesh_obj))

    return piece


def export_solution(solution: solver.Solution, output_dir: Path) -> None:
    """Export a complete Solution to VTK XML format as separate files per layer.

    Args:
        solution: Complete solution containing meshes and potential fields
        output_dir: Directory where VTU files should be written (one per layer)
    """
    log.info(f"Exporting solution to ParaView format: {output_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep track of used filenames to handle duplicates
    used_names: Set[str] = set()

    # Process each layer solution as a separate file
    total_files = 0
    total_pieces = 0

    for layer_idx, layer_solution in enumerate(solution.layer_solutions):
        # Get layer name from the problem
        layer_name = solution.problem.layers[layer_idx].name
        log.debug(f"Processing layer '{layer_name}' with {len(layer_solution.meshes)} meshes")

        # Skip layers with no meshes
        meshes_and_potentials = [
            (mesh_obj, potential)
            for mesh_obj, potential in
            zip(layer_solution.meshes, layer_solution.potentials)
        ]

        if not meshes_and_potentials:
            log.warning(f"Skipping layer '{layer_name}' - no non-empty meshes")
            continue

        # Generate sanitized filename
        filename = _sanitize_filename(layer_name, used_names)
        output_file = output_dir / f"{filename}.vtu"

        # Create root structure for this layer
        root = create_vtk_root()
        unstructured_grid = SubElement(root, "UnstructuredGrid")

        # Add all meshes in this layer as pieces
        layer_pieces = 0
        for mesh_obj, potential in meshes_and_potentials:
            piece = create_piece(mesh_obj, potential)
            unstructured_grid.append(piece)
            layer_pieces += 1

        log.debug(f"Layer '{layer_name}' -> {output_file} ({layer_pieces} pieces)")

        # Write XML to file
        tree = lxml.etree.ElementTree(root)
        tree.write(
            str(output_file),
            xml_declaration=True,
            encoding="utf-8",
            pretty_print=True
        )

        total_files += 1
        total_pieces += layer_pieces

    log.info(f"Exported {total_pieces} mesh pieces across {total_files} layer files to {output_dir}")
