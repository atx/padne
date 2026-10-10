"""
ParaView VTK XML export functionality for FEM simulation results.

This module provides functions to export padne's FEM simulation results to the
VTK XML UnstructuredGrid format, compatible with ParaView and other VTK-based
visualization tools.
"""

import logging
import re
from pathlib import Path
from typing import Iterable

import lxml.etree
import numpy as np
from lxml.etree import Element, SubElement

from . import mesh, solver

log = logging.getLogger(__name__)


def _sanitize_filename(name: str, used_names: set[str], fallback_prefix: str = "layer") -> str:
    """Sanitize a layer name into a filename (no extension) unique in used_names, adding it."""
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
    """Append an ASCII DataArray of the given VTK type (e.g. "Float64") to parent."""
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
    """Create the root VTKFile element for an UnstructuredGrid."""
    root = Element("VTKFile")
    root.set("type", "UnstructuredGrid")
    root.set("version", "0.1")
    root.set("byte_order", "LittleEndian")
    return root


def create_point_data(potentials: mesh.ZeroForm) -> Element:
    """Create a PointData element holding the voltage at each vertex."""
    point_data = Element("PointData")
    point_data.set("Scalars", "voltage")

    create_data_array(point_data, "Float64", potentials.values.tolist(), name="voltage")
    return point_data


def create_points(mesh_obj: mesh.Mesh) -> Element:
    """Create a Points element with z=0 and Y negated for ParaView orientation."""
    points = Element("Points")

    pos = mesh_obj.positions()
    coordinates = np.column_stack([pos[:, 0], -pos[:, 1], np.zeros(len(pos))])
    create_data_array(points, "Float64", coordinates.ravel().tolist(), number_of_components=3)
    return points


def create_cells(mesh_obj: mesh.Mesh) -> Element:
    """Create a Cells element with triangle connectivity, offsets and types."""
    cells = Element("Cells")
    triangles = mesh_obj.triangles()

    create_data_array(cells, "Int32", triangles.ravel().tolist(), name="connectivity")
    offset_values = [3 * (i + 1) for i in range(len(triangles))]
    create_data_array(cells, "Int32", offset_values, name="offsets")

    # Types array (all triangles = type 5)
    type_values = [5] * len(triangles)
    create_data_array(cells, "UInt8", type_values, name="types")

    return cells


def create_piece(mesh_obj: mesh.Mesh, potentials: mesh.ZeroForm) -> Element:
    """Create a Piece element for one triangular mesh with its voltage field."""
    num_points = len(mesh_obj.vertices)
    num_cells = len(mesh_obj.faces)

    piece = Element("Piece")
    piece.set("NumberOfPoints", str(num_points))
    piece.set("NumberOfCells", str(num_cells))

    # Add sub-elements
    piece.append(create_point_data(potentials))
    piece.append(create_points(mesh_obj))
    piece.append(create_cells(mesh_obj))

    return piece


def export_solution(solution: solver.Solution, output_dir: Path) -> None:
    """Export a Solution to VTK XML, one .vtu file per layer."""
    log.info(f"Exporting solution to ParaView format: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    total_files = 0
    total_pieces = 0
    for layer, layer_solution in zip(solution.problem.layers, solution.layer_solutions):
        layer_pieces = len(layer_solution.meshes)
        log.debug(f"Processing layer '{layer.name}' with {layer_pieces} meshes")
        if not layer_pieces:
            log.warning(f"Skipping layer '{layer.name}' - no non-empty meshes")
            continue

        output_file = output_dir / f"{_sanitize_filename(layer.name, used_names)}.vtu"
        root = create_vtk_root()
        unstructured_grid = SubElement(root, "UnstructuredGrid")
        for mesh_obj, potential in zip(layer_solution.meshes, layer_solution.potentials):
            unstructured_grid.append(create_piece(mesh_obj, potential))
        log.debug(f"Layer '{layer.name}' -> {output_file} ({layer_pieces} pieces)")

        lxml.etree.ElementTree(root).write(
            str(output_file),
            xml_declaration=True,
            encoding="utf-8",
            pretty_print=True
        )
        total_files += 1
        total_pieces += layer_pieces

    log.info(f"Exported {total_pieces} mesh pieces across {total_files} layer files to {output_dir}")
