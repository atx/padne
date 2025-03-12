
import shapely

from dataclasses import dataclass


# This file should specify the data structure that gets passed to the mesher
# and later to the FEM solver.


@dataclass(frozen=True)
class Layer:
    # TODO: Theoretically, the solver does not need to distinguish layers,
    # just a bunch of polygons with resistivity connected via lumped elements.
    # However, the renderer _will_ need to know this.
    # So, figuring out how to handle this correctly in the data model is important.
    # A layer is simply a shapely multipolygon with some metadata
    # At some point, this will get triangulated. Not quite sure what the data structure
    shape: shapely.geometry.MultiPolygon
    name: str

    # Hmm, TODO: maybe it should be polygon specific? For now out of scope
    resistivity: float


@dataclass(frozen=True)
class Lumped:
    """
    This is a two-terminal device that is connected to two physical points in the circuit.
    """
    a_layer: Layer
    a_point: shapely.geometry.Point

    b_layer: Layer
    b_point: shapely.geometry.Point


@dataclass(frozen=True)
class Resistor(Lumped):
    resistance: float


@dataclass(frozen=True)
class VoltageSource(Lumped):
    voltage: float


@dataclass(frozen=True)
class CurrentSource(Lumped):
    current: float


@dataclass(frozen=True)
class Problem:
    layers: list[Layer]
    resistors: list[Resistor]
    voltage_sources: list[VoltageSource]
    current_sources: list[CurrentSource]
