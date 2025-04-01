
import enum
import shapely

from dataclasses import dataclass
from typing import Protocol


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
    # This is in Siemens
    # Note that this is computed by
    # conductivity [S/m] * thickness [m]
    conductance: float


@dataclass(frozen=True)
class Terminal:
    """
    Represents a connection point of a lumped element to a layer.
    """
    layer: Layer
    point: shapely.geometry.Point


@dataclass(frozen=True)
class BaseLumped(Protocol):

    def __post_init__(self):
        assert self.terminals, "Lumped elements must have terminals"

    @property
    def terminals(self) -> list[Terminal]:
        ...


@dataclass(frozen=True)
class Resistor(BaseLumped):
    a: Terminal
    b: Terminal
    resistance: float

    @property
    def terminals(self) -> list[Terminal]:
        return [self.a, self.b]


@dataclass(frozen=True)
class VoltageSource(BaseLumped):
    p: Terminal
    n: Terminal
    voltage: float

    @property
    def terminals(self) -> list[Terminal]:
        return [self.p, self.n]


@dataclass(frozen=True)
class CurrentSource(BaseLumped):
    f: Terminal
    t: Terminal
    current: float

    @property
    def terminals(self) -> list[Terminal]:
        return [self.f, self.t]


@dataclass(frozen=True)
class VoltageRegulator(BaseLumped):
    """
    This is effectivelly a CCCS, but the current is determined by a current
    through a voltage source.
    """
    v_p: Terminal
    v_n: Terminal

    i_f: Terminal
    i_t: Terminal

    @property
    def terminals(self) -> list[Terminal]:
        return [self.v_p, self.v_n, self.i_f, self.i_t]


@dataclass(frozen=True)
class Problem:
    layers: list[Layer]
    lumpeds: list[BaseLumped]
