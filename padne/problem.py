
import shapely

from dataclasses import dataclass, field
from typing import Protocol


# This file should specify the data structure that gets passed to the mesher
# and later to the FEM solver.


@dataclass(frozen=True)
class Layer:
    """
    Represents a single copper layer of the input circuit board.
    """
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


@dataclass(frozen=True, eq=False)
class NodeID:
    """
    Opaque identifier for a node in the network.
    """
    pass


@dataclass(frozen=True)
class Connection:
    """
    Represents a connection between an internal node of the Network and
    a point in the layer.
    """
    layer: Layer
    point: shapely.geometry.Point
    node_id: NodeID = field(default_factory=NodeID)


@dataclass(frozen=True)
class BaseLumped(Protocol):
    """
    Represents a lumped element in the network.
    """

    def __post_init__(self):
        assert self.terminals, "Lumped elements must have terminals"

    @property
    def terminals(self) -> list[NodeID]:
        ...

    @property
    def is_source(self) -> bool:
        return False

    @property
    def extra_variable_count(self) -> int:
        return 0


@dataclass(frozen=True)
class Network:
    connections: list[Connection]
    elements: list[BaseLumped]
    nodes: dict[NodeID, int] = field(init=False)
    has_source: bool = field(init=False)

    def __post_init__(self):
        # Initialize the nodes
        node_set = set()
        for element in self.elements:
            for terminal in element.terminals:
                if not isinstance(terminal, NodeID):
                    raise TypeError("Terminal must be a NodeID")
                node_set.add(terminal)

        # Do not allow floating elements
        for connection in self.connections:
            if connection.node_id not in node_set:
                raise ValueError("Connection must be connected to at least one element")

        keys = list(node_set)
        nodes = {key: i for i, key in enumerate(keys)}
        # This bypasses the frozen dataclass restriction
        object.__setattr__(self, "nodes", nodes)
        # Check if the network has a source
        has_source = any(element.is_source for element in self.elements)
        object.__setattr__(self, "has_source", has_source)


@dataclass(frozen=True)
class Resistor(BaseLumped):
    a: NodeID
    b: NodeID
    resistance: float

    @property
    def terminals(self) -> list[NodeID]:
        return [self.a, self.b]


@dataclass(frozen=True)
class VoltageSource(BaseLumped):
    p: NodeID
    n: NodeID
    voltage: float

    @property
    def terminals(self) -> list[NodeID]:
        return [self.p, self.n]

    @property
    def is_source(self) -> bool:
        return True

    @property
    def extra_variable_count(self) -> int:
        return 1


@dataclass(frozen=True)
class CurrentSource(BaseLumped):
    f: NodeID
    t: NodeID
    current: float

    @property
    def terminals(self) -> list[NodeID]:
        return [self.f, self.t]

    @property
    def is_source(self) -> bool:
        return True


@dataclass(frozen=True)
class Problem:
    layers: list[Layer]
    networks: list[Network]
