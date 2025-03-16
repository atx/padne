
import numpy as np
from dataclasses import dataclass
from typing import Optional, Iterable, Hashable

# The purpose of this module is to generate triangular meshes from Shapely
# (multi)polygons


@dataclass(frozen=True)
class Vector:
    dx: float
    dy: float

    def dot(self, other: "Vector") -> float:
        return self.dx * other.dx + self.dy * other.dy

    def __add__(self, other: "Vector") -> "Vector":
        if not isinstance(other, Vector):
            raise TypeError("Addition is only defined for Vectors")
        return Vector(self.dx + other.dx, self.dy + other.dy)

    def __mul__(self, scalar: float) -> "Vector":
        return Vector(self.dx * scalar, self.dy * scalar)

    def __rmul__(self, scalar: float) -> "Vector":
        return self.__mul__(scalar)

    def __neg__(self) -> "Vector":
        return Vector(-self.dx, -self.dy)

    def __xor__(self, other: "Vector") -> float:
        # Should there be a special 2-vector type?
        return self.dx * other.dy - self.dy * other.dx

    def __abs__(self):
        return np.sqrt(self.dx ** 2 + self.dy ** 2)


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance(self, other: "Point") -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __sub__(self, other: "Point") -> Vector:
        if not isinstance(other, Point):
            raise TypeError("Subtraction is only defined for Points")
        return Vector(self.x - other.x, self.y - other.y)


@dataclass(eq=False)
class Vertex:
    p: Point
    out: Optional["HalfEdge"] = None

    def orbit(self) -> Iterable["Vertex"]:
        edge = self.out
        while True:
            yield edge
            edge = edge.twin.next
            if edge == self.out:
                break


@dataclass(eq=False)
class HalfEdge:
    origin: Vertex
    twin: Optional["HalfEdge"] = None
    next: Optional["HalfEdge"] = None
    face: Optional["Face"] = None

    @property
    def is_boundary(self) -> bool:
        return self.face is None

    @property
    def prev(self) -> "HalfEdge":
        return self.twin.next.twin


@dataclass(eq=False)
class Face:
    edge: HalfEdge = None

    @property
    def edges(self):
        edge = self.edge
        while True:
            yield edge
            edge = edge.next
            if edge == self.edge:
                break

    @property
    def vertices(self):
        for edge in self.edges:
            yield edge.origin

    @property
    def area(self) -> float:
        """
        Compute the area using the shoelace formula.
        Returns the absolute value to ensure positive area regardless of vertex order.
        """
        area = 0.0
        for edge in self.edges:
            p1 = edge.origin.p
            p2 = edge.next.origin.p
            area += (p1.x * p2.y - p2.x * p1.y)
        return 0.5 * abs(area)


class IndexMap[T: Hashable]:
    """
    A simple class that maps objects to indices and vice versa.
    """

    def __init__(self):
        self._obj_to_idx: dict[T, int] = {}
        self._idx_to_obj: list[T] = []

    def add(self, obj: T) -> int:
        # TODO: Maybe we do not want to allow adding the same object multiple
        # times? This breaks the set .add interface, but we are realistically
        # never going to do it and it could catch bugs
        if obj not in self._obj_to_idx:
            idx = len(self._idx_to_obj)
            self._obj_to_idx[obj] = idx
            self._idx_to_obj.append(obj)
        return self._obj_to_idx[obj]

    def to_index(self, obj: T) -> int:
        return self._obj_to_idx[obj]

    def to_object(self, idx: int) -> T:
        return self._idx_to_obj[idx]

    def __len__(self) -> int:
        return len(self._idx_to_obj)

    def items(self) -> Iterable[tuple[int, T]]:
        for idx, obj in enumerate(self._idx_to_obj):
            yield idx, obj


class Mesh:
    def __init__(self):
        self.vertices = IndexMap[Vertex]()
        self.halfedges = IndexMap[HalfEdge]()
        self.faces = IndexMap[Face]()
        self._edge_map: dict[tuple[int, int], HalfEdge] = {}

    def make_vertex(self, p: Point) -> Vertex:
        v = Vertex(p)
        self.vertices.add(v)
        return v

    def connect_vertices(self, v1: Vertex, v2: Vertex) -> HalfEdge:
        """
        Return a half-edge between the two vertices v1 and v2. If the half
        edge does not exist, create it. The twin half-edge is also created.
        """
        key12 = (self.vertices.to_index(v1), self.vertices.to_index(v2))
        key21 = (key12[1], key12[0])
        if key12 in self._edge_map:
            return self._edge_map[key12]

        # Assert that the twin also does not exist. It should not be possible
        # to have one without the other
        assert key21 not in self._edge_map, "Inconsistent half edge state"

        e12 = HalfEdge(v1)
        e21 = HalfEdge(v2)
        e12.twin = e21
        e21.twin = e12
        self.halfedges.add(e12)
        self.halfedges.add(e21)

        self._edge_map[key12] = e12
        self._edge_map[key21] = e21

        # Update the vertex out pointers
        if v1.out is None:
            v1.out = e12
        if v2.out is None:
            v2.out = e21

        return e12

    def triangle_from_vertices(self, v1: Vertex, v2: Vertex, v3: Vertex) -> Face:
        """
        Create a triangle face from the three vertices.
        """
        e12 = self.connect_vertices(v1, v2)
        e23 = self.connect_vertices(v2, v3)
        e31 = self.connect_vertices(v3, v1)

        f = Face(e12)
        e12.next = e23
        e23.next = e31
        e31.next = e12

        e12.face = e23.face = e31.face = f

        self.faces.add(f)

        return f

    def euler_characteristic(self) -> int:
        return len(self.vertices) - len(self.halfedges) // 2 + len(self.faces)
