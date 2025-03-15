
import numpy as np
from dataclasses import dataclass
from typing import Optional, Iterable

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


@dataclass
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


@dataclass
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


@dataclass
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
