
import collections
import numpy as np
import shapely.geometry
from dataclasses import dataclass, field
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
        
    def to_shapely(self) -> shapely.geometry.Point:
        """
        Convert this Point to a shapely.geometry.Point.
        
        Returns:
            A shapely Point with the same coordinates
        """
        return shapely.geometry.Point(self.x, self.y)


@dataclass(eq=False, repr=False)
class Vertex:
    p: Point
    out: Optional["HalfEdge"] = None

    def orbit(self) -> Iterable["HalfEdge"]:
        edge = self.out
        while True:
            yield edge
            edge = edge.twin.next
            if edge == self.out:
                break


@dataclass(eq=False, repr=False)
class HalfEdge:
    origin: Vertex
    twin: Optional["HalfEdge"] = None
    next: Optional["HalfEdge"] = None
    prev: Optional["HalfEdge"] = None
    face: Optional["Face"] = None

    @property
    def is_boundary(self) -> bool:
        return self.face.is_boundary

    @staticmethod
    def connect(e1: "HalfEdge", e2: "HalfEdge") -> None:
        e1.next = e2
        e2.prev = e1

    def walk(self):
        edge = self
        while True:
            yield edge
            edge = edge.next
            if edge == self:
                break


@dataclass(eq=False)
class Face:
    edge: HalfEdge = None
    is_boundary: bool = False

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

    def __iter__(self) -> Iterable[T]:
        return iter(self._idx_to_obj)

    def __contains__(self, obj: T) -> bool:
        """Check if an object is in the index map."""
        return obj in self._obj_to_idx

    def items(self) -> Iterable[tuple[int, T]]:
        for idx, obj in enumerate(self._idx_to_obj):
            yield idx, obj


class Mesh:
    def __init__(self):
        self.vertices = IndexMap[Vertex]()
        self.halfedges = IndexMap[HalfEdge]()
        self.faces = IndexMap[Face]()
        self.boundaries = IndexMap[Face]()
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
            assert key21 in self._edge_map, "Inconsistent half edge state"
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

    def euler_characteristic(self) -> int:
        return len(self.vertices) - len(self.halfedges) // 2 + len(self.faces)

    @classmethod
    def from_triangle_soup(cls,
                           points: list[Point],
                           triangles: list[tuple[int, int, int]]) -> "Mesh":
        mesh = cls()

        # First we create the vertices
        vertices = [mesh.make_vertex(p) for p in points]

        for tri in triangles:
            assert len(tri) == 3
            v1, v2, v3 = [vertices[i] for i in tri]

            vertex_edge_pairs = [(v1, v2), (v2, v3), (v3, v1)]
            # Create the _interior_ half-edges
            face = Face()
            mesh.faces.add(face)
            current_hedges = []
            for u, v in vertex_edge_pairs:
                hedge = mesh.connect_vertices(u, v)
                u.out = hedge
                face.edge = hedge  # We just get the last one
                hedge.face = face
                current_hedges.append(hedge)

            # Next, connect the interior hedges in a loop
            for h1, h2 in zip(current_hedges, current_hedges[1:] + [current_hedges[0]]):
                HalfEdge.connect(h1, h2)

        # Now, comes the final stage, where we need to produce the boundary
        # edges. We do this by iterating over all half-edges and checking if
        # they have a twin. Then we handle the boundary connectivity update

        for hedge in mesh.halfedges:
            if hedge.face is not None:
                continue
            # Okay, we have a boundary hedge. Now we need to effectively
            # "walk around" the boundary. We assume that there is always
            # at most one "outgoing" boundary edge per vertex
            face = Face(is_boundary=True)
            mesh.boundaries.add(face)
            face.edge = hedge
            hedge.face = face

            hedge_prev = hedge
            while True:
                vertex_next = hedge_prev.twin.origin
                # Now, we need to find the next boundary edge
                hedge_next_list = [
                    h for h in mesh.halfedges
                    if h.origin == vertex_next and h.face is None
                ]
                if len(hedge_next_list) == 0:
                    # We have reached the end of the boundary
                    break
                if len(hedge_next_list) > 1:
                    raise ValueError("Non-manifold mesh")

                hedge_next = hedge_next_list[0]

                assert hedge_next.next is None  # Sanity check, should not happen since we checked earlier

                HalfEdge.connect(hedge_prev, hedge_next)
                hedge_next.face = face
                hedge_prev = hedge_next

            # And finally, connect the last edge to the first
            HalfEdge.connect(hedge_prev, hedge)

        return mesh


@dataclass
class ZeroForm:
    mesh: Mesh
    values: dict[Vertex, float] = field(
        default_factory=lambda: collections.defaultdict(float),
        repr=False,
    )

    def __getitem__(self, vertex: Vertex) -> float:
        if vertex not in self.mesh.vertices:
            raise KeyError("Vertex not in mesh")
        return self.values[vertex]

    def __setitem__(self, vertex: Vertex, value: float) -> None:
        if vertex not in self.mesh.vertices:
            raise KeyError("Vertex not in mesh")
        self.values[vertex] = value

    def __add__(self, other: "ZeroForm") -> "ZeroForm":
        """Add two ZeroForms element-wise.
        
        Args:
            other: The ZeroForm to add to this one
            
        Returns:
            A new ZeroForm with the sum of values
            
        Raises:
            ValueError: If the two ZeroForms are on different meshes
        """
        if self.mesh is not other.mesh:
            raise ValueError("Cannot add ZeroForms on different meshes")
        
        result = ZeroForm(self.mesh)
        for vertex in self.mesh.vertices:
            result[vertex] = self[vertex] + other[vertex]
        return result

    def __sub__(self, other: "ZeroForm") -> "ZeroForm":
        """Subtract another ZeroForm element-wise.
        
        Args:
            other: The ZeroForm to subtract from this one
            
        Returns:
            A new ZeroForm with the difference of values
            
        Raises:
            ValueError: If the two ZeroForms are on different meshes
        """
        if self.mesh is not other.mesh:
            raise ValueError("Cannot subtract ZeroForms on different meshes")
        
        result = ZeroForm(self.mesh)
        for vertex in self.mesh.vertices:
            result[vertex] = self[vertex] - other[vertex]
        return result

    def __mul__(self, scalar: float) -> "ZeroForm":
        """Multiply this ZeroForm by a scalar.
        
        Args:
            scalar: The scalar value to multiply by
            
        Returns:
            A new ZeroForm with scaled values
        """
        result = ZeroForm(self.mesh)
        for vertex in self.mesh.vertices:
            result[vertex] = self[vertex] * scalar
        return result

    def __rmul__(self, scalar: float) -> "ZeroForm":
        """Right multiplication by a scalar.
        
        Args:
            scalar: The scalar value to multiply by
            
        Returns:
            A new ZeroForm with scaled values
        """
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "ZeroForm":
        """Divide this ZeroForm by a scalar.
        
        Args:
            scalar: The scalar value to divide by
            
        Returns:
            A new ZeroForm with divided values
            
        Raises:
            ZeroDivisionError: If scalar is zero
        """
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide ZeroForm by zero")
        
        result = ZeroForm(self.mesh)
        for vertex in self.mesh.vertices:
            result[vertex] = self[vertex] / scalar
        return result

    def __neg__(self) -> "ZeroForm":
        """Negate all values in this ZeroForm.
        
        Returns:
            A new ZeroForm with negated values
        """
        result = ZeroForm(self.mesh)
        for vertex in self.mesh.vertices:
            result[vertex] = -self[vertex]
        return result


class Mesher:
    """
    This class is responsible for generating a mesh from a Shapely polygon.
    Works through the triangle library.
    """

    def __init__(self, minimum_angle: float = 20.0, maximum_area: float = 0.1):
        self.minimum_angle = minimum_angle
        self.maximum_area = maximum_area

    def _make_triangle_args(self) -> str:
        r = ""
        r += "p"  # Planar straight line graph
        r += f"q{self.minimum_angle}"  # Quality mesh generation with minimum angle
        r += f"a{self.maximum_area}"  # Imposes a maximum triangle area constraint
        return r

    def poly_to_mesh(self,
                     poly: shapely.geometry.Polygon,
                     seed_points: list[Point] = []) -> Mesh:
        """
        Convert a Shapely polygon to a triangular mesh.
        
        Args:
            poly: A Shapely polygon, potentially with holes
            
        Returns:
            A Mesh object representing the triangulated polygon
        """
        import triangle as tr

        # This serves to deduplicate vertices.
        # In theory, deduplication should not be needed
        vertices = []
        segments = []

        def insert_linear_ring(ring):
            assert ring.is_closed
            if not ring.is_ccw:
                ring = shapely.geometry.LinearRing(reversed(ring.coords))
            # Add the first point
            i_first = len(vertices)

            for p in ring.coords[:-1]:
                vertices.append(p)

            n = len(ring.coords) - 1

            for i in range(n):
                segments.append((i_first + i, i_first + (i + 1) % n))

        insert_linear_ring(poly.exterior)

        hole_points = []
        for hole in poly.interiors:
            # Note that we need to convert the LinearRing into a Polygon here.
            # Otherwise representative_point returns a point that lies _on the linear ring_,
            # not within the space enclosed by the ring!
            rep_point = shapely.geometry.Polygon(hole).representative_point()
            hole_points.append(rep_point.coords[0])

            insert_linear_ring(hole)

        # Insert the seed points to the vertices list
        # Those points are not part of any segment
        vertices_set = set(vertices)
        for seed_point in seed_points:
            pt = (seed_point.x, seed_point.y)
            # We have to check we are not passing duplicate points to triangle.
            # This causes a malformed mesh to be generated at best and a segfault at worst.
            # TODO: It seems that even a 1e-100 difference in coordinates is sufficient
            # to prevent this bug. Not quite sure whether we should allow that though
            if pt in vertices_set:
                continue
            vertices.append(pt)
            # This is to make sure duplicate seed points are also handled
            vertices_set.add(pt)

        tri_input = {
            "vertices": np.array(vertices),
            "segments": np.array(segments),
        }

        if hole_points:
            # There is a bug in the library that makes it crash when
            # an empty holes array is passed
            tri_input["holes"] = np.array(hole_points)
        
        tri_output = tr.triangulate(tri_input, self._make_triangle_args())

        mesh = Mesh.from_triangle_soup(
            [Point(*p) for p in tri_output['vertices']],
            tri_output['triangles']
        )

        return mesh
