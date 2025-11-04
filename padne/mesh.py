
import collections
import numpy as np
import shapely.geometry
import padne._cgal as cgal

from dataclasses import dataclass, field
from typing import Optional, Iterable, Iterator, Hashable

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

    def __abs__(self) -> float:
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

    def orbit(self) -> Iterator["HalfEdge"]:
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

    def __getstate__(self):
        # We _do not_ pickle the twin/next/prev halfedges explicitly
        # to avoid reaching recursion depth limits
        # The Mesh class performs additional bookkeeping and rehydration
        # to ensure that the topology is properly unpickled.
        state = self.__dict__.copy()
        censor_keys = ["next", "prev", "twin"]
        for key in censor_keys:
            state[key] = id(state[key])
        return state

    @property
    def is_boundary(self) -> bool:
        return self.face.is_boundary

    @staticmethod
    def connect(e1: "HalfEdge", e2: "HalfEdge") -> None:
        e1.next = e2
        e2.prev = e1

    def walk(self) -> Iterator["HalfEdge"]:
        edge = self
        while True:
            yield edge
            edge = edge.next
            if edge == self:
                break

    def cotan(self) -> float:
        """
        Compute the cotangent weight for this half-edge.
        """
        vertex_i = self.origin
        # Grab the vertex on the other side of the edge
        vertex_k = self.twin.origin
        ratio = 0.
        for other in [self.next.next, self.twin.next.next]:
            if other.next.face.is_boundary:
                # Do not include boundary edges
                continue
            vi = vertex_i.p - other.origin.p
            vk = vertex_k.p - other.origin.p
            ratio += abs(vi.dot(vk) / (vi ^ vk)) / 2
        return ratio


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
    def centroid(self) -> Point:
        """
        Compute the centroid of the face using the average of vertex coordinates.
        """
        x_sum = 0.0
        y_sum = 0.0
        count = 0
        for vertex in self.vertices:
            x_sum += vertex.p.x
            y_sum += vertex.p.y
            count += 1
        return Point(x_sum / count, y_sum / count)

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

    def __iter__(self) -> Iterator[T]:
        return iter(self._idx_to_obj)

    def __contains__(self, obj: T) -> bool:
        """Check if an object is in the index map."""
        return obj in self._obj_to_idx

    def items(self) -> Iterator[tuple[int, T]]:
        for idx, obj in enumerate(self._idx_to_obj):
            yield idx, obj


class Mesh:
    def __init__(self):
        self.vertices = IndexMap[Vertex]()
        self.halfedges = IndexMap[HalfEdge]()
        self.faces = IndexMap[Face]()
        self.boundaries = IndexMap[Face]()
        self._edge_map: dict[tuple[int, int], HalfEdge] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        # Will be important for rehydrating the mesh
        ids_to_hedges = {
            id(hedge): hedge for hedge in state["halfedges"]
        }
        state["_ids_to_hedges"] = ids_to_hedges
        return state

    def __setstate__(self, state):
        _ids_to_hedges = state.pop("_ids_to_hedges")
        # Rehydrate the halfedges
        for hedge in state["halfedges"]:
            # This should be set to id(...) in the __getstate__ method
            # of HalfEdge
            assert isinstance(hedge.next, int) and isinstance(hedge.prev, int) \
                and isinstance(hedge.twin, int), "HalfEdge state is not properly serialized"
            hedge.next = _ids_to_hedges[hedge.next]
            hedge.prev = _ids_to_hedges[hedge.prev]
            hedge.twin = _ids_to_hedges[hedge.twin]

        self.__dict__.update(state)

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

        boundary_hedges = set()
        vertex_to_boundary_hedge = {}
        for hedge in mesh.halfedges:
            if hedge.face is not None:
                continue
            boundary_hedges.add(hedge)

            if hedge.origin in vertex_to_boundary_hedge:
                raise ValueError("Non-manifold mesh")

            vertex_to_boundary_hedge[hedge.origin] = hedge

        boundary_hedges = set(hedge for hedge in mesh.halfedges if hedge.face is None)
        while boundary_hedges:
            hedge = boundary_hedges.pop()

            # Okay, we have an as of yet unprocessed boundary hedge. Now we
            # need to effectively "walk aournd" the boundary. We assume that
            # there is always at most one "outgoing" boundary edge per vertex

            face = Face(is_boundary=True)
            mesh.boundaries.add(face)
            face.edge = hedge
            hedge.face = face

            hedge_prev = hedge
            while True:
                vertex_next = hedge_prev.twin.origin
                hedge_next = vertex_to_boundary_hedge.get(vertex_next)
                if hedge_next not in boundary_hedges:
                    # We have reached the end of the boundary
                    break

                boundary_hedges.remove(hedge_next)

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

    def d(self) -> "OneForm":
        """
        Compute the exterior derivative (gradient) of this 0-form.

        For a function f on vertices, the exterior derivative df is a 1-form where:
        (df)[edge] = f(target_vertex) - f(source_vertex)

        Returns:
            A OneForm representing the gradient of this function
        """
        one_form = OneForm(self.mesh)

        # Process each half-edge
        for hedge in self.mesh.halfedges:
            # For edge from A to B: df[edge] = f(B) - f(A)
            target_value = self[hedge.twin.origin]  # Value at target vertex
            source_value = self[hedge.origin]       # Value at source vertex
            one_form[hedge] = target_value - source_value

        return one_form


@dataclass
class OneForm:
    """
    A discrete 1-form defined on the (h)edges of a mesh.
    """
    mesh: Mesh
    values: dict[HalfEdge, float] = field(
        default_factory=dict,
        repr=False,
    )

    def __getitem__(self, hedge: HalfEdge) -> float:
        """Get the value of the 1-form on a half-edge."""
        if hedge not in self.mesh.halfedges:
            raise KeyError("HalfEdge not in mesh")

        return self.values.get(hedge, 0.0)

    def __setitem__(self, hedge: HalfEdge, value: float) -> None:
        """Set the value of the 1-form on a half-edge, ensuring antisymmetry."""
        if hedge not in self.mesh.halfedges:
            raise KeyError("HalfEdge not in mesh")

        # Set value for hedge and -value for its twin
        self.values[hedge] = value
        self.values[hedge.twin] = -value

    def __add__(self, other: "OneForm") -> "OneForm":
        """Add two OneForm objects element-wise."""
        if self.mesh is not other.mesh:
            raise ValueError("Cannot add OneForms on different meshes")

        result = OneForm(self.mesh)
        for hedge in self.mesh.halfedges:
            result[hedge] = self[hedge] + other[hedge]
        return result

    def __sub__(self, other: "OneForm") -> "OneForm":
        """Subtract another OneForm element-wise."""
        if self.mesh is not other.mesh:
            raise ValueError("Cannot subtract OneForms on different meshes")

        result = OneForm(self.mesh)
        for hedge in self.mesh.halfedges:
            result[hedge] = self[hedge] - other[hedge]
        return result

    def __mul__(self, scalar: float) -> "OneForm":
        """Multiply this OneForm by a scalar."""
        result = OneForm(self.mesh)
        for hedge in self.mesh.halfedges:
            result[hedge] = self[hedge] * scalar
        return result

    def __rmul__(self, scalar: float) -> "OneForm":
        """Right multiplication by a scalar."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "OneForm":
        """Divide this OneForm by a scalar."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide OneForm by zero")

        result = OneForm(self.mesh)
        for hedge in self.mesh.halfedges:
            result[hedge] = self[hedge] / scalar
        return result

    def __neg__(self) -> "OneForm":
        """Negate all values in this OneForm."""
        result = OneForm(self.mesh)
        for hedge in self.mesh.halfedges:
            result[hedge] = -self[hedge]
        return result


@dataclass
class TwoForm:
    """
    A discrete 2-form defined on the faces of a mesh.
    """
    mesh: Mesh
    values: dict[Face, float] = field(
        default_factory=dict,
        repr=False,
    )

    def __getitem__(self, face: Face) -> float:
        """Get the value of the 2-form on a face."""
        if face not in self.mesh.faces and face not in self.mesh.boundaries:
            raise KeyError("Face not in mesh")

        # Boundary faces always return 0.0
        if face in self.mesh.boundaries:
            return 0.0

        return self.values.get(face, 0.0)

    def __setitem__(self, face: Face, value: float) -> None:
        """Set the value of the 2-form on a face."""
        if face not in self.mesh.faces:
            raise KeyError("Face not in mesh.faces (boundary faces not supported)")

        self.values[face] = value

    def __add__(self, other: "TwoForm") -> "TwoForm":
        """Add two TwoForm objects element-wise."""
        if self.mesh is not other.mesh:
            raise ValueError("Cannot add TwoForms on different meshes")

        result = TwoForm(self.mesh)
        for face in self.mesh.faces:
            result[face] = self[face] + other[face]
        return result

    def __sub__(self, other: "TwoForm") -> "TwoForm":
        """Subtract another TwoForm element-wise."""
        if self.mesh is not other.mesh:
            raise ValueError("Cannot subtract TwoForms on different meshes")

        result = TwoForm(self.mesh)
        for face in self.mesh.faces:
            result[face] = self[face] - other[face]
        return result

    def __mul__(self, scalar: float) -> "TwoForm":
        """Multiply this TwoForm by a scalar."""
        result = TwoForm(self.mesh)
        for face in self.mesh.faces:
            result[face] = self[face] * scalar
        return result

    def __rmul__(self, scalar: float) -> "TwoForm":
        """Right multiplication by a scalar."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "TwoForm":
        """Divide this TwoForm by a scalar."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide TwoForm by zero")

        result = TwoForm(self.mesh)
        for face in self.mesh.faces:
            result[face] = self[face] / scalar
        return result

    def __neg__(self) -> "TwoForm":
        """Negate all values in this TwoForm."""
        result = TwoForm(self.mesh)
        for face in self.mesh.faces:
            result[face] = -self[face]
        return result


PolyBoundaryDistanceMap = cgal.PolyBoundaryDistanceMap
CGALPolygon = cgal.CGALPolygon


class MeshingException(RuntimeError):
    """
    Exception raised when CGAL mesh generation fails due to invalid geometry.

    This includes cases such as:
    - Self-intersecting polygons with unauthorized constraint intersections
    - Degenerate edges that are too short (near-duplicate vertices)
    - Other geometric degeneracies that prevent mesh generation

    With CGAL_DEBUG enabled, these issues are detected early through CGAL's
    internal precondition checking, preventing crashes and providing clear
    error messages.
    """
    pass


class Mesher:
    """
    This class is responsible for generating a mesh from a Shapely polygon.
    Works through the triangle library.
    """

    @dataclass(frozen=True)
    class Config:
        """Configuration parameters for mesh generation."""
        minimum_angle: float = 20.0
        maximum_size: float = 0.6
        # Variable density parameters
        variable_density_min_distance: float = 0.5
        variable_density_max_distance: float = 3.0
        variable_size_maximum_factor: float = 3.0
        distance_map_quantization: float = 1.0

        # Static relaxed configuration for disconnected copper triangulation
        RELAXED = None  # Will be initialized after class definition

        @property
        def is_variable_density(self) -> bool:
            """Return True if variable density meshing is enabled."""
            return self.variable_size_maximum_factor != 1.0

        def __post_init__(self):
            """Validate configuration parameters."""
            if not (0 <= self.minimum_angle <= 60):
                raise ValueError(f"minimum_angle must be between 0 and 60 degrees, got {self.minimum_angle}")

            if self.maximum_size < 0:
                raise ValueError(f"maximum_size must be non-negative, got {self.maximum_size}")

            if self.variable_density_min_distance < 0:
                raise ValueError(f"variable_density_min_distance must be non-negative, got {self.variable_density_min_distance}")

            if self.variable_density_max_distance <= self.variable_density_min_distance:
                raise ValueError(f"variable_density_max_distance ({self.variable_density_max_distance}) must be greater than variable_density_min_distance ({self.variable_density_min_distance})")

            if self.variable_size_maximum_factor < 1.0:
                raise ValueError(f"variable_size_maximum_factor must be >= 1.0, got {self.variable_size_maximum_factor}")

            if self.distance_map_quantization <= 0:
                raise ValueError(f"distance_map_quantization must be positive, got {self.distance_map_quantization}")

    def __init__(self, config: Optional['Mesher.Config'] = None):
        self.config = config if config is not None else Mesher.Config()

    def _prepare_polygon_for_cgal(self,
                                  poly: shapely.geometry.Polygon,
                                  seed_points: list[Point] = []) -> tuple[list, list, list]:
        """
        Convert a Shapely polygon to vertices, segments, and seeds for CGAL.

        Args:
            poly: A Shapely polygon, potentially with holes
            seed_points: Additional seed points to include

        Returns:
            Tuple of (vertices, segments, seeds) for CGAL functions
        """
        # This serves to deduplicate vertices.
        # In theory, deduplication should not be needed
        vertices = []
        segments = []
        seeds = [
            (seed_point.x, seed_point.y)
            for seed_point in seed_points
        ]
        seeds.append(poly.representative_point().coords[0])

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

        for hole in poly.interiors:
            insert_linear_ring(hole)

        return vertices, segments, seeds

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
        import padne._cgal as cgal

        vertices, segments, seeds = self._prepare_polygon_for_cgal(poly, seed_points)

        try:
            # Create distance map for variable density meshing only if enabled
            if self.config.is_variable_density:
                distance_map = cgal.PolyBoundaryDistanceMap(poly, self.config.distance_map_quantization)
            else:
                distance_map = None

            cgal_output = cgal.mesh(self.config, vertices, segments, seeds, distance_map)
        except RuntimeError as e:
            # Re-raise as MeshingException to provide clearer error context
            raise MeshingException(str(e)) from e

        mesh = Mesh.from_triangle_soup(
            [Point(*p) for p in cgal_output['vertices']],
            cgal_output['triangles']
        )

        return mesh


Mesher.Config.RELAXED = Mesher.Config(
    minimum_angle=5.0,
    maximum_size=0,
    variable_size_maximum_factor=1.0
)
