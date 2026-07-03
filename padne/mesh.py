
import numpy as np
import shapely.geometry
import padne._cgal as cgal
import padne._mesh as _mesh

from dataclasses import dataclass, field
from typing import Optional, Iterator

# The purpose of this module is to generate triangular meshes from Shapely
# (multi)polygons

index_type = np.uint32


@dataclass(frozen=True)
class Vector:
    dx: float
    dy: float

    def dot(self, other: "Vector") -> float:
        return self.dx * other.dx + self.dy * other.dy

    def __add__(self, other: object) -> "Vector":
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

    def __sub__(self, other: object) -> Vector:
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


# The half-edge data structures are implemented in C++ (padne/cpp/_mesh.cpp)
# as struct-of-arrays index storage. Vertex/HalfEdge/Face are lightweight
# value-type handles into that storage
#
# The algorithmic methods below are implemented in Python and attached onto
# the bound types; they are kept verbatim from the original pure-Python
# implementation. They will be moved into C++ in later commits.
Mesh = _mesh.Mesh
Vertex = _mesh.Vertex
HalfEdge = _mesh.HalfEdge
Face = _mesh.Face


def _extend(cls, name):
    """Attach a method or property to one of the C++-bound mesh types."""
    def deco(obj):
        setattr(cls, name, obj)
        return obj
    return deco


@_extend(Vertex, "p")
@property
def _(self) -> Point:
    return Point(self._x, self._y)


@_extend(Vertex, "orbit")
def _(self) -> Iterator["HalfEdge"]:
    edge = self.out
    while True:
        yield edge
        edge = edge.twin.next
        if edge == self.out:
            break


@_extend(HalfEdge, "is_boundary")
@property
def _(self) -> bool:
    return self.face.is_boundary


@_extend(HalfEdge, "connect")
@staticmethod
def _(e1: "HalfEdge", e2: "HalfEdge") -> None:
    e1.next = e2
    e2.prev = e1


@_extend(HalfEdge, "walk")
def _(self) -> Iterator["HalfEdge"]:
    edge = self
    while True:
        yield edge
        edge = edge.next
        if edge == self:
            break


@_extend(Face, "edges")
@property
def _(self):
    edge = self.edge
    while True:
        yield edge
        edge = edge.next
        if edge == self.edge:
            break


@_extend(Face, "vertices")
@property
def _(self):
    for edge in self.edges:
        yield edge.origin


@_extend(Face, "centroid")
@property
def _(self) -> Point:
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


@_extend(Face, "area")
@property
def _(self) -> float:
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


@_extend(Mesh, "from_triangle_soup")
@classmethod
def _(cls,
      points: list[Point] | np.ndarray,
      triangles: list[tuple[int, int, int]] | np.ndarray) -> "Mesh":
    """
    Build a half-edge mesh from a triangle soup. The topology construction
    itself happens in C++ with the GIL released (see build_from_triangle_soup
    in _mesh.cpp); this wrapper only normalizes the inputs into arrays.
    """
    if not isinstance(points, np.ndarray):
        points = np.array([(p.x, p.y) for p in points], dtype=np.float64)
    points = points.reshape(-1, 2)
    triangles = np.asarray(triangles, dtype=np.uint32).reshape(-1, 3)

    mesh = cls()
    _mesh.build_from_triangle_soup(mesh, points, triangles)
    return mesh


@dataclass
class ZeroForm:
    mesh: Mesh
    values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.values = np.zeros(len(self.mesh.vertices), dtype=np.float64)

    def __getitem__(self, vertex: Vertex) -> float:
        if vertex not in self.mesh.vertices:
            raise KeyError("Vertex not in mesh")
        return float(self.values[vertex.i])

    def __setitem__(self, vertex: Vertex, value: float) -> None:
        if vertex not in self.mesh.vertices:
            raise KeyError("Vertex not in mesh")
        self.values[vertex.i] = value

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
        result.values = self.values + other.values
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
        result.values = self.values - other.values
        return result

    def __mul__(self, scalar: float) -> "ZeroForm":
        """Multiply this ZeroForm by a scalar.

        Args:
            scalar: The scalar value to multiply by

        Returns:
            A new ZeroForm with scaled values
        """
        result = ZeroForm(self.mesh)
        result.values = self.values * scalar
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
        result.values = self.values / scalar
        return result

    def __neg__(self) -> "ZeroForm":
        """Negate all values in this ZeroForm.

        Returns:
            A new ZeroForm with negated values
        """
        result = ZeroForm(self.mesh)
        result.values = -self.values
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

        for hedge in self.mesh.halfedges:
            # For edge from A to B: df[edge] = f(B) - f(A)
            assert hedge.twin is not None
            target_value = self.values[hedge.twin.origin.i]
            source_value = self.values[hedge.origin.i]
            one_form.values[hedge.i] = target_value - source_value

        return one_form


@dataclass
class OneForm:
    """
    A discrete 1-form defined on the (h)edges of a mesh.
    """
    mesh: Mesh
    values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.values = np.zeros(len(self.mesh.halfedges), dtype=np.float64)

    def __getitem__(self, hedge: HalfEdge) -> float:
        """Get the value of the 1-form on a half-edge."""
        if hedge not in self.mesh.halfedges:
            raise KeyError("HalfEdge not in mesh")
        return float(self.values[hedge.i])

    def __setitem__(self, hedge: HalfEdge, value: float) -> None:
        """Set the value of the 1-form on a half-edge, ensuring antisymmetry."""
        if hedge not in self.mesh.halfedges:
            raise KeyError("HalfEdge not in mesh")
        assert hedge.twin is not None
        self.values[hedge.i] = value
        self.values[hedge.twin.i] = -value

    def __add__(self, other: "OneForm") -> "OneForm":
        """Add two OneForm objects element-wise."""
        if self.mesh is not other.mesh:
            raise ValueError("Cannot add OneForms on different meshes")
        result = OneForm(self.mesh)
        result.values = self.values + other.values
        return result

    def __sub__(self, other: "OneForm") -> "OneForm":
        """Subtract another OneForm element-wise."""
        if self.mesh is not other.mesh:
            raise ValueError("Cannot subtract OneForms on different meshes")
        result = OneForm(self.mesh)
        result.values = self.values - other.values
        return result

    def __mul__(self, scalar: float) -> "OneForm":
        """Multiply this OneForm by a scalar."""
        result = OneForm(self.mesh)
        result.values = self.values * scalar
        return result

    def __rmul__(self, scalar: float) -> "OneForm":
        """Right multiplication by a scalar."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "OneForm":
        """Divide this OneForm by a scalar."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide OneForm by zero")
        result = OneForm(self.mesh)
        result.values = self.values / scalar
        return result

    def __neg__(self) -> "OneForm":
        """Negate all values in this OneForm."""
        result = OneForm(self.mesh)
        result.values = -self.values
        return result


@dataclass
class TwoForm:
    """
    A discrete 2-form defined on the faces of a mesh.
    """
    mesh: Mesh
    values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.values = np.zeros(len(self.mesh.faces), dtype=np.float64)

    def __getitem__(self, face: Face) -> float:
        """Get the value of the 2-form on a face."""
        if face not in self.mesh.faces and face not in self.mesh.boundaries:
            raise KeyError("Face not in mesh")
        # Boundary faces always return 0.0
        if face in self.mesh.boundaries:
            return 0.0
        return float(self.values[face.i])

    def __setitem__(self, face: Face, value: float) -> None:
        """Set the value of the 2-form on a face."""
        if face not in self.mesh.faces:
            raise KeyError("Face not in mesh.faces (boundary faces not supported)")
        self.values[face.i] = value

    def __add__(self, other: "TwoForm") -> "TwoForm":
        """Add two TwoForm objects element-wise."""
        if self.mesh is not other.mesh:
            raise ValueError("Cannot add TwoForms on different meshes")
        result = TwoForm(self.mesh)
        result.values = self.values + other.values
        return result

    def __sub__(self, other: "TwoForm") -> "TwoForm":
        """Subtract another TwoForm element-wise."""
        if self.mesh is not other.mesh:
            raise ValueError("Cannot subtract TwoForms on different meshes")
        result = TwoForm(self.mesh)
        result.values = self.values - other.values
        return result

    def __mul__(self, scalar: float) -> "TwoForm":
        """Multiply this TwoForm by a scalar."""
        result = TwoForm(self.mesh)
        result.values = self.values * scalar
        return result

    def __rmul__(self, scalar: float) -> "TwoForm":
        """Right multiplication by a scalar."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "TwoForm":
        """Divide this TwoForm by a scalar."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide TwoForm by zero")
        result = TwoForm(self.mesh)
        result.values = self.values / scalar
        return result

    def __neg__(self) -> "TwoForm":
        """Negate all values in this TwoForm."""
        result = TwoForm(self.mesh)
        result.values = -self.values
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
                                  seed_points: list[Point | shapely.geometry.Point] = []) -> tuple[list, list, list]:
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
                     seed_points: list[Point | shapely.geometry.Point] = []) -> Mesh:
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
            np.asarray(cgal_output['vertices'], dtype=np.float64),
            cgal_output['triangles']
        )

        return mesh


Mesher.Config.RELAXED = Mesher.Config(
    minimum_angle=5.0,
    maximum_size=0,
    variable_size_maximum_factor=1.0
)
