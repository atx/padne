import pytest
import shapely.geometry

from padne import mesh, problem, solver
from padne.ui import VertexSpatialIndex, FaceSpatialIndex


class TestSpatialIndex:
    """Tests for VertexSpatialIndex and FaceSpatialIndex."""

    def _make_simple_triangle_layer_solution(self):
        """Create a simple triangle mesh with known values for testing."""
        points = [mesh.Point(0, 0), mesh.Point(1, 0), mesh.Point(0.5, 1)]
        triangles = [(0, 1, 2)]
        msh = mesh.Mesh.from_triangle_soup(points, triangles)

        vertices = list(msh.vertices)

        # ZeroForm with values 1.0, 2.0, 3.0 at the three vertices
        zero_form = mesh.ZeroForm(msh)
        for i, v in enumerate(vertices):
            zero_form[v] = float(i + 1)

        # TwoForm with value 42.0 at the single face
        two_form = mesh.TwoForm(msh)
        face = list(msh.faces)[0]
        two_form[face] = 42.0

        layer_solution = solver.LayerSolution(
            meshes=[msh],
            potentials=[zero_form],
            power_densities=[two_form],
            disconnected_meshes=[]
        )

        shape = shapely.geometry.MultiPolygon([
            shapely.geometry.Polygon([(0, 0), (1, 0), (0.5, 1)])
        ])
        layer = problem.Layer(shape=shape, name="test", conductance=1.0)

        return layer, layer_solution, vertices

    def test_vertex_spatial_index_basic(self):
        """Query point near a vertex returns that vertex's value."""
        layer, layer_solution, vertices = self._make_simple_triangle_layer_solution()

        index = VertexSpatialIndex.from_layer_data(layer, layer_solution)

        # Query near vertex 0 at (0, 0) - should return value close to 1.0
        value = index.query_nearest(0.05, 0.05)
        assert value is not None
        assert value == pytest.approx(1.0)

    def test_face_spatial_index_basic(self):
        """Query point near face centroid returns that face's value."""
        layer, layer_solution, _ = self._make_simple_triangle_layer_solution()

        index = FaceSpatialIndex.from_layer_data(layer, layer_solution)

        # The centroid of triangle (0,0), (1,0), (0.5,1) is at (0.5, 1/3)
        value = index.query_nearest(0.5, 0.33)
        assert value is not None
        assert value == pytest.approx(42.0)

    def test_spatial_index_outside_geometry(self):
        """Query point outside layer shape returns None."""
        layer, layer_solution, _ = self._make_simple_triangle_layer_solution()

        vertex_index = VertexSpatialIndex.from_layer_data(layer, layer_solution)
        face_index = FaceSpatialIndex.from_layer_data(layer, layer_solution)

        # Point far outside the triangle
        assert vertex_index.query_nearest(10.0, 10.0) is None
        assert face_index.query_nearest(10.0, 10.0) is None

    def test_spatial_index_empty_mesh(self):
        """Empty LayerSolution returns None for any query."""
        shape = shapely.geometry.MultiPolygon([
            shapely.geometry.box(0, 0, 1, 1)
        ])
        layer = problem.Layer(shape=shape, name="empty", conductance=1.0)

        layer_solution = solver.LayerSolution(
            meshes=[],
            potentials=[],
            power_densities=[],
            disconnected_meshes=[]
        )

        vertex_index = VertexSpatialIndex.from_layer_data(layer, layer_solution)
        face_index = FaceSpatialIndex.from_layer_data(layer, layer_solution)

        assert vertex_index.query_nearest(0.5, 0.5) is None
        assert face_index.query_nearest(0.5, 0.5) is None

    def _make_dense_mesh_layer_solution(self):
        """Create a denser mesh using Mesher on a rectangle."""
        rect = shapely.geometry.box(0, 0, 10, 10)

        mesher = mesh.Mesher()
        msh = mesher.poly_to_mesh(rect)

        # ZeroForm: f(x, y) = x + y
        zero_form = mesh.ZeroForm(msh)
        for v in msh.vertices:
            zero_form[v] = v.p.x + v.p.y

        # TwoForm: f(x, y) = x * y at centroid
        two_form = mesh.TwoForm(msh)
        for face in msh.faces:
            c = face.centroid
            two_form[face] = c.x * c.y

        layer_solution = solver.LayerSolution(
            meshes=[msh],
            potentials=[zero_form],
            power_densities=[two_form],
            disconnected_meshes=[]
        )

        shape = shapely.geometry.MultiPolygon([rect])
        layer = problem.Layer(shape=shape, name="dense", conductance=1.0)

        return layer, layer_solution, msh

    def test_vertex_spatial_index_dense_mesh(self):
        """Dense mesh with coordinate-based values returns correct nearest values."""
        layer, layer_solution, msh = self._make_dense_mesh_layer_solution()

        index = VertexSpatialIndex.from_layer_data(layer, layer_solution)

        # Query at (5, 5) - expected value is approximately 10.0 (x + y)
        value = index.query_nearest(5.0, 5.0)
        assert value is not None
        # With a dense mesh, nearest vertex should be very close to query point
        assert value == pytest.approx(10.0, abs=1.0)

        # Query at corner (0, 0) - expected value is approximately 0.0
        value_corner = index.query_nearest(0.1, 0.1)
        assert value_corner is not None
        assert value_corner == pytest.approx(0.0, abs=0.5)

        # Query at (10, 10) - expected value is approximately 20.0
        value_far = index.query_nearest(9.9, 9.9)
        assert value_far is not None
        assert value_far == pytest.approx(20.0, abs=1.0)

    def test_face_spatial_index_dense_mesh(self):
        """Dense mesh with coordinate-based face values returns correct nearest values."""
        layer, layer_solution, msh = self._make_dense_mesh_layer_solution()

        index = FaceSpatialIndex.from_layer_data(layer, layer_solution)

        # Query at (5, 5) - expected value is approximately 25.0 (x * y)
        value = index.query_nearest(5.0, 5.0)
        assert value is not None
        assert value == pytest.approx(25.0, abs=5.0)

        # Query near corner (1, 1) - expected value is approximately 1.0
        value_corner = index.query_nearest(1.0, 1.0)
        assert value_corner is not None
        assert value_corner == pytest.approx(1.0, abs=2.0)
