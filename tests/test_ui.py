import pytest
import numpy as np
import shapely.geometry

from padne import mesh, problem, solver
from padne.ui import (
    VertexSpatialIndex, FaceSpatialIndex, collect_contact_coverage, MeshViewer,
    color_scale_uses_log, color_scale_bounds, color_scale_value_at,
    color_scale_fraction_of, prepare_ui_data,
)


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


class TestContactCoverage:
    """Tests for the SMT contact coverage overlay helper."""

    def _make_problem_and_solution(self, has_source: bool):
        rect = shapely.geometry.box(0, 0, 10, 10)
        layer = problem.Layer(shape=shapely.geometry.MultiPolygon([rect]),
                              name="F.Cu", conductance=1.0, thickness=0.035)

        # A real mesh so there are vertices to cover.
        msh = mesh.Mesher().poly_to_mesh(rect)
        layer_solution = solver.LayerSolution(
            meshes=[msh], potentials=[], power_densities=[], disconnected_meshes=[])

        centre = problem.Connection(layer=layer, point=shapely.geometry.Point(5, 5))
        other = problem.Connection(layer=layer, point=shapely.geometry.Point(1, 1))
        if has_source:
            element = problem.CurrentSource(f=centre.node_id, t=other.node_id, current=1.0)
        else:
            element = problem.Resistor(a=centre.node_id, b=other.node_id, resistance=1.0)
        network = problem.Network(connections=[centre, other], elements=[element])

        region = shapely.geometry.MultiPolygon([shapely.geometry.box(4, 4, 6, 6)])
        prob = problem.Problem(layers=[layer], networks=[network],
                               refinement_regions=[("F.Cu", region)])
        return prob, [layer_solution], region

    def test_coverage_covers_region_and_is_red_for_sources(self):
        prob, layer_solutions, region = self._make_problem_and_solution(has_source=True)
        coverage = collect_contact_coverage(prob, layer_solutions)

        assert "F.Cu" in coverage
        points = coverage["F.Cu"]
        assert points, "Expected coverage points inside the pad region"
        for (x, y), color in points:
            assert region.contains(shapely.geometry.Point(x, y))
            assert color == (1.0, 0.0, 0.0)

    def test_coverage_is_gray_for_passive_networks(self):
        prob, layer_solutions, _ = self._make_problem_and_solution(has_source=False)
        coverage = collect_contact_coverage(prob, layer_solutions)
        assert coverage["F.Cu"]
        assert all(color == (0.5, 0.5, 0.5) for _, color in coverage["F.Cu"])

    def test_no_regions_means_no_coverage(self):
        prob, layer_solutions, _ = self._make_problem_and_solution(has_source=True)
        no_regions = problem.Problem(layers=prob.layers, networks=prob.networks)
        assert collect_contact_coverage(no_regions, layer_solutions) == {}


class TestCurrentDensityUnit:
    """The current-density mode's unit must match the quantity it shows."""

    def _unit_for(self, thickness):
        layer = problem.Layer(
            shape=shapely.geometry.MultiPolygon([shapely.geometry.box(0, 0, 1, 1)]),
            name="F.Cu", conductance=2082.0, thickness=thickness)
        layer_solution = solver.LayerSolution(
            meshes=[], potentials=[], power_densities=[], disconnected_meshes=[])
        solution = solver.Solution(
            problem=problem.Problem(layers=[layer], networks=[]),
            layer_solutions=[layer_solution],
            solver_info=solver.SolverInfo(ground_node_current=0.0, residual_norm=0.0),
        )
        mode = MeshViewer.CurrentDensityRenderingMode()
        mode.set_solution(solution)
        return mode.unit

    def test_unit_a_per_mm2_with_thickness(self):
        assert self._unit_for(0.035) == "A/mm²"

    def test_unit_a_per_mm_without_thickness(self):
        assert self._unit_for(None) == "A/mm"


class TestColorScaleMapping:
    """The log/linear slider mapping and the percentile cap."""

    def test_non_negative_data_uses_log(self):
        assert color_scale_uses_log(0.0, 10.0)
        assert not color_scale_uses_log(-1.0, 10.0)
        assert not color_scale_uses_log(0.0, 0.0)

    def test_log_bounds_span_the_configured_decades(self):
        lo, hi = color_scale_bounds(0.0, 10.0)
        assert hi == 10.0
        assert lo == pytest.approx(10.0 * 10.0 ** -4.0)

    def test_log_bounds_respect_a_positive_data_min(self):
        lo, hi = color_scale_bounds(1.0, 10.0)
        assert lo == pytest.approx(1.0)

    def test_log_value_and_fraction_round_trip(self):
        for fraction in (0.0, 0.25, 0.5, 1.0):
            value = color_scale_value_at(fraction, 0.0, 10.0)
            assert color_scale_fraction_of(value, 0.0, 10.0) == pytest.approx(fraction)
        # Constant ratio per unit fraction (the point of a log scale).
        assert (color_scale_value_at(0.5, 0.0, 10.0)
                == pytest.approx(np.sqrt(color_scale_value_at(0.0, 0.0, 10.0)
                                         * color_scale_value_at(1.0, 0.0, 10.0))))

    def test_signed_data_falls_back_to_linear(self):
        assert color_scale_value_at(0.0, -1.0, 1.0) == pytest.approx(-1.0)
        assert color_scale_value_at(0.5, -1.0, 1.0) == pytest.approx(0.0)
        assert color_scale_value_at(1.0, -1.0, 1.0) == pytest.approx(1.0)
        assert color_scale_fraction_of(0.0, -1.0, 1.0) == pytest.approx(0.5)

    def test_degenerate_range_does_not_crash(self):
        assert color_scale_value_at(0.5, 5.0, 5.0) == pytest.approx(5.0)
        assert color_scale_fraction_of(5.0, 5.0, 5.0) == 0.0


class TestPercentileCap:
    """The 'cap to current layer percentile' rendering-mode helper."""

    class _Index:
        def __init__(self, values):
            self.values = values

    def test_caps_max_to_the_layer_percentile(self):
        mode = MeshViewer.PowerDensityRenderingMode()
        mode.min_value, mode.max_value = 0.0, 1000.0
        mode.data_min_value, mode.data_max_value = 0.0, 1000.0
        mode.spatial_indices = {"F.Cu": self._Index(list(range(1, 1001)))}

        assert mode.cap_max_to_percentile("F.Cu", 99.9) is True
        expected = float(np.percentile(range(1, 1001), 99.9))
        assert mode.max_value == pytest.approx(expected)
        assert mode.data_max_value == pytest.approx(expected)

    def test_no_values_means_no_cap(self):
        mode = MeshViewer.PowerDensityRenderingMode()
        mode.spatial_indices = {}
        assert mode.cap_max_to_percentile("F.Cu", 99.9) is False


class TestRenderingModeDataRange:
    """Each mode tracks its autoscaled data range separately from the edit range."""

    def test_autoscale_sets_the_data_range(self):
        mode = MeshViewer.PowerDensityRenderingMode()
        mode.spatial_indices = {"F.Cu": TestPercentileCap._Index([0.0, 2.0, 4.0])}
        solution = solver.Solution(
            problem=problem.Problem(layers=[], networks=[]),
            layer_solutions=[],
            solver_info=solver.SolverInfo(ground_node_current=0.0, residual_norm=0.0),
        )
        mode.autoscale_values(solution)
        assert (mode.data_min_value, mode.data_max_value) == (0.0, 4.0)
        assert (mode.min_value, mode.max_value) == (0.0, 4.0)
        # Editing the current range must not move the data range.
        mode.min_value, mode.max_value = 1.0, 3.0
        assert (mode.data_min_value, mode.data_max_value) == (0.0, 4.0)


class TestPrepareUiData:
    """The GL-free UI preparation builds modes/indices without Qt or OpenGL."""

    def test_builds_modes_and_spatial_indices(self):
        # A hand-built mesh keeps this independent of the CGAL mesher.
        points = [mesh.Point(0, 0), mesh.Point(1, 0), mesh.Point(0.5, 1)]
        msh = mesh.Mesh.from_triangle_soup(points, [(0, 1, 2)])

        zero_form = mesh.ZeroForm(msh)
        for i, vertex in enumerate(msh.vertices):
            zero_form[vertex] = float(i + 1)
        two_form = mesh.TwoForm(msh)
        for face in msh.faces:
            two_form[face] = 1.0

        layer_solution = solver.LayerSolution(
            meshes=[msh], potentials=[zero_form], power_densities=[two_form],
            disconnected_meshes=[])
        layer = problem.Layer(
            shape=shapely.geometry.MultiPolygon([
                shapely.geometry.Polygon([(0, 0), (1, 0), (0.5, 1)])]),
            name="F.Cu", conductance=1.0)
        solution = solver.Solution(
            problem=problem.Problem(layers=[layer], networks=[]),
            layer_solutions=[layer_solution],
            solver_info=solver.SolverInfo(ground_node_current=0.0, residual_norm=0.0))

        prepared = prepare_ui_data(solution)

        assert prepared.modes
        for mode in prepared.modes:
            assert mode.solution is solution
            assert "F.Cu" in mode.spatial_indices
            assert mode.max_value >= mode.min_value

    def test_prepare_populates_contact_coverage(self):
        # Guards against the coverage overlay being computed but never wired in.
        prob, layer_solutions, region = (
            TestContactCoverage()._make_problem_and_solution(has_source=True))
        solution = solver.Solution(
            problem=prob, layer_solutions=layer_solutions,
            solver_info=solver.SolverInfo(ground_node_current=0.0, residual_norm=0.0))

        prepared = prepare_ui_data(solution)

        assert prepared.contact_coverage.get("F.Cu"), "coverage not populated"
