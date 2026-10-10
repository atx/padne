import pytest
import numpy as np
import shapely.geometry
import threading
from dataclasses import fields

from padne import mesh, problem, solver, parallel, ui
from padne.ui import (
    VertexSpatialIndex, FaceSpatialIndex, RenderedMesh, MeshViewer,
    LinearScale, LogScale, prepare_ui_data,
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
            solver_info=solver.SolverInfo(ground_node_current=0.0, residual_norm=0.0,
                                          relative_residual=0.0),
        )
        mode = MeshViewer.CurrentDensityRenderingMode()
        mode.set_solution(solution)
        return mode.unit

    def test_unit_a_per_mm2_with_thickness(self):
        assert self._unit_for(0.035) == "A/mm²"

    def test_unit_a_per_mm_without_thickness(self):
        assert self._unit_for(None) == "A/mm"


class TestSliderScale:

    def test_linear_round_trip(self):
        scale = LinearScale()
        assert scale.value_at(0.0, -1.0, 1.0) == pytest.approx(-1.0)
        assert scale.value_at(0.5, -1.0, 1.0) == pytest.approx(0.0)
        assert scale.value_at(1.0, -1.0, 1.0) == pytest.approx(1.0)
        assert scale.fraction_of(0.0, -1.0, 1.0) == pytest.approx(0.5)

    def test_linear_degenerate_range(self):
        assert LinearScale().fraction_of(5.0, 5.0, 5.0) == 0.0

    def test_log_spans_the_decades_below_hi(self):
        scale = LogScale(decades=4.0)
        assert scale.value_at(1.0, 0.0, 10.0) == pytest.approx(10.0)
        # Just above fraction 0 the value is ~decades below hi
        assert scale.value_at(1e-9, 0.0, 10.0) == pytest.approx(10.0 * 1e-4)
        # Constant ratio per unit fraction
        assert scale.value_at(0.5, 0.0, 10.0) == pytest.approx(10.0 * 1e-2)

    def test_log_fraction_zero_is_lo(self):
        scale = LogScale(decades=4.0)
        assert scale.value_at(0.0, 0.0, 10.0) == 0.0
        assert scale.fraction_of(0.0, 0.0, 10.0) == 0.0
        assert scale.fraction_of(1e-6, 0.0, 10.0) == 0.0

    def test_log_round_trip(self):
        scale = LogScale(decades=4.0)
        for fraction in (0.0, 0.25, 0.5, 1.0):
            value = scale.value_at(fraction, 0.0, 10.0)
            assert scale.fraction_of(value, 0.0, 10.0) == pytest.approx(fraction)


class TestRenderingModeColorScale:
    """Clamping and percentile cap rules of BaseRenderingMode."""

    def _mode(self, capped):
        mode = MeshViewer.PowerDensityRenderingMode(capped=capped)
        mode.data_range = (0.0, 100.0)
        mode.percentile_max = 10.0
        mode.autoscale()
        return mode

    def test_autoscale_respects_the_cap(self):
        capped, uncapped = self._mode(capped=True), self._mode(capped=False)
        assert (capped.min_value, capped.max_value) == (0.0, 10.0)
        assert (uncapped.min_value, uncapped.max_value) == (0.0, 100.0)

    def test_edits_are_clamped_to_the_slider_range(self):
        mode = self._mode(capped=True)
        mode.set_max(50.0)
        assert mode.max_value == 10.0
        mode.set_min(-5.0)
        assert mode.min_value == 0.0

    def test_min_pushes_max_and_max_pushes_min(self):
        mode = self._mode(capped=False)
        mode.set_max(20.0)
        mode.set_min(30.0)
        assert (mode.min_value, mode.max_value) == (30.0, 30.0)
        mode.set_max(5.0)
        assert (mode.min_value, mode.max_value) == (5.0, 5.0)

    def test_capping_clamps_without_rescaling(self):
        mode = self._mode(capped=False)
        mode.set_min(3.0)
        mode.set_max(60.0)
        mode.set_capped(True)
        assert (mode.min_value, mode.max_value) == (3.0, 10.0)
        mode.set_min(20.0)
        assert (mode.min_value, mode.max_value) == (10.0, 10.0)

    def test_uncapping_keeps_the_values(self):
        mode = self._mode(capped=True)
        mode.set_min(2.0)
        mode.set_capped(False)
        assert (mode.min_value, mode.max_value) == (2.0, 10.0)
        assert mode.slider_range == (0.0, 100.0)


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
            solver_info=solver.SolverInfo(ground_node_current=0.0, residual_norm=0.0,
                                          relative_residual=0.0))

        prepared = prepare_ui_data(solution)

        assert prepared.modes
        for mode in prepared.modes:
            assert mode.solution is solution
            assert "F.Cu" in mode.spatial_indices
            assert mode.max_value >= mode.min_value



@pytest.mark.parametrize("mode_cls, layer_values, expected", [
    (ui.MeshViewer.VoltageRenderingMode, [], (0.0, 1.0)),
    (ui.MeshViewer.VoltageRenderingMode, [[]], (0.0, 1.0)),
    (ui.MeshViewer.VoltageRenderingMode, [[2.0, 2.0]], (2.0, 3.0)),
    (ui.MeshViewer.VoltageRenderingMode, [[5.0, 7.0], [], [-1.0]], (-1.0, 7.0)),
    (ui.MeshViewer.PowerDensityRenderingMode, [[5.0, 7.0]], (0.0, 7.0)),
])
def test_compute_min_max(mode_cls, layer_values, expected):
    shape = shapely.geometry.MultiPolygon()
    mode = mode_cls(spatial_indices={
        f"L{i}": ui.BaseSpatialIndex(None, np.array(values), shape)
        for i, values in enumerate(layer_values)
    })
    assert mode._compute_min_max() == expected


def _ring_mesh():
    # An exterior boundary, a hole, and shared interior edges. Face traversal
    # starts at the mesh's face.edge, which need not be the soup's first vertex.
    return mesh.Mesh.from_triangle_soup(
        [mesh.Point(x, y) for x, y in
         [(0, 0), (4, 0), (4, 4), (0, 4), (1, 1), (3, 1), (3, 3), (1, 3)]],
        [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
         (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)],
    )


def _reference_render(msh, values):
    """Independent half-edge traversal oracle for the former rendering path."""
    triangles, colors, edges, boundary = [], [], [], []
    for face in msh.faces:
        for edge in face.edges:
            p, q = edge.origin.p, edge.next.origin.p
            triangles.extend((p.x, p.y))
            colors.append(values[face] if isinstance(values, mesh.TwoForm)
                          else values[edge.origin])
            target = boundary if edge.twin.is_boundary else edges
            target.extend((p.x, p.y, q.x, q.y))
    return RenderedMesh.PreparedData(
        np.asarray(triangles, dtype=np.float32), np.asarray(colors, dtype=np.float32),
        np.asarray(edges, dtype=np.float32), np.full(len(edges) // 4 * 6, 0.9, dtype=np.float32),
        np.asarray(boundary, dtype=np.float32), np.full(len(boundary) // 4 * 6, 0.9, dtype=np.float32),
    )


def _assert_render_equal(actual, expected):
    for f in fields(RenderedMesh.PreparedData):
        a, b = getattr(actual, f.name), getattr(expected, f.name)
        np.testing.assert_array_equal(a, b)
        assert a.dtype == np.float32
        assert a.flags.c_contiguous


@pytest.mark.parametrize("empty", [False, True])
def test_bulk_render_preserves_face_edge_and_field_order(empty):
    msh = mesh.Mesh() if empty else _ring_mesh()
    voltage, power = mesh.ZeroForm(msh), mesh.TwoForm(msh)
    voltage.values[:] = np.arange(len(msh.vertices)) * -0.137 + 1.23456789
    power.values[:] = np.arange(len(msh.faces)) * 0.197
    for values, prepare in [(voltage, RenderedMesh.prepare_zero_form),
                            (power, RenderedMesh.prepare_two_form)]:
        _assert_render_equal(prepare(msh, values), _reference_render(msh, values))
    zero = mesh.ZeroForm(msh)
    _assert_render_equal(RenderedMesh.prepare_mesh(msh), _reference_render(msh, zero))

    expected_mask = np.asarray([
        [edge.twin.is_boundary for edge in face.edges] for face in msh.faces
    ], dtype=np.uint8).reshape(-1, 3)
    np.testing.assert_array_equal(msh.triangle_boundary_mask(), expected_mask)
    if not empty:
        assert expected_mask.sum() == 8  # Four exterior edges and four hole edges.


def _ui_solution():
    layers, solutions = [], []
    ring = shapely.geometry.Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    for i, name in enumerate(["F.Cu", "B.Cu", "empty"]):
        msh = _ring_mesh()
        voltage, power = mesh.ZeroForm(msh), mesh.TwoForm(msh)
        voltage.values[:] = np.arange(8) * 0.1 + i * 10
        power.values[:] = np.arange(8) * 0.3 + i * 20
        other, empty = _ring_mesh(), mesh.Mesh()
        other_voltage, other_power = mesh.ZeroForm(other), mesh.TwoForm(other)
        other_voltage.values[:] = voltage.values + 0.5
        other_power.values[:] = power.values + 0.7
        layers.append(problem.Layer(shapely.geometry.MultiPolygon([ring]), name, 1.0))
        # Two meshes check concatenation ordering; the last layer has no fields.
        solutions.append(solver.LayerSolution(
            meshes=[msh, other, empty] if i < 2 else [],
            potentials=[voltage, other_voltage, mesh.ZeroForm(empty)] if i < 2 else [],
            power_densities=[power, other_power, mesh.TwoForm(empty)] if i < 2 else [],
            disconnected_meshes=[_ring_mesh()],
        ))
    return solver.Solution(problem.Problem(layers, []), solutions,
                           solver.SolverInfo(0.0, 0.0, 0.0))


def test_set_solution_precomputes_global_ranges():
    prepared = prepare_ui_data(_ui_solution())
    for mode in prepared.modes:
        values = np.concatenate([index.values for index in mode.spatial_indices.values()])
        assert mode.data_range[1] == pytest.approx(values.max())
        assert mode.percentile_max == pytest.approx(np.percentile(values, mode.cap_percentile))
        assert (mode.min_value, mode.max_value) == mode.slider_range
    voltage, power, current = prepared.modes
    assert not voltage.capped
    assert power.capped and current.capped


def test_bulk_spatial_indices_preserve_order_and_snapshot_values():
    sol = _ui_solution()
    layer, ls = sol.problem.layers[0], sol.layer_solutions[0]
    for index_type, form_values, coords in [
        (VertexSpatialIndex, ls.potentials,
         [[v.p.x, v.p.y] for msh in ls.meshes for v in msh.vertices]),
        (FaceSpatialIndex, ls.power_densities,
         [[f.centroid.x, f.centroid.y] for msh in ls.meshes for f in msh.faces]),
    ]:
        index = index_type.from_layer_data(layer, ls)
        np.testing.assert_allclose(index.tree.data, coords, rtol=0, atol=1e-15)
        expected = np.concatenate([f.values for f in form_values])
        np.testing.assert_array_equal(index.values, expected)
        form_values[0].values[:] = -100
        np.testing.assert_array_equal(index.values, expected)
        assert index.query_nearest(2, 2) is None  # Hole is not copper.
        assert index.query_nearest(10, 10) is None
        assert isinstance(index.query_nearest(0.1, 0.1), float)


def test_parallel_ui_matches_serial_and_shares_geometry(monkeypatch):
    sol = _ui_solution()
    monkeypatch.setattr(parallel, "_default", parallel.Parallel(parallel.Config(jobs=1)))
    serial = prepare_ui_data(sol)
    # Force the small correctness fixture through the same worker path as a
    # large board. Fail immediately if CPU preparation accidentally touches GL.
    monkeypatch.setattr(ui, "_UI_PARALLEL_MIN_SIZE", 1)
    monkeypatch.setattr(ui.gl, "glGenVertexArrays",
                        lambda *_: pytest.fail("GL called during CPU preparation"))
    calls = []
    original_map = parallel.thread_map

    def observed_map(fn, items):
        worker_ids = set()
        lock = threading.Lock()

        def observed(item):
            with lock:
                worker_ids.add(threading.get_ident())
            return fn(item)

        result = original_map(observed, items)
        calls.append(worker_ids)
        return result

    monkeypatch.setattr(parallel, "thread_map", observed_map)
    monkeypatch.setattr(parallel, "_default", parallel.Parallel(parallel.Config(jobs=4)))
    threaded = prepare_ui_data(sol)
    assert len(calls) == 4  # Geometry plus vertex, power and current density indexes.
    assert all(ids and threading.get_ident() not in ids for ids in calls)
    for a, b in zip(serial.modes, threaded.modes):
        assert (a.min_value, a.max_value) == (b.min_value, b.max_value)
        assert b.rendered_meshes == b.disconnected_rendered_meshes == {}
        for layer in sol.problem.layers:
            name = layer.name
            np.testing.assert_array_equal(a.spatial_indices[name].values,
                                          b.spatial_indices[name].values)
            if a.spatial_indices[name].tree is not None:
                np.testing.assert_array_equal(a.spatial_indices[name].tree.data,
                                              b.spatial_indices[name].tree.data)
            for attr in ["_prepared_rendered_meshes", "_prepared_disconnected_rendered_meshes"]:
                for expected, actual in zip(getattr(a, attr)[name], getattr(b, attr)[name]):
                    _assert_render_equal(actual, expected)
    voltage, power, current = threaded.modes
    for other in [power, current]:
        assert voltage._prepared_disconnected_rendered_meshes is other._prepared_disconnected_rendered_meshes
        for a, b in zip(voltage._prepared_rendered_meshes["F.Cu"],
                        other._prepared_rendered_meshes["F.Cu"]):
            for name in ["triangle_vertices", "edge_vertices", "edge_colors",
                         "boundary_vertices", "boundary_colors"]:
                assert getattr(a, name) is getattr(b, name)
                assert not getattr(a, name).flags.writeable
    for mode in threaded.modes:
        assert mode.spatial_indices["empty"].tree is None
