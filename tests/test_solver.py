import pytest
import itertools
import shapely.geometry
import numpy as np

from padne import solver, problem, mesh, kicad

from conftest import for_all_kicad_projects


# Helper function to find the voltage at the vertex closest to a terminal
def find_vertex_value(sol: solver.Solution, term: problem.Terminal) -> float:
    target_layer_idx = sol.problem.layers.index(term.layer)
    target_point_shapely = term.point # Terminal point is already shapely

    layer_sol = sol.layer_solutions[target_layer_idx]
    
    best_dist = float('inf')
    found_value = None

    for msh, values in zip(layer_sol.meshes, layer_sol.values):
        for vertex in msh.vertices:
            # vertex.p is mesh.Point, convert to shapely for distance comparison
            dist = vertex.p.to_shapely().distance(target_point_shapely)
            if dist < best_dist:
                best_dist = dist
                found_value = values[vertex]
    
    # Ensure a vertex was found reasonably close
    # This tolerance should match or be slightly larger than the one used in the solver
    assert best_dist < 1e-4, f"Could not find a close vertex for terminal {term} (dist={best_dist})"
    assert found_value is not None
    return found_value


class TestSolverMeshLayer:
    def test_mesh_layer_simple_geometry(self, kicad_test_projects):
        """Test that mesh_layer correctly meshes layers from the simple_geometry project."""
        # Get the simple_geometry project
        project = kicad_test_projects["simple_geometry"]
        
        # Load the problem from the KiCad project
        prob = kicad.load_kicad_project(project.pro_path)
        
        # Create a mesher with default settings
        mesher = mesh.Mesher()
        
        # For each layer in the problem, test mesh_layer
        for layer in prob.layers:
            # Call the function under test
            meshes = solver.mesh_layer(mesher, prob, layer)
            
            # Check that we got the expected result
            assert isinstance(meshes, list), "mesh_layer should return a list of meshes"
            
            # The simple_geometry project should have a specific number of separated copper regions
            # Specifically, it has two meshes (one for each region)
            assert len(meshes) == 2, f"Expected 2 meshes for layer {layer.name}, got {len(meshes)}"
            
            # Verify each mesh has the right properties
            for m in meshes:
                assert isinstance(m, mesh.Mesh), "Each item should be a Mesh instance"
                assert len(m.vertices) > 0, "Mesh should have vertices"
                assert len(m.faces) > 0, "Mesh should have faces"
                
                # Check mesh topology is valid
                euler = m.euler_characteristic()
                assert euler == 1, f"Euler characteristic should be 1 for a valid mesh, got {euler}"
                
                # Check that all faces have proper area
                for face in m.faces:
                    assert face.area > 0, "Each face should have positive area"
    
    def test_mesh_layer_with_seed_points(self, kicad_test_projects):
        """Test that mesh_layer correctly handles seed points from lumped elements."""
        # Get the simple_geometry project
        project = kicad_test_projects["simple_geometry"]
        
        # Load the problem from the KiCad project
        prob = kicad.load_kicad_project(project.pro_path)
        
        # Create a mesher with default settings
        mesher = mesh.Mesher()
        
        # Test that collect_seed_points extracts the right points
        for layer in prob.layers:
            # Call the function we're testing
            seed_points = solver.collect_seed_points(prob, layer)
            
            # Simple_geometry has 2 lumped elements with 4 terminals total
            assert len(seed_points) == 4, f"Expected 4 seed points for layer {layer.name}, got {len(seed_points)}"
            
            # Each point should be a mesh.Point
            for point in seed_points:
                assert isinstance(point, mesh.Point), "Seed point should be a mesh.Point instance"
            
            # Verify the meshes have vertices at or very near the seed points
            meshes = solver.mesh_layer(mesher, prob, layer)
            
            # For each seed point, verify there's a vertex very close to it in one of the meshes
            for seed_point in seed_points:
                found = False
                for m in meshes:
                    for vertex in m.vertices:
                        if vertex.p.distance(seed_point) < 1e-6:  # Very small tolerance
                            found = True
                            break
                    if found:
                        break
                
                assert found, f"Seed point {seed_point} should be represented in the mesh"


class TestSyntheticProblems:

    def test_linear_rectangle(self):
        # In this test, synthesize a Problem instance that has a singular layer
        # which has a single rectangle with wide aspect ratio in it.
        # In addition, add three lumped voltage sources of 1V, that are connected
        # along the shorter sides of the rectangle.
        
        # 1. Create the rectangle shape
        rect_width = 10.0
        rect_height = 1.0
        rectangle = shapely.geometry.box(0, 0, rect_width, rect_height)
        
        # 2. Create the Layer
        layer = problem.Layer(
            shape=shapely.geometry.MultiPolygon([rectangle]),
            name="TestLayer",
            conductance=1.0  # Conductance value doesn't strongly affect voltage source test
        )
        
        # 3. Define Terminal points
        points_left = [
            shapely.geometry.Point(0, 0.05 * rect_height),
            shapely.geometry.Point(0, 0.25 * rect_height),
            shapely.geometry.Point(0, 0.50 * rect_height),
            shapely.geometry.Point(0, 0.75 * rect_height),
            shapely.geometry.Point(0, 0.95 * rect_height),
        ]
        points_right = [
            shapely.geometry.Point(rect_width, 0.05 * rect_height),
            shapely.geometry.Point(rect_width, 0.25 * rect_height),
            shapely.geometry.Point(rect_width, 0.50 * rect_height),
            shapely.geometry.Point(rect_width, 0.75 * rect_height),
            shapely.geometry.Point(rect_width, 0.95 * rect_height),
        ]
        
        # 4. Create Terminals
        terminals_left = [problem.Terminal(layer=layer, point=p) for p in points_left]
        terminals_right = [problem.Terminal(layer=layer, point=p) for p in points_right]
        
        # 5. Create Voltage Sources (1V each, positive on right, negative on left)
        voltage_sources = [
            problem.VoltageSource(p=term_r, n=term_l, voltage=1.0)
            for term_l, term_r in zip(terminals_left, terminals_right)
        ]
        
        # 6. Create the Problem
        prob_synthetic = problem.Problem(layers=[layer], lumpeds=voltage_sources)
        
        # 7. Solve the Problem
        solution = solver.solve(prob_synthetic)
        
        # 8. Verify the Solution
        assert solution is not None
        assert isinstance(solution, solver.Solution)
        assert len(solution.layer_solutions) == 1
        
        # Check each voltage source constraint
        for vs in voltage_sources:
            voltage_p = find_vertex_value(solution, vs.p)
            voltage_n = find_vertex_value(solution, vs.n)
            
            # Verify the voltage difference matches the source voltage
            assert voltage_p - voltage_n == pytest.approx(vs.voltage), \
                f"Voltage difference for {vs} does not match expected value."
                
        # Optional: Check general voltage trend (should increase from left to right)
        # Find average voltage near left and right edges
        avg_v_left = np.mean([find_vertex_value(solution, t) for t in terminals_left])
        avg_v_right = np.mean([find_vertex_value(solution, t) for t in terminals_right])
        assert avg_v_right > avg_v_left # Voltage should generally increase

        # Verify that at each vertex, the voltage is approximately proportional
        # to its x coordinate
        layer_solution = solution.layer_solutions[0]
        mesh_obj = layer_solution.meshes[0] # Assuming only one mesh for this simple case
        values = layer_solution.values[0]
        
        # Calculate the expected slope (voltage change per unit x)
        # Use the average voltages calculated earlier for robustness
        expected_slope = (avg_v_right - avg_v_left) / rect_width
        
        for vertex in mesh_obj.vertices:
            # Expected voltage based on linear interpolation from the average left voltage
            expected_voltage = avg_v_left + vertex.p.x * expected_slope
            actual_voltage = values[vertex]
            
            # Use pytest.approx with a reasonable absolute tolerance
            # The tolerance might need adjustment based on mesh density and solver accuracy
            # TODO: The tolerance here is minimal possible with the current solver
            # and mesher. I feel like it should be _way_ more accurate...
            assert actual_voltage == pytest.approx(expected_voltage, abs=0.05), \
                f"Voltage at vertex {vertex.p} ({actual_voltage:.3f}) is not proportional to x ({expected_voltage:.3f})"


class TestLaplaceOperator:

    @staticmethod
    def assert_matrix_is_laplacian(L):
        N = L.shape[0]
        assert L.shape == (N, N), "Laplace operator should be square"
        # Check that the diagonal entries are negative
        for i in range(N):
            assert L[i, i] < 0, f"Diagonal entry {i} should be negative"

        # Check that the off-diagonal entries are non-negative
        for i, j in itertools.product(range(N), range(N)):
            if i != j:
                assert L[i, j] >= 0, f"Off-diagonal entry ({i}, {j}) should be non-negative"
            assert L[i, j] == L[j, i], f"Laplace operator should be symmetric ({i}, {j})"

        # And finally, check that the diagonal is the sum of the off-diagonal entries
        for i in range(N):
            row_sum = np.sum(L[i, :])
            assert abs(row_sum) < 1e-5, f"Row {i} does not sum to zero (sum={row_sum})"


    def test_laplace_operator_unit_square_with_center(self):
        """
        Test the laplace_operator function using a unit square with a central vertex.
        The resulting mesh has 4 triangles, and we can analytically compute the
        expected Laplace operator matrix.
        """
        # Create a simple mesh: unit square with a central vertex
        # Points at the corners of the square and one at the center
        points = [
            mesh.Point(0.0, 0.0),  # bottom left (0)
            mesh.Point(1.0, 0.0),  # bottom right (1)
            mesh.Point(1.0, 1.0),  # top right (2)
            mesh.Point(0.0, 1.0),  # top left (3)
            mesh.Point(0.5, 0.5),  # center (4)
        ]
        
        # Define the triangles (counter-clockwise order)
        triangles = [
            (0, 1, 4),  # bottom triangle
            (1, 2, 4),  # right triangle
            (2, 3, 4),  # top triangle
            (3, 0, 4),  # left triangle
        ]
        
        # Create the mesh
        test_mesh = mesh.Mesh.from_triangle_soup(points, triangles)
        
        # Call the function under test
        L = solver.laplace_operator(test_mesh)

        assert L.shape == (5, 5), "Laplace operator should be a 5x5 matrix"
        
        # Convert to dense matrix for easier testing
        L_dense = L.toarray()

        self.assert_matrix_is_laplacian(L_dense)

        # Manually calculate the expected Laplace matrix
        # For this regular structure with right isosceles triangles:
        # - Each corner vertex connects to two other vertices (center and adjacent corners)
        # - The center vertex connects to all four corners
        # - For right isosceles triangles, the cotangent of the angle is 1.0
        
        # For the center vertex (index 4):
        # It connects to vertices 0, 1, 2, 3 with cotangent weights
        # Each triangle has two 45° angles (cotangent = 1) and one 90° angle (cotangent = 0)
        # So the center vertex gets 4 connections, each with weight 0.5 (average of cotangents)
        
        # For each corner vertex (indices 0-3):
        # It connects to the center and two adjacent corners
        # The connections to adjacent corners have weight 0 (90° angle, cotangent = 0)
        # The connection to center has weight 0.5 (same as above)
        
        # Create the expected matrix (initialized to zeros)
        expected_L = np.zeros((5, 5), dtype=np.float32)

        # Fill the diagonal entries (negative sum of off-diagonal entries in the same row)
        # Center vertex (index 4) connects to all corners with weight 1.0
        # since
        # 1/2 * (cot 45 + cot 45) = 1.0
        expected_L[4, 0] = 1
        expected_L[4, 1] = 1
        expected_L[4, 2] = 1
        expected_L[4, 3] = 1
        expected_L[4, 4] = -4.0  # -sum(0.5 * 4)
        
        # Corner vertices
        # Each corner vertex connects to the center with weight 1.0 (as above)
        # and to two adjacent corners with weight 0.0 (cot 90° = 0)
        for i in range(4):
            # Connection to center
            expected_L[i, 4] = 1.0
            expected_L[i, i] = -1.0
        
        # Verify the Laplace operator matches our expectations
        np.testing.assert_allclose(L_dense, expected_L, rtol=1e-5, atol=1e-5,
                                   err_msg="Laplace operator matrix does not match expected values")


class TestVertexIndexer:
    def test_index_store_create(self):
        """
        Test that VertexIndexer.create correctly maps vertices from multiple meshes to global indices.
        """
        # Create two simple meshes
        # First mesh: triangle
        points_1 = [
            mesh.Point(0.0, 0.0),
            mesh.Point(1.0, 0.0),
            mesh.Point(0.0, 1.0)
        ]
        triangles_1 = [(0, 1, 2)]
        mesh_1 = mesh.Mesh.from_triangle_soup(points_1, triangles_1)
        
        # Second mesh: square (made of two triangles)
        points_2 = [
            mesh.Point(2.0, 0.0),
            mesh.Point(3.0, 0.0),
            mesh.Point(3.0, 1.0),
            mesh.Point(2.0, 1.0)
        ]
        triangles_2 = [(0, 1, 2), (0, 2, 3)]
        mesh_2 = mesh.Mesh.from_triangle_soup(points_2, triangles_2)
        
        # Create the VertexIndexer
        index_store = solver.VertexIndexer.create([mesh_1, mesh_2])
        
        # Verify basic properties
        # Total number of vertices across both meshes
        expected_total_vertices = len(mesh_1.vertices) + len(mesh_2.vertices)
        assert len(index_store.global_index_to_vertex_index) == expected_total_vertices
        
        # Verify mapping from mesh/vertex to global index
        # First mesh vertices should have global indices 0, 1, 2
        for vertex_idx in range(len(mesh_1.vertices)):
            global_idx = index_store.mesh_vertex_index_to_global_index[(0, vertex_idx)]
            assert 0 <= global_idx < len(mesh_1.vertices)
            
        # Second mesh vertices should have global indices 3, 4, 5, 6
        for vertex_idx in range(len(mesh_2.vertices)):
            global_idx = index_store.mesh_vertex_index_to_global_index[(1, vertex_idx)]
            assert len(mesh_1.vertices) <= global_idx < expected_total_vertices
        
        # Verify mapping from global index back to mesh/vertex
        for global_idx in range(expected_total_vertices):
            mesh_idx, vertex_idx = index_store.global_index_to_vertex_index[global_idx]
            
            # Check the mapping is consistent
            assert index_store.mesh_vertex_index_to_global_index[(mesh_idx, vertex_idx)] == global_idx
            
            # Check we're referencing the correct mesh
            # Note: this is technically not part of the VertexIndexer API, but
            # it's a good sanity check. Can be removed if needed.
            if global_idx < len(mesh_1.vertices):
                assert mesh_idx == 0
            else:
                assert mesh_idx == 1
                
            # Verify vertex index is valid for the referenced mesh
            if mesh_idx == 0:
                assert 0 <= vertex_idx < len(mesh_1.vertices)
            else:
                assert 0 <= vertex_idx < len(mesh_2.vertices)


class TestSolverEndToEnd:

    @for_all_kicad_projects(exclude=["tht_component"])
    def test_all_test_projects_solve(self, project):
        """Test that solver.solve works on all test projects."""
        # Load the problem from the KiCad project
        prob = kicad.load_kicad_project(project.pro_path)

        # Call the function under test
        solution = solver.solve(prob)

        assert solution is not None
        assert isinstance(solution, solver.Solution)

        # Check that every layer has a solution
        assert len(solution.layer_solutions) == len(prob.layers)

        # Next, we iterate over all the solutions and check that the ZeroForms
        # live in the corresponding meshes
        for layer_solution in solution.layer_solutions:
            assert len(layer_solution.meshes) == len(layer_solution.values)

            for msh, value in zip(layer_solution.meshes, layer_solution.values):
                for vertex in msh.vertices:
                    # This checks both that the value is valid number and
                    # that it is finite
                    # Note that isinstance check for float is not good enough
                    # here, since the solver may decide to return np.float32 or something
                    assert np.isfinite(value[vertex])

    @for_all_kicad_projects(exclude=["tht_component", "long_trace_current"])
    def test_voltage_sources_work(self, project):
        # Load the problem from the KiCad project
        prob = kicad.load_kicad_project(project.pro_path)
        # Check if there is a voltage source in the project
        has_voltage_source = any(
            isinstance(el, problem.VoltageSource) for el in prob.lumpeds
        )
        if not has_voltage_source:
            pytest.skip("No voltage sources in this project.")

        # Call the function under test
        solution = solver.solve(prob)

        assert solution is not None
        assert isinstance(solution, solver.Solution)

        # Check that every layer has a solution
        assert len(solution.layer_solutions) == len(prob.layers)

        # Check each voltage source
        for elem in prob.lumpeds:
            if isinstance(elem, problem.VoltageSource):
                voltage_p = find_vertex_value(solution, elem.p)
                voltage_n = find_vertex_value(solution, elem.n)
                
                # Verify the voltage difference matches the source voltage
                assert voltage_p - voltage_n == pytest.approx(elem.voltage, abs=0.001), \
                    f"Voltage difference for {elem} does not match expected value."

    def test_long_trace_current_source(self, kicad_test_projects):
        project = kicad_test_projects["long_trace_current"]
        # Load the problem and solve it
        prob = kicad.load_kicad_project(project.pro_path)
        solution = solver.solve(prob)
        
        # Find the current source
        assert len(prob.lumpeds) == 1
        current_source = prob.lumpeds[0]
        
        assert current_source is not None, "No current source found in the test project"
        
        # Get voltages at the terminals
        voltage_from = find_vertex_value(solution, current_source.f)
        voltage_to = find_vertex_value(solution, current_source.t)
        
        # Check voltage difference is approximately 0.24 mV
        voltage_diff = abs(voltage_from - voltage_to)
        assert voltage_diff == pytest.approx(0.24, abs=0.01), \
            f"Voltage difference for {current_source} does not match expected value (diff={voltage_diff})"
