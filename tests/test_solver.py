import pytest
import itertools
import shapely.geometry
import math
import numpy as np
import scipy.sparse

from padne import solver, problem, mesh, kicad

from conftest import for_all_kicad_projects


class TestNetworkSolver:

    def create_test_system(self, network: problem.Network) -> tuple[solver.NodeIndexer, scipy.sparse.lil_matrix, np.ndarray]:
        # For reasons unknown, network.nodes is not a list of NodeIDs
        # but instead a dict mapping NodeIDs to numbers from 0 to N-1.
        # So we just use that here
        node_to_global_index = network.nodes
        extra_source_to_global_index = {}
        for elem in network.elements:
            if elem.extra_variable_count == 0:
                continue
            if elem.extra_variable_count > 1:
                raise ValueError(f"Element {elem} has more than one extra variable, which is not supported")
            extra_source_to_global_index[elem] = len(extra_source_to_global_index) \
                    + len(node_to_global_index)

        node_indexer = solver.NodeIndexer(
            node_to_global_index=node_to_global_index,
            extra_source_to_global_index=extra_source_to_global_index
        )

        N = len(node_indexer.node_to_global_index) + len(node_indexer.extra_source_to_global_index)
        L = scipy.sparse.lil_matrix((N, N), dtype=solver.DTYPE)
        r = np.zeros(N, dtype=solver.DTYPE)
        return node_indexer, L, r

    def solve_network(self, network):
        """Solves the given Network and returns a Solution."""
        node_indexer, L, r = self.create_test_system(network)
        solver.stamp_network_into_system(network, node_indexer, L, r)
        # Drop the first row and column, which correspond to the ground node
        # TODO: It is unclear how is it possible that this works fine
        # in the main solver code, but crashes here.
        # However, the main solver code should also force a ground node
        # to improve numerical stability, so this is a mystery likely not worth
        # solving...
        print(L.todense())
        L = L.todense()[1:,1:]
        v = np.linalg.solve(L, r[1:])  # Solve the system
        v = np.concatenate(([0.0], v))  # Add ground node voltage back
        node_to_value = {
            node_id: v[node_indexer.node_to_global_index[node_id]]
            for node_id in node_indexer.node_to_global_index.keys()
        }
        return node_indexer, v, node_to_value

    def test_current_into_resistor(self):
        n_f = problem.NodeID()
        n_t = problem.NodeID()

        csrc = problem.CurrentSource(
            f=n_f, t=n_t, current=1.1
        )
        res = problem.Resistor(
            a=n_f, b=n_t, resistance=2.2
        )

        network = problem.Network(
            connections=[],
            elements=[csrc, res],
        )

        _, _, s = self.solve_network(network)
        v_diff = s[n_t] - s[n_f]
        assert v_diff == pytest.approx(csrc.current * res.resistance, abs=1e-6)

    def test_voltage_into_resistor(self):
        n_p = problem.NodeID()
        n_n = problem.NodeID()

        vsrc = problem.VoltageSource(
            p=n_p, n=n_n, voltage=3.3
        )
        res = problem.Resistor(
            a=n_p, b=n_n, resistance=2.2
        )

        network = problem.Network(
            connections=[],
            elements=[vsrc, res],
        )

        node_indexer, v, s = self.solve_network(network)
        v_diff = s[n_p] - s[n_n]
        assert v_diff == pytest.approx(vsrc.voltage, abs=1e-6)

        i_vsrc = node_indexer.extra_source_to_global_index[vsrc]
        current_through_resistor = v[i_vsrc]
        assert current_through_resistor == pytest.approx(vsrc.voltage / res.resistance, abs=1e-6)

    def test_voltage_regulator(self):
        n_p = problem.NodeID()
        n_n = problem.NodeID()
        n_f = problem.NodeID()
        n_t = problem.NodeID()

        res_vsrc = problem.Resistor(
            a=n_p, b=n_n, resistance=2.2
        )
        res_csrc = problem.Resistor(
            a=n_f, b=n_t, resistance=1.4
        )

        # This is just so that we do not have two floating connected components
        res_coupling = problem.Resistor(
            a=n_t, b=n_n, resistance=100000
        )

        reg = problem.VoltageRegulator(
            v_p=n_p, v_n=n_n,
            s_f=n_f, s_t=n_t,
            voltage=3.3,
            gain=0.3,
        )

        network = problem.Network(
            connections=[],
            elements=[res_csrc, res_vsrc, res_coupling, reg],
        )

        node_indexer, v, s = self.solve_network(network)
        v_at_vsrc_side = s[n_p] - s[n_n]
        assert v_at_vsrc_side == pytest.approx(reg.voltage, abs=1e-6)
        vsrc_current = v[node_indexer.extra_source_to_global_index[reg]]
        assert vsrc_current == pytest.approx(reg.voltage / res_vsrc.resistance, abs=1e-6)
        expected_current_at_csrc = vsrc_current * reg.gain
        expected_voltage_at_csrc = expected_current_at_csrc * res_csrc.resistance
        v_at_csrc_side = s[n_f] - s[n_t]
        assert v_at_csrc_side == pytest.approx(expected_voltage_at_csrc, abs=1e-6)


# Helper function to find the voltage at the vertex closest to a connection point
def find_vertex_value(sol: solver.Solution, conn: problem.Connection) -> float:
    """Finds the voltage at the mesh vertex closest to the given Connection point."""
    target_layer_idx = sol.problem.layers.index(conn.layer)
    target_point_shapely = conn.point # Connection point is already shapely

    layer_sol = sol.layer_solutions[target_layer_idx]
    
    best_dist = float('inf')
    found_value = None
    closest_vertex_point = None # For debugging

    for msh, values in zip(layer_sol.meshes, layer_sol.values):
        for vertex in msh.vertices:
            # vertex.p is mesh.Point, convert to shapely for distance comparison
            dist = vertex.p.to_shapely().distance(target_point_shapely)
            if dist < best_dist:
                best_dist = dist
                found_value = values[vertex]
                closest_vertex_point = vertex.p # For debugging
    
    # Ensure a vertex was found reasonably close
    # This tolerance should match or be slightly larger than the one used in the solver's KDTree query
    assert best_dist < 1e-4, \
        f"Could not find a close vertex for connection at {target_point_shapely} on layer {conn.layer.name} " \
        f"(closest found: {closest_vertex_point} with dist={best_dist})"
    assert found_value is not None
    return found_value


class TestConnectivityGraph:

    def test_simple_geometry(self, kicad_test_projects):
        # Grab the simple_geometry project, use it to construct a ConnectivityGraph
        project = kicad_test_projects["simple_geometry"]
        prob = kicad.load_kicad_project(project.pro_path)

        cg = solver.ConnectivityGraph.create_from_problem(prob)
        assert len(cg.nodes) == 2
        connected = cg.compute_connected_nodes()
        assert len(connected) == 2

    def test_different_layer_and_net_same_xy(self, kicad_test_projects):
        # We had a bug that caused a geometry to not get garbage collected
        # when it was on a different layer but there was a terminal on a
        # different layer that had XY coordinates within that shape.
        project = kicad_test_projects["different_layer_and_net_same_xy"]
        prob = kicad.load_kicad_project(project.pro_path)

        cg = solver.ConnectivityGraph.create_from_problem(prob)
        assert len(cg.nodes) == 3
        connected = cg.compute_connected_nodes()
        assert len(connected) == 2

    def test_complicated_case(self, kicad_test_projects):
        project = kicad_test_projects["disconnected_components"]
        prob = kicad.load_kicad_project(project.pro_path)

        cg = solver.ConnectivityGraph.create_from_problem(prob)
        assert len(cg.nodes) == 11
        connected = cg.compute_connected_nodes()
        assert len(connected) == 5
        # Check that there are 3 connected components on the F.Cu layer
        # and 2 on the B.Cu layer
        # Beware: This assumes the order of the layers is F.Cu, B.Cu
        assert len([n for n in connected if n.layer_i == 0]) == 3
        assert len([n for n in connected if n.layer_i == 1]) == 2


class TestSolverMeshLayer:
    def test_generate_meshes_for_problem_simple_geometry(self, kicad_test_projects):
        """Test that generate_meshes_for_problem correctly meshes layers from the simple_geometry project."""
        # Get the simple_geometry project
        project = kicad_test_projects["simple_geometry"]
        
        # Load the problem from the KiCad project
        prob = kicad.load_kicad_project(project.pro_path)
        
        # Create a mesher with default settings
        mesher = mesh.Mesher()
        
        # Create connectivity graph and find connected layer-mesh pairs
        cg = solver.ConnectivityGraph.create_from_problem(prob)
        connected_layer_mesh_pairs = solver.find_connected_layer_geom_indices(cg)
        
        # Call the function under test with the required argument
        meshes, mesh_index_to_layer_index = solver.generate_meshes_for_problem(
            prob, mesher, connected_layer_mesh_pairs)
        
        # Check that we got the expected result
        assert isinstance(meshes, list), "generate_meshes_for_problem should return a list of meshes"
        
        # The simple_geometry project should have a specific number of separated copper regions
        # Specifically, it has two meshes (one for each region)
        assert len(meshes) == 2, f"Expected 2 meshes total, got {len(meshes)}"
        
        # Verify the mesh_index_to_layer_index mapping
        assert len(mesh_index_to_layer_index) == len(meshes), "Each mesh should have a corresponding layer index"
        
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
    
    def test_generate_meshes_with_seed_points(self, kicad_test_projects):
        """Test that generate_meshes_for_problem correctly handles seed points from lumped elements."""
        # Get the simple_geometry project
        project = kicad_test_projects["simple_geometry"]
        
        # Load the problem from the KiCad project
        prob = kicad.load_kicad_project(project.pro_path)
        
        # Create a mesher with default settings
        mesher = mesh.Mesher()
        
        # Create connectivity graph and find connected layer-mesh pairs
        cg = solver.ConnectivityGraph.create_from_problem(prob)
        connected_layer_mesh_pairs = solver.find_connected_layer_geom_indices(cg)
        
        # Test that collect_seed_points extracts the right points
        for layer in prob.layers:
            seed_points = solver.collect_seed_points(prob, layer)
            
            # Simple_geometry has 2 lumped elements with 4 terminals total
            assert len(seed_points) == 4, f"Expected 4 seed points for layer {layer.name}, got {len(seed_points)}"
            
            # Each point should be a mesh.Point
            for point in seed_points:
                assert isinstance(point, mesh.Point), "Seed point should be a mesh.Point instance"
        
        # Call generate_meshes_for_problem with the required argument
        meshes, mesh_index_to_layer_index = solver.generate_meshes_for_problem(
            prob, mesher, connected_layer_mesh_pairs)
        
        # For each connection point in the problem, verify there's a mesh vertex very close to its location
        for network in prob.networks:
            for connection in network.connections:
                layer_index = prob.layers.index(connection.layer)
                relevant_meshes = [meshes[i] for i, l_idx in enumerate(mesh_index_to_layer_index) if l_idx == layer_index]

                # Convert connection point (already shapely) to mesh.Point for distance comparison
                conn_point_mesh = mesh.Point(connection.point.x, connection.point.y)
                
                # Check if any mesh on the correct layer has a vertex close to this connection point
                found = False
                for m in relevant_meshes:
                    for vertex in m.vertices:
                        if vertex.p.distance(conn_point_mesh) < 1e-6:  # Very small tolerance
                            found = True
                            break
                    if found:
                        break
                
                assert found, (
                    f"Connection point {conn_point_mesh} on layer {connection.layer.name} "
                    f"should be represented by a vertex in the mesh"
                )


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

        # 3. Define connection points
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

        # 4. Create Networks, each containing one Voltage Source and its Connections
        networks = []
        connections_left = [] # Keep track for later verification
        connections_right = [] # Keep track for later verification
        for p_left, p_right in zip(points_left, points_right):
            # Create connections for this source
            conn_left = problem.Connection(layer=layer, point=p_left)
            conn_right = problem.Connection(layer=layer, point=p_right)
            connections_left.append(conn_left)
            connections_right.append(conn_right)

            # Create the voltage source element using the connection NodeIDs
            vsource_element = problem.VoltageSource(
                p=conn_right.node_id, # Positive on the right
                n=conn_left.node_id,  # Negative on the left
                voltage=1.0
            )

            # Create the network for this source
            network = problem.Network(
                connections=[conn_left, conn_right],
                elements=[vsource_element]
            )
            networks.append(network)

        # 5. Create the Problem
        prob_synthetic = problem.Problem(layers=[layer], networks=networks)

        # 6. Solve the Problem
        solution = solver.solve(prob_synthetic)

        # 7. Verify the Solution
        assert solution is not None
        assert isinstance(solution, solver.Solution)
        assert len(solution.layer_solutions) == 1

        # Check each voltage source constraint by iterating through networks
        for network in networks:
            # Assuming each network has exactly one voltage source as constructed
            vsource = network.elements[0]
            assert isinstance(vsource, problem.VoltageSource)

            # Find the connections associated with this source within its network
            conn_p = next(c for c in network.connections if c.node_id == vsource.p)
            conn_n = next(c for c in network.connections if c.node_id == vsource.n)

            voltage_p = find_vertex_value(solution, conn_p)
            voltage_n = find_vertex_value(solution, conn_n)

            # Verify the voltage difference matches the source voltage
            assert voltage_p - voltage_n == pytest.approx(vsource.voltage), \
                f"Voltage difference for {vsource} does not match expected value."

        # Optional: Check general voltage trend (should increase from left to right)
        # Find average voltage near left and right edges using the stored connections
        avg_v_left = np.mean([find_vertex_value(solution, c) for c in connections_left])
        avg_v_right = np.mean([find_vertex_value(solution, c) for c in connections_right])
        assert avg_v_right > avg_v_left # Voltage should generally increase

        # Verify that at each vertex, the voltage is approximately proportional
        # to its x coordinate
        layer_solution = solution.layer_solutions[0]
        # Handle possibility of multiple meshes if mesher splits the rectangle
        all_vertices = []
        all_values = {}
        for msh, values_form in zip(layer_solution.meshes, layer_solution.values):
            all_vertices.extend(msh.vertices)
            for v in msh.vertices:
                all_values[v] = values_form[v]

        # Calculate the expected slope (voltage change per unit x)
        # Use the average voltages calculated earlier for robustness
        expected_slope = (avg_v_right - avg_v_left) / rect_width

        for vertex in all_vertices:
            # Expected voltage based on linear interpolation from the average left voltage
            expected_voltage = avg_v_left + vertex.p.x * expected_slope
            actual_voltage = all_values[vertex]

            # Use pytest.approx with a reasonable absolute tolerance
            # The tolerance might need adjustment based on mesh density and solver accuracy
            # TODO: The tolerance here is minimal possible with the current solver
            # and mesher. I feel like it should be _way_ more accurate...
            assert actual_voltage == pytest.approx(expected_voltage, abs=0.05), \
                f"Voltage at vertex {vertex.p} ({actual_voltage:.3f}) is not proportional to x ({expected_voltage:.3f})"

    def test_coaxial_structure(self):
        """
        Test the solver against a coaxial (annular) structure with an analytical solution.
        Inner boundary fixed at 1V relative to outer boundary at 0V.
        """
        # Parameters for the coaxial structure
        inner_radius = 1.0
        outer_radius = 9.0
        segments_per_quadrant = 16  # Gives 32 segments for the full circle
        
        # Create the annular shape (ring) using shapely
        inner_circle = shapely.geometry.Point(0, 0).buffer(inner_radius, quad_segs=segments_per_quadrant)
        outer_circle = shapely.geometry.Point(0, 0).buffer(outer_radius, quad_segs=segments_per_quadrant)
        annular_ring = outer_circle.difference(inner_circle)
        
        # Ensure we have a MultiPolygon as expected by the Layer constructor
        if annular_ring.geom_type == "Polygon":
            annular_ring = shapely.geometry.MultiPolygon([annular_ring])
        
        # Create layer for the annulus
        layer = problem.Layer(
            shape=annular_ring,
            name="AnnulusLayer",
            conductance=1.0
        )
        
        # Extract boundary points directly from the polygon
        polygon = annular_ring.geoms[0]  # Get the first (and only) polygon
        outer_boundary_pts = list(polygon.exterior.coords)[:-1]  # Remove duplicate last point
        inner_boundary_pts = list(polygon.interiors[0].coords)[:-1]  # First interior ring
        
        # Convert points to (angle, point) pairs for sorting
        def to_angle_point_pair(point):
            x, y = point
            angle = math.atan2(y, x)
            # Ensure angles range from 0 to 2π instead of -π to π
            if angle < 0:
                angle += 2 * math.pi
            return (angle, point)
        
        # Sort inner and outer boundary points by angle
        inner_pts_with_angles = sorted([to_angle_point_pair(pt) for pt in inner_boundary_pts])
        outer_pts_with_angles = sorted([to_angle_point_pair(pt) for pt in outer_boundary_pts])
        
        # Create Connections and store them
        networks = []
        outer_connections = []
        inner_connections = []
        
        # Pair inner and outer points by their sorted order and create Connections
        for (_, inner_pt), (_, outer_pt) in zip(inner_pts_with_angles, outer_pts_with_angles):
            inner_conn = problem.Connection(layer=layer,
                                          point=shapely.geometry.Point(inner_pt))
            outer_conn = problem.Connection(layer=layer,
                                          point=shapely.geometry.Point(outer_pt))
            
            outer_connections.append(outer_conn)
            inner_connections.append(inner_conn)

        # Verify that the outer connections are outer_distance away from the origin
        for conn in outer_connections:
            x, y = conn.point.x, conn.point.y
            distance = math.sqrt(x**2 + y**2)
            assert distance == pytest.approx(outer_radius, abs=0.001), \
                f"Outer connection {conn} is not at the expected outer radius (distance={distance})"
        # Verify that the inner connections are inner_distance away from the origin
        for conn in inner_connections:
            x, y = conn.point.x, conn.point.y
            distance = math.sqrt(x**2 + y**2)
            assert distance == pytest.approx(inner_radius, abs=0.001), \
                f"Inner connection {conn} is not at the expected inner radius (distance={distance})"

        # Next, we go in circle around the boundary, forcing the voltage to be equal
        # at the outer connections by creating 0V sources between adjacent connections.
        for c_a, c_b in zip(outer_connections, outer_connections[1:] + [outer_connections[0]]):
            vsource_element = problem.VoltageSource(p=c_a.node_id, n=c_b.node_id, voltage=0.0)
            network = problem.Network(connections=[c_a, c_b], elements=[vsource_element])
            networks.append(network)

        # And do the same thing for the inner connections
        for c_a, c_b in zip(inner_connections, inner_connections[1:] + [inner_connections[0]]):
            vsource_element = problem.VoltageSource(p=c_a.node_id, n=c_b.node_id, voltage=0.0)
            network = problem.Network(connections=[c_a, c_b], elements=[vsource_element])
            networks.append(network)

        # And finally, connect the first inner connection to the first outer connection with 1V
        c_inner_first = inner_connections[0]
        c_outer_first = outer_connections[0]
        vsource_element = problem.VoltageSource(p=c_inner_first.node_id, n=c_outer_first.node_id, voltage=1.0)
        network = problem.Network(connections=[c_inner_first, c_outer_first], elements=[vsource_element])
        networks.append(network)

        # Create the Problem and solve
        prob_coaxial = problem.Problem(layers=[layer], networks=networks)
        solution = solver.solve(prob_coaxial)
        
        # Verify the solution
        assert solution is not None
        assert len(solution.layer_solutions) == 1
        
        # Analytical solution function for potential in a coaxial structure
        def analytical_solution(x, y):
            r = math.sqrt(x**2 + y**2)
            # Avoid log(0) or division by zero if r is exactly inner_radius or outer_radius
            if r <= inner_radius: return 1.0
            if r >= outer_radius: return 0.0
            # Calculate ideal potential with outer boundary at 0V and inner at 1V
            ideal_potential = math.log(outer_radius / r) / math.log(outer_radius / inner_radius)
            # Adjust by the reference potential offset
            return ideal_potential

        # This check checks that we have implemented the analytical solution correctly
        assert analytical_solution(inner_radius, 0) == 1.0
        assert analytical_solution(outer_radius, 0) == 0.0
        
        # Compare numerical solution with analytical solution
        layer_solution = solution.layer_solutions[0]
        
        # Check voltages at outer connections - should all be approximately the same
        outer_potentials = [find_vertex_value(solution, conn) for conn in outer_connections]
        reference_potential = outer_potentials[0]
        for pot in outer_potentials:
            assert pot == pytest.approx(reference_potential, abs=0.001), \
                f"Outer boundary potential inconsistency: {pot} vs reference {reference_potential}"
        # Check voltages at inner boundary - should be approximately 1V higher than reference
        inner_potentials = [find_vertex_value(solution, conn) for conn in inner_connections]
        for pot in inner_potentials:
            assert pot == pytest.approx(reference_potential + 1.0, abs=0.001), \
                f"Inner boundary potential inconsistency: {pot} vs reference {reference_potential + 1.0}"
        
        # TODO: I suspect that there is systematic bias here somewhere. In reality,
        # we should be getting better than 0.03V accuracy, but I don't know why we are not.
        # It seems that shifting the outer_radius and inner_radius in the
        # analytical_solution function definition does help and allow us to match the actual result exactly
        
        for mesh_idx, (msh, values) in enumerate(zip(layer_solution.meshes, layer_solution.values)):
            for vertex in msh.vertices:
                numerical_value = values[vertex] - reference_potential
                x, y = vertex.p.x, vertex.p.y
                r = math.sqrt(x**2 + y**2)
                
                # Skip vertices very close to boundaries where numerical errors might be larger
                # or where analytical solution might be sensitive
                boundary_margin = 0.1 # Slightly increased margin
                if r > outer_radius - boundary_margin or r < inner_radius + boundary_margin:
                    continue
                    
                analytical_value = analytical_solution(x, y)
                # Check each point against analytical solution with reasonable tolerance
                assert numerical_value == pytest.approx(analytical_value, abs=0.03), \
                    f"Error too large at point ({x:.2f}, {y:.2f}), r={r:.2f}: " \
                    f"numerical={numerical_value:.4f}, analytical={analytical_value:.4f}"
        

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

    @for_all_kicad_projects(exclude=["tht_component",
                                     "unterminated_current_loop"])
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

    @for_all_kicad_projects(exclude=["tht_component",
                                     "long_trace_current",
                                     "unterminated_current_loop"])
    def test_voltage_sources_work(self, project):
        # Load the problem from the KiCad project
        prob = kicad.load_kicad_project(project.pro_path)

        # Call the function under test
        solution = solver.solve(prob)

        assert solution is not None
        assert isinstance(solution, solver.Solution)

        # Check that every layer has a solution
        assert len(solution.layer_solutions) == len(prob.layers)

        found_voltage_source = False
        # Check each voltage source within its network
        for network in prob.networks:
            # Simplify: Look for networks containing exactly one VoltageSource element
            # More complex networks could be handled, but this covers many test cases.
            if len(network.elements) == 1 and isinstance(network.elements[0], problem.VoltageSource):
                found_voltage_source = True
                vsource = network.elements[0]

                # Find the Connection objects corresponding to the p and n NodeIDs
                try:
                    p_conn = next(c for c in network.connections if c.node_id == vsource.p)
                    n_conn = next(c for c in network.connections if c.node_id == vsource.n)
                except StopIteration:
                    pytest.fail(f"Could not find connections for VoltageSource {vsource} in network {network}")

                # Find the voltage at the mesh vertices closest to the connection points
                voltage_p = find_vertex_value(solution, p_conn)
                voltage_n = find_vertex_value(solution, n_conn)

                # Verify the voltage difference matches the source voltage
                assert voltage_p - voltage_n == pytest.approx(vsource.voltage, abs=0.001), \
                    f"Voltage difference for {vsource} (p@{p_conn.point}, n@{n_conn.point}) does not match expected value."

        # If no suitable voltage source network was found in the project, skip the test
        if not found_voltage_source:
            pytest.skip(f"No networks containing only a VoltageSource found in project {project.name}.")

    def test_long_trace_current_source(self, kicad_test_projects):
        project = kicad_test_projects["long_trace_current"]
        # Load the problem and solve it
        prob = kicad.load_kicad_project(project.pro_path)
        solution = solver.solve(prob)
        
        # Find the current source network and element
        current_source_element = None
        current_source_network = None
        for network in prob.networks:
            # Assuming this project has one network with one current source
            if len(network.elements) == 1 and isinstance(network.elements[0], problem.CurrentSource):
                current_source_element = network.elements[0]
                current_source_network = network
                break

        assert current_source_element is not None, "No current source element found in the test project"
        assert current_source_network is not None, "No network containing the current source found"

        # Find the Connection objects corresponding to the f and t NodeIDs
        try:
            f_conn = next(c for c in current_source_network.connections if c.node_id == current_source_element.f)
            t_conn = next(c for c in current_source_network.connections if c.node_id == current_source_element.t)
        except StopIteration:
            pytest.fail(f"Could not find connections for CurrentSource {current_source_element} in network {current_source_network}")
        
        # Get voltages at the connection points
        voltage_from = find_vertex_value(solution, f_conn)
        voltage_to = find_vertex_value(solution, t_conn)
        
        # Check voltage difference is approximately 0.24 V
        voltage_diff = abs(voltage_from - voltage_to)
        assert voltage_diff == pytest.approx(0.24, abs=0.01), \
            f"Voltage difference for {current_source_element} does not match expected value (diff={voltage_diff})"

    def test_complicated_trace_current_source(self, kicad_test_projects):
        project = kicad_test_projects["complicated_trace_current"]

        prob = kicad.load_kicad_project(project.pro_path)
        solution = solver.solve(prob)

        # This trace is composed from multiple segments with varying widths
        # Width at each 10mm point (21 points from 0mm to 200mm inclusive)
        widths = [
            0.2, 0.2,
            6.0, 6.0, 6.0,
            0.2, 0.2, 0.2, 0.2,
            2.0, 2.0, 2.0,
            4.0, 4.0,
            0.2, 0.2,
            1.0, 2.0, 1.0, 0.2, 0.2
        ]
        assert len(widths) == 21, "Width array should have 21 elements"

        # Find the current source network and element
        current_source_element = None
        current_source_network = None
        for network in prob.networks:
            # Assuming this project has one network with one current source
            if len(network.elements) == 1 and isinstance(network.elements[0], problem.CurrentSource):
                current_source_element = network.elements[0]
                current_source_network = network
                break

        assert current_source_element is not None, "No current source element found in the test project"
        assert current_source_network is not None, "No network containing the current source found"

        # Find the Connection objects corresponding to the f and t NodeIDs
        try:
            f_conn = next(c for c in current_source_network.connections if c.node_id == current_source_element.f)
            t_conn = next(c for c in current_source_network.connections if c.node_id == current_source_element.t)
        except StopIteration:
            pytest.fail(f"Could not find connections for CurrentSource {current_source_element} in network {current_source_network}")
        
        # Get voltages at the connection points
        voltage_from = find_vertex_value(solution, f_conn)
        voltage_to = find_vertex_value(solution, t_conn)
        
        # Calculate voltage difference between terminals
        voltage_diff = voltage_to - voltage_from

        def tapered_segment_resistance(w_start, w_end, length):
            """Calculate resistance of a tapered trace segment."""
            # Assuming the trace is on the first layer for conductance
            # TODO: Make this more robust if multiple layers exist
            sheet_resistance = 1.0 / prob.layers[0].conductance # Resistance per square (ohms/sq)

            if abs(w_start - w_end) < 1e-9: # Check for floating point equality
                # Straight segment
                return sheet_resistance * length / w_start
            else:
                # Tapered segment - use average width formula for resistance
                # R = rho * L / (t * W_avg) where rho/t is sheet_resistance
                w_avg = (w_end - w_start) / math.log(w_end / w_start)
                return sheet_resistance * length / w_avg

        def expected_voltage_at_position(position_mm):
            """Calculate expected voltage drop up to a given position along the trace."""
            at_position = 0.0
            total_resistance = 0.0
            segment_length = 10.0 # Length of each segment in mm

            while (remaining_length := position_mm - at_position) > 1e-9: # Use tolerance for float comparison
                # Get current segment index
                i_segment = int(at_position / segment_length)
                # Ensure index is within bounds
                if i_segment >= len(widths) - 1:
                    break # Reached end of defined widths

                w_start = widths[i_segment]
                w_end_full = widths[i_segment + 1]

                # Handle partial segments
                if remaining_length >= segment_length - 1e-9: # Use tolerance
                    this_segment_length = segment_length
                    w_end = w_end_full
                else:
                    this_segment_length = remaining_length
                    # Linearly interpolate width for partial segment
                    w_end = w_start + (w_end_full - w_start) * (remaining_length / segment_length)

                # Add this segment's resistance
                segment_resistance = tapered_segment_resistance(w_start, w_end, this_segment_length)
                total_resistance += segment_resistance
                at_position += this_segment_length

            # Calculate voltage drop based on resistance and current
            # Voltage drop = I * R. If current flows f->t, V_t - V_f = I * R
            return current_source_element.current * total_resistance

        # Calculate expected voltage drop across the entire trace (200mm)
        total_expected_voltage = expected_voltage_at_position(200.0)
        
        # Compare simulated vs analytical results for overall drop
        assert voltage_diff == pytest.approx(total_expected_voltage, rel=0.1), \
            f"Voltage drop mismatch: simulated={voltage_diff:.3f}V, analytical={total_expected_voltage:.3f}V"

        # TODO: Also add a test for individual trace segments. This seems to not work
        # that well though...

    def test_superposition_principle(self, kicad_test_projects):
        """Test that superposition principle holds for a circuit with voltage and current sources."""
        # Get the project with combined voltage and current sources
        project = kicad_test_projects["voltage_source_into_current_sink"]

        # Load the original problem with both sources
        full_problem = kicad.load_kicad_project(project.pro_path)

        # --- Identify the voltage source, current source, and their networks ---
        voltage_source_element = None
        voltage_source_network = None
        current_source_element = None
        current_source_network = None

        for network in full_problem.networks:
            for element in network.elements:
                if isinstance(element, problem.VoltageSource):
                    if voltage_source_element is not None:
                        pytest.fail("Found more than one voltage source")
                    voltage_source_element = element
                    voltage_source_network = network
                elif isinstance(element, problem.CurrentSource):
                    if current_source_element is not None:
                        pytest.fail("Found more than one current source")
                    current_source_element = element
                    current_source_network = network

        assert voltage_source_element is not None, "Expected exactly one voltage source"
        assert current_source_element is not None, "Expected exactly one current source"

        # --- Solve the full problem ---
        full_solution = solver.solve(full_problem)

        # --- Create and solve problem with only voltage source active ---
        voltage_only_networks = []
        for network in full_problem.networks:
            new_elements = []
            for element in network.elements:
                if element == current_source_element:
                    # Replace current source with 0A version
                    new_elements.append(problem.CurrentSource(
                        f=element.f, t=element.t, current=0.0
                    ))
                else:
                    new_elements.append(element)
            # Create new network with original connections and potentially modified elements
            voltage_only_networks.append(problem.Network(
                connections=network.connections, elements=new_elements
            ))

        voltage_only_problem = problem.Problem(
            layers=full_problem.layers,
            networks=voltage_only_networks
        )
        voltage_only_solution = solver.solve(voltage_only_problem)

        # --- Create and solve problem with only current source active ---
        current_only_networks = []
        for network in full_problem.networks:
            new_elements = []
            for element in network.elements:
                if element == voltage_source_element:
                    # Replace voltage source with 0V version
                    new_elements.append(problem.VoltageSource(
                        p=element.p, n=element.n, voltage=0.0
                    ))
                else:
                    new_elements.append(element)
            # Create new network with original connections and potentially modified elements
            current_only_networks.append(problem.Network(
                connections=network.connections, elements=new_elements
            ))

        current_only_problem = problem.Problem(
            layers=full_problem.layers,
            networks=current_only_networks
        )
        current_only_solution = solver.solve(current_only_problem)

        # --- Choose test points (Connections of the sources) ---
        test_connections = []
        try:
            # Connections for the voltage source
            test_connections.append(next(c for c in voltage_source_network.connections if c.node_id == voltage_source_element.p))
            test_connections.append(next(c for c in voltage_source_network.connections if c.node_id == voltage_source_element.n))
            # Connections for the current source
            test_connections.append(next(c for c in current_source_network.connections if c.node_id == current_source_element.f))
            test_connections.append(next(c for c in current_source_network.connections if c.node_id == current_source_element.t))
        except StopIteration:
             pytest.fail("Could not find all connections for the sources")

        # Remove duplicates if sources share connections
        test_connections = list(set(test_connections))

        # --- Compare solutions at each test point ---
        for connection in test_connections:
            v_full = find_vertex_value(full_solution, connection)
            v_voltage = find_vertex_value(voltage_only_solution, connection)
            v_current = find_vertex_value(current_only_solution, connection)

            # Verify superposition (with tolerance appropriate for floating-point)
            v_superposition = v_voltage + v_current
            assert v_full == pytest.approx(v_superposition, abs=1e-3), \
                f"Superposition failed at connection {connection}: " \
                f"full={v_full:.6f}, voltage={v_voltage:.6f}, " \
                f"current={v_current:.6f}, sum={v_superposition:.6f}"

        # --- Verify specific expected voltage values in the full solution ---
        # Find connections for the original voltage source again
        vsource_p_conn = next(c for c in voltage_source_network.connections if c.node_id == voltage_source_element.p)
        vsource_n_conn = next(c for c in voltage_source_network.connections if c.node_id == voltage_source_element.n)

        v_source_p = find_vertex_value(full_solution, vsource_p_conn)
        v_source_n = find_vertex_value(full_solution, vsource_n_conn)
        assert v_source_p - v_source_n == pytest.approx(voltage_source_element.voltage, abs=1e-4), \
            "Voltage source constraint not satisfied in full solution"
        
    def test_disconnected_component_gets_dropped(self, kicad_test_projects):
        project = kicad_test_projects["floating_copper"]
        
        # Load the problem from the KiCad project
        prob = kicad.load_kicad_project(project.pro_path)
        
        # Call the function under test
        solution = solver.solve(prob)
        
        assert solution is not None
        
        # Verify this has a single layer (this project should only have F.Cu)
        assert len(solution.layer_solutions) == 1
        
        layer_solution = solution.layer_solutions[0]
        
        # Verify it has only one mesh in the layer solution
        # (one connected component with electrical elements, the other should be dropped)
        assert len(layer_solution.meshes) == 2
        
        # Verify the mesh has vertices and values
        assert len(layer_solution.meshes[0].vertices) > 0
        assert len(layer_solution.values[0].values) > 0

    def test_unconnected_via_mesh_isolation(self, kicad_test_projects):
        """
        Test that meshes remain properly isolated and maintain correct voltage despite unconnected vias.
        This test will FAIL with the current implementation and PASS after the fix is applied.
        """
        # Get the unconnected_via project
        project = kicad_test_projects["unconnected_via"]
        
        # Load the problem from the KiCad project
        prob = kicad.load_kicad_project(project.pro_path)
        
        # Solve the problem
        solution = solver.solve(prob)
        
        # Find the voltage source network and element
        voltage_source_element = None
        voltage_source_network = None
        for network in prob.networks:
            # Assuming this project has one network with one voltage source
            if len(network.elements) == 1 and isinstance(network.elements[0], problem.VoltageSource):
                voltage_source_element = network.elements[0]
                voltage_source_network = network
                break

        assert voltage_source_element is not None, "No voltage source element found in the unconnected_via project"
        assert voltage_source_network is not None, "No network containing the voltage source found"

        # Find the Connection objects corresponding to the p and n NodeIDs
        try:
            p_conn = next(c for c in voltage_source_network.connections if c.node_id == voltage_source_element.p)
            n_conn = next(c for c in voltage_source_network.connections if c.node_id == voltage_source_element.n)
        except StopIteration:
            pytest.fail(f"Could not find connections for VoltageSource {voltage_source_element} in network {voltage_source_network}")
        
        # Get reference voltages at the source connection points
        neg_voltage = find_vertex_value(solution, n_conn)
        expected_diff = voltage_source_element.voltage  # Should be 1.0V
        
        # Get the layer and mesh containing the positive connection
        pos_layer_idx = prob.layers.index(p_conn.layer)
        pos_layer_sol = solution.layer_solutions[pos_layer_idx]
        
        # Find the mesh containing the positive connection point
        pos_mesh_idx = None
        pos_conn_point_mesh = mesh.Point(p_conn.point.x, p_conn.point.y)
        
        for i, msh in enumerate(pos_layer_sol.meshes):
            for vertex in msh.vertices:
                # Use a small tolerance to find the mesh containing the connection point
                if vertex.p.distance(pos_conn_point_mesh) < 1e-4:
                    pos_mesh_idx = i
                    break
            if pos_mesh_idx is not None:
                break
        
        assert pos_mesh_idx is not None, "Could not find mesh containing positive connection point"
        
        # Check that ALL vertices in the positive mesh have consistent voltage
        pos_mesh = pos_layer_sol.meshes[pos_mesh_idx]
        pos_values = pos_layer_sol.values[pos_mesh_idx]
        
        # With the bug: some vertices might have incorrect voltage due to via shorting
        # After fix: all vertices should have same voltage as positive terminal
        for vertex in pos_mesh.vertices:
            vertex_voltage = pos_values[vertex]
            # Verify voltage relative to negative terminal
            voltage_diff = vertex_voltage - neg_voltage
            
            # This assertion will fail if any vertex in the positive mesh
            # has been improperly connected through a "dead" via
            assert voltage_diff == pytest.approx(expected_diff, abs=0.01), \
                f"Vertex at ({vertex.p.x:.2f}, {vertex.p.y:.2f}) has incorrect voltage: " \
                f"{voltage_diff:.3f}V vs expected {expected_diff:.1f}V"

    def test_two_big_planes_voltage_source(self, kicad_test_projects):
        project = kicad_test_projects["two_big_planes"]
        # This project has two large planes with a voltage source between them
        # The idea of this test is to verify that that the voltage difference
        # between the two planes (meshes) is approximately equal to the voltage
        # of the voltage source.
        prob = kicad.load_kicad_project(project.pro_path)
        solution = solver.solve(prob)

        assert solution is not None, "Solver failed to produce a solution"

        # Find the voltage source network and element
        voltage_source_element = None
        voltage_source_network = None
        found_networks_with_vs = 0
        for network in prob.networks:
            for element in network.elements:
                if isinstance(element, problem.VoltageSource):
                    if voltage_source_element is not None:
                         pytest.fail("Found more than one voltage source element")
                    voltage_source_element = element
                    voltage_source_network = network
                    found_networks_with_vs += 1
                # Check for other unexpected elements (like resistors from vias)
                # For this specific test, we assume only the voltage source exists.
                elif not isinstance(element, problem.VoltageSource):
                     pytest.fail(f"Found unexpected element type {type(element)} in network")

        assert voltage_source_element is not None, "No voltage source element found in the project"
        assert voltage_source_network is not None, "No network containing the voltage source found"
        # Verify it's the only network (as expected for this specific test project)
        assert len(prob.networks) == 1, "Expected exactly one network"
        # Verify the network contains only the voltage source
        assert len(voltage_source_network.elements) == 1, "Expected network to contain only the voltage source"

        expected_voltage_diff = voltage_source_element.voltage

        # Assuming this project has one layer with two disconnected meshes
        assert len(solution.layer_solutions) == 1, "Expected exactly one layer solution"
        layer_solution = solution.layer_solutions[0]
        # The connectivity graph should drop the unconnected plane if grounding is applied correctly
        # However, the solver might still mesh both if they contain connection points.
        # Let's check if we have two meshes as the geometry suggests.
        assert len(layer_solution.meshes) == 2, "Expected exactly two meshes (planes) in the layer"

        # Get the meshes and their corresponding voltage values
        mesh1, mesh2 = layer_solution.meshes
        values1, values2 = layer_solution.values

        # Verify voltage is constant within each mesh and get representative voltages
        def check_mesh_voltage(msh, values):
            assert len(msh.vertices) > 0, "Mesh should have vertices"
            first_vertex = next(iter(msh.vertices)) # Get an arbitrary vertex
            ref_voltage = values[first_vertex]
            for vertex in msh.vertices:
                # Use a very tight tolerance for constant voltage check within a plane
                # This assumes the grounding fix prevents floating potential issues.
                assert values[vertex] == pytest.approx(ref_voltage, abs=1e-10), \
                    f"Voltage inconsistency within mesh: {values[vertex]} vs {ref_voltage}"
            return ref_voltage

        voltage_plane1 = check_mesh_voltage(mesh1, values1)
        voltage_plane2 = check_mesh_voltage(mesh2, values2)

        # Verify the voltage difference between the planes matches the source
        actual_voltage_diff = abs(voltage_plane1 - voltage_plane2)
        # Use a very tight tolerance for the difference between planes
        assert actual_voltage_diff == pytest.approx(expected_voltage_diff, abs=1e-10), \
            f"Voltage difference between planes ({actual_voltage_diff}) does not match source ({expected_voltage_diff})"

        # Additionally, check that the source terminals land on the correct planes
        # and have the expected voltage difference
        # Find the Connection objects corresponding to the p and n NodeIDs
        try:
            p_conn = next(c for c in voltage_source_network.connections if c.node_id == voltage_source_element.p)
            n_conn = next(c for c in voltage_source_network.connections if c.node_id == voltage_source_element.n)
        except StopIteration:
            pytest.fail(f"Could not find connections for VoltageSource {voltage_source_element} in network {voltage_source_network}")

        voltage_p = find_vertex_value(solution, p_conn)
        voltage_n = find_vertex_value(solution, n_conn)

        # Use a very tight tolerance for the terminal voltage difference
        assert voltage_p - voltage_n == pytest.approx(expected_voltage_diff, abs=1e-10), \
            "Voltage difference across source terminals does not match expected value"

        # Check which plane corresponds to which terminal voltage
        # Use a slightly larger tolerance here to account for find_vertex_value lookup
        # if the terminal isn't exactly on a vertex.
        if abs(voltage_p - voltage_plane1) < 1e-6:
            assert abs(voltage_n - voltage_plane2) < 1e-6, "Terminal N should be on Plane 2"
        elif abs(voltage_p - voltage_plane2) < 1e-6:
            assert abs(voltage_n - voltage_plane1) < 1e-6, "Terminal N should be on Plane 1"
        else:
            pytest.fail("Voltage source positive terminal does not match either plane's voltage")

    def test_simple_consumer(self, kicad_test_projects):
        project = kicad_test_projects["simple_consumer"]
        # This project is designed to test the CONSUMER directive. The idea is that
        # We have the following test points:
        # TP6 @ 100,50
        # TP7 @ 120,50
        # TP8 @ 140,50

        # TP2 @ 100,150
        # TP3 @ 120,150
        # TP4 @ 140,150

        # TP5 @ 180,50
        # TP1 @ 180,150

        # There is a consumer pushing 3A from
        # TP2,TP3,TP4 straight into TP1
        # Together, these are flowing via a trace from TP1 to TP5
        # and from TP5 back into the TP2,TP3,TP4
        # We test that it works by checking that the voltage between
        # TP6 and TP2
        # TP7 and TP3
        # TP8 and TP4
        # is 0.24 (1A split)
        # and that the voltage between
        # TP1 and TP5
        # is 3*0.24 (since it is shared)
        prob = kicad.load_kicad_project(project.pro_path)
        solution = solver.solve(prob)

        assert solution is not None, "Solver failed to produce a solution"

        # Find the connection objects for each test point
        # For this specific test, we know the test points are at these specific coordinates
        expected_coords = {
            "TP1": (180, 150), "TP2": (100, 150), "TP3": (120, 150), "TP4": (140, 150),
            "TP5": (180, 50), "TP6": (100, 50), "TP7": (120, 50), "TP8": (140, 50),
        }

        # Find connections that match these coordinates
        tp_connections = {}
        
        for network in prob.networks:
            for conn in network.connections:
                # Find which test point this connection corresponds to
                for tp_name, coords in expected_coords.items():
                    if abs(conn.point.x - coords[0]) < 1e-3 and abs(conn.point.y - coords[1]) < 1e-3:
                        tp_connections[tp_name] = conn
                        break
        
        # Ensure we found all test points
        assert len(tp_connections) == 8, f"Did not find all test points. Found: {list(tp_connections.keys())}"

        # Get voltages at each test point
        tp_voltages = {name: find_vertex_value(solution, conn) for name, conn in tp_connections.items()}

        # Perform assertions
        # Voltage drop across individual 1A paths (TP2-TP6, TP3-TP7, TP4-TP8)
        # Assuming current flows from TP{2,3,4} towards TP1, and returns via TP5 towards TP{6,7,8}
        # The voltage at TP{2,3,4} should be higher than TP{6,7,8} respectively.
        # The resistance of the path segment is 0.24 Ohm (from project description).
        # V_drop = I * R = 1A * 0.24 Ohm = 0.24V
        expected_individual_drop = 0.24
        assert tp_voltages["TP6"] - tp_voltages["TP2"] == pytest.approx(expected_individual_drop, abs=0.01)
        assert tp_voltages["TP7"] - tp_voltages["TP3"] == pytest.approx(expected_individual_drop, abs=0.01)
        assert tp_voltages["TP8"] - tp_voltages["TP4"] == pytest.approx(expected_individual_drop, abs=0.01)

        # Voltage drop across the shared 3A path (TP1-TP5)
        # Current flows from TP1 towards TP5. V(TP1) should be higher than V(TP5).
        # V_drop = I_total * R = 3A * 0.24 Ohm = 0.72V
        expected_shared_drop = 3 * 0.24
        assert tp_voltages["TP1"] - tp_voltages["TP5"] == pytest.approx(expected_shared_drop, abs=0.02), \
            f"Voltage drop TP1-TP5: {tp_voltages['TP1'] - tp_voltages['TP5']:.3f}V vs expected {expected_shared_drop:.3f}V"

    def test_unterminated_current_loop_warning(self, kicad_test_projects):
        project = kicad_test_projects["unterminated_current_loop"]
        prob = kicad.load_kicad_project(project.pro_path)
        with pytest.warns(solver.SolverWarning, match="Ground node voltage is not zero"):
            solution = solver.solve(prob)
        # TODO: Ideally, we would sanity check that the solution object is
        # at least reasonably structured
