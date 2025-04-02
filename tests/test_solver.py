import pytest
import shapely.geometry
import numpy as np
from padne import solver, problem, mesh, kicad
from pathlib import Path


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

    def test_all_test_projects_solve(self, kicad_test_projects):
        """Test that solver.solve works on all test projects."""
        for project in kicad_test_projects.values():
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


    def test_voltage_sources_work(self, kicad_test_projects):
        for project in kicad_test_projects.values():
            # Load the problem from the KiCad project
            prob = kicad.load_kicad_project(project.pro_path)
            # Check if there is a voltage source in the project
            has_voltage_source = any(
                isinstance(el, problem.VoltageSource) for el in prob.lumpeds
            )
            if not has_voltage_source:
                continue

            # Call the function under test
            solution = solver.solve(prob)

            assert solution is not None
            assert isinstance(solution, solver.Solution)

            # Check that every layer has a solution
            assert len(solution.layer_solutions) == len(prob.layers)

            # Helper function to find the voltage at the vertex closest to a terminal
            # TODO: This should be abstracted elsewhere
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
                assert best_dist < 1e-5, f"Could not find a close vertex for terminal {term}"
                assert found_value is not None
                return found_value

            # Check each voltage source
            for elem in prob.lumpeds:
                if isinstance(elem, problem.VoltageSource):
                    voltage_p = find_vertex_value(solution, elem.p)
                    voltage_n = find_vertex_value(solution, elem.n)
                    
                    # Verify the voltage difference matches the source voltage
                    assert voltage_p - voltage_n == pytest.approx(elem.voltage), \
                        f"Voltage difference for {elem} does not match expected value."
