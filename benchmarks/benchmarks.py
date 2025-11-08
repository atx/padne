
import os
import pathlib
import sys
import tempfile
from pathlib import Path
import shapely.geometry
from padne.mesh import Mesher
from padne import kicad, solver
import padne._cgal as cgal
import pcbnew

from tests.conftest import _kicad_test_projects


class MesherSuite:
    """Benchmarks for the Mesher class with different geometries and configurations."""

    def setup(self, *_):
        """Create the three test geometries once for all benchmarks."""
        # 1. Normal rectangle: 30x30mm
        self.normal_rectangle = shapely.geometry.box(0, 0, 30, 30)

        # 2. Large rectangle: 100x100mm
        self.large_rectangle = shapely.geometry.box(0, 0, 100, 100)

        # 3. Large rectangle with circular hole: 80x80mm with 50mm diameter hole
        outer = shapely.geometry.box(0, 0, 80, 80)
        # Create circle centered at (40, 40) with 25mm radius (50mm diameter)
        hole = shapely.geometry.Point(40, 40).buffer(25)
        self.rectangle_with_hole = outer.difference(hole)

        # Store geometries in a dict for parametrized access
        self.geometries = {
            'normal_rect': self.normal_rectangle,
            'large_rect': self.large_rectangle,
            'rect_with_hole': self.rectangle_with_hole
        }

        fixed_density_kwargs = {
            "minimum_angle": 20.0,
            "maximum_size": 0.6,
        }
        # This is to support older versions that do not have this parameter
        if hasattr(Mesher.Config, 'variable_size_maximum_factor'):
            fixed_density_kwargs["variable_size_maximum_factor"] = 1.0

        # Store configurations in a dict for parametrized access
        self.configs = {
            'default': Mesher.Config(),
            'relaxed': Mesher.Config.RELAXED,
            'fixed_density': Mesher.Config(**fixed_density_kwargs)
        }

    def time_mesh_generation(self, geometry_name, config_name):
        """Time mesh generation for different geometries and configurations."""
        geometry = self.geometries[geometry_name]
        config = self.configs[config_name]

        mesher = Mesher(config)
        mesher.poly_to_mesh(geometry)

    # Define parameters for the benchmark
    time_mesh_generation.params = (
        ['normal_rect', 'large_rect', 'rect_with_hole'],  # geometry names
        ['default', 'relaxed', 'fixed_density']  # config names
    )
    time_mesh_generation.param_names = ['geometry', 'config']

    def track_triangle_count(self, geometry_name, config_name):
        """Track the number of triangles generated for different geometries and configurations."""
        geometry = self.geometries[geometry_name]
        config = self.configs[config_name]

        mesher = Mesher(config)
        mesh = mesher.poly_to_mesh(geometry)
        return len(mesh.faces)

    # Define parameters for the benchmark
    track_triangle_count.params = (
        ['normal_rect', 'large_rect', 'rect_with_hole'],  # geometry names
        ['default', 'relaxed', 'fixed_density']  # config names
    )
    track_triangle_count.param_names = ['geometry', 'config']

    def track_mesh_sizeof(self, geometry_name, config_name):
        """Track memory usage of Mesh objects for different geometries and configurations."""
        # Note: we cannot use mem_ type benchmark here, since asv uses asizeof which
        # internally attempts to pickle the Mesh. Unfortunately it pickles it
        # using a special pickler that does not respect __getstate__
        # and everything breaks due to that.
        geometry = self.geometries[geometry_name]
        config = self.configs[config_name]

        mesher = Mesher(config)
        mesh = mesher.poly_to_mesh(geometry)

        def sizeof_indexmap(indexmap):
            total = sys.getsizeof(indexmap)
            for key, value in indexmap.items():
                total += sys.getsizeof(key)
                total += sys.getsizeof(value)
            return total

        imaps_to_measure = [
            mesh.vertices,
            mesh.halfedges,
            mesh.faces,
            mesh.boundaries
        ]
        imaps_total_sizes = sum(sizeof_indexmap(imap) for imap in imaps_to_measure)
        return sys.getsizeof(mesh) + imaps_total_sizes

    track_mesh_sizeof.params = (
        ['normal_rect', 'large_rect', 'rect_with_hole'],  # geometry names
        ['default', 'relaxed', 'fixed_density']  # config names
    )
    track_mesh_sizeof.param_names = ['geometry', 'config']

    def peakmem_mesh_generation(self, geometry_name, config_name):
        """Track peak memory usage during mesh generation for different geometries and configurations."""
        geometry = self.geometries[geometry_name]
        config = self.configs[config_name]

        mesher = Mesher(config)
        mesher.poly_to_mesh(geometry)

    peakmem_mesh_generation.params = (
        ['normal_rect', 'large_rect', 'rect_with_hole'],  # geometry names
        ['default', 'relaxed', 'fixed_density']  # config names
    )
    peakmem_mesh_generation.param_names = ['geometry', 'config']


class KicadSuite:
    """Benchmarks for KiCad project loading with different test projects."""

    def setup(self, *_):
        """Discover and set up test KiCad projects using the existing test infrastructure."""
        # Get all available test projects
        self.test_projects = _kicad_test_projects()

    def time_kicad_project_loading(self, project_name):
        """Time KiCad project loading for different test projects."""
        project = self.test_projects[project_name]
        # Use the project's pro_path to call load_kicad_project
        kicad.load_kicad_project(project.pro_path)

    # Define parameters for the benchmark
    time_kicad_project_loading.params = ['two_big_planes', 'via_tht_4layer']
    time_kicad_project_loading.param_names = ['project']


class SolverSuite:
    """Benchmarks for the FEM solver and solution memory usage."""

    def setup_cache(self):
        """Cache loaded projects and pre-computed solutions for benchmarks."""
        test_projects = _kicad_test_projects()
        # Expanded project list to include simple_geometry
        project_names = ['simple_geometry', 'two_big_planes', 'via_tht_4layer']

        cache = {'problems': {}, 'solutions': {}}

        for project_name in project_names:
            project = test_projects[project_name]
            problem = kicad.load_kicad_project(project.pro_path)
            cache['problems'][project_name] = problem
            # Pre-compute solutions for memory benchmarks
            cache['solutions'][project_name] = solver.solve(problem)

        return cache

    def time_solver_solve(self, cache, project_name):
        """Time the complete FEM solving pipeline."""
        problem = cache['problems'][project_name]
        solver.solve(problem)

    time_solver_solve.params = ['simple_geometry', 'two_big_planes', 'via_tht_4layer']
    time_solver_solve.param_names = ['project']

    def peakmem_solver_solve(self, cache, project_name):
        """Track peak memory usage for complete solve pipeline."""
        problem = cache['problems'][project_name]
        solver.solve(problem)

    peakmem_solver_solve.params = ['simple_geometry', 'two_big_planes', 'via_tht_4layer']
    peakmem_solver_solve.param_names = ['project']

    def mem_problem_definition(self, cache, project_name):
        """Track memory usage of Problem objects."""
        return cache['problems'][project_name]

    mem_problem_definition.params = ['simple_geometry', 'two_big_planes', 'via_tht_4layer']
    mem_problem_definition.param_names = ['project']

    def mem_layer_solutions(self, cache, project_name):
        """Track memory usage of LayerSolution objects."""
        solution = cache['solutions'][project_name]
        return solution.layer_solutions

    mem_layer_solutions.params = ['simple_geometry', 'two_big_planes', 'via_tht_4layer']
    mem_layer_solutions.param_names = ['project']


class KicadRenderSuite:
    """Benchmarks for KiCad rendering and geometry extraction operations."""

    def setup(self, *_):
        # Load test projects
        test_projects = _kicad_test_projects()

        # Select projects with varying complexity
        project_names = [
            'two_big_planes', 'via_tht_4layer', 'castellated_vias_internal_cutout',
        ]

        # Load boards using pcbnew
        self.boards = {}
        for project_name in project_names:
            project = test_projects[project_name]
            board = pcbnew.LoadBoard(str(project.pcb_path))
            self.boards[project_name] = board

        # Create a persistent temp directory for this benchmark run
        self.temp_dir = tempfile.mkdtemp(prefix='padne_bench_')
        self.temp_path = Path(self.temp_dir)

        # Map layer names to layer IDs
        self.layer_map = {
            'F_Cu': pcbnew.F_Cu,
            'B_Cu': pcbnew.B_Cu,
            'In1_Cu': pcbnew.In1_Cu
        }

        # Pre-generate all Gerber files needed for benchmarking
        self.gerber_files = {}

        for project_name, board in self.boards.items():
            for layer_name, layer_id in self.layer_map.items():
                # Check if this layer exists on the board
                if not board.GetEnabledLayers().Contains(layer_id):
                    continue

                # Generate Gerber file
                gerber_path = self.temp_path / f"{project_name}_{layer_name}.gbr"
                kicad.plot_board_layer_to_gerber(board, layer_id, gerber_path)

                # Store path for later use
                self.gerber_files[(project_name, layer_name)] = gerber_path

    def teardown(self, *_):
        import shutil
        shutil.rmtree(self.temp_dir)

    def time_plot_board_layer_to_gerber(self, project_name, layer_name):
        """Time Gerber generation for different layers and projects."""
        board = self.boards[project_name]
        layer_id = self.layer_map[layer_name]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / f"layer_{layer_name}.gbr"
            kicad.plot_board_layer_to_gerber(board, layer_id, output_path)

    # Parameters for plot_board_layer_to_gerber
    time_plot_board_layer_to_gerber.params = (
        ['two_big_planes', 'via_tht_4layer'],
        ['F_Cu', 'B_Cu']
    )
    time_plot_board_layer_to_gerber.param_names = ['project', 'layer']

    def time_gerber_file_to_shapely(self, project_name, layer_name):
        """Time Shapely conversion from pre-generated Gerber files."""
        # Use pre-generated Gerber file to isolate conversion timing
        gerber_path = self.gerber_files.get((project_name, layer_name))

        kicad.gerber_file_to_shapely(gerber_path)

    # Parameters for gerber_file_to_shapely
    time_gerber_file_to_shapely.params = (
        ['two_big_planes', 'via_tht_4layer'],
        ['F_Cu', 'B_Cu']
    )
    time_gerber_file_to_shapely.param_names = ['project', 'layer']

    def time_extract_board_outline(self, project_name):
        """Time board outline extraction for different PCB complexities."""
        board = self.boards[project_name]
        kicad.extract_board_outline(board)

    # Parameters for extract_board_outline
    time_extract_board_outline.params = ['castellated_vias_internal_cutout']
    time_extract_board_outline.param_names = ['project']


class LaplaceOperatorSuite:
    """Benchmarks for Laplace operator sparse matrix assembly."""

    def setup(self, *_):
        """Create synthetic meshes for Laplace operator benchmarks."""
        # Create synthetic rectangle geometries
        self.small_rectangle = shapely.geometry.box(0, 0, 10, 10)   # 10x10mm
        self.big_rectangle = shapely.geometry.box(0, 0, 100, 100)   # 100x100mm
        self.long_rectangle = shapely.geometry.box(0, 0, 1, 100)    # 1x100mm

        self.geometries = {
            'small_rectangle': self.small_rectangle,
            'big_rectangle': self.big_rectangle,
            'long_rectangle': self.long_rectangle
        }

        # Explicitly specify default Mesher config parameters to avoid benchmark
        # performance changes if defaults change
        config = Mesher.Config(
            minimum_angle=20.0,
            maximum_size=0.6,
            variable_density_min_distance=0.5,
            variable_density_max_distance=3.0,
            variable_size_maximum_factor=3.0,
            distance_map_quantization=1.0
        )

        # Generate meshes for each geometry
        mesher = Mesher(config)
        self.meshes = {}
        for name, geometry in self.geometries.items():
            self.meshes[name] = mesher.poly_to_mesh(geometry)

    def time_laplace_operator_assembly(self, geometry_name):
        """Time sparse matrix assembly of Laplace operator.

        Measures the performance of building CSR sparse matrices from mesh
        topology, including gradient operator construction and matrix transposes.
        Tests synthetic rectangles with predictable mesh characteristics.
        """
        mesh = self.meshes[geometry_name]
        solver.laplace_operator(mesh)

    time_laplace_operator_assembly.params = ['small_rectangle', 'big_rectangle', 'long_rectangle']
    time_laplace_operator_assembly.param_names = ['geometry']


class ConnectivitySuite:
    """Benchmarks for electrical connectivity analysis."""

    def setup(self, *_):
        """Load test projects for connectivity analysis."""
        test_projects = _kicad_test_projects()
        project_names = ['simple_geometry', 'disconnected_components', 'two_big_planes', 'via_tht_4layer', 'many_meshes']

        # Load problem definitions
        self.problems = {}
        for project_name in project_names:
            project = test_projects[project_name]
            self.problems[project_name] = kicad.load_kicad_project(project.pro_path)

    def time_connectivity_graph_construction(self, project_name):
        """Time construction of connectivity graph from layer geometry."""
        problem = self.problems[project_name]
        # Construct STRtrees and connectivity graph
        strtrees = solver.construct_strtrees_from_layers(problem.layers)
        solver.ConnectivityGraph.create_from_problem(problem, strtrees)

    time_connectivity_graph_construction.params = ['simple_geometry', 'disconnected_components', 'via_tht_4layer', 'many_meshes']
    time_connectivity_graph_construction.param_names = ['project']


class MeshGenerationSuite:
    """Benchmarks for mesh generation from KiCad projects."""

    def setup_cache(self):
        """Cache loaded projects and pre-computed data for mesh generation benchmarks."""
        test_projects = _kicad_test_projects()
        project_names = ['many_meshes', 'via_tht_4layer', 'simple_geometry']

        cache = {
            'problems': {},
            'strtrees': {},
            'connected_layer_mesh_pairs': {}
        }

        for project_name in project_names:
            project = test_projects[project_name]
            problem = kicad.load_kicad_project(project.pro_path)
            cache['problems'][project_name] = problem

            # Pre-compute STRtrees
            strtrees = solver.construct_strtrees_from_layers(problem.layers)
            cache['strtrees'][project_name] = strtrees

            # Pre-compute connectivity graph and connected pairs
            connectivity_graph = solver.ConnectivityGraph.create_from_problem(problem, strtrees)
            connected_pairs = solver.find_connected_layer_geom_indices(connectivity_graph)
            cache['connected_layer_mesh_pairs'][project_name] = connected_pairs

        return cache

    def time_generate_meshes_for_problem(self, cache, project_name):
        """Time mesh generation for different KiCad projects."""
        problem = cache['problems'][project_name]
        strtrees = cache['strtrees'][project_name]
        connected_pairs = cache['connected_layer_mesh_pairs'][project_name]

        # Create fresh Mesher instance with default config for each run
        mesher = Mesher()
        solver.generate_meshes_for_problem(problem, mesher, connected_pairs, strtrees)

    time_generate_meshes_for_problem.params = ['many_meshes', 'via_tht_4layer', 'simple_geometry']
    time_generate_meshes_for_problem.param_names = ['project']

    def track_mesh_count(self, cache, project_name):
        """Track the number of meshes generated for different projects."""
        problem = cache['problems'][project_name]
        strtrees = cache['strtrees'][project_name]
        connected_pairs = cache['connected_layer_mesh_pairs'][project_name]

        mesher = Mesher()
        meshes, _ = solver.generate_meshes_for_problem(problem, mesher, connected_pairs, strtrees)
        return len(meshes)

    track_mesh_count.params = ['many_meshes', 'via_tht_4layer', 'simple_geometry']
    track_mesh_count.param_names = ['project']


class DistanceMapSuite:
    """Benchmarks for distance map computation and queries."""

    def setup(self, *_):
        """Create polygons and distance maps for benchmarking."""
        # Create the same geometries as MeshMemorySuite
        self.small_rectangle = shapely.geometry.box(0, 0, 30, 30)
        self.large_rectangle = shapely.geometry.box(0, 0, 100, 100)
        outer = shapely.geometry.box(0, 0, 80, 80)
        hole = shapely.geometry.Point(40, 40).buffer(25)
        self.rectangle_with_hole = outer.difference(hole)

        self.geometries = {
            'small_rect': self.small_rectangle,
            'large_rect': self.large_rectangle,
            'rect_with_hole': self.rectangle_with_hole
        }

        # Pre-compute distance maps for query benchmarks
        self.distance_maps = {}
        quantization = 0.5  # Default quantization
        for name, geometry in self.geometries.items():
            self.distance_maps[name] = cgal.PolyBoundaryDistanceMap(geometry, quantization)

        # Pre-compute query points (center of each geometry)
        self.query_points = {
            'small_rect': (15.0, 15.0),
            'large_rect': (50.0, 50.0),
            'rect_with_hole': (40.0, 40.0)
        }

    def time_distance_map_creation(self, geometry_name):
        """Time distance map computation for different polygon geometries."""
        geometry = self.geometries[geometry_name]
        cgal.PolyBoundaryDistanceMap(geometry, 0.5)

    time_distance_map_creation.params = ['small_rect', 'large_rect', 'rect_with_hole']
    time_distance_map_creation.param_names = ['geometry']

    def time_distance_map_queries(self, geometry_name):
        """Time distance map query performance with bilinear interpolation."""
        distance_map = self.distance_maps[geometry_name]
        x, y = self.query_points[geometry_name]
        # Perform multiple queries to get meaningful timing
        for _ in range(1000):
            distance_map.query(x, y)

    time_distance_map_queries.params = ['small_rect', 'large_rect', 'rect_with_hole']
    time_distance_map_queries.param_names = ['geometry']
