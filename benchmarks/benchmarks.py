
import os
import pathlib
import sys
import tempfile
from pathlib import Path
import shapely.geometry
from padne.mesh import Mesher
from padne import kicad, solver
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

    def setup_cache(self, *_):
        test_projects = _kicad_test_projects()
        project_names = self.time_solver_solve.params
        loaded_cache = {}
        for project_name in project_names:
            project = test_projects[project_name]
            loaded_cache[project_name] = kicad.load_kicad_project(project.pro_path)

        # Hopefully this will get properly pickled...
        return loaded_cache

    def time_solver_solve(self, loaded_projects, project_name):
        project = loaded_projects[project_name]
        # Just solve the project, we don't care about the result here
        solver.solve(project)

    time_solver_solve.params = ['two_big_planes', 'via_tht_4layer']
    time_solver_solve.param_names = ['project']


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
