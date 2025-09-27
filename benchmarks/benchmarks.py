
import os
import pathlib
import sys
import shapely.geometry
from padne.mesh import Mesher
from padne import kicad, solver

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
