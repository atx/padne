
import shapely.geometry
from padne.mesh import Mesher


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
