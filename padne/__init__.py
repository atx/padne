
from . import kicad, mesh, solver, problem, units, colormaps

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "kicad",
    "mesh",
    "solver",
    "problem",
    "units",
    "colormaps",
]
