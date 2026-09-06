import importlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Make the lazily-resolved submodules visible to static analysis.
    from . import (  # noqa: F401
        colormaps, context, kicad, mesh, parallel, problem, solver, units,
    )

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

# Submodules are resolved lazily (PEP 562) so that lightweight consumers --
# in particular spawned worker processes (see padne.parallel) -- do not pay
# for the pcbnew/pygerber/scipy imports unless they actually use them.
_SUBMODULES = ["kicad", "mesh", "solver", "problem", "units", "colormaps"]

__all__ = ["__version__", "kicad", "mesh", "solver", "problem", "units", "colormaps"]


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
