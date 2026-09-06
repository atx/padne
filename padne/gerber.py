"""
Gerber file to Shapely geometry conversion.

This lives in its own module (rather than padne.kicad) so that
parallel.process_map workers can import it without paying for the heavy
pcbnew import that padne.kicad performs at module load time.
"""

import logging
import pygerber.gerber.api
import pygerber.vm
import shapely
import shapely.affinity
import shapely.geometry

from pathlib import Path
from typing import Optional, Union

from .context import stage_timer

log = logging.getLogger(__name__)


def ensure_geometry_is_multipolygon(geometry: Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]) -> shapely.geometry.MultiPolygon:
    """Convert Polygon to MultiPolygon if needed, ensuring consistent interface."""
    if geometry.geom_type == "Polygon":
        return shapely.geometry.MultiPolygon([geometry])
    if geometry.geom_type != "MultiPolygon":
        raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}")
    return geometry


def render_with_shapely(gerber_data: pygerber.gerber.api.GerberFile
                        ) -> shapely.geometry.MultiPolygon:
    # We have to call all of this manually, since we need to manually configure the
    # amount of segments in our arcs
    rvmc = gerber_data._get_rvmc()

    def angle_length_to_segment_count(angle_length: float) -> int:
        return int(abs(angle_length) * 0.4 + 10)

    result = pygerber.vm.render(
        rvmc,
        backend="shapely",
        angle_length_to_segment_count=angle_length_to_segment_count
    )
    return result.shape


@stage_timer
def gerber_file_to_shapely(gerber_path: Path) -> Optional[shapely.geometry.MultiPolygon]:
    """Loads data from a Gerber file and converts it to a Shapely geometry."""
    gerber_data = pygerber.gerber.api.GerberFile.from_file(gerber_path)
    try:
        geometry = render_with_shapely(gerber_data)
    except AssertionError:
        # This is a bug in pygerber, which gets triggered if the
        # gerber file is empty. We should fix this in pygerber ideally
        # TODO: Figure out if there is at least a way to check if the
        # gerber file is empty before we try to render it
        return None

    # For reasons to be determined, the geometry generated like this has
    # a flipped y axis. Flip it back.
    geometry = shapely.affinity.scale(geometry, 1.0, -1.0, origin=(0, 0))

    # First, we try to clean up the geometry by inflating and deflating.
    # This should remove any tiny slivers or gaps, usually caused by
    # pygerber not quite matching starts and ends of consecutive traces.
    # (see the test case "broken_trace_geometry" for an example)
    geometry = geometry.buffer(1e-4).buffer(-1e-4)

    # Simplify the geometry to remove almost-duplicate points
    # This is unfortunately a "bug" in pygerber, where drawing
    # a circle is implemented by drawing an arbitrary degree arc,
    # which sometimes results to the "starting" and "ending" points
    # not being exactly the same such as
    # (-1.0, 0.0) vs  (-1.0, 1.2246467991473532e-16)
    # Again, it would be nice to fix this in pygerber, but that
    # is a task for another day...
    geometry = geometry.simplify(tolerance=1e-4, preserve_topology=True)
    # Unfortunately, the above simplification can sometimes miss issues
    # with the polygon. Setting preserve_topology=False fixes it, but
    # who knows what other issues it may cause. Running a dedicated
    # point deduplication step seems to fix the issue, but again,
    # could potentially break the geometry. The "degenerate_hole_geometry"
    # test project exhibits this issue.
    geometry = shapely.remove_repeated_points(geometry, tolerance=1e-8)

    # If the layer has only a single connected component, convert it to a MultiPolygon
    geometry = ensure_geometry_is_multipolygon(geometry)

    return geometry
