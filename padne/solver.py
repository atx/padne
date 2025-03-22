

from dataclasses import dataclass
import shapely.geometry

from . import problem, mesh


@dataclass
class LayerSolution:
    meshes: list[mesh.Mesh]
    values: list[list[float]]


@dataclass
class Solution:
    problem: problem.Problem
    layer_solutions: list[LayerSolution]


def collect_seed_points(problem: problem.Problem, layer: problem.Layer) -> list[mesh.Point]:
    """
    Collect all seed points (component pads) that are on this layer.
    
    Args:
        problem: The entire problem containing all lumped elements
        layer: The specific layer to collect seed points for
        
    Returns:
        List of Points to be used as mesh seed points
    """
    seed_points = []
    for elem in problem.lumpeds:
        # Check if this lumped element connects to our layer
        if elem.a_layer == layer:
            # Convert from shapely Point to our mesh.Point
            seed_points.append(mesh.Point(elem.a_point.x, elem.a_point.y))
        if elem.b_layer == layer:
            seed_points.append(mesh.Point(elem.b_point.x, elem.b_point.y))
    return seed_points


def mesh_layer(mesher: mesh.Mesher, problem: problem.Problem, layer: problem.Layer) -> list[mesh.Mesh]:
    """
    Generate meshes for a single layer.
    
    Args:
        mesher: The mesh generator to use
        problem: The entire problem containing all lumped elements
        layer: The specific layer to mesh
        
    Returns:
        List of Mesh objects, one for each connected region in the layer
    """
    seed_points = collect_seed_points(problem, layer)
    ret = []
    for subshape in layer.shape.geoms:
        # Filter seed points to only those contained in this subshape
        seed_points_in_subshape = [
            p for p in seed_points if subshape.intersects(shapely.geometry.Point(p.x, p.y))
        ]
        m = mesher.poly_to_mesh(subshape, seed_points_in_subshape)
        ret.append(m)
    return ret


def solve(problem: problem.Problem) -> Solution:
    """
    Solve the given PCB problem to find voltage and current distribution.
    
    Args:
        problem: The Problem object containing layers and lumped elements
        
    Returns:
        A Solution object with the computed results
    """
    mesher = mesh.Mesher()
    
    # Implement a dummy solver that does everything in regards to meshing,
    # but instead of doing FEM it just assigns random values to the nodes.
    # AI!
