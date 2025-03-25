

from dataclasses import dataclass
import shapely.geometry
import numpy as np
import scipy.sparse

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


def solve(prob: problem.Problem) -> Solution:
    """
    Solve the given PCB problem to find voltage and current distribution.
    
    Args:
        problem: The Problem object containing layers and lumped elements
        
    Returns:
        A Solution object with the computed results
    """
    # References:
    # https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf
    # http://mobile.rodolphe-vaillant.fr/entry/101/definition-laplacian-matrix-for-triangle-meshes
    # TODO: Eliminate disconnected regions
    mesher = mesh.Mesher()

    # As a first step, we flatten the Layer-Mesh structure to get a list of meshes
    # (and store which layer they originally come from)

    meshes = []
    mesh_index_to_layer_index: list[int] = []

    for layer_i, layer in enumerate(prob.layers):
        layer_meshes = mesh_layer(mesher, prob, layer)
        meshes.extend(layer_meshes)
        mesh_index_to_layer_index.extend([layer_i] * len(layer_meshes))

    # In the next step, we assign a global index to each vertex in every mesh
    # this is needed since we need to somehow map the vertex indices to the
    # matrix indices in the final system of equations
    global_index_to_vertex_index: list[tuple[int, int]] = []
    mesh_vertex_index_to_global_index: dict[tuple[int, int], int] = {}
    for mesh_idx, msh in enumerate(meshes):
        for vertex_idx, vertex in enumerate(msh.vertices):
            global_index = len(global_index_to_vertex_index)
            global_index_to_vertex_index.append((mesh_idx, vertex_idx))
            mesh_vertex_index_to_global_index[(mesh_idx, vertex_idx)] = global_index

    voltage_source_count = sum(
        1 for elem in prob.lumpeds
        if elem.type == problem.Lumped.Type.VOLTAGE
    )

    # We are solving the equation L * v = r
    # where L is the "laplace operator",
    # v is the voltage vector and
    # r is the right-hand side "source" vector
    # TODO: This needs to be
    # decremented by 1 for each connected component (we are just going to get one for now)
    N = len(global_index_to_vertex_index) + voltage_source_count # - 1
    L = scipy.sparse.dok_matrix((N, N), dtype=np.float32)
    r = np.zeros(N, dtype=np.float32)

    # Okay, now we enumerate over every vertex
    for i, (mesh_idx, vertex_idx) in enumerate(global_index_to_vertex_index):
        vertex = meshes[mesh_idx].vertices.to_object(vertex_idx)
        for edge in vertex.orbit():
            vertex_other = edge.twin.origin
            vertex_other_idx = meshes[mesh_idx].vertices.to_index(vertex_other)
            k = mesh_vertex_index_to_global_index[(mesh_idx, vertex_other_idx)]

            if not edge.face.is_boundary and not edge.twin.face.is_boundary:
                # We are in "the interior"
                ratio = 0.
                for ed in [edge.next.next, edge.twin.next.next]:
                    va = vertex.p - ed.origin.p
                    vb = vertex_other.p - ed.origin.p
                    ratio += abs(va.dot(vb) / (va ^ vb)) / 2
            else:
                # TODO: This boundary handling comes from my original code 
                # written in 2019. It is very likely wrong.
                # Do considerable amount of thinking here to figure out
                # the correct way to force the normal derivative to zero
                # Questionable:
                eop = edge.next.next if not edge.face.is_boundary else edge.twin.next.next
                va = vertex.p - eop.origin.p
                vb = vertex_other.p - eop.origin.p
                ratio = abs(va.dot(vb) / (va ^ vb)) / 2
            L[i, i] -= ratio
            # Note that we are iterating over everything, so the (k, i) pair gets 
            # set in a different iteration
            L[i, k] += ratio

    def get_vertex_global_index_by_point(layer: problem.Layer,
                                         pt: shapely.geometry.Point) -> int:
        # TODO: This is not exactly efficient.
        # At some point, we are going to either need to use a spatial index
        # or track the seed points through the meshing process
        for i, (mesh_idx, vertex_idx) in enumerate(global_index_to_vertex_index):
            # Does the mesh index match the layer index?
            if mesh_index_to_layer_index[mesh_idx] != prob.layers.index(layer):
                continue
            # *quack quack*
            dist = meshes[mesh_idx].vertices.to_object(vertex_idx).p.distance(pt)
            if dist < 1e-6:
                return i
        raise ValueError("Vertex not found")

    # Now we need to process the lumped elements
    voltage_source_i = 0
    for elem in prob.lumpeds:
        i_a = get_vertex_global_index_by_point(elem.a_layer, elem.a_point)
        i_b = get_vertex_global_index_by_point(elem.b_layer, elem.b_point)

        # TODO: Note that these need to be multiplied by a conductance factor,
        # since our laplace matrix is unitless.
        # TODO: Maybe we actually want to actually scale the L matrix since different
        # layers have different conductances anyway?
        match elem.type:
            case problem.Lumped.Type.VOLTAGE:
                # THIS IS WRONG, BUT JUST FOR NOW
                # Normally, we need to inject additional rows and columns for this
                # case
                i_v = len(global_index_to_vertex_index) + voltage_source_i
                voltage_source_i += 1

                L[i_v, i_a] = -1
                L[i_a, i_v] = -1

                L[i_v, i_b] = 1
                L[i_b, i_v] = 1

                r[i_v] = elem.value
            case problem.Lumped.Type.CURRENT:
                r[i_a] = elem.value / elem.a_layer.conductance
                r[i_b] = -elem.value / elem.b_layer.conductance
            case problem.Lumped.Type.RESISTANCE:
                val = 1 / elem.value / elem.a_layer.conductance
                L[i_a, i_a] -= val
                L[i_b, i_b] -= val
                L[i_a, i_b] += val
                L[i_b, i_a] += val

    # Now we need to solve the system of equations
    # We are going to use a direct solver for now
    v = scipy.sparse.linalg.spsolve(L.tocsc(), r)

    # Great, now just convert it back to a Solution
    layer_solutions = []
    for layer_i, layer in enumerate(prob.layers):
        # TODO: Also unfuck this a bit
        layer_values = []
        for mesh_idx, msh in enumerate(meshes):
            if mesh_index_to_layer_index[mesh_idx] != layer_i:
                continue
            vertex_values = []
            for vertex_idx, vertex in enumerate(msh.vertices):
                global_index = mesh_vertex_index_to_global_index[(mesh_idx, vertex_idx)]
                vertex_values.append(v[global_index])
            layer_values.append(vertex_values)

        layer_solutions.append(LayerSolution(meshes=meshes, values=layer_values))

    # Return the complete solution
    return Solution(problem=prob, layer_solutions=layer_solutions)
