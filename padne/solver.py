

import numpy as np
import scipy.sparse
import shapely.geometry

from dataclasses import dataclass, field

from . import problem, mesh


@dataclass
class LayerSolution:
    meshes: list[mesh.Mesh]
    values: list[mesh.ZeroForm]


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
        for terminal in elem.terminals:
            if terminal.layer == layer:
                seed_points.append(mesh.Point(terminal.point.x, terminal.point.y))
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


def laplace_operator(mesh: mesh.Mesh) -> scipy.sparse.dok_matrix:
    """
    Compute the Laplace operator for a given mesh. This is in "mesh-local"
    indices, so the variable indices are given by the mesh.vertices indices.
    """
    N = len(mesh.vertices)
    L = scipy.sparse.dok_matrix((N, N), dtype=np.float32)

    for i, vertex_i in enumerate(mesh.vertices):
        for edge in vertex_i.orbit():
            # Grab the vertex on the other side of the edge
            vertex_k = edge.twin.origin
            k = mesh.vertices.to_index(vertex_k)
            ratio = 0.
            for ed in [edge.next.next, edge.twin.next.next]:
                if ed.next.face.is_boundary:
                    # Do not include boundary edges
                    continue
                vi = vertex_i.p - ed.origin.p
                vk = vertex_k.p - ed.origin.p
                ratio += abs(vi.dot(vk) / (vi ^ vk)) / 2
            L[i, i] -= ratio
            # Note that we are iterating over everything, so the (k, i) pair gets
            # set in a different iteration
            L[i, k] += ratio

    return L


@dataclass
class IndexStore:
    global_index_to_vertex_index: list[tuple[int, int]] = field(default_factory=list)
    mesh_vertex_index_to_global_index: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def create(cls, meshes: list[mesh.Mesh]) -> "IndexStore":
        store = cls()
        for mesh_idx, msh in enumerate(meshes):
            for vertex_idx, msh in enumerate(msh.vertices):
                global_index = len(store.global_index_to_vertex_index)
                store.global_index_to_vertex_index.append((mesh_idx, vertex_idx))
                store.mesh_vertex_index_to_global_index[(mesh_idx, vertex_idx)] = global_index
        return store


def generate_meshes_for_problem(prob: problem.Problem, mesher: mesh.Mesher) -> list[list[mesh.Mesh], list[int]]:
    meshes: list[mesh.Mesh] = []
    mesh_index_to_layer_index: list[int] = []

    for layer_i, layer in enumerate(prob.layers):
        layer_meshes = mesh_layer(mesher, prob, layer)
        meshes.extend(layer_meshes)
        mesh_index_to_layer_index.extend([layer_i] * len(layer_meshes))
    return meshes, mesh_index_to_layer_index


def make_terminal_index(prob: problem.Problem,
                        meshes: list[mesh.Mesh],
                        mesh_index_to_layer_index: list[int],
                        store: IndexStore) -> dict[problem.Terminal, int]:
    """
    Create a mapping from terminals to their global indices.
    
    Args:
        prob: The Problem object containing layers and lumped elements
        store: The IndexStore object containing the global indices
        
    Returns:
        A dictionary mapping terminals to their global indices
    """
    # TODO: This function mildly cursed. We need to somehow consolidate
    # the arguments.
    terminals = [
        t for lumped in prob.lumpeds for t in lumped.terminals
    ]
    terminal_index: dict[problem.Terminal, int] = {}
    for terminal in terminals:
        for i, (mesh_idx, vertex_idx) in enumerate(store.global_index_to_vertex_index):
            if mesh_index_to_layer_index[mesh_idx] != prob.layers.index(terminal.layer):
                continue
            dist = meshes[mesh_idx].vertices.to_object(vertex_idx).p.distance(terminal.point)
            if dist > 1e-6:
                continue
            # Found the terminal
            if terminal in terminal_index:
                raise ValueError("Duplicate terminal vertex found, this should not happen.")
            terminal_index[terminal] = i
    return terminal_index


def process_lumped_elements(lumpeds: list[problem.BaseLumped],
                            terminal_index: dict[problem.Terminal, int],
                            voltage_source_i: int,
                            L: scipy.sparse.dok_matrix,
                            r: np.ndarray) -> None:
    for elem in lumpeds:
        # TODO: Maybe we actually want to actually scale the L matrix since different
        # layers have different conductances anyway?
        # It is unclear how this ends up affecting the final solution.
        match elem:
            case problem.Resistor(a=a, b=b, resistance=resistance):
                # TODO: What if the conductances of the layers differ?
                val = 1 / resistance / a.layer.conductance
                i_a = terminal_index[a]
                i_b = terminal_index[b]

                L[i_a, i_a] -= val
                L[i_b, i_b] -= val
                L[i_a, i_b] += val
                L[i_b, i_a] += val
            case problem.VoltageSource(p=p, n=n, voltage=voltage):
                i_v = voltage_source_i
                voltage_source_i += 1

                i_p = terminal_index[p]
                i_n = terminal_index[n]

                L[i_v, i_p] = 1
                L[i_p, i_v] = 1

                L[i_v, i_n] = -1
                L[i_n, i_v] = -1

                r[i_v] = voltage
            case problem.CurrentSource(f=f, t=t, current=current):
                i_f = terminal_index[f]
                i_t = terminal_index[t]

                r[i_f] = current / f.layer.conductance
                r[i_t] = -current / t.layer.conductance

            case problem.VoltageRegulator():
                raise NotImplementedError("Voltage regulators are not yet supported")


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
    meshes, mesh_index_to_layer_index = generate_meshes_for_problem(prob, mesher)

    # In the next step, we assign a global index to each vertex in every mesh
    # this is needed since we need to somehow map the vertex indices to the
    # matrix indices in the final system of equations
    store = IndexStore.create(meshes)

    voltage_source_count = sum(
        1 for elem in prob.lumpeds
        if isinstance(elem, problem.VoltageSource)
    )

    # We are solving the equation L * v = r
    # where L is the "laplace operator",
    # v is the voltage vector and
    # r is the right-hand side "source" vector
    # TODO: Maybe we need to force a ground somewhere? Honestly, I
    # feel like as long as the solver can handle it, we can just leave everything
    # floating and let the UI figure out. This can possibly lead to some
    # numerical instability, so it needs more stress testing.
    N = len(store.global_index_to_vertex_index) + voltage_source_count
    L = scipy.sparse.dok_matrix((N, N), dtype=np.float32)
    r = np.zeros(N, dtype=np.float32)

    # Now we compute the Laplace operator for each mesh
    for mesh_idx, msh in enumerate(meshes):

        L_msh = laplace_operator(msh)

        # Glue them together into the global matrix
        for local_i, local_j in zip(*L_msh.nonzero()):
            global_i = store.mesh_vertex_index_to_global_index[(mesh_idx, local_i)]
            global_j = store.mesh_vertex_index_to_global_index[(mesh_idx, local_j)]
            L[global_i, global_j] = L_msh[local_i, local_j]

    # Create a mapping from terminals to the global index of the vertex they are connected to
    terminal_index = make_terminal_index(prob, meshes, mesh_index_to_layer_index, store)

    # Now we need to process the lumped elements
    process_lumped_elements(
        prob.lumpeds,
        terminal_index,
        len(store.global_index_to_vertex_index),
        L,
        r
    )

    # Now we need to solve the system of equations
    # We are going to use a direct solver for now
    # TODO: This is a symmetric positive definite matrix, so we can theoretically
    # use something like Conjugate Gradient
    v = scipy.sparse.linalg.spsolve(L.tocsc(), r)

    # Great, now just convert it back to a Solution
    layer_solutions = []
    for layer_i, layer in enumerate(prob.layers):
        layer_meshes = []
        layer_values = []
        for mesh_idx, msh in enumerate(meshes):
            if mesh_index_to_layer_index[mesh_idx] != layer_i:
                continue
            layer_meshes.append(msh)
            # Create a ZeroForm for this mesh's vertices
            vertex_values = mesh.ZeroForm(msh)  # Initialize ZeroForm with the mesh
            for vertex_idx, vertex in enumerate(msh.vertices):
                global_index = store.mesh_vertex_index_to_global_index[(mesh_idx, vertex_idx)]
                vertex_values[vertex] = v[global_index]  # Set values using indexing
            layer_values.append(vertex_values)

        layer_solutions.append(LayerSolution(meshes=layer_meshes, values=layer_values))

    # Return the complete solution
    return Solution(problem=prob, layer_solutions=layer_solutions)
