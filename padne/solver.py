

import collections
import itertools
import logging
import numpy as np
import scipy.sparse
import scipy.spatial
import shapely.geometry

from dataclasses import dataclass, field

from . import problem, mesh

log = logging.getLogger(__name__)


@dataclass
class LayerSolution:
    meshes: list[mesh.Mesh]
    values: list[mesh.ZeroForm]


@dataclass
class Solution:
    problem: problem.Problem
    layer_solutions: list[LayerSolution]


@dataclass
class ConnectivityGraph:
    nodes: list["Node"] = field(default_factory=list)

    @dataclass(eq=False)
    class Node:
        layer_i: int  # Index of the layer in the Problem
        geom_i: int   # Index of this particular polygon in the layer.shape.geoms list
        is_root: bool
        neighbors: set["Node"] = field(default_factory=set)

    @classmethod
    def create_from_problem(cls, problem: problem.Problem) -> "ConnectivityGraph":
        nodes = []
        lumped_to_nodes: dict[problem.BaseLumped, list["Node"]] = \
            collections.defaultdict(list)
        # First, we construct the individual nodes and figure out what
        # lumped elements are connected to them
        # This is probably computationally reasonably okay,
        # since a point in a polygon check is fast enough and we should not have
        # that many polygons (say, tens of thousands at most hopefully)
        for layer_i, layer in enumerate(problem.layers):
            for geom_i, geom in enumerate(layer.shape.geoms):
                # Create the node
                is_root = False
                connected_elements = []
                for element in problem.lumpeds:
                    intersects = any(
                        geom.intersects(t.point)
                        for t in element.terminals
                    )
                    if not intersects:
                        continue
                    is_root = is_root or element.is_source
                    connected_elements.append(element)

                node = cls.Node(layer_i=layer_i, geom_i=geom_i, is_root=is_root)
                for element in connected_elements:
                    lumped_to_nodes[element].append(node)
                nodes.append(node)

        # Now we need to connect the nodes together
        for element, node_list in lumped_to_nodes.items():
            for node_a, node_b in itertools.combinations(node_list, 2):
                node_a.neighbors.add(node_b)
                node_b.neighbors.add(node_a)

        return cls(nodes=nodes)

    def compute_connected_nodes(self) -> list[Node]:
        """
        Return a list of all nodes that are either root nodes themselves
        or are connected to a root node via any connection.
        """
        open_set = set([n for n in self.nodes if n.is_root])
        closed_set = set()

        while open_set:
            node = open_set.pop()
            closed_set.add(node)
            for neighbor in node.neighbors:
                if neighbor not in closed_set:
                    open_set.add(neighbor)

        return list(closed_set)


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


def laplace_operator(mesh: mesh.Mesh) -> scipy.sparse.dok_matrix:
    """
    Compute the Laplace operator for a given mesh. This is in "mesh-local"
    indices, so the variable indices are given by the mesh.vertices indices.
    """
    N = len(mesh.vertices)

    row_is = []
    col_is = []
    values = []
    diagonal_entries = np.zeros(N, dtype=np.float32)

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

            if ratio == 0:
                # I do not think this happens all that often, except for maybe
                # some degenerate cases
                continue
            
            # Note that we are iterating over everything, so the (k, i) pair gets
            # set in a different iteration
            # The below is equivalent to:
            # L[i, i] -= ratio
            # L[i, k] += ratio
            row_is.append(i)
            col_is.append(k)
            values.append(ratio)
            diagonal_entries[i] -= ratio

    # Insert the diagonal entries
    for i, val in enumerate(diagonal_entries):
        row_is.append(i)
        col_is.append(i)
        values.append(val)

    L = scipy.sparse.coo_matrix((values, (row_is, col_is)), shape=(N, N), dtype=np.float32)

    return L


@dataclass
class VertexIndexer:
    global_index_to_vertex_index: list[tuple[int, int]] = field(default_factory=list)
    mesh_vertex_index_to_global_index: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def create(cls, meshes: list[mesh.Mesh]) -> "VertexIndexer":
        vindex = cls()
        for mesh_idx, msh in enumerate(meshes):
            for vertex_idx, msh in enumerate(msh.vertices):
                global_index = len(vindex.global_index_to_vertex_index)
                vindex.global_index_to_vertex_index.append((mesh_idx, vertex_idx))
                vindex.mesh_vertex_index_to_global_index[(mesh_idx, vertex_idx)] = global_index
        return vindex


def find_connected_layer_geom_indices(connectivity_graph: ConnectivityGraph) -> set[tuple[int, int]]:
    connected_nodes = connectivity_graph.compute_connected_nodes()

    layer_mesh_pairs = set()
    for node in connected_nodes:
        layer_i = node.layer_i
        geom_i = node.geom_i
        layer_mesh_pairs.add((layer_i, geom_i))

    return layer_mesh_pairs


def generate_meshes_for_problem(prob: problem.Problem,
                                mesher: mesh.Mesher,
                                connected_layer_mesh_pairs: set[tuple[int, int]]) -> list[list[mesh.Mesh], list[int]]:
    meshes: list[mesh.Mesh] = []
    mesh_index_to_layer_index: list[int] = []

    for layer_i, layer in enumerate(prob.layers):
        seed_points_in_layer = collect_seed_points(prob, layer)
        for geom_i, geom in enumerate(layer.shape.geoms):
            if (layer_i, geom_i) not in connected_layer_mesh_pairs:
                # This layer is not connected to any lumped elements, skip it
                # for now. Eventually, we may want to just simply triangulate
                # it and pass it to the UI for rendering
                continue
            # This layer is connected to at least one lumped element, so we need to mesh it
            seed_points_in_geom = [
                p for p in seed_points_in_layer
                if layer.shape.geoms[geom_i].intersects(shapely.geometry.Point(p.x, p.y))
            ]

            assert seed_points_in_geom, "No seed points in this geometry, this should not happen"

            m = mesher.poly_to_mesh(
                layer.shape.geoms[geom_i],
                seed_points_in_geom
            )
            meshes.append(m)
            mesh_index_to_layer_index.append(layer_i)

    return meshes, mesh_index_to_layer_index


def make_terminal_index(prob: problem.Problem,
                        meshes: list[mesh.Mesh],
                        mesh_index_to_layer_index: list[int],
                        vindex: VertexIndexer) -> dict[problem.Terminal, int]:
    """
    Create a mapping from terminals to their global indices.
    
    Args:
        prob: The Problem object containing layers and lumped elements
        vindex: The VertexIndexer object containing the global indices
        
    Returns:
        A dictionary mapping terminals to their global indices
    """
    # TODO: This function mildly cursed. We need to somehow consolidate
    # the arguments.

    # First, construct 2D kdtree for the mesh vertices
    layer_to_kdtree = {}
    layer_global_index_and_vertex = {}
    for layer_idx in range(len(prob.layers)):

        layer_vertices = []
        for i_mesh, msh in enumerate(meshes):

            if mesh_index_to_layer_index[i_mesh] != layer_idx:
                continue

            for i_vertex, vertex in enumerate(msh.vertices):
                global_index = vindex.mesh_vertex_index_to_global_index[(i_mesh, i_vertex)]
                layer_vertices.append((global_index, vertex.p))

        if not layer_vertices:
            # No vertices in this layer, skip it
            # In theory, there _could_ be a terminal that attempts to bind to
            # an empty layer. This is going to crash weirdly after, but
            # we are not going to handle it for now.
            continue

        layer_global_index_and_vertex[layer_idx] = layer_vertices
        layer_to_kdtree[layer_idx] = scipy.spatial.KDTree(
            [(p.x, p.y) for _, p in layer_vertices],
            leafsize=32
        )

    terminals = [
        t for lumped in prob.lumpeds for t in lumped.terminals
    ]
    terminal_index: dict[problem.Terminal, int] = {}

    for terminal in terminals:
        # Find the layer index for this terminal
        layer_idx = prob.layers.index(terminal.layer)
        kdtree = layer_to_kdtree[layer_idx]

        # Find the closest vertex to this terminal
        _, vertex_idx_in_kdtree = kdtree.query((terminal.point.x, terminal.point.y), k=1)
        vertex_global_idx = layer_global_index_and_vertex[layer_idx][vertex_idx_in_kdtree][0]

        # Check if this terminal is already in the index
        if terminal in terminal_index and terminal_index[terminal] != vertex_global_idx:
            raise ValueError("Duplicate terminal vertices found, this should not happen.")
        terminal_index[terminal] = vertex_global_idx

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


def process_mesh_laplace_operators(meshes: list[mesh.Mesh],
                                   vindex: VertexIndexer,
                                   L: scipy.sparse.dok_matrix) -> None:
    for i_mesh, msh in enumerate(meshes):
        L_msh = laplace_operator(msh)

        # Glue them together into the global matrix
        for i, j, v in zip(L_msh.row, L_msh.col, L_msh.data):
            global_i = vindex.mesh_vertex_index_to_global_index[(i_mesh, i)]
            global_j = vindex.mesh_vertex_index_to_global_index[(i_mesh, j)]
            # TODO: Is there any possibility that the COO matrix contains duplicates?
            L[global_i, global_j] += v


def produce_layer_solutions(layers: list[problem.Layer],
                            vindex: VertexIndexer,
                            meshes: list[mesh.Mesh],
                            mesh_index_to_layer_index: list[int],
                            v: np.array) -> list[LayerSolution]:
    layer_solutions = []
    for i_layer, layer in enumerate(layers):
        layer_meshes = []
        layer_values = []
        for i_mesh, msh in enumerate(meshes):
            if mesh_index_to_layer_index[i_mesh] != i_layer:
                continue

            # Initialize an empty ZeroForm on this Mesh
            vertex_values = mesh.ZeroForm(msh)
            # and fill it with values from the global value array (solution of the system)
            for i_vertex, vertex in enumerate(msh.vertices):
                global_index = vindex.mesh_vertex_index_to_global_index[(i_mesh, i_vertex)]
                vertex_values[vertex] = v[global_index]

            # Append to the layer values
            layer_values.append(vertex_values)
            layer_meshes.append(msh)

        layer_solutions.append(LayerSolution(meshes=layer_meshes, values=layer_values))

    return layer_solutions


def filter_lumped_elements_in_dead_regions(prob: problem.Problem,
                                           connected_layer_mesh_pairs: set[tuple[int, int]]) -> list[problem.BaseLumped]:

    filtered_lumpeds = []
    for lumped in prob.lumpeds:
        has_a_dead_terminal = False

        for terminal in lumped.terminals:
            layer_idx = prob.layers.index(terminal.layer)
            for geom_i, geom in enumerate(terminal.layer.shape.geoms):
                if not geom.intersects(terminal.point):
                    continue

                if (layer_idx, geom_i) not in connected_layer_mesh_pairs:
                    has_a_dead_terminal = True
                    break

        if not has_a_dead_terminal:
            filtered_lumpeds.append(lumped)

    return filtered_lumpeds


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

    # As a first step, we flatten the Layer-Mesh tree to get a flat list of meshes.
    # We also keep track of which layer each mesh belongs to.
    # This will be needed later when we construct the final solution object.
    log.info("Meshing...")
    connectivity_graph = ConnectivityGraph.create_from_problem(prob)
    connected_layer_mesh_pairs = find_connected_layer_geom_indices(connectivity_graph)
    meshes, mesh_index_to_layer_index = \
        generate_meshes_for_problem(prob, mesher, connected_layer_mesh_pairs)

    # In the next step, we assign a global index to each vertex in every mesh.
    # This is needed since we need to somehow map the vertex indices to the
    # matrix indices in the final system of equations
    log.info("Indexing vertices and terminals")
    vindex = VertexIndexer.create(meshes)

    # And for the last index, we create a mapping from terminals (= the lumped element endpoints)
    # to the vertices they are connected to. This could be done later, but is kept here
    # in order to improve clarity.
    terminal_index = make_terminal_index(prob, meshes, mesh_index_to_layer_index, vindex)

    # Now we need to filter out the lumped elements that are not connected to any
    # of the meshes that we are driving with a source.
    filtered_lumpeds = filter_lumped_elements_in_dead_regions(
        prob, connected_layer_mesh_pairs
    )
    log.info(f"Filtered lumped elements: {len(filtered_lumpeds)}/{len(prob.lumpeds)}")

    # Next, we need to create the matrix of the system of equations and its right hand side.
    
    # Since we are using modified nodal analysis, every voltage source yields
    # an additional variable in the system of equations. Do note that this
    # extra dimension is a _voltage_ on the right hand side and a _current_ in the
    # unknowns space (flipped from the other variables).
    voltage_source_count = sum(
        1 for elem in filtered_lumpeds
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
    N = len(vindex.global_index_to_vertex_index) + voltage_source_count
    L = scipy.sparse.dok_matrix((N, N), dtype=np.float32)
    r = np.zeros(N, dtype=np.float32)

    # Now we compute the Laplace operator for each mesh and insert it into the
    # global L matrix.
    log.info("Constructing the Laplace operators")
    process_mesh_laplace_operators(meshes, vindex, L)

    # Now we need to process the lumped elements, inserting them to the L matrix
    # and the right hand side
    log.info("Processing lumped elements")
    process_lumped_elements(
        filtered_lumpeds,
        terminal_index,
        len(vindex.global_index_to_vertex_index),
        L,
        r
    )

    # Now we need to solve the system of equations
    # We are going to use a direct solver for now
    # TODO: This is a symmetric positive definite matrix, so we can theoretically
    # use something like Conjugate Gradient. Unfortunately, this requires a strictly PD
    # matrix. We can technically get that fairly easily, but it requires forcing a ground
    # for every connected component.
    log.info("Solving the system of equations")
    v = scipy.sparse.linalg.spsolve(L.tocsc(), r)

    # And now we just grab the final solution vector and reconstruct it back
    # into a solution object for easier consumption by the caller.
    log.info("Producing the solution object")
    layer_solutions = produce_layer_solutions(
        prob.layers,
        vindex,
        meshes,
        mesh_index_to_layer_index,
        v
    )

    return Solution(problem=prob, layer_solutions=layer_solutions)
