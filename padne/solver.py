

import itertools
import logging
import numpy as np
import scipy.sparse
import scipy.spatial
import shapely
import shapely.geometry
import warnings

from dataclasses import dataclass, field

from . import problem, mesh

log = logging.getLogger(__name__)


DTYPE = np.float64


class SolverWarning(Warning):
    """
    A warning that is raised by the solver when it encounters a problem
    that does not prevent it from solving the problem, but may indicate
    a potential issue with the problem definition.
    """
    pass


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
        is_root: bool = False
        neighbors: set["ConnectivityGraph.Node"] = field(default_factory=set)

    @classmethod
    def create_from_problem(cls, problem: problem.Problem) -> "ConnectivityGraph":
        # TODO: This should be refactored and split into multiple functions
        # First, construct an STRTree for every layer
        strtrees = []
        for layer in problem.layers:
            # We need to break up the layer Multipolygons into the individual components
            # for the STRTree construction.
            geoms = [geom for geom in layer.shape.geoms]
            strtree = shapely.strtree.STRtree(geoms)
            strtrees.append(strtree)

        # Next, we construct Node objects for ever layer geometry in the layers
        # that is, a list nodes_by_layers[layer_i][geom_i] gives us the
        # Node that coresponds to the layer_i-th layers geom_i-th geometry
        # object.
        nodes_by_layers = []
        for layer_i, layer in enumerate(problem.layers):
            nodes_by_layers.append(
                [cls.Node(layer_i=layer_i, geom_i=geom_i)
                 for geom_i, geom in enumerate(layer.shape.geoms)]
            )

        # And finally, we walk through each of the networks, figure out
        # which Nodes are connected to each of the Connection and then
        # consider those Nodes connected to each other.
        for network in problem.networks:
            nodes_in_this_network = []
            for conn in network.connections:
                # Find the layer index for this connection
                layer_i = problem.layers.index(conn.layer)
                kdtree = strtrees[layer_i]

                # Find the closest vertex to this connection
                candidates = kdtree.query(conn.point)

                for geom_i in candidates:
                    # Check if this connection is already in the index
                    if not conn.layer.shape.geoms[geom_i].intersects(conn.point):
                        continue
                    intersecting_node = nodes_by_layers[layer_i][geom_i]
                    nodes_in_this_network.append(intersecting_node)

                    if network.has_source:
                        intersecting_node.is_root = True
            # Wire the nodes together
            for node_a, node_b in itertools.combinations(nodes_in_this_network, 2):
                node_a.neighbors.add(node_b)
                node_b.neighbors.add(node_a)

        # And finally flatten the list of nodes into a single list
        nodes = [
            node for xs in nodes_by_layers for node in xs
        ]

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
    for network in problem.networks:
        for conn in network.connections:
            # Check if this connection is on our layer
            if conn.layer == layer:
                seed_points.append(mesh.Point(conn.point.x, conn.point.y))
    return seed_points


def laplace_operator(mesh: mesh.Mesh) -> scipy.sparse.coo_matrix:
    """
    Compute the Laplace operator for a given mesh. This is in "mesh-local"
    indices, so the variable indices are given by the mesh.vertices indices.
    """
    N = len(mesh.vertices)

    row_is = []
    col_is = []
    values = []
    diagonal_entries = np.zeros(N, dtype=DTYPE)

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

    L = scipy.sparse.coo_matrix((values, (row_is, col_is)), shape=(N, N), dtype=DTYPE)

    return L


@dataclass
class VertexIndexer:
    global_index_to_vertex_index: list[tuple[int, int]] = field(default_factory=list)
    mesh_vertex_index_to_global_index: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def create(cls, meshes: list[mesh.Mesh]) -> "VertexIndexer":
        vindex = cls()
        for mesh_idx, msh in enumerate(meshes):
            for vertex_idx, _ in enumerate(msh.vertices):
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
        seed_points_in_layer = [
            shapely.geometry.Point(p.x, p.y)
            for p in collect_seed_points(prob, layer)
        ]
        for geom_i, geom in enumerate(layer.shape.geoms):
            if (layer_i, geom_i) not in connected_layer_mesh_pairs:
                # This layer is not connected to any lumped elements, skip it
                # for now. Eventually, we may want to just simply triangulate
                # it and pass it to the UI for rendering
                continue
            # This layer is connected to at least one lumped element, so we need to mesh it
            seed_points_in_geom = [
                p for p in seed_points_in_layer
                if layer.shape.geoms[geom_i].intersects(p)
            ]

            assert seed_points_in_geom, "No seed points in this geometry, this should not happen"

            m = mesher.poly_to_mesh(
                layer.shape.geoms[geom_i],
                seed_points_in_geom
            )
            meshes.append(m)
            mesh_index_to_layer_index.append(layer_i)

    return meshes, mesh_index_to_layer_index


@dataclass
class NodeIndexer:
    node_to_global_index: dict[problem.NodeID, int] = field(default_factory=dict)
    extra_source_to_global_index: dict[problem.BaseLumped, int] = field(default_factory=dict)
    internal_node_count: int = 0

    @classmethod
    def _construct_kdtrees(cls,
                           prob: problem.Problem,
                           meshes: list[mesh.Mesh],
                           mesh_index_to_layer_index: list[int],
                           vindex: VertexIndexer) -> tuple[dict[int, scipy.spatial.KDTree], dict]:
        """
        Construct a kdtree for each layer in the problem.
        """
        # Maps a layer to a kdtree of _all_ vertices in _all_ meshes in that layer
        layer_to_kdtree = {}
        # Maps a layer to a list of (global_index, vertex) tuples
        # This can be used to retrieve the original vertex from the index that
        # gets returned by the kdtree query
        layer_global_index_and_vertex = {}

        for layer_i in range(len(prob.layers)):
            layer_vertices = []

            for mesh_i, msh in enumerate(meshes):
                if mesh_index_to_layer_index[mesh_i] != layer_i:
                    continue

                for vertex_i, vertex in enumerate(msh.vertices):
                    global_index = vindex.mesh_vertex_index_to_global_index[(mesh_i, vertex_i)]
                    layer_vertices.append((global_index, vertex.p))
            if not layer_vertices:
                # No vertices in this layer, skip it
                # In theory, there _could_ be a terminal that attempts to bind to
                # an empty layer. This is going to crash weirdly after, but
                # we are not going to handle it for now.
                continue

            layer_global_index_and_vertex[layer_i] = layer_vertices
            layer_to_kdtree[layer_i] = scipy.spatial.KDTree(
                [(p.x, p.y) for _, p in layer_vertices],
                leafsize=32,
            )

        return layer_to_kdtree, layer_global_index_and_vertex

    @classmethod
    def create(cls,
               prob: problem.Problem,
               meshes: list[mesh.Mesh],
               mesh_index_to_layer_index: list[int],
               vindex: VertexIndexer) -> "NodeIndexer":

        layer_to_kdtree, layer_global_index_and_vertex = cls._construct_kdtrees(
            prob,
            meshes,
            mesh_index_to_layer_index,
            vindex
        )

        # Contains both the Connection-related nodes and the
        # "virtual" nodes that only live inside a Network
        node_to_global_index = {}

        # First, we index the NodeIDs that are used in a Connection
        connections = [
            conn for network in prob.networks for conn in network.connections
        ]
        for conn in connections:
            layer_i = prob.layers.index(conn.layer)
            kdtree = layer_to_kdtree[layer_i]

            _, vertex_idx_in_kdtree = kdtree.query((conn.point.x, conn.point.y), k=1)
            vertex_global_idx = layer_global_index_and_vertex[layer_i][vertex_idx_in_kdtree][0]
            node = conn.node_id

            # Check that we are not overwriting an existing node with different
            # vertex index. This should never happen in practice
            if node in node_to_global_index and node_to_global_index[node] != vertex_global_idx:
                raise ValueError("Duplicate connection vertices found, this should not happen.")
            node_to_global_index[node] = vertex_global_idx

        # Next, we allocate new indices for all the yet to be allocated nodes
        nodes = [
            node for network in prob.networks for node in network.nodes
            if node not in node_to_global_index
        ]
        internal_node_count = len(nodes)
        i_at = len(vindex.global_index_to_vertex_index)
        for node in nodes:
            node_to_global_index[node] = i_at
            i_at += 1

        # And finally we need to allocate indices for the voltage sources
        # (those need an extra variable)
        extra_sources = [
            elem for network in prob.networks for elem in network.elements
        ]
        extra_source_to_global_index = {}
        for elem in extra_sources:

            if elem.extra_variable_count > 1:
                # TODO: Store a (elem, index) pair in the global index or something
                raise NotImplementedError("Extra variable count > 1 not supported yet")

            for _ in range(elem.extra_variable_count):
                extra_source_to_global_index[elem] = i_at
                i_at += 1

        return cls(
            node_to_global_index=node_to_global_index,
            extra_source_to_global_index=extra_source_to_global_index,
            internal_node_count=internal_node_count
        )


def stamp_network_into_system(network: problem.Network,
                              node_indexer: NodeIndexer,
                              L: scipy.sparse.lil_matrix,
                              r: np.ndarray) -> None:
    for element in network.elements:
        match element:
            case problem.Resistor(a=a, b=b, resistance=resistance):
                i_a = node_indexer.node_to_global_index[a]
                i_b = node_indexer.node_to_global_index[b]

                # (V_b - V_a) / R term
                L[i_a, i_a] -= 1 / resistance
                L[i_a, i_b] += 1 / resistance
                # (V_a - V_b) / R term
                L[i_b, i_b] -= 1 / resistance
                L[i_b, i_a] += 1 / resistance
            case problem.CurrentSource(f=f, t=t, current=current):
                i_f = node_indexer.node_to_global_index[f]
                i_t = node_indexer.node_to_global_index[t]

                # Σ(V_i - V_f) / R =  this
                r[i_f] += current
                # Σ(V_i - V_t) / R = -this
                r[i_t] += -current
            case problem.VoltageSource(p=p, n=n, voltage=voltage):
                i_p = node_indexer.node_to_global_index[p]
                i_n = node_indexer.node_to_global_index[n]
                i_v = node_indexer.extra_source_to_global_index[element]

                # Okay, so, here, the idea is to introduce another variable (I_v).
                # This time, the _unknown variable is the current_
                # and the _right hand side variable is the voltage_.
                # So, effectively, we get these equations:
                # V_p - V_n = voltage
                L[i_v, i_p] = 1
                L[i_v, i_n] = -1
                r[i_v] = voltage
                # add and subtract the I_v current from the equations
                # for the i_p and i_n nodes. Imagine we placed it to the right hand
                # side where source currents live, but since it is an unknown,
                # it has to live in the system matrix
                # TODO: Explain this better
                L[i_p, i_v] = 1
                L[i_n, i_v] = -1
            case problem.VoltageRegulator(v_p=v_p, v_n=v_n,
                                          s_f=s_f, s_t=s_t,
                                          voltage=voltage,
                                          gain=gain):
                i_v_p = node_indexer.node_to_global_index[v_p]
                i_v_n = node_indexer.node_to_global_index[v_n]

                i_s_f = node_indexer.node_to_global_index[s_f]
                i_s_t = node_indexer.node_to_global_index[s_t]

                i_v = node_indexer.extra_source_to_global_index[element]

                # First, we setup the voltage source part. This is identical
                # to the VoltageSource case above.
                L[i_v, i_v_p] = 1
                L[i_v, i_v_n] = -1
                L[i_v_p, i_v] = 1
                L[i_v_n, i_v] = -1
                r[i_v] = voltage

                # Now, we need to take bearings. The variable at the index i_v
                # is the _current_ flowing from the output of the regulator.
                # What we need to do is cause that current to be mirrored
                # at the input of the regulator
                # (i_s_f, i_s_t) pair.
                L[i_s_f, i_v] = gain
                L[i_s_t, i_v] = -gain

            case _:
                raise NotImplementedError(f"Unsupported node type {element}")


def setup_ground_node(i_gnd: int,
                      L: scipy.sparse.lil_matrix,
                      r: np.ndarray):
    # This effectively wires a voltage source of 0V from i_gnd to a
    # virtual (not in the matrix) "ground" node.
    # A more useful way of thinking about this is that:
    # 1. We construct a VoltageSource as above.
    # 2. We imagine that the voltage at its negative terminal is 0V.
    # 3. This means that whatever happens at the corresponding _column_ is going to result in 0 being added anyway
    # 4. This also means there is no point in keeping the corresponding row,
    #    since we already decided that the corresponding voltage is going to be zero
    # 5. So, we drop both the row and column for the ground node.
    # It's worth noting that there is still a "ground current" variable
    # --- this is the variable at -1 index in the system.
    L[-1, i_gnd] = 1
    L[i_gnd, -1] = 1
    r[-1] = 0  # Ground node voltage is 0


def process_mesh_laplace_operators(meshes: list[mesh.Mesh],
                                   conductances: list[float],
                                   vindex: VertexIndexer,
                                   L: scipy.sparse.lil_matrix) -> None:
    for i_mesh, (msh, conductance) in enumerate(zip(meshes, conductances)):
        L_msh = conductance * laplace_operator(msh)

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


def filter_networks_in_dead_regions(prob: problem.Problem,
                                    connected_layer_mesh_pairs: set[tuple[int, int]]) -> list[problem.Network]:
    filtered_networks = []
    for network in prob.networks:
        has_a_dead_terminal = False

        for conn in network.connections:
            layer_idx = prob.layers.index(conn.layer)

            for geom_i, geom in enumerate(conn.layer.shape.geoms):
                if not geom.intersects(conn.point):
                    continue

                if (layer_idx, geom_i) not in connected_layer_mesh_pairs:
                    has_a_dead_terminal = True
                    break

        if not has_a_dead_terminal:
            filtered_networks.append(network)

    return filtered_networks


def find_best_ground_node_index(prob: problem.Problem, node_indexer: NodeIndexer) -> int:
    max_voltage = float('-inf')
    ground_node_index = 0  # Default to the first node

    for network in prob.networks:
        for element in network.elements:
            if not isinstance(element, problem.VoltageSource):
                continue
            # We are looking for the node with the highest voltage
            if element.voltage > max_voltage:
                max_voltage = element.voltage
                ground_node_index = node_indexer.node_to_global_index[element.n]

    log.debug(f"Selected ground node index: {ground_node_index}")

    return ground_node_index


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
    log.info("Indexing vertices and connections")
    vindex = VertexIndexer.create(meshes)

    log.info("Processing lumped element networks")
    # Now we need to filter out the lumped element networks that are not connected
    # to any of the meshes that we are driving with a source.
    filtered_networks = filter_networks_in_dead_regions(
        prob, connected_layer_mesh_pairs
    )
    log.info(f"Filtered networks: {len(filtered_networks)}/{len(prob.networks)}")
    # Next, we construct the _internal_ system of equations for each of the
    # network.
    log.info("Constructing node index for networks")
    node_indexer = NodeIndexer.create(prob, meshes, mesh_index_to_layer_index, vindex)

    # We are solving the equation L * v = r
    # where L is the "laplace operator",
    # v is the voltage vector and
    # r is the right-hand side "source" vector
    # TODO: Maybe we need to force a ground somewhere? Honestly, I
    # feel like as long as the solver can handle it, we can just leave everything
    # floating and let the UI figure out. This can possibly lead to some
    # numerical instability, so it needs more stress testing.
    N = len(vindex.global_index_to_vertex_index) + \
        node_indexer.internal_node_count + \
        len(node_indexer.extra_source_to_global_index) + \
        1  # +1 for the ground node
    L = scipy.sparse.lil_matrix((N, N), dtype=DTYPE)
    r = np.zeros(N, dtype=DTYPE)

    # Now we compute the Laplace operator for each mesh and insert it into the
    # global L matrix.
    log.info("Constructing the Laplace operators")
    # TODO: I am not a big fan of just passing a raw list of conductances
    # around like this...
    mesh_conductances = [
        prob.layers[mesh_index_to_layer_index[i]].conductance
        for i in range(len(meshes))
    ]
    process_mesh_laplace_operators(meshes, mesh_conductances, vindex, L)

    # Now, we process the Networks, directly inserting them in-place into the
    # system matrix. Esthetically, it would be nicer to construct them first
    # and then insert them, but this requires a bit of extra work with regards
    # to handling nodes that have Connections and nodes that do not.
    log.info("Processing networks")
    for network in filtered_networks:
        # First, we need to insert the network elements into the system matrix
        # This is done in-place, so we do not need to worry about the size of the
        # matrix. The only thing we need to worry about is that the indices are
        # correct.
        stamp_network_into_system(network, node_indexer, L, r)

    # TODO: Implement a better way to pick the ground node.
    i_gnd = find_best_ground_node_index(prob, node_indexer)
    setup_ground_node(i_gnd, L, r)

    # Now we need to solve the system of equations
    # We are going to use a direct solver for now
    # TODO: This is a symmetric positive definite matrix, so we can theoretically
    # use something like Conjugate Gradient. Unfortunately, this requires a strictly PD
    # matrix. We can technically get that fairly easily, but it requires forcing a ground
    # for every connected component.
    log.info("Solving the system of equations")
    v = scipy.sparse.linalg.spsolve(L.tocsc(), r)

    if not np.isclose(v[-1], 0):
        # This is a warning, but we still continue to produce the solution object
        # since it may still be useful for the user.
        warnings.warn(
            "Ground node voltage is not zero, this may indicate a problem with the system.",
            SolverWarning
        )

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
