import pytest
import numpy as np
from padne.mesh import Vector, Point, Vertex, HalfEdge, Face, IndexMap, Mesh


class TestVector:
    def test_vector_creation(self):
        v = Vector(1.0, 2.0)
        assert v.dx == 1.0
        assert v.dy == 2.0

    def test_dot_product(self):
        v1 = Vector(1.0, 2.0)
        v2 = Vector(3.0, 4.0)
        assert v1.dot(v2) == 1.0 * 3.0 + 2.0 * 4.0 == 11.0
        
        # Test with zero vector
        zero_vec = Vector(0.0, 0.0)
        assert v1.dot(zero_vec) == 0.0

    def test_vector_addition(self):
        v1 = Vector(1.0, 2.0)
        v2 = Vector(3.0, 4.0)
        result = v1 + v2
        assert isinstance(result, Vector)
        assert result.dx == 4.0
        assert result.dy == 6.0
        
        # Test addition with non-vector
        with pytest.raises(TypeError):
            v1 + 5

    def test_vector_multiplication(self):
        v = Vector(2.0, 3.0)
        result = v * 2.5
        assert isinstance(result, Vector)
        assert result.dx == 5.0
        assert result.dy == 7.5
        
        # Test right multiplication
        result = 2.5 * v
        assert isinstance(result, Vector)
        assert result.dx == 5.0
        assert result.dy == 7.5

    def test_vector_negation(self):
        v = Vector(2.0, -3.0)
        result = -v
        assert isinstance(result, Vector)
        assert result.dx == -2.0
        assert result.dy == 3.0

    def test_vector_cross_product(self):
        v1 = Vector(1.0, 0.0)
        v2 = Vector(0.0, 1.0)
        # Cross product in 2D is scalar
        assert v1 ^ v2 == 1.0
        assert v2 ^ v1 == -1.0
        
        # Parallel vectors have cross product of 0
        v3 = Vector(2.0, 0.0)
        assert v1 ^ v3 == 0.0

    def test_vector_magnitude(self):
        v1 = Vector(3.0, 4.0)
        assert abs(v1) == 5.0
        
        v2 = Vector(0.0, 0.0)
        assert abs(v2) == 0.0


class TestPoint:
    def test_point_creation(self):
        p = Point(1.0, 2.0)
        assert p.x == 1.0
        assert p.y == 2.0

    def test_distance(self):
        p1 = Point(0.0, 0.0)
        p2 = Point(3.0, 4.0)
        assert p1.distance(p2) == 5.0
        
        # Distance to self should be 0
        assert p1.distance(p1) == 0.0

    def test_point_subtraction(self):
        p1 = Point(5.0, 10.0)
        p2 = Point(2.0, 4.0)
        result = p1 - p2
        assert isinstance(result, Vector)
        assert result.dx == 3.0
        assert result.dy == 6.0
        
        # Test subtraction with non-point
        with pytest.raises(TypeError):
            p1 - 5


class TestMeshStructure:
    def test_create_simple_mesh(self):
        # Create a simple triangular mesh
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)
        p3 = Point(0.0, 1.0)
        
        # Create vertices
        v1 = Vertex(p1)
        v2 = Vertex(p2)
        v3 = Vertex(p3)
        
        # Create half-edges
        e12 = HalfEdge(v1)
        e23 = HalfEdge(v2)
        e31 = HalfEdge(v3)
        
        # Link half-edges
        e12.next = e23
        e23.next = e31
        e31.next = e12
        
        # Create face
        f = Face(e12)
        
        # Set faces for edges
        e12.face = f
        e23.face = f
        e31.face = f
        
        # Set outgoing edges for vertices
        v1.out = e12
        v2.out = e23
        v3.out = e31
        
        # Test structure
        assert f.edge == e12
        assert e12.next == e23
        assert e23.next == e31
        assert e31.next == e12
        
        assert e12.origin == v1
        assert e23.origin == v2
        assert e31.origin == v3
        
        assert e12.face == f
        assert e23.face == f
        assert e31.face == f

    def test_half_edge_boundary_property(self):
        v = Vertex(Point(0.0, 0.0))
        e1 = HalfEdge(v, face=None)
        e2 = HalfEdge(v, face=Face())
        
        assert e1.is_boundary == True
        assert e2.is_boundary == False

    def test_half_edge_prev_property(self):
        # Create a simple mesh with twin edges
        p1, p2 = Point(0.0, 0.0), Point(1.0, 0.0)
        v1, v2 = Vertex(p1), Vertex(p2)
        
        e12 = HalfEdge(v1)
        e21 = HalfEdge(v2)
        
        # Set twins
        e12.twin = e21
        e21.twin = e12
        
        # Set next
        e21.next = e12
        
        # Test prev
        assert e12.prev == e21

    def test_face_edges(self):
        # Create a triangular face
        v1 = Vertex(Point(0.0, 0.0))
        v2 = Vertex(Point(1.0, 0.0))
        v3 = Vertex(Point(0.0, 1.0))
        
        e1 = HalfEdge(v1)
        e2 = HalfEdge(v2)
        e3 = HalfEdge(v3)
        
        e1.next = e2
        e2.next = e3
        e3.next = e1
        
        f = Face(e1)
        
        # Test edges iterator
        edges = list(f.edges)
        assert len(edges) == 3
        assert edges[0] == e1
        assert edges[1] == e2
        assert edges[2] == e3

    def test_face_vertices(self):
        # Create a triangular face
        v1 = Vertex(Point(0.0, 0.0))
        v2 = Vertex(Point(1.0, 0.0))
        v3 = Vertex(Point(0.0, 1.0))
        
        e1 = HalfEdge(v1)
        e2 = HalfEdge(v2)
        e3 = HalfEdge(v3)
        
        e1.next = e2
        e2.next = e3
        e3.next = e1
        
        f = Face(e1)
        
        # Test vertices iterator
        vertices = list(f.vertices)
        assert len(vertices) == 3
        assert vertices[0] == v1
        assert vertices[1] == v2
        assert vertices[2] == v3

    def test_vertex_orbit(self):
        # Create a vertex with multiple outgoing edges
        v_center = Vertex(Point(0.0, 0.0))
        v1 = Vertex(Point(1.0, 0.0))
        v2 = Vertex(Point(0.0, 1.0))
        v3 = Vertex(Point(-1.0, 0.0))
        
        # Create half-edges
        e_out1 = HalfEdge(v_center)
        e_out2 = HalfEdge(v_center)
        e_out3 = HalfEdge(v_center)
        
        e_in1 = HalfEdge(v1)
        e_in2 = HalfEdge(v2)
        e_in3 = HalfEdge(v3)
        
        # Set twins
        e_out1.twin = e_in1
        e_in1.twin = e_out1
        
        e_out2.twin = e_in2
        e_in2.twin = e_out2
        
        e_out3.twin = e_in3
        e_in3.twin = e_out3
        
        # Connect the edges
        e_in1.next = e_out2
        e_in2.next = e_out3
        e_in3.next = e_out1
        
        # Set the outgoing edge for center vertex
        v_center.out = e_out1
        
        # Test orbit
        orbit_edges = list(v_center.orbit())
        assert len(orbit_edges) == 3
        assert orbit_edges[0] == e_out1
        assert orbit_edges[1] == e_out2
        assert orbit_edges[2] == e_out3

    def test_vertex_hashability(self):
        # Create vertices
        v1 = Vertex(Point(1.0, 2.0))
        v2 = Vertex(Point(1.0, 2.0))  # Same point, different object
        v3 = Vertex(Point(3.0, 4.0))
        
        # Test hashability
        hash(v1)  # This should not raise an exception
        
        # Test dictionary use
        vertex_dict = {}
        vertex_dict[v1] = "vertex 1"
        vertex_dict[v3] = "vertex 3"
        
        # Verify dictionary behavior
        assert len(vertex_dict) == 2
        assert vertex_dict[v1] == "vertex 1"
        assert vertex_dict[v3] == "vertex 3"
        
        # This depends on equality implementation - if vertices with same point are equal, this will pass
        # Otherwise it would add a new entry
        vertex_dict[v2] = "vertex 2"
        assert len(vertex_dict) == 3  # Since v1 and v2 are different objects

    def test_halfedge_hashability(self):
        # Create vertices and half-edges
        v1 = Vertex(Point(0.0, 0.0))
        v2 = Vertex(Point(1.0, 0.0))
        
        e1 = HalfEdge(v1)
        e2 = HalfEdge(v1)  # Same origin, different object
        e3 = HalfEdge(v2)
        
        # Test hashability
        hash(e1)  # This should not raise an exception
        
        # Test dictionary use
        edge_dict = {}
        edge_dict[e1] = "edge 1"
        edge_dict[e3] = "edge 3"
        
        # Verify dictionary behavior
        assert len(edge_dict) == 2
        assert edge_dict[e1] == "edge 1"
        assert edge_dict[e3] == "edge 3"
        
        # Add another edge with same origin
        edge_dict[e2] = "edge 2"
        assert len(edge_dict) == 3  # Since e1 and e2 are different objects

    def test_face_hashability(self):
        # Create faces
        f1 = Face()
        f2 = Face()
        
        # Test hashability
        hash(f1)  # This should not raise an exception
        
        # Test dictionary use
        face_dict = {}
        face_dict[f1] = "face 1"
        face_dict[f2] = "face 2"
        
        # Verify dictionary behavior
        assert len(face_dict) == 2
        assert face_dict[f1] == "face 1"
        assert face_dict[f2] == "face 2"


class TestMesh:
    def test_initialization(self):
        """Test that a new mesh is correctly initialized."""
        mesh = Mesh()
        assert len(mesh.vertices) == 0
        assert len(mesh.halfedges) == 0
        assert len(mesh.faces) == 0
        assert len(mesh._edge_map) == 0

    def test_make_vertex(self):
        """Test vertex creation and registration."""
        mesh = Mesh()
        p = Point(1.0, 2.0)
        
        v = mesh.make_vertex(p)
        
        assert v.p == p
        assert len(mesh.vertices) == 1
        assert mesh.vertices.to_object(0) == v

    def test_connect_vertices_new(self):
        """Test connecting vertices with no existing connections."""
        mesh = Mesh()
        p1, p2 = Point(0.0, 0.0), Point(1.0, 0.0)
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)
        
        e12 = mesh.connect_vertices(v1, v2)
        
        # Check properties of the edge
        assert e12.origin == v1
        assert e12.twin.origin == v2
        assert e12.twin.twin == e12
        
        # Check registration
        assert len(mesh.halfedges) == 2
        assert e12 in [mesh.halfedges.to_object(i) for i in range(len(mesh.halfedges))]
        assert e12.twin in [mesh.halfedges.to_object(i) for i in range(len(mesh.halfedges))]
        
        # Check edge map
        v1_idx, v2_idx = mesh.vertices.to_index(v1), mesh.vertices.to_index(v2)
        assert mesh._edge_map[(v1_idx, v2_idx)] == e12
        assert mesh._edge_map[(v2_idx, v1_idx)] == e12.twin

    def test_connect_vertices_existing(self):
        """Test that connecting already connected vertices returns the existing edge."""
        mesh = Mesh()
        p1, p2 = Point(0.0, 0.0), Point(1.0, 0.0)
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)
        
        e1 = mesh.connect_vertices(v1, v2)
        e2 = mesh.connect_vertices(v1, v2)
        
        assert e1 == e2
        assert len(mesh.halfedges) == 2  # Only one pair of half-edges created

    def test_triangle_from_vertices(self):
        """Test creating a triangle from three vertices."""
        mesh = Mesh()
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)
        p3 = Point(0.0, 1.0)
        
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)
        v3 = mesh.make_vertex(p3)
        
        face = mesh.triangle_from_vertices(v1, v2, v3)
        
        # Check that face was created properly
        assert isinstance(face, Face)
        assert face.edge.origin == v1
        assert face.edge.next.origin == v2
        assert face.edge.next.next.origin == v3
        assert face.edge.next.next.next == face.edge  # Loop back to start
        
        # Check that edges point to this face
        assert face.edge.face == face
        assert face.edge.next.face == face
        assert face.edge.next.next.face == face
        
        # Check that the face was registered
        assert len(mesh.faces) == 1
        assert mesh.faces.to_object(0) == face
        
        # Check half-edges
        assert len(mesh.halfedges) == 6  # 3 edges * 2 half-edges each

    def test_create_multiple_triangles(self):
        """Test creating multiple connected triangles."""
        mesh = Mesh()
        
        # Create a square with two triangles
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)
        p3 = Point(1.0, 1.0)
        p4 = Point(0.0, 1.0)
        
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)
        v3 = mesh.make_vertex(p3)
        v4 = mesh.make_vertex(p4)
        
        f1 = mesh.triangle_from_vertices(v1, v2, v3)
        f2 = mesh.triangle_from_vertices(v1, v3, v4)
        
        # Check registration
        assert len(mesh.faces) == 2
        assert len(mesh.vertices) == 4
        
        # Since some edges are shared, we expect fewer than 12 half-edges
        # Each triangle adds 3 edges (6 half-edges), but they share 1 edge (2 half-edges)
        assert len(mesh.halfedges) == 10
        
        # Check that edge v1->v3 is shared between the triangles
        edge_v1v3 = None
        for edge in f1.edges:
            if edge.origin == v3 and edge.next.origin == v1:
                edge_v1v3 = edge
                break
        
        assert edge_v1v3 is not None
        assert edge_v1v3.face == f1
        assert edge_v1v3.twin.face == f2

    def test_euler_characteristic(self):
        """Test calculation of the Euler characteristic."""
        mesh = Mesh()
        
        # Empty mesh
        assert mesh.euler_characteristic() == 0
        
        # Single triangle (V=3, E=3, F=1 => 3-3+1 = 1)
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)
        p3 = Point(0.0, 1.0)
        
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)
        v3 = mesh.make_vertex(p3)
        
        mesh.triangle_from_vertices(v1, v2, v3)
        
        assert mesh.euler_characteristic() == 1
        
        # Add a second triangle sharing an edge (V=4, E=5, F=2 => 4-5+2 = 1)
        p4 = Point(1.0, 1.0)
        v4 = mesh.make_vertex(p4)
        mesh.triangle_from_vertices(v2, v4, v3)
        
        assert mesh.euler_characteristic() == 1

    def test_edge_lookup(self):
        """Test that edge lookup works correctly."""
        mesh = Mesh()
        
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)
        p3 = Point(0.0, 1.0)
        
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)
        v3 = mesh.make_vertex(p3)
        
        # Create a triangle with direct edge connections first
        e12 = mesh.connect_vertices(v1, v2)
        e23 = mesh.connect_vertices(v2, v3)
        e31 = mesh.connect_vertices(v3, v1)
        
        # Edge lookup should return the same edges
        assert mesh.connect_vertices(v1, v2) == e12
        assert mesh.connect_vertices(v2, v3) == e23
        assert mesh.connect_vertices(v3, v1) == e31
        
        # Edge lookup should also work in reverse
        assert mesh.connect_vertices(v2, v1).twin == e12
        assert mesh.connect_vertices(v3, v2).twin == e23
        assert mesh.connect_vertices(v1, v3).twin == e31

    def test_connect_vertices_updates_out(self):
        """Test that connect_vertices updates vertex out references."""
        mesh = Mesh()
        
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)
        
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)
        
        # Initially vertices should have no out edge
        assert v1.out is None
        assert v2.out is None
        
        e12 = mesh.connect_vertices(v1, v2)
        
        # After connecting, vertices should have out edges
        # Note: This depends on the implementation details of connect_vertices
        # so we might need to adjust this test if the implementation changes
        assert v1.out is not None
        assert v2.out is not None

    def test_triangle_from_vertices_reuses_edges(self):
        """Test that triangle_from_vertices reuses existing edges."""
        mesh = Mesh()
        
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)
        p3 = Point(0.0, 1.0)
        
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)
        v3 = mesh.make_vertex(p3)
        
        # Create edge first
        e12 = mesh.connect_vertices(v1, v2)
        
        # Create triangle that should reuse this edge
        face = mesh.triangle_from_vertices(v1, v2, v3)
        
        edge_count = len(mesh.halfedges) // 2  # Count actual edges, not half-edges
        
        # Should have exactly 3 edges (not 4) since one was reused
        assert edge_count == 3
        
        # The triangle's first edge should be the reused edge
        assert face.edge == e12

    def test_duplicate_triangles(self):
        """Test that creating identical triangles doesn't duplicate data."""
        mesh = Mesh()
        
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)
        p3 = Point(0.0, 1.0)
        
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)
        v3 = mesh.make_vertex(p3)
        
        f1 = mesh.triangle_from_vertices(v1, v2, v3)
        f2 = mesh.triangle_from_vertices(v1, v2, v3)
        
        # The faces are new objects but should reuse the same edges
        assert f1 != f2
        assert len(mesh.faces) == 2
        assert len(mesh.halfedges) == 6  # Still only 3 edges (6 half-edges)

    def test_non_manifold_edge_handling(self):
        """Test behavior when creating non-manifold edges."""
        mesh = Mesh()
        
        # Create a triangle
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)
        p3 = Point(0.0, 1.0)
        p4 = Point(0.0, -1.0)
        
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)
        v3 = mesh.make_vertex(p3)
        v4 = mesh.make_vertex(p4)
        
        # Create first triangle
        f1 = mesh.triangle_from_vertices(v1, v2, v3)
        
        # Create second triangle sharing an edge but in same direction
        # This creates a non-manifold edge where one edge is used by two faces
        # Implementation details will determine the exact behavior here
        f2 = mesh.triangle_from_vertices(v1, v2, v4)
        
        # Both faces should exist
        assert len(mesh.faces) == 2
        
        # The edge from v1 to v2 should be used by both faces
        # How this is handled depends on implementation, so we can't make
        # many assertions about structure integrity here

    def test_complex_mesh(self):
        """Test creating a more complex mesh structure."""
        mesh = Mesh()
        
        # Create a cube-like structure (without top face)
        # Bottom face
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)
        p3 = Point(1.0, 1.0)
        p4 = Point(0.0, 1.0)
        
        # Top vertices
        p5 = Point(0.0 + 10, 0.0)
        p6 = Point(1.0 + 10, 0.0)
        p7 = Point(1.0 + 10, 1.0)
        p8 = Point(0.0 + 10, 1.0)
        
        # Create vertices
        vertices = []
        for p in [p1, p2, p3, p4, p5, p6, p7, p8]:
            vertices.append(mesh.make_vertex(Point(p.x, p.y)))  # Using only x,y since our mesh is 2D
        
        # Create base face (square with diagonal)
        f1 = mesh.triangle_from_vertices(vertices[0], vertices[1], vertices[2])
        f2 = mesh.triangle_from_vertices(vertices[0], vertices[2], vertices[3])
        
        # Create side faces
        f3 = mesh.triangle_from_vertices(vertices[0], vertices[1], vertices[4])
        f4 = mesh.triangle_from_vertices(vertices[1], vertices[5], vertices[4])
        
        f5 = mesh.triangle_from_vertices(vertices[1], vertices[2], vertices[5])
        f6 = mesh.triangle_from_vertices(vertices[2], vertices[6], vertices[5])
        
        f7 = mesh.triangle_from_vertices(vertices[2], vertices[3], vertices[6])
        f8 = mesh.triangle_from_vertices(vertices[3], vertices[7], vertices[6])
        
        f9 = mesh.triangle_from_vertices(vertices[3], vertices[0], vertices[7])
        f10 = mesh.triangle_from_vertices(vertices[0], vertices[4], vertices[7])
        
        # Check face count
        assert len(mesh.faces) == 10
        
        # Edge count should be less than what we'd need for separate triangles
        # due to shared edges
        assert len(mesh.halfedges) / 2 < 3 * 10  # Less than 30 edges for 10 triangles
        
        # Verify Euler characteristic for this mesh
        # V = 8, E varies based on shared edges, F = 10
        # For a topological cube with one face removed: V - E + F = 2 - 1 = 1
        # (may vary depending on the exact triangle configuration)
        assert mesh.euler_characteristic() == 1


class TestIndexMap:
    def test_initialization(self):
        # Test that an empty IndexMap has length 0
        index_map = IndexMap()
        assert len(index_map) == 0

    def test_add_single_object(self):
        # Test adding a single object
        index_map = IndexMap()
        idx = index_map.add("test")
        assert idx == 0
        assert len(index_map) == 1

    def test_add_multiple_objects(self):
        # Test adding multiple different objects
        index_map = IndexMap()
        idx1 = index_map.add("one")
        idx2 = index_map.add("two")
        idx3 = index_map.add("three")
        
        assert idx1 == 0
        assert idx2 == 1
        assert idx3 == 2
        assert len(index_map) == 3

    def test_add_duplicate_objects(self):
        # Test that adding the same object twice returns the same index
        index_map = IndexMap()
        idx1 = index_map.add("repeated")
        idx2 = index_map.add("repeated")
        
        assert idx1 == idx2
        assert len(index_map) == 1

    def test_to_index(self):
        # Test retrieving index from object
        index_map = IndexMap()
        index_map.add("apple")
        index_map.add("banana")
        
        assert index_map.to_index("apple") == 0
        assert index_map.to_index("banana") == 1
        
        with pytest.raises(KeyError):
            # Non-existent object should raise KeyError
            index_map.to_index("orange")

    def test_to_object(self):
        # Test retrieving object from index
        index_map = IndexMap()
        index_map.add("apple")
        index_map.add("banana")
        
        assert index_map.to_object(0) == "apple"
        assert index_map.to_object(1) == "banana"
        
        with pytest.raises(IndexError):
            # Invalid index should raise IndexError
            index_map.to_object(2)

    def test_items(self):
        # Test the items iterator
        index_map = IndexMap()
        objects = ["one", "two", "three"]
        for obj in objects:
            index_map.add(obj)
        
        items = list(index_map.items())
        assert len(items) == 3
        
        for i, (idx, obj) in enumerate(items):
            assert idx == i
            assert obj == objects[i]

    def test_different_object_types(self):
        # Test with various object types
        index_map = IndexMap()
        
        # String
        str_idx = index_map.add("string")
        assert str_idx == 0
        
        # Integer
        int_idx = index_map.add(42)
        assert int_idx == 1
        
        # Tuple (hashable)
        tuple_idx = index_map.add((1, 2, 3))
        assert tuple_idx == 2
        
        # Custom object (Point)
        point = Point(1.0, 2.0)
        point_idx = index_map.add(point)
        assert point_idx == 3
        
        # Verify all objects
        assert index_map.to_object(str_idx) == "string"
        assert index_map.to_object(int_idx) == 42
        assert index_map.to_object(tuple_idx) == (1, 2, 3)
        assert index_map.to_object(point_idx) == point
