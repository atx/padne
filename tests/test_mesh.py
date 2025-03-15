import pytest
import numpy as np
from padne.mesh import Vector, Point, Vertex, HalfEdge, Face


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
