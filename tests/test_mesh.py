import pytest
import numpy as np
import pickle
import random
import shapely.geometry

from padne import kicad

from padne.mesh import Vector, Point, Vertex, HalfEdge, Face, IndexMap, Mesh, \
    Mesher, ZeroForm, OneForm, TwoForm, PolyBoundaryDistanceMap, CGALPolygon
from unittest.mock import patch, Mock

from conftest import for_all_kicad_projects


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

    def test_point_to_shapely(self):
        """Test conversion from Point to shapely.geometry.Point."""
        p = Point(3.5, 4.2)
        shapely_point = p.to_shapely()

        # Verify the type
        assert isinstance(shapely_point, shapely.geometry.Point)

        # Verify the coordinates
        assert shapely_point.x == 3.5
        assert shapely_point.y == 4.2

        # Test with integer coordinates
        p2 = Point(0, 0)
        shapely_point2 = p2.to_shapely()
        assert shapely_point2.x == 0
        assert shapely_point2.y == 0

        # Test with negative coordinates
        p3 = Point(-1.5, -2.7)
        shapely_point3 = p3.to_shapely()
        assert shapely_point3.x == -1.5
        assert shapely_point3.y == -2.7


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
        e1 = HalfEdge(v, face=Face())
        e2 = HalfEdge(v, face=Face(is_boundary=True))

        assert not e1.is_boundary
        assert e2.is_boundary

    def test_face_area(self):
        """Test area calculation for different face configurations."""
        # Create a simple triangular face
        v1 = Vertex(Point(0.0, 0.0))
        v2 = Vertex(Point(2.0, 0.0))
        v3 = Vertex(Point(0.0, 2.0))

        e1 = HalfEdge(v1)
        e2 = HalfEdge(v2)
        e3 = HalfEdge(v3)

        e1.next = e2
        e2.next = e3
        e3.next = e1

        f = Face(e1)

        # Area should be 2.0 (base * height / 2)
        assert f.area == 2.0

        # Test square face
        v1 = Vertex(Point(0.0, 0.0))
        v2 = Vertex(Point(2.0, 0.0))
        v3 = Vertex(Point(2.0, 2.0))
        v4 = Vertex(Point(0.0, 2.0))

        e1 = HalfEdge(v1)
        e2 = HalfEdge(v2)
        e3 = HalfEdge(v3)
        e4 = HalfEdge(v4)

        e1.next = e2
        e2.next = e3
        e3.next = e4
        e4.next = e1

        f = Face(e1)

        # Area should be 4.0 (2 * 2)
        assert f.area == 4.0

        # Test face with negative area (vertices in clockwise order)
        v1 = Vertex(Point(0.0, 0.0))
        v2 = Vertex(Point(0.0, 2.0))
        v3 = Vertex(Point(2.0, 0.0))

        e1 = HalfEdge(v1)
        e2 = HalfEdge(v2)
        e3 = HalfEdge(v3)

        e1.next = e2
        e2.next = e3
        e3.next = e1

        f = Face(e1)

        # Area should still be positive (2.0)
        assert f.area == 2.0

        # Test degenerate face (collinear points)
        v1 = Vertex(Point(0.0, 0.0))
        v2 = Vertex(Point(1.0, 0.0))
        v3 = Vertex(Point(2.0, 0.0))

        e1 = HalfEdge(v1)
        e2 = HalfEdge(v2)
        e3 = HalfEdge(v3)

        e1.next = e2
        e2.next = e3
        e3.next = e1

        f = Face(e1)

        # Area should be 0.0
        assert f.area == 0.0

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


def assert_mesh_topology_okay(mesh):
    # Walk over _every half edge_ and check that the loop it defines has
    # 3 edges if it is part of a face and any finite amount of edges if it
    # is a boundary edge
    for halfedge in mesh.halfedges:
        walk_len = len(list(halfedge.walk()))
        if halfedge.is_boundary:
            assert walk_len >= 3
        else:
            assert walk_len == 3

    # Check that boundary loops have no self intersections (same vertex twice)
    for boundary in mesh.boundaries:
        # Get all vertices in this boundary loop
        boundary_vertices = []
        for edge in boundary.edges:
            boundary_vertices.append(edge.origin)

        # Check that no vertex appears twice
        vertex_set = set()
        for vertex in boundary_vertices:
            assert vertex not in vertex_set, "Boundary loop has self-intersection"
            vertex_set.add(vertex)

    # Validate the Euler's formula for the characteristic
    # For a 2D mesh with b boundaries (including the outer boundary):
    # χ = V - E + F = 2 - b
    # Rearranging: V - E + F - (2 - b) = 0
    num_vertices = len(mesh.vertices)
    num_edges = len(mesh.halfedges) // 2  # Each edge has 2 half-edges
    num_faces = len(mesh.faces)
    num_boundaries = len(mesh.boundaries)

    # Calculate Euler characteristic in two ways
    euler_calc = num_vertices - num_edges + num_faces
    euler_expected = 2 - num_boundaries

    assert euler_calc == mesh.euler_characteristic(), "Calculated Euler characteristic doesn't match mesh.euler_characteristic()"
    assert euler_calc == euler_expected, f"Euler's formula violated: V({num_vertices}) - E({num_edges}) + F({num_faces}) = {euler_calc}, expected {euler_expected} (2 - {num_boundaries})"


def assert_mesh_structure_valid(mesh):
    # Check that every face has 3 edges
    for face in mesh.faces:
        assert len(list(face.edges)) == 3
        assert face.edge is not None
        assert face.edge.face == face

    # Check that every edge has a twin
    for halfedge in mesh.halfedges:
        assert halfedge.twin is not None
        # Every halfedge should have an assigned face
        assert halfedge.face is not None
        # Check that at most one of the halfedges in a twin pair is a boundary
        # edge
        assert not (halfedge.is_boundary and halfedge.twin.is_boundary)

    # Check that every edge has a next and previous edge
    for halfedge in mesh.halfedges:
        assert halfedge.next is not None
        assert halfedge.prev is not None

    # Check that every edge has an origin vertex
    for halfedge in mesh.halfedges:
        assert halfedge.origin is not None


class TestMesh:
    def test_initialization(self):
        """Test that a new mesh is correctly initialized."""
        mesh = Mesh()
        assert len(mesh.vertices) == 0
        assert len(mesh.halfedges) == 0
        assert len(mesh.faces) == 0
        assert len(mesh._edge_map) == 0

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

    def test_connect_vertices_existing_twin(self):
        """Test that connecting vertices with existing edges in opposite direction returns the existing edge."""
        mesh = Mesh()
        p1, p2 = Point(0.0, 0.0), Point(1.0, 0.0)
        v1 = mesh.make_vertex(p1)
        v2 = mesh.make_vertex(p2)

        e1 = mesh.connect_vertices(v1, v2)
        e2 = mesh.connect_vertices(v2, v1)

        assert e1.twin == e2
        assert e1 == e2.twin
        assert len(mesh.halfedges) == 2

    def test_single_triangle(self):
        """Test creating a single triangle mesh."""
        points = [
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(0.0, 1.0)
        ]
        triangles = [(0, 1, 2)]

        mesh = Mesh.from_triangle_soup(points, triangles)

        # Check basic mesh properties
        assert len(mesh.vertices) == 3
        assert len(mesh.faces) == 1
        assert len(mesh.halfedges) == 6  # 3 edges * 2 half-edges each
        assert len(mesh.boundaries) == 1

        # Get the vertices in order they appear in the face
        face = mesh.faces.to_object(0)
        face_verts = list(face.vertices)

        # Check the triangle vertices are in CCW order
        v1, v2, v3 = face_verts
        cross = (v2.p - v1.p) ^ (v3.p - v1.p)
        assert cross > 0, "Triangle vertices not in CCW order"

        # Walk around the face
        assert len(list(face.edge.walk())) == 3
        # Walk around the boundary
        assert len(list(face.edge.twin.walk())) == 3

        assert_mesh_topology_okay(mesh)
        assert_mesh_structure_valid(mesh)

    def test_create_multiple_triangles(self):
        """Test creating multiple connected triangles."""
        # Create a square with two triangles
        points = [
            Point(0.0, 0.0),  # 0
            Point(1.0, 0.0),  # 1
            Point(1.0, 1.0),  # 2
            Point(0.0, 1.0)   # 3
        ]

        triangles = [
            (0, 1, 2),  # First triangle
            (0, 2, 3)   # Second triangle
        ]

        mesh = Mesh.from_triangle_soup(points, triangles)

        # Check registration
        assert len(mesh.vertices) == 4
        assert len(mesh.faces) == 2
        assert len(mesh.boundaries) == 1

        # Since some edges are shared, we expect fewer than 12 half-edges
        # Each triangle adds 3 edges (6 half-edges), but they share 1 edge (2 half-edges)
        assert len(mesh.halfedges) == 10

        # Check that the faces have the expected vertices
        face_points = []
        for face in mesh.faces:
            face_points.append({vertex.p for vertex in face.vertices})

        expected_faces = [
            {points[0], points[1], points[2]},
            {points[0], points[2], points[3]}
        ]

        assert all(face in face_points for face in expected_faces)

        # Check that all faces have 3 edges
        for face in mesh.faces:
            assert len(list(face.edges)) == 3

        # Check boundary
        boundary_edges = [e for e in mesh.halfedges if e.is_boundary]
        assert len(list(boundary_edges[0].walk())) == 4

        assert_mesh_topology_okay(mesh)
        assert_mesh_structure_valid(mesh)

    def test_square_from_four(self):
        """Test creating a square from four triangles around a center point."""
        # Create points for a star-shaped square
        points = [
            Point(0.0, 0.0),   # 0: Center
            Point(0.0, 1.0),   # 1: Top
            Point(-1.0, 0.0),  # 2: Left
            Point(0.0, -1.0),  # 3: Bottom
            Point(1.0, 0.0)    # 4: Right
        ]

        triangles = [
            (0, 4, 1),  # Center, Right, Top
            (0, 1, 2),  # Center, Top, Left
            (0, 2, 3),  # Center, Left, Bottom
            (0, 3, 4)   # Center, Bottom, Right
        ]

        mesh = Mesh.from_triangle_soup(points, triangles)

        # Check basic mesh properties
        assert len(mesh.vertices) == 5
        assert len(mesh.faces) == 4
        assert len(mesh.boundaries) == 1

        # Check Euler characteristic
        # V=5, E=8, F=4 => χ=5-8+4=1
        assert mesh.euler_characteristic() == 1

        # Check boundary
        boundary_edges = [e for e in mesh.halfedges if e.is_boundary]
        assert len(list(boundary_edges[0].walk())) == 4  # Square boundary has 4 edges

        assert_mesh_topology_okay(mesh)
        assert_mesh_structure_valid(mesh)

    def test_euler_characteristic(self):
        """Test calculation of the Euler characteristic."""
        # Empty mesh
        mesh = Mesh()
        assert mesh.euler_characteristic() == 0

        # Single triangle (V=3, E=3, F=1 => 3-3+1 = 1)
        points1 = [Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)]
        triangles1 = [(0, 1, 2)]
        mesh1 = Mesh.from_triangle_soup(points1, triangles1)
        assert mesh1.euler_characteristic() == 1

        # Add a second triangle sharing an edge (V=4, E=5, F=2 => 4-5+2 = 1)
        points2 = [Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0), Point(1.0, 1.0)]
        triangles2 = [(0, 1, 2), (1, 3, 2)]
        mesh2 = Mesh.from_triangle_soup(points2, triangles2)
        assert mesh2.euler_characteristic() == 1

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

    def test_non_manifold_edge_handling(self):
        """Test behavior when creating non-manifold edges."""
        # Create points for a non-manifold construction
        points = [
            Point(0.0, 0.0),  # 0
            Point(1.0, 0.0),  # 1
            Point(0.0, 1.0),  # 2
            Point(0.0, -1.0)  # 3
        ]

        # Create triangles that share an edge with the same orientation
        # This would create a non-manifold edge
        triangles = [
            (0, 1, 2),  # First triangle
            (0, 1, 3)   # Second triangle using same edge (0,1)
        ]

        # The from_triangle_soup method should handle this correctly
        # Either by creating a valid mesh or raising an error

        with pytest.raises(ValueError):
            Mesh.from_triangle_soup(points, triangles)

    def test_complex_mesh(self):
        """Test creating a more complex mesh structure."""
        # Define points for a complex shape
        points = [
            Point(-1.0, -1.0),  # 0: Inner square bottom-left
            Point(1.0, -1.0),   # 1: Inner square bottom-right
            Point(1.0, 1.0),    # 2: Inner square top-right
            Point(-1.0, 1.0),   # 3: Inner square top-left
            Point(-2.0, 0.0),   # 4: Left spike
            Point(0.0, -2.0),   # 5: Bottom spike
            Point(2.0, 0.0),    # 6: Right spike
            Point(0.0, 2.0)     # 7: Top spike
        ]

        # Define triangles for our star-like shape
        triangles = [
            (0, 1, 2),  # Inner square bottom triangle
            (0, 2, 3),  # Inner square top triangle
            (0, 3, 4),  # Left spike
            (1, 0, 5),  # Bottom spike
            (2, 1, 6),  # Right spike
            (3, 2, 7)   # Top spike
        ]

        mesh = Mesh.from_triangle_soup(points, triangles)

        # Check mesh properties
        assert len(mesh.vertices) == 8
        assert len(mesh.faces) == 6
        assert len(mesh.halfedges) == 13 * 2  # Each edge appears twice
        assert len(mesh.boundaries) == 1

        assert mesh.euler_characteristic() == 1

        assert_mesh_topology_okay(mesh)
        assert_mesh_structure_valid(mesh)

    def test_mesh_with_hole(self):
        # Define points for a shape with a hole
        points = [
            Point(0.0, 0.0),   # 0: Outer square bottom-left
            Point(4.0, 0.0),   # 1: Outer square bottom-right
            Point(4.0, 4.0),   # 2: Outer square top-right
            Point(0.0, 4.0),   # 3: Outer square top-left
            Point(1.0, 1.0),   # 4: Inner square bottom-left
            Point(3.0, 1.0),   # 5: Inner square bottom-right
            Point(3.0, 3.0),   # 6: Inner square top-right
            Point(1.0, 3.0),   # 7: Inner square top-left
        ]

        # Define triangles for our shape with hole
        # We need to triangulate around the hole
        triangles = [
            # Bottom side
            (0, 1, 4),
            (1, 5, 4),
            # Right side
            (1, 2, 5),
            (2, 6, 5),
            # Top side
            (2, 3, 6),
            (3, 7, 6),
            # Left side
            (3, 0, 7),
            (0, 4, 7)
        ]

        mesh = Mesh.from_triangle_soup(points, triangles)

        # Check mesh properties
        assert len(mesh.vertices) == 8
        assert len(mesh.faces) == 8  # We have 8 triangles now
        assert len(mesh.halfedges) == 16 * 2  # Each edge appears twice
        assert len(mesh.boundaries) == 2  # Outer boundary and hole boundary

        assert mesh.euler_characteristic() == 0  # For a surface with one hole, χ = 2-2g-b = 0

        assert_mesh_topology_okay(mesh)
        assert_mesh_structure_valid(mesh)



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

    def test_contains(self):
        # Test the 'in' operator
        index_map = IndexMap()
        index_map.add("apple")
        index_map.add("banana")

        assert "apple" in index_map
        assert "banana" in index_map
        assert "orange" not in index_map

        # Test with other types
        index_map = IndexMap()
        index_map.add(1)
        index_map.add((2, 3))

        assert 1 in index_map
        assert (2, 3) in index_map
        assert "string" not in index_map
        assert 2 not in index_map

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


def assert_meshes_equivalent(mesh1: Mesh, mesh2: Mesh):
    """
    Asserts that two mesh objects are equivalent, checking their structure
    and data after a process like pickling and unpickling.
    """
    # Compare basic counts
    assert len(mesh1.vertices) == len(mesh2.vertices)
    assert len(mesh1.halfedges) == len(mesh2.halfedges)
    assert len(mesh1.faces) == len(mesh2.faces)
    assert len(mesh1.boundaries) == len(mesh2.boundaries)
    assert len(mesh1._edge_map) == len(mesh2._edge_map)

    # Compare Vertices
    for i in range(len(mesh1.vertices)):
        v1 = mesh1.vertices.to_object(i)
        v2 = mesh2.vertices.to_object(i)
        assert v1.p == v2.p  # Point data
        if v1.out is not None:
            assert v2.out is not None
            assert mesh1.halfedges.to_index(v1.out) == mesh2.halfedges.to_index(v2.out)
        else:
            assert v2.out is None

    # Compare HalfEdges
    for i in range(len(mesh1.halfedges)):
        h1 = mesh1.halfedges.to_object(i)
        h2 = mesh2.halfedges.to_object(i)

        assert mesh1.vertices.to_index(h1.origin) == mesh2.vertices.to_index(h2.origin)

        assert h1.twin is not None and h2.twin is not None
        assert mesh1.halfedges.to_index(h1.twin) == mesh2.halfedges.to_index(h2.twin)

        assert h1.next is not None and h2.next is not None
        assert mesh1.halfedges.to_index(h1.next) == mesh2.halfedges.to_index(h2.next)

        assert h1.prev is not None and h2.prev is not None
        assert mesh1.halfedges.to_index(h1.prev) == mesh2.halfedges.to_index(h2.prev)

        if h1.face is not None:
            assert h2.face is not None
            assert h1.face.is_boundary == h2.face.is_boundary
            if h1.face.is_boundary:
                assert mesh1.boundaries.to_index(h1.face) == mesh2.boundaries.to_index(h2.face)
            else:
                assert mesh1.faces.to_index(h1.face) == mesh2.faces.to_index(h2.face)
        else:
            assert h2.face is None

    # Compare Faces (interior faces)
    for i in range(len(mesh1.faces)):
        f1 = mesh1.faces.to_object(i)
        f2 = mesh2.faces.to_object(i)
        assert f1.is_boundary == f2.is_boundary # Should be False for mesh.faces
        assert not f1.is_boundary
        if f1.edge is not None:
            assert f2.edge is not None
            assert mesh1.halfedges.to_index(f1.edge) == mesh2.halfedges.to_index(f2.edge)
        else:
            assert f2.edge is None

    # Compare Boundaries (which are also Face objects, but stored in mesh.boundaries)
    for i in range(len(mesh1.boundaries)):
        b1 = mesh1.boundaries.to_object(i)
        b2 = mesh2.boundaries.to_object(i)
        assert b1.is_boundary == b2.is_boundary # Should be True for mesh.boundaries
        assert b1.is_boundary
        if b1.edge is not None:
            assert b2.edge is not None
            assert mesh1.halfedges.to_index(b1.edge) == mesh2.halfedges.to_index(b2.edge)
        else:
            assert b2.edge is None

    # Compare _edge_map by converting HalfEdge values to their indices
    edge_map1_indexed = {key: mesh1.halfedges.to_index(value) for key, value in mesh1._edge_map.items()}
    edge_map2_indexed = {key: mesh2.halfedges.to_index(value) for key, value in mesh2._edge_map.items()}
    assert edge_map1_indexed == edge_map2_indexed

    # Final validation of the unpickled mesh structure
    assert_mesh_topology_okay(mesh2)
    assert_mesh_structure_valid(mesh2)


class TestZeroForm:
    @pytest.fixture
    def simple_mesh(self):
        """Create a simple mesh for testing ZeroForm operations."""
        # Create a simple triangular mesh
        points = [
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(0.0, 1.0),
            Point(1.0, 1.0)
        ]
        triangles = [(0, 1, 2), (1, 3, 2)]

        return Mesh.from_triangle_soup(points, triangles)

    def test_zeroform_initialization(self, simple_mesh):
        """Test basic initialization of a ZeroForm."""
        zf = ZeroForm(simple_mesh)

        # Initial values should be zero
        for vertex in simple_mesh.vertices:
            assert zf[vertex] == 0.0

    def test_zeroform_set_get(self, simple_mesh):
        """Test setting and getting values in a ZeroForm."""
        zf = ZeroForm(simple_mesh)

        # Set some values
        values = {}
        for i, vertex in enumerate(simple_mesh.vertices):
            value = float(i + 1)
            zf[vertex] = value
            values[vertex] = value

        # Check values were set correctly
        for vertex in simple_mesh.vertices:
            assert zf[vertex] == values[vertex]

    def test_zeroform_invalid_vertex(self, simple_mesh):
        """Test that accessing an invalid vertex raises an error."""
        zf = ZeroForm(simple_mesh)

        # Create a vertex that's not in the mesh
        invalid_vertex = Vertex(Point(999.0, 999.0))

        with pytest.raises(KeyError):
            zf[invalid_vertex] = 1.0

        with pytest.raises(KeyError):
            value = zf[invalid_vertex]

    def test_zeroform_addition(self, simple_mesh):
        """Test addition of two ZeroForms."""
        zf1 = ZeroForm(simple_mesh)
        zf2 = ZeroForm(simple_mesh)

        # Set different values in the two forms
        for i, vertex in enumerate(simple_mesh.vertices):
            zf1[vertex] = float(i)
            zf2[vertex] = float(i * 2)

        # Add them
        result = zf1 + zf2

        # Check the result
        for i, vertex in enumerate(simple_mesh.vertices):
            assert result[vertex] == float(i) + float(i * 2)

    def test_zeroform_subtraction(self, simple_mesh):
        """Test subtraction of two ZeroForms."""
        zf1 = ZeroForm(simple_mesh)
        zf2 = ZeroForm(simple_mesh)

        # Set different values in the two forms
        for i, vertex in enumerate(simple_mesh.vertices):
            zf1[vertex] = float(i * 10)
            zf2[vertex] = float(i * 2)

        # Subtract
        result = zf1 - zf2

        # Check the result
        for i, vertex in enumerate(simple_mesh.vertices):
            assert result[vertex] == float(i * 10) - float(i * 2) == float(i * 8)

    def test_zeroform_scalar_multiplication(self, simple_mesh):
        """Test multiplication of a ZeroForm by a scalar."""
        zf = ZeroForm(simple_mesh)

        # Set some values
        for i, vertex in enumerate(simple_mesh.vertices):
            zf[vertex] = float(i)

        # Multiply by scalar
        scalar = 3.5
        result = zf * scalar

        # Check result
        for i, vertex in enumerate(simple_mesh.vertices):
            assert result[vertex] == float(i) * scalar

        # Test right multiplication
        result2 = scalar * zf
        for vertex in simple_mesh.vertices:
            assert result2[vertex] == result[vertex]

    def test_zeroform_division(self, simple_mesh):
        """Test division of a ZeroForm by a scalar."""
        zf = ZeroForm(simple_mesh)

        # Set some values
        for i, vertex in enumerate(simple_mesh.vertices):
            zf[vertex] = float(i * 10)

        # Divide by scalar
        scalar = 2.0
        result = zf / scalar

        # Check result
        for i, vertex in enumerate(simple_mesh.vertices):
            assert result[vertex] == float(i * 10) / scalar

        # Test division by zero
        with pytest.raises(ZeroDivisionError):
            result = zf / 0.0

    def test_zeroform_negation(self, simple_mesh):
        """Test negation of a ZeroForm."""
        zf = ZeroForm(simple_mesh)

        # Set some values
        for i, vertex in enumerate(simple_mesh.vertices):
            zf[vertex] = float(i * 10 - 15)  # Include positive and negative values

        # Negate
        result = -zf

        # Check result
        for i, vertex in enumerate(simple_mesh.vertices):
            assert result[vertex] == -(float(i * 10 - 15))

    def test_zeroform_different_meshes(self, simple_mesh):
        """Test operations between ZeroForms on different meshes."""
        # Create another mesh
        points = [Point(0.0, 0.0), Point(2.0, 0.0), Point(0.0, 2.0)]
        triangles = [(0, 1, 2)]
        other_mesh = Mesh.from_triangle_soup(points, triangles)

        zf1 = ZeroForm(simple_mesh)
        zf2 = ZeroForm(other_mesh)

        # Operations between forms on different meshes should fail
        with pytest.raises(ValueError):
            result = zf1 + zf2

        with pytest.raises(ValueError):
            result = zf1 - zf2


def assert_mesh_minimum_angle(mesh, min_angle):
   """Check that all faces have angles greater or equal to min_angle."""
   min_angle_rad = np.radians(min_angle)
   tolerance = 1e-6  # Tolerance for floating point comparisons

   for face in mesh.faces:
       vertices = list(face.vertices)
       assert len(vertices) == 3, "Face is not a triangle"

       p0, p1, p2 = vertices[0].p, vertices[1].p, vertices[2].p

       # Calculate vectors for the sides
       v01 = p1 - p0
       v02 = p2 - p0
       v12 = p2 - p1

       # Calculate lengths of sides
       len01 = abs(v01)
       len02 = abs(v02)
       len12 = abs(v12)

       # Avoid division by zero for degenerate triangles (should not happen in valid mesh)
       if len01 < tolerance or len02 < tolerance or len12 < tolerance:
           # Or raise an error, or handle as appropriate
           continue

       # Calculate angles using dot product formula: cos(theta) = (a . b) / (|a| * |b|)
       # Angle at p0
       cos_angle0 = v01.dot(v02) / (len01 * len02)
       # Clamp value to [-1, 1] due to potential floating point errors
       cos_angle0 = np.clip(cos_angle0, -1.0, 1.0)
       angle0 = np.arccos(cos_angle0)

       # Angle at p1 (using vectors v10 = -v01 and v12)
       cos_angle1 = (-v01).dot(v12) / (len01 * len12)
       cos_angle1 = np.clip(cos_angle1, -1.0, 1.0)
       angle1 = np.arccos(cos_angle1)

       # Angle at p2 (using vectors v20 = -v02 and v21 = -v12)
       cos_angle2 = (-v02).dot(-v12) / (len02 * len12)
       cos_angle2 = np.clip(cos_angle2, -1.0, 1.0)
       angle2 = np.arccos(cos_angle2)

       # Check if all angles meet the minimum requirement
       assert angle0 >= min_angle_rad - tolerance, f"Angle {np.degrees(angle0)} at vertex {p0} is less than {min_angle}"
       assert angle1 >= min_angle_rad - tolerance, f"Angle {np.degrees(angle1)} at vertex {p1} is less than {min_angle}"
       assert angle2 >= min_angle_rad - tolerance, f"Angle {np.degrees(angle2)} at vertex {p2} is less than {min_angle}"


def assert_mesh_maximum_edge_length(mesh, max_size, tolerance=1e-6):
    """Verify all edges in mesh are at most max_size in length."""
    for hedge in mesh.halfedges:
        # Calculate edge length
        start_vertex = hedge.origin
        end_vertex = hedge.next.origin
        edge_length = start_vertex.p.distance(end_vertex.p)

        # Check constraint (with tolerance for floating-point arithmetic)
        assert edge_length <= max_size + tolerance, \
            f"Edge exceeds maximum size: {edge_length:.6f} > {max_size:.6f} " \
            f"(from {start_vertex.p} to {end_vertex.p})"


class TestMesher:

    def test_simple_square(self):
        """Test meshing a simple square polygon."""
        # Create a square
        square = shapely.geometry.box(0, 0, 1, 1)

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(square)

        # Verify mesh properties
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert len(mesh.halfedges) > 0
        assert_mesh_minimum_angle(mesh, mesher.config.minimum_angle)  # Check minimum angle constraint

        # A simple square should be triangulated into at least 2 triangles
        assert len(mesh.faces) >= 2

    def test_triangle(self):
        """Test meshing a triangle polygon."""
        # Create a triangle
        triangle = shapely.geometry.Polygon([(0, 0), (1, 0), (0, 1)])

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(triangle)

        # Verify mesh properties
        assert isinstance(mesh, Mesh)

        # A simple triangle might be represented as one face
        # or more depending on quality constraints
        assert len(mesh.faces) >= 1
        assert_mesh_minimum_angle(mesh, mesher.config.minimum_angle)

        # Check that all vertices are within the polygon bounds
        for _, vertex in mesh.vertices.items():
            x, y = vertex.p.x, vertex.p.y
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            # Note that this actually fails, one of the vertices is very slightly
            # outside of the bounds due to floating point error
            assert y <= -x + 1 + 1e-6  # This is the line connecting (0,1) and (1,0)

    def test_triangulate_simple_polygon(self):
        """Test triangulation without mesh refinement using relaxed config."""
        # Create a simple L-shaped polygon
        coords = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 0)]
        polygon = shapely.geometry.Polygon(coords)

        mesher = Mesher(Mesher.Config.RELAXED)
        mesh = mesher.poly_to_mesh(polygon)

        # Verify mesh properties
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

        # Since it's just triangulation without refinement,
        # we should have minimal triangles
        # An L-shape needs at least 4 triangles to cover
        assert len(mesh.faces) >= 4

        # All vertices should be within the polygon
        for _, vertex in mesh.vertices.items():
            point = shapely.geometry.Point(vertex.p.x, vertex.p.y)
            assert polygon.contains(point) or polygon.boundary.contains(point)

    def test_triangulate_with_hole(self):
        """Test triangulation of a polygon with a hole using relaxed config."""
        # Create a square with a square hole
        exterior = [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]
        interior = [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]
        polygon = shapely.geometry.Polygon(exterior, [interior])

        mesher = Mesher(Mesher.Config.RELAXED)
        mesh = mesher.poly_to_mesh(polygon)

        # Verify mesh properties
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

        # All vertices should be within the polygon (but not in the hole)
        for _, vertex in mesh.vertices.items():
            point = shapely.geometry.Point(vertex.p.x, vertex.p.y)
            assert polygon.contains(point) or polygon.boundary.contains(point)

    def test_polygon_with_hole(self):
        """Test meshing a polygon with a hole."""
        # Create a square with a square hole
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        interior = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]

        poly_with_hole = shapely.geometry.Polygon(exterior, [interior])
        assert poly_with_hole.interiors  # Verify that the hole exists

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(poly_with_hole)

        # Verify mesh properties
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert mesh.euler_characteristic() == 0
        assert_mesh_minimum_angle(mesh, mesher.config.minimum_angle)

        for vertex in mesh.vertices:
            x = vertex.p.x
            y = vertex.p.y
            assert not (4 < x < 6 and 4 < y < 6)

    def test_polygon_with_multiple_holes(self):
        """Test meshing a polygon with multiple holes."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole1 = [(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]  # First hole
        hole2 = [(6, 6), (8, 6), (8, 8), (6, 8), (6, 6)]  # Second hole

        poly_with_holes = shapely.geometry.Polygon(exterior, [hole1, hole2])
        assert len(poly_with_holes.interiors) == 2  # Verify that both holes exist

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(poly_with_holes)

        # Verify mesh properties
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert_mesh_minimum_angle(mesh, mesher.config.minimum_angle)

        for vertex in mesh.vertices:
            x = vertex.p.x
            y = vertex.p.y

            assert not (2 < x < 4 and 2 < y < 4)
            assert not (6 < x < 8 and 6 < y < 8)

        assert mesh.euler_characteristic() == -1

    def test_concave_polygon(self):
        """Test meshing a concave polygon."""
        # Create a concave polygon (like a 'C' shape)
        concave = shapely.geometry.Polygon([
            (0, 0), (10, 0), (10, 10), (0, 10),
            (0, 8), (8, 8), (8, 2), (0, 2), (0, 0)
        ])

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(concave)

        # Verify mesh properties
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert mesh.euler_characteristic() == 1
        assert_mesh_minimum_angle(mesh, mesher.config.minimum_angle)

        # Check that all vertices are contained within the original polygon
        for vertex in mesh.vertices:
            concave.contains(shapely.geometry.Point(vertex.p.x, vertex.p.y))

    def test_mesh_quality_constraints(self):
        """Test that mesh quality constraints are respected."""
        # Create a square
        square = shapely.geometry.box(0, 0, 1, 1)

        # Create two meshers with different quality constraints
        low_quality_mesher = Mesher(Mesher.Config(minimum_angle=5.0, maximum_size=0.1))
        high_quality_mesher = Mesher(Mesher.Config(minimum_angle=30.0, maximum_size=0.01))

        low_quality_mesh = low_quality_mesher.poly_to_mesh(square)
        high_quality_mesh = high_quality_mesher.poly_to_mesh(square)

        assert_mesh_minimum_angle(low_quality_mesh, low_quality_mesher.config.minimum_angle)
        assert_mesh_minimum_angle(high_quality_mesh, high_quality_mesher.config.minimum_angle)

        # The higher quality mesh should have more triangles due to stricter constraints
        assert len(high_quality_mesh.faces) > len(low_quality_mesh.faces)

    def test_seed_points_simple(self):
        """Test that seed points are correctly incorporated into the mesh."""
        # Create a square
        square = shapely.geometry.box(0, 0, 10, 10)

        # Define seed points inside the square
        seed_points = [
            Point(2.5, 2.5),
            Point(7.5, 7.5),
            Point(5.0, 5.0)
        ]

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(square, seed_points)

        # Verify that the seed points are included in the mesh vertices
        seed_points_found = 0
        for vertex in mesh.vertices:
            for seed_point in seed_points:
                if vertex.p.distance(seed_point) < 1e-6:
                    seed_points_found += 1
                    break

        assert seed_points_found == len(seed_points), "Not all seed points were included in the mesh"

    def test_seed_points_affect_triangulation(self):
        """Test that adding seed points changes the triangulation."""
        # Create a square
        square = shapely.geometry.box(0, 0, 1, 1)

        # Create mesh without seed points
        mesher = Mesher(Mesher.Config(minimum_angle=20.0, maximum_size=0.1))
        mesh_without_seeds = mesher.poly_to_mesh(square, [])

        # Create mesh with a seed point in the middle
        seed_points = [Point(0.23, 0.41)]
        mesh_with_seeds = mesher.poly_to_mesh(square, seed_points)

        # The mesh with the seed point should have more vertices
        assert len(mesh_with_seeds.vertices) > len(mesh_without_seeds.vertices)

        # Verify that the seed point is included
        seed_found = False
        for vertex in mesh_with_seeds.vertices:
            if vertex.p.distance(seed_points[0]) < 1e-6:
                seed_found = True
                break

        assert seed_found, "Seed point was not included in the mesh"

    def test_seed_points_on_boundary(self):
        """Test behavior with seed points on the polygon boundary."""
        # Create a square
        square = shapely.geometry.box(0, 0, 1, 1)

        # Define seed points on the boundary
        boundary_points = [
            Point(0.5, 0.0),  # Bottom edge
            Point(1.0, 0.5),  # Right edge
            Point(0.0, 0.5)   # Left edge
        ]

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(square, boundary_points)

        # Check that boundary points are included
        boundary_points_found = 0
        for vertex in mesh.vertices:
            for point in boundary_points:
                if vertex.p.distance(point) < 1e-6:
                    boundary_points_found += 1
                    break

        assert boundary_points_found == len(boundary_points), "Not all boundary seed points were included"

    def test_seed_points_with_holes(self):
        """Test seed points with a polygon that has holes."""
        # Create a square with a square hole
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        interior = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]
        poly_with_hole = shapely.geometry.Polygon(exterior, [interior])

        # Define seed points in the valid region (not in the hole)
        seed_points = [
            Point(2, 2),    # Bottom-left quadrant
            Point(8, 2),    # Bottom-right quadrant
            Point(8, 8),    # Top-right quadrant
            Point(2, 8),    # Top-left quadrant
        ]

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(poly_with_hole, seed_points)

        # Check that seed points are included
        seed_points_found = 0
        for vertex in mesh.vertices:
            for point in seed_points:
                if abs(vertex.p.x - point.x) < 1e-6 and abs(vertex.p.y - point.y) < 1e-6:
                    seed_points_found += 1
                    break

        assert seed_points_found == len(seed_points), "Not all seed points were included in the mesh with hole"

    def test_clockwise_polygon(self):
        """Test meshing a polygon with clockwise orientation."""
        # Create a polygon with clockwise orientation
        clockwise_polygon = shapely.geometry.Polygon([
            (0, 0),        # Bottom left
            (0, 10),       # Top left
            (10, 10),      # Top right
            (10, 0),       # Bottom right
            (0, 0)         # Back to bottom left
        ], holes=None)

        # Verify the polygon is actually clockwise
        assert not shapely.geometry.LinearRing(clockwise_polygon.exterior.coords).is_ccw

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(clockwise_polygon)

        # Verify mesh properties
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert mesh.euler_characteristic() == 1

        # Verify that all mesh triangles are valid
        assert_mesh_minimum_angle(mesh, mesher.config.minimum_angle)
        assert_mesh_topology_okay(mesh)
        assert_mesh_structure_valid(mesh)

        # Check that vertices are within the original polygon bounds
        for vertex in mesh.vertices:
            x, y = vertex.p.x, vertex.p.y
            assert 0 <= x <= 10
            assert 0 <= y <= 10
            assert clockwise_polygon.intersects(shapely.geometry.Point(x, y))

    def test_tiny_polygon(self):
        """Test meshing a very small polygon."""
        # Create a tiny square
        tiny_square = shapely.geometry.box(0, 0, 1e-6, 1e-6)

        mesher = Mesher(Mesher.Config(maximum_size=1e-7))  # Small enough for the tiny square
        mesh = mesher.poly_to_mesh(tiny_square)

        # Verify that something was meshed
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert mesh.euler_characteristic() == 1
        assert_mesh_minimum_angle(mesh, mesher.config.minimum_angle)

    def test_seed_points_in_polygon_vertex(self):
        seed_points = [
            shapely.geometry.Point(0.0, 0.0),
        ]

        rectangle = shapely.geometry.box(0, 0, 1.0, 1.0)

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(rectangle, seed_points)

        for vertex in mesh.vertices:
            assert 0 <= vertex.p.x <= 1.0
            assert 0 <= vertex.p.y <= 1.0

            for edge in vertex.orbit():
                # Passing a seed point to the mesher that is also a vertex
                # of the mesh caused a malformed mesh to be produced
                assert edge is not None

        assert_mesh_structure_valid(mesh)
        assert_mesh_topology_okay(mesh)

    @pytest.mark.parametrize("max_size", [0.5, 1.0, 2.0])
    @for_all_kicad_projects(include=["two_big_planes",
                                     "degenerate_hole_geometry",
                                     "via_tht_4layer"])
    def test_maximum_edge_length_constraint(self, project, max_size):
        """Test that mesher enforces maximum edge length constraint."""
        # Load project geometry
        problem = kicad.load_kicad_project(project.pro_path)

        # Create mesher with fixed density (no variable sizing)
        mesher = Mesher(Mesher.Config(
            minimum_angle=20.0,
            maximum_size=max_size,
            variable_size_maximum_factor=1.0  # Disable variable density
        ))

        # Test each layer's geometry
        for layer in problem.layers:
            for polygon in layer.shape.geoms:
                mesh = mesher.poly_to_mesh(polygon)
                assert_mesh_maximum_edge_length(mesh, max_size)


class TestMeshPickling:

    def test_pickle_single_triangle_mesh(self):
        """Test pickling and unpickling a mesh with a single triangle."""
        points = [
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(0.0, 1.0)
        ]
        triangles = [(0, 1, 2)]
        original_mesh = Mesh.from_triangle_soup(points, triangles)

        pickled_mesh = pickle.dumps(original_mesh)
        unpickled_mesh = pickle.loads(pickled_mesh)

        assert_meshes_equivalent(original_mesh, unpickled_mesh)
        # Also check Euler characteristic for the unpickled mesh
        assert unpickled_mesh.euler_characteristic() == 1

    def test_references_preserved(self):
        """Test that if we pickle multiple objects simultaneously, the Vertex/HalfEdge/Face objects references get preserved"""
        points = [
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(0.0, 1.0)
        ]
        triangles = [(0, 1, 2)]
        original_mesh = Mesh.from_triangle_soup(points, triangles)

        # Create a few references to the same objects
        v = original_mesh.vertices.to_object(1)
        h = original_mesh.halfedges.to_object(3)
        f = original_mesh.faces.to_object(0)

        pickled_mesh = pickle.dumps((original_mesh, v, h, f))
        (unpickled_mesh, vl, hl, fl) = pickle.loads(pickled_mesh)

        assert unpickled_mesh.vertices.to_object(1) == vl
        assert unpickled_mesh.halfedges.to_object(3) == hl
        assert unpickled_mesh.faces.to_object(0) == fl

    def test_pickle_complex_mesh_with_hole(self):
        """Test pickling and unpickling a more complex mesh with a hole."""
        points = [
            Point(0.0, 0.0),   # 0: Outer square bottom-left
            Point(4.0, 0.0),   # 1: Outer square bottom-right
            Point(4.0, 4.0),   # 2: Outer square top-right
            Point(0.0, 4.0),   # 3: Outer square top-left
            Point(1.0, 1.0),   # 4: Inner square bottom-left
            Point(3.0, 1.0),   # 5: Inner square bottom-right
            Point(3.0, 3.0),   # 6: Inner square top-right
            Point(1.0, 3.0),   # 7: Inner square top-left
        ]
        triangles = [
            (0, 1, 4), (1, 5, 4), # Bottom strip
            (1, 2, 5), (2, 6, 5), # Right strip
            (2, 3, 6), (3, 7, 6), # Top strip
            (3, 0, 7), (0, 4, 7)  # Left strip
        ]
        original_mesh = Mesh.from_triangle_soup(points, triangles)

        pickled_mesh = pickle.dumps(original_mesh)
        unpickled_mesh = pickle.loads(pickled_mesh)

        assert_meshes_equivalent(original_mesh, unpickled_mesh)
        # Also check Euler characteristic for the unpickled mesh
        # For a surface with one hole (genus 1, 2 boundaries if we count outer and inner)
        # V - E + F = 2 - 2g - b_loops = 2 - 2*0 - num_boundaries (if planar embedding)
        # Here, num_boundaries = 2 (outer, inner hole)
        # So, V - E + F = 2 - 2 = 0
        assert unpickled_mesh.euler_characteristic() == 0

    def test_seed_points_in_hole_vertex(self):
        seed_points = [
            shapely.geometry.Point(4, 4),
        ]

        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        interior = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]
        poly_with_hole = shapely.geometry.Polygon(exterior, [interior])

        mesher = Mesher()
        mesh = mesher.poly_to_mesh(poly_with_hole, seed_points)

        assert_mesh_structure_valid(mesh)
        assert_mesh_topology_okay(mesh)

    def test_duplicate_seed_points(self):
        """Test that duplicate seed points are handled correctly."""
        # Create a square polygon for testing
        square = shapely.geometry.box(0, 0, 1, 1)

        # Create a list of seed points with duplicates
        seed_points = [
            Point(0.25, 0.25),  # First seed point
            Point(0.75, 0.75),  # Second seed point
            Point(0.25, 0.25),  # Duplicate of first seed point
            Point(0.5, 0.5),  # Another seed point
            Point(0.75, 0.75),  # Duplicate of second seed point
            Point(0.25, 0.25),   # Another duplicate of first seed point
        ]

        # Create a mesher and generate mesh with duplicate seed points
        mesher = Mesher()
        mesh = mesher.poly_to_mesh(square, seed_points)

        # Verify the mesh is valid
        assert isinstance(mesh, Mesh)
        assert_mesh_structure_valid(mesh)
        assert_mesh_topology_okay(mesh)

        # Count how many vertices are near our seed points
        for seed_point in [Point(0.25, 0.25), Point(0.75, 0.75)]:
            seed_point_vertices = 0
            for vertex in mesh.vertices:
                if vertex.p.distance(seed_point) < 1e-6:
                    seed_point_vertices += 1
                    break
            assert seed_point_vertices == 1, "Duplicate seed points weren't properly handled"

    def test_pickle_large_mesh(self):
        # This test should construct a large-ish mesh and try to pickle it
        # to ensure that the pickling process does not fail on maximum recursion
        # depth.

        # Create a square polygon
        square = shapely.geometry.box(0, 0, 10, 10)

        # Use a Mesher to create a reasonably complex mesh
        # A smaller maximum_size will result in more triangles.
        mesher = Mesher(Mesher.Config(minimum_angle=20.0, maximum_size=0.1))
        original_mesh = mesher.poly_to_mesh(square)

        # Ensure the mesh is non-trivial
        assert len(original_mesh.faces) > 50, "Generated mesh is too small for a 'large' mesh test."

        pickled_mesh = pickle.dumps(original_mesh)
        unpickled_mesh = pickle.loads(pickled_mesh)

        assert_meshes_equivalent(original_mesh, unpickled_mesh)
        # For a simple square, Euler characteristic should be 1
        assert unpickled_mesh.euler_characteristic() == 1


class TestOneForm:
    @pytest.fixture
    def simple_mesh(self):
        """Create a simple mesh for testing OneForm operations."""
        # Create a simple triangular mesh
        points = [
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(0.0, 1.0),
            Point(1.0, 1.0)
        ]
        triangles = [(0, 1, 2), (1, 3, 2)]

        return Mesh.from_triangle_soup(points, triangles)

    def test_oneform_initialization(self, simple_mesh):
        """Test basic initialization of a OneForm."""
        of = OneForm(simple_mesh)

        # Initial values should be zero
        for hedge in simple_mesh.halfedges:
            assert of[hedge] == 0.0

    def test_oneform_antisymmetry(self, simple_mesh):
        """Test that OneForm maintains antisymmetry: form[hedge] == -form[hedge.twin]."""
        of = OneForm(simple_mesh)

        # Set values and test antisymmetry
        for i, hedge in enumerate(simple_mesh.halfedges):
            # Only set values for half the edges to avoid setting both hedge and hedge.twin
            if i % 2 == 0:
                value = float(i + 1)
                of[hedge] = value

                # Verify antisymmetry immediately
                assert of[hedge] == value
                assert of[hedge.twin] == -value
                assert abs(of[hedge] + of[hedge.twin]) < 1e-12

    def test_oneform_set_get(self, simple_mesh):
        """Test setting and getting values in a OneForm."""
        of = OneForm(simple_mesh)

        # Collect values we're setting to verify later
        set_values = {}

        for i, hedge in enumerate(simple_mesh.halfedges):
            # Only set on canonical hedges to avoid conflicts
            if i % 2 == 0:
                value = float(i * 2.5)
                of[hedge] = value
                set_values[hedge] = value

        # Verify all set values are correct
        for hedge, expected_value in set_values.items():
            assert of[hedge] == expected_value
            assert of[hedge.twin] == -expected_value

    def test_oneform_invalid_halfedge(self, simple_mesh):
        """Test that accessing an invalid half-edge raises an error."""
        of = OneForm(simple_mesh)

        # Create a half-edge that's not in the mesh
        invalid_vertex = Vertex(Point(999.0, 999.0))
        invalid_hedge = HalfEdge(invalid_vertex)

        with pytest.raises(KeyError):
            of[invalid_hedge] = 1.0

        with pytest.raises(KeyError):
            value = of[invalid_hedge]

    def test_oneform_addition(self, simple_mesh):
        """Test addition of two OneForms."""
        of1 = OneForm(simple_mesh)
        of2 = OneForm(simple_mesh)

        # Set different values in the two forms
        for i, hedge in enumerate(simple_mesh.halfedges):
            if i % 2 == 0:  # Only set canonical edges
                of1[hedge] = float(i)
                of2[hedge] = float(i * 2)

        # Add them
        result = of1 + of2

        # Check the result maintains antisymmetry and correct values
        for i, hedge in enumerate(simple_mesh.halfedges):
            if i % 2 == 0:
                expected = float(i) + float(i * 2)
                assert result[hedge] == expected
                assert result[hedge.twin] == -expected

    def test_oneform_subtraction(self, simple_mesh):
        """Test subtraction of two OneForms."""
        of1 = OneForm(simple_mesh)
        of2 = OneForm(simple_mesh)

        # Set different values in the two forms
        for i, hedge in enumerate(simple_mesh.halfedges):
            if i % 2 == 0:
                of1[hedge] = float(i * 10)
                of2[hedge] = float(i * 3)

        # Subtract
        result = of1 - of2

        # Check the result
        for i, hedge in enumerate(simple_mesh.halfedges):
            if i % 2 == 0:
                expected = float(i * 10) - float(i * 3)
                assert result[hedge] == expected
                assert result[hedge.twin] == -expected

    def test_oneform_scalar_multiplication(self, simple_mesh):
        """Test multiplication of a OneForm by a scalar."""
        of = OneForm(simple_mesh)

        # Set some values
        for i, hedge in enumerate(simple_mesh.halfedges):
            if i % 2 == 0:
                of[hedge] = float(i)

        # Multiply by scalar
        scalar = 3.5
        result = of * scalar

        # Check result
        for i, hedge in enumerate(simple_mesh.halfedges):
            if i % 2 == 0:
                expected = float(i) * scalar
                assert result[hedge] == expected
                assert result[hedge.twin] == -expected

        # Test right multiplication
        result2 = scalar * of
        for hedge in simple_mesh.halfedges:
            assert result2[hedge] == result[hedge]

    def test_oneform_division(self, simple_mesh):
        """Test division of a OneForm by a scalar."""
        of = OneForm(simple_mesh)

        # Set some values
        for i, hedge in enumerate(simple_mesh.halfedges):
            if i % 2 == 0:
                of[hedge] = float(i * 10)

        # Divide by scalar
        scalar = 2.0
        result = of / scalar

        # Check result
        for i, hedge in enumerate(simple_mesh.halfedges):
            if i % 2 == 0:
                expected = float(i * 10) / scalar
                assert result[hedge] == expected
                assert result[hedge.twin] == -expected

        # Test division by zero
        with pytest.raises(ZeroDivisionError):
            result = of / 0.0

    def test_oneform_negation(self, simple_mesh):
        """Test negation of a OneForm."""
        of = OneForm(simple_mesh)

        # Set some values
        for i, hedge in enumerate(simple_mesh.halfedges):
            if i % 2 == 0:
                of[hedge] = float(i * 10 - 15)  # Include positive and negative values

        # Negate
        result = -of

        # Check result
        for i, hedge in enumerate(simple_mesh.halfedges):
            if i % 2 == 0:
                expected = -(float(i * 10 - 15))
                assert result[hedge] == expected
                assert result[hedge.twin] == -expected

    def test_oneform_different_meshes(self, simple_mesh):
        """Test operations between OneForms on different meshes."""
        # Create another mesh
        points = [Point(0.0, 0.0), Point(2.0, 0.0), Point(0.0, 2.0)]
        triangles = [(0, 1, 2)]
        other_mesh = Mesh.from_triangle_soup(points, triangles)

        of1 = OneForm(simple_mesh)
        of2 = OneForm(other_mesh)

        # Operations between forms on different meshes should fail
        with pytest.raises(ValueError):
            result = of1 + of2

        with pytest.raises(ValueError):
            result = of1 - of2


class TestExteriorDerivative:
    @pytest.fixture
    def triangle_mesh(self):
        """Create a simple triangle mesh for testing exterior derivative."""
        points = [
            Point(0.0, 0.0),  # v0
            Point(1.0, 0.0),  # v1
            Point(0.0, 1.0),  # v2
        ]
        triangles = [(0, 1, 2)]
        return Mesh.from_triangle_soup(points, triangles)

    @pytest.fixture
    def square_mesh(self):
        """Create a square mesh for testing exterior derivative."""
        points = [
            Point(0.0, 0.0),  # v0
            Point(1.0, 0.0),  # v1
            Point(0.0, 1.0),  # v2
            Point(1.0, 1.0),  # v3
        ]
        triangles = [(0, 1, 2), (1, 3, 2)]
        return Mesh.from_triangle_soup(points, triangles)

    def test_exterior_derivative_constant_function(self, triangle_mesh):
        """Test exterior derivative of a constant function (should be zero)."""
        zf = ZeroForm(triangle_mesh)

        # Set constant value
        constant_value = 5.0
        for vertex in triangle_mesh.vertices:
            zf[vertex] = constant_value

        # Compute exterior derivative
        df = zf.d()

        # For constant function, gradient should be zero everywhere
        for hedge in triangle_mesh.halfedges:
            assert abs(df[hedge]) < 1e-12

    def test_exterior_derivative_linear_function_x(self, square_mesh):
        """Test exterior derivative of f(x,y) = x."""
        zf = ZeroForm(square_mesh)

        # Set f(x,y) = x at each vertex
        for vertex in square_mesh.vertices:
            zf[vertex] = vertex.p.x

        # Compute exterior derivative
        df = zf.d()

        # For f(x,y) = x, we expect df/dx = 1, df/dy = 0
        # So on horizontal edges (dy=0), df should be ±1 depending on orientation
        # On vertical edges (dx=0), df should be 0

        tolerance = 1e-12
        for hedge in square_mesh.halfedges:
            origin = hedge.origin.p
            target = hedge.twin.origin.p

            # Calculate the actual difference
            expected_value = target.x - origin.x
            actual_value = df[hedge]

            assert abs(actual_value - expected_value) < tolerance

            # Verify antisymmetry
            assert abs(df[hedge] + df[hedge.twin]) < tolerance

    def test_exterior_derivative_linear_function_y(self, square_mesh):
        """Test exterior derivative of f(x,y) = y."""
        zf = ZeroForm(square_mesh)

        # Set f(x,y) = y at each vertex
        for vertex in square_mesh.vertices:
            zf[vertex] = vertex.p.y

        # Compute exterior derivative
        df = zf.d()

        # For f(x,y) = y, we expect df/dx = 0, df/dy = 1
        tolerance = 1e-12
        for hedge in square_mesh.halfedges:
            origin = hedge.origin.p
            target = hedge.twin.origin.p

            # Calculate the actual difference
            expected_value = target.y - origin.y
            actual_value = df[hedge]

            assert abs(actual_value - expected_value) < tolerance

            # Verify antisymmetry
            assert abs(df[hedge] + df[hedge.twin]) < tolerance

    def test_exterior_derivative_linear_function_xy(self, square_mesh):
        """Test exterior derivative of f(x,y) = x + 2*y."""
        zf = ZeroForm(square_mesh)

        # Set f(x,y) = x + 2*y at each vertex
        for vertex in square_mesh.vertices:
            zf[vertex] = vertex.p.x + 2.0 * vertex.p.y

        # Compute exterior derivative
        df = zf.d()

        # For f(x,y) = x + 2*y, gradient is (1, 2)
        # So df[edge] = (target_x + 2*target_y) - (origin_x + 2*origin_y)
        #             = (target_x - origin_x) + 2*(target_y - origin_y)
        tolerance = 1e-12
        for hedge in square_mesh.halfedges:
            origin = hedge.origin.p
            target = hedge.twin.origin.p

            expected_value = (target.x - origin.x) + 2.0 * (target.y - origin.y)
            actual_value = df[hedge]

            assert abs(actual_value - expected_value) < tolerance

            # Verify antisymmetry
            assert abs(df[hedge] + df[hedge.twin]) < tolerance

    def test_exterior_derivative_quadratic_function(self, triangle_mesh):
        """Test exterior derivative of f(x,y) = x² + y²."""
        zf = ZeroForm(triangle_mesh)

        # Set f(x,y) = x² + y² at each vertex
        for vertex in triangle_mesh.vertices:
            x, y = vertex.p.x, vertex.p.y
            zf[vertex] = x*x + y*y

        # Compute exterior derivative
        df = zf.d()

        # The discrete gradient should approximate the continuous gradient (2x, 2y)
        # Verify that antisymmetry is maintained
        for hedge in triangle_mesh.halfedges:
            assert abs(df[hedge] + df[hedge.twin]) < 1e-12

    def test_exterior_derivative_antisymmetry_preserved(self, square_mesh):
        """Test that exterior derivative always preserves antisymmetry."""
        zf = ZeroForm(square_mesh)

        # Set some arbitrary function values
        import random
        random.seed(42)  # For reproducible tests
        for vertex in square_mesh.vertices:
            zf[vertex] = random.uniform(-10, 10)

        # Compute exterior derivative
        df = zf.d()

        # Verify antisymmetry for all edges
        for hedge in square_mesh.halfedges:
            assert abs(df[hedge] + df[hedge.twin]) < 1e-12

    def test_exterior_derivative_linearity(self, square_mesh):
        """Test that exterior derivative is linear: d(af + bg) = a*df + b*dg."""
        # Create two functions
        f = ZeroForm(square_mesh)
        g = ZeroForm(square_mesh)

        # Set f(x,y) = x and g(x,y) = y
        for vertex in square_mesh.vertices:
            f[vertex] = vertex.p.x
            g[vertex] = vertex.p.y

        # Scalars
        a, b = 3.0, -2.0

        # Compute d(af + bg)
        combined = a * f + b * g
        d_combined = combined.d()

        # Compute a*df + b*dg
        df = f.d()
        dg = g.d()
        expected = a * df + b * dg

        # Compare
        tolerance = 1e-12
        for hedge in square_mesh.halfedges:
            assert abs(d_combined[hedge] - expected[hedge]) < tolerance

    def test_exterior_derivative_zero_form(self, triangle_mesh):
        """Test exterior derivative of zero function."""
        zf = ZeroForm(triangle_mesh)
        # Leave all values as zero (default)

        df = zf.d()

        # Derivative of zero should be zero
        for hedge in triangle_mesh.halfedges:
            assert abs(df[hedge]) < 1e-12


class TestTwoForm:
    @pytest.fixture
    def simple_mesh(self):
        """Create a simple mesh for testing TwoForm operations."""
        # Create a simple triangular mesh
        points = [
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(0.0, 1.0),
            Point(1.0, 1.0)
        ]
        triangles = [(0, 1, 2), (1, 3, 2)]

        return Mesh.from_triangle_soup(points, triangles)

    def test_twoform_initialization(self, simple_mesh):
        """Test basic initialization of a TwoForm."""
        tf = TwoForm(simple_mesh)

        # Initial values should be zero
        for face in simple_mesh.faces:
            assert tf[face] == 0.0

        # Boundary faces should always return 0.0
        for boundary in simple_mesh.boundaries:
            assert tf[boundary] == 0.0

    def test_twoform_set_get(self, simple_mesh):
        """Test setting and getting values in a TwoForm."""
        tf = TwoForm(simple_mesh)

        # Set some values on interior faces
        values = {}
        for i, face in enumerate(simple_mesh.faces):
            value = float(i + 1) * 2.5
            tf[face] = value
            values[face] = value

        # Check values were set correctly
        for face in simple_mesh.faces:
            assert tf[face] == values[face]

        # Boundary faces should always return 0.0 and cannot be set
        for boundary in simple_mesh.boundaries:
            assert tf[boundary] == 0.0
            with pytest.raises(KeyError):
                tf[boundary] = 5.0

    def test_twoform_invalid_face(self, simple_mesh):
        """Test that accessing an invalid face raises an error."""
        tf = TwoForm(simple_mesh)

        # Create a face that's not in the mesh
        invalid_vertex = Vertex(Point(999.0, 999.0))
        invalid_hedge = HalfEdge(invalid_vertex)
        invalid_face = Face(invalid_hedge)

        with pytest.raises(KeyError):
            tf[invalid_face] = 1.0

        with pytest.raises(KeyError):
            value = tf[invalid_face]

    def test_twoform_addition(self, simple_mesh):
        """Test addition of two TwoForms."""
        tf1 = TwoForm(simple_mesh)
        tf2 = TwoForm(simple_mesh)

        # Set different values in the two forms
        for i, face in enumerate(simple_mesh.faces):
            tf1[face] = float(i)
            tf2[face] = float(i * 2)

        # Add them
        result = tf1 + tf2

        # Check the result on interior faces
        for i, face in enumerate(simple_mesh.faces):
            assert result[face] == float(i) + float(i * 2)

        # Boundary faces should remain 0.0 after addition
        for boundary in simple_mesh.boundaries:
            assert result[boundary] == 0.0

    def test_twoform_subtraction(self, simple_mesh):
        """Test subtraction of two TwoForms."""
        tf1 = TwoForm(simple_mesh)
        tf2 = TwoForm(simple_mesh)

        # Set different values in the two forms
        for i, face in enumerate(simple_mesh.faces):
            tf1[face] = float(i * 10)
            tf2[face] = float(i * 2)

        # Subtract
        result = tf1 - tf2

        # Check the result on interior faces
        for i, face in enumerate(simple_mesh.faces):
            assert result[face] == float(i * 10) - float(i * 2) == float(i * 8)

        # Boundary faces should remain 0.0 after subtraction
        for boundary in simple_mesh.boundaries:
            assert result[boundary] == 0.0

    def test_twoform_scalar_multiplication(self, simple_mesh):
        """Test multiplication of a TwoForm by a scalar."""
        tf = TwoForm(simple_mesh)

        # Set some values
        for i, face in enumerate(simple_mesh.faces):
            tf[face] = float(i)

        # Multiply by scalar
        scalar = 3.5
        result = tf * scalar

        # Check result on interior faces
        for i, face in enumerate(simple_mesh.faces):
            assert result[face] == float(i) * scalar

        # Boundary faces should remain 0.0 after multiplication
        for boundary in simple_mesh.boundaries:
            assert result[boundary] == 0.0

        # Test right multiplication
        result2 = scalar * tf
        for face in simple_mesh.faces:
            assert result2[face] == result[face]
        for boundary in simple_mesh.boundaries:
            assert result2[boundary] == 0.0

    def test_twoform_division(self, simple_mesh):
        """Test division of a TwoForm by a scalar."""
        tf = TwoForm(simple_mesh)

        # Set some values
        for i, face in enumerate(simple_mesh.faces):
            tf[face] = float(i * 10)

        # Divide by scalar
        scalar = 2.0
        result = tf / scalar

        # Check result on interior faces
        for i, face in enumerate(simple_mesh.faces):
            assert result[face] == float(i * 10) / scalar

        # Boundary faces should remain 0.0 after division
        for boundary in simple_mesh.boundaries:
            assert result[boundary] == 0.0

        # Test division by zero
        with pytest.raises(ZeroDivisionError):
            result = tf / 0.0

    def test_twoform_negation(self, simple_mesh):
        """Test negation of a TwoForm."""
        tf = TwoForm(simple_mesh)

        # Set some values
        for i, face in enumerate(simple_mesh.faces):
            tf[face] = float(i * 10 - 15)  # Include positive and negative values

        # Negate
        result = -tf

        # Check result on interior faces
        for i, face in enumerate(simple_mesh.faces):
            assert result[face] == -(float(i * 10 - 15))

        # Boundary faces should remain 0.0 after negation
        for boundary in simple_mesh.boundaries:
            assert result[boundary] == 0.0

    def test_twoform_different_meshes(self, simple_mesh):
        """Test operations between TwoForms on different meshes."""
        # Create another mesh
        points = [Point(0.0, 0.0), Point(2.0, 0.0), Point(0.0, 2.0)]
        triangles = [(0, 1, 2)]
        other_mesh = Mesh.from_triangle_soup(points, triangles)

        tf1 = TwoForm(simple_mesh)
        tf2 = TwoForm(other_mesh)

        # Operations between forms on different meshes should fail
        with pytest.raises(ValueError):
            result = tf1 + tf2

        with pytest.raises(ValueError):
            result = tf1 - tf2

    def test_twoform_with_complex_mesh(self):
        """Test TwoForm with a more complex mesh structure."""
        # Create a mesh with holes (more complex topology)
        points = [
            Point(0.0, 0.0),   # 0: Outer square bottom-left
            Point(4.0, 0.0),   # 1: Outer square bottom-right
            Point(4.0, 4.0),   # 2: Outer square top-right
            Point(0.0, 4.0),   # 3: Outer square top-left
            Point(1.0, 1.0),   # 4: Inner square bottom-left
            Point(3.0, 1.0),   # 5: Inner square bottom-right
            Point(3.0, 3.0),   # 6: Inner square top-right
            Point(1.0, 3.0),   # 7: Inner square top-left
        ]
        triangles = [
            (0, 1, 4), (1, 5, 4), # Bottom strip
            (1, 2, 5), (2, 6, 5), # Right strip
            (2, 3, 6), (3, 7, 6), # Top strip
            (3, 0, 7), (0, 4, 7)  # Left strip
        ]
        complex_mesh = Mesh.from_triangle_soup(points, triangles)

        tf = TwoForm(complex_mesh)

        # Test that we can set values on interior faces only
        for i, face in enumerate(complex_mesh.faces):
            tf[face] = float(i * 3)

        # Verify boundary faces cannot be set and always return 0.0
        for boundary in complex_mesh.boundaries:
            assert tf[boundary] == 0.0
            with pytest.raises(KeyError):
                tf[boundary] = 100.0

        # Test arithmetic operations on complex mesh
        tf2 = TwoForm(complex_mesh)
        for face in complex_mesh.faces:
            tf2[face] = 5.0

        result = tf + tf2
        assert isinstance(result, TwoForm)

        # Verify arithmetic worked correctly on interior faces
        for i, face in enumerate(complex_mesh.faces):
            assert result[face] == float(i * 3) + 5.0

        # Verify boundary faces remain 0.0
        for boundary in complex_mesh.boundaries:
            assert result[boundary] == 0.0

    def test_twoform_single_triangle(self):
        """Test TwoForm with a single triangle mesh."""
        points = [Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)]
        triangles = [(0, 1, 2)]
        triangle_mesh = Mesh.from_triangle_soup(points, triangles)

        tf = TwoForm(triangle_mesh)

        # Should have 1 interior face and 1 boundary
        assert len(triangle_mesh.faces) == 1
        assert len(triangle_mesh.boundaries) == 1

        # Test setting values on interior face only
        interior_face = triangle_mesh.faces.to_object(0)
        boundary_face = triangle_mesh.boundaries.to_object(0)

        tf[interior_face] = 42.0

        # Boundary face should always return 0.0 and cannot be set
        assert tf[boundary_face] == 0.0
        with pytest.raises(KeyError):
            tf[boundary_face] = -17.0

        assert tf[interior_face] == 42.0
        assert tf[boundary_face] == 0.0

        # Test arithmetic
        tf2 = TwoForm(triangle_mesh)
        tf2[interior_face] = 8.0

        result = tf - tf2
        assert result[interior_face] == 42.0 - 8.0
        assert result[boundary_face] == 0.0

    def test_twoform_default_values(self, simple_mesh):
        """Test that TwoForm returns 0.0 for faces that haven't been set."""
        tf = TwoForm(simple_mesh)

        # All interior faces should default to 0.0
        for face in simple_mesh.faces:
            assert tf[face] == 0.0

        # Boundary faces should always return 0.0
        for boundary in simple_mesh.boundaries:
            assert tf[boundary] == 0.0

        # Set one face and verify others remain 0.0
        first_face = simple_mesh.faces.to_object(0)
        tf[first_face] = 99.0

        # Check that the set face has the correct value
        assert tf[first_face] == 99.0

        # Check that other interior faces still default to 0.0
        for i, face in enumerate(simple_mesh.faces):
            if i != 0:  # Skip the face we set
                assert tf[face] == 0.0

        # Boundary faces should still return 0.0
        for boundary in simple_mesh.boundaries:
            assert tf[boundary] == 0.0


class TestPolyBoundaryDistanceMap:
    """Test suite for PolyBoundaryDistanceMap functionality."""

    def test_basic_rectangle_distance_map(self):
        """Test distance map for simple rectangle."""
        poly = shapely.geometry.box(0, 0, 10, 10)
        dist_map = PolyBoundaryDistanceMap(poly, 0.2)

        # Center should be ~5.0 (distance to nearest boundary)
        center_dist = dist_map.query(5.0, 5.0)
        assert 4.9 <= center_dist <= 5.1, f"Center distance {center_dist} not ~5.0"

        # Point near edge should be ~1.0
        near_edge = dist_map.query(1.0, 5.0)
        assert 0.8 <= near_edge <= 1.2, f"Near edge distance {near_edge} not ~1.0"

        # Outside points should be 0
        assert dist_map.query(-1.0, 5.0) == 0.0, "Outside point should return 0"
        assert dist_map.query(15.0, 5.0) == 0.0, "Outside point should return 0"

        # Boundary points should be small due to quantization
        boundary_dist = dist_map.query(0.0, 5.0)
        assert boundary_dist < 0.25, f"Boundary point distance {boundary_dist} too large"

    def test_rectangle_with_hole(self):
        """Test distance map for rectangle with rectangular hole."""
        outer_box = shapely.geometry.box(0, 0, 10, 10)
        inner_box = shapely.geometry.box(3, 3, 7, 7)
        poly = outer_box.difference(inner_box)

        dist_map = PolyBoundaryDistanceMap(poly, 0.2)

        # Center of hole should be 0 (outside)
        hole_center = dist_map.query(5.0, 5.0)
        assert hole_center == 0.0, f"Hole center should be 0, got {hole_center}"
        hole_center = dist_map.query(5.0, 6.0)
        assert hole_center == 0.0, f"Hole should be 0, got {hole_center}"

        # Point between boundaries should have reasonable distance
        between = dist_map.query(1.0, 5.0)
        assert 0.5 <= between <= 2.5, f"Distance between boundaries {between} unreasonable"

        # Outside outer boundary should be 0
        assert dist_map.query(-1.0, 5.0) == 0.0, "Outside outer boundary should be 0"

        # On hole boundary should be small
        hole_boundary = dist_map.query(3.0, 5.0)
        assert hole_boundary < 0.5, f"Hole boundary distance {hole_boundary} too large"

    def test_circle_distance_map(self):
        """Test distance map for circular polygon."""
        center_pt = shapely.geometry.Point(5, 5)
        circle = center_pt.buffer(3)  # Radius 3 circle at (5,5)

        dist_map = PolyBoundaryDistanceMap(circle, 0.1)

        # Center should be ~3.0 (radius)
        center_dist = dist_map.query(5.0, 5.0)
        assert 2.9 <= center_dist <= 3.1, f"Circle center distance {center_dist} not ~3.0"

        # Point at radius 1 should have distance ~2.0 to boundary
        inner_point = dist_map.query(6.0, 5.0)  # 1 unit from center
        assert 1.5 <= inner_point <= 2.5, f"Inner point distance {inner_point} not ~2.0"

        # Outside circle should be 0
        outside = dist_map.query(10.0, 5.0)
        assert outside == 0.0, f"Outside circle should be 0, got {outside}"

    def test_circle_with_hole(self):
        """Test distance map for circle with circular hole."""
        outer_circle = shapely.geometry.Point(5, 5).buffer(4)
        inner_circle = shapely.geometry.Point(5, 5).buffer(2)
        annulus = outer_circle.difference(inner_circle)

        dist_map = PolyBoundaryDistanceMap(annulus, 0.1)

        # Center should be 0 (in hole)
        center = dist_map.query(5.0, 5.0)
        assert center == 0.0, f"Center of hole should be 0, got {center}"

        # Point in annulus should have reasonable distance
        annulus_point = dist_map.query(8.0, 5.0)  # Between inner and outer
        assert annulus_point > 0.5, f"Annulus point distance {annulus_point} too small"

        # Outside outer circle should be 0
        outside = dist_map.query(15.0, 5.0)
        assert outside == 0.0, f"Outside should be 0, got {outside}"

    def test_distance_continuity(self):
        """Test that distance values are continuous (no wild discontinuities)."""
        poly = shapely.geometry.box(0, 0, 10, 10)
        quantization = 0.5
        dist_map = PolyBoundaryDistanceMap(poly, quantization)

        # Sample along horizontal line at 10x finer resolution
        sample_step = quantization / 10
        y = 5.0  # Middle of rectangle

        distances = []
        x_values = []

        # Sample from x=1 to x=9 (staying inside)
        x = 1.0
        while x <= 9.0:
            dist = dist_map.query(x, y)
            distances.append(dist)
            x_values.append(x)
            x += sample_step

        # Check for wild discontinuities
        # Even without the bilinear interpolation, jumps should be limited
        # by `quantization`, since walking away by a single quantization should
        # not change distance more than quantization.
        max_allowed_jump = quantization * 0.2  # Allow some reasonable variation

        for i in range(1, len(distances)):
            diff = abs(distances[i] - distances[i-1])
            assert diff < max_allowed_jump, \
                f"Wild discontinuity at x={x_values[i]:.3f}: " \
                f"distance jumped from {distances[i-1]:.3f} to {distances[i]:.3f}"

        # Distances should be symmetric around center for this rectangle
        center_idx = len(distances) // 2
        left_dists = distances[:center_idx]
        right_dists = list(reversed(distances[center_idx+1:]))

        # Allow some tolerance for symmetry due to quantization
        for i, (left, right) in enumerate(zip(left_dists, right_dists)):
            diff = abs(left - right)
            assert diff < quantization, \
                f"Asymmetric distances at offset {i}: left={left:.3f}, right={right:.3f}"

    def test_quantization_effects(self):
        """Test that finer quantization gives more accurate results."""
        poly = shapely.geometry.box(0, 0, 10, 10)

        coarse_map = PolyBoundaryDistanceMap(poly, 1.0)
        fine_map = PolyBoundaryDistanceMap(poly, 0.1)

        # Test center point - fine should be closer to expected 5.0
        coarse_center = coarse_map.query(5.0, 5.0)
        fine_center = fine_map.query(5.0, 5.0)

        expected = 5.0
        coarse_error = abs(coarse_center - expected)
        fine_error = abs(fine_center - expected)

        assert 0 < fine_error < coarse_error, \
            f"Fine quantization should be more accurate: " \
            f"coarse_error={coarse_error:.3f}, fine_error={fine_error:.3f}"


class TestCGALPolygon:
    """Test the CGALPolygon class for correctness against Shapely."""

    def test_synthetic_geometry(self):
        """Test CGALPolygon with synthetic rectangle-with-hole geometry."""
        import padne._cgal as cgal
        from shapely.geometry import Polygon, Point

        # Create rectangle with hole: outer (0,0) to (10,10), hole (3,3) to (7,7)
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        shapely_poly = Polygon(outer, [hole])
        cgal_poly = cgal.CGALPolygon(shapely_poly)

        # Test predetermined points
        test_points = [
            # Points clearly inside outer boundary
            (1, 1), (9, 9), (2, 5), (8, 2),
            # Points inside the hole (should return False)
            (5, 5), (4, 4), (6, 6),
            # Points on outer boundary
            (0, 0), (10, 5), (5, 10), (0, 5),
            # Points on hole boundary
            (3, 3), (7, 5), (5, 7), (3, 5),
            # Points outside everything
            (-1, -1), (11, 11), (-5, 5), (12, 3)
        ]

        for x, y in test_points:
            point = Point(x, y)

            # Test contains
            cgal_contains = cgal_poly.contains(x, y)
            shapely_contains = shapely_poly.covers(point)
            assert cgal_contains == shapely_contains, \
                f"Contains mismatch at ({x}, {y}): CGAL={cgal_contains}, Shapely={shapely_contains}"

            # Test distance (only for inside points)
            cgal_dist = cgal_poly.distance_to_boundary(x, y)
            shapely_dist = point.distance(shapely_poly.boundary)

            assert cgal_dist == pytest.approx(shapely_dist, rel=1e-4), \
                f"Distance mismatch at ({x}, {y}): CGAL={cgal_dist:.6f}, Shapely={shapely_dist:.6f}"

    @for_all_kicad_projects(include=["degenerate_hole_geometry", "simple_geometry", "via_tht_4layer"])
    def test_real_geometry(self, project):
        """Test CGALPolygon with real PCB geometries from KiCad test projects."""
        from shapely.geometry import Point

        random.seed(42)  # Deterministic random sampling

        problem = kicad.load_kicad_project(project.pro_path)

        for layer_idx, layer in enumerate(problem.layers):
            for poly_idx, polygon in enumerate(layer.shape.geoms):
                cgal_poly = CGALPolygon(polygon)
                bounds = polygon.bounds

                # Test 100 random points within bounding box
                for point_idx in range(100):
                    x = random.uniform(bounds[0], bounds[2])
                    y = random.uniform(bounds[1], bounds[3])
                    point = Point(x, y)

                    # Test contains
                    cgal_contains = cgal_poly.contains(x, y)
                    shapely_contains = polygon.covers(point)
                    assert cgal_contains == shapely_contains, \
                        f"Contains mismatch, layer {layer_idx}, " \
                        f"polygon {poly_idx}, point {point_idx} at ({x:.6f}, {y:.6f}): " \
                        f"CGAL={cgal_contains}, Shapely={shapely_contains}"

                    cgal_dist = cgal_poly.distance_to_boundary(x, y)
                    shapely_dist = point.distance(polygon.boundary)
                    assert cgal_dist == pytest.approx(shapely_dist, rel=1e-4)


class TestVariableDensityMeshing:
    """Test variable density meshing configuration."""

    def test_is_variable_density_property(self):
        """Test the is_variable_density property correctly identifies variable density mode."""
        # Default config should have variable density enabled
        default_config = Mesher.Config()
        assert default_config.is_variable_density is True
        assert default_config.variable_size_maximum_factor == 3.0

        # Config with factor=1.0 should disable variable density
        disabled_config = Mesher.Config(variable_size_maximum_factor=1.0)
        assert disabled_config.is_variable_density is False

        # Config with factor != 1.0 should enable variable density
        enabled_config = Mesher.Config(variable_size_maximum_factor=2.5)
        assert enabled_config.is_variable_density is True

    @patch('padne._cgal.PolyBoundaryDistanceMap')
    def test_variable_density_disabled_no_distance_map_construction(self, mock_distance_map):
        """Test that PolyBoundaryDistanceMap is not constructed when variable density is disabled."""
        # Create a simple rectangle for testing
        poly = shapely.geometry.box(0, 0, 10, 10)

        # Create mesher with variable density disabled
        config = Mesher.Config(variable_size_maximum_factor=1.0)
        assert not config.is_variable_density
        mesher = Mesher(config)

        # Mock the cgal.mesh function to avoid actual meshing
        with patch('padne._cgal.mesh') as mock_cgal_mesh:
            mock_cgal_mesh.return_value = {'vertices': [], 'triangles': []}

            # Call poly_to_mesh
            mesher.poly_to_mesh(poly)

            # Verify PolyBoundaryDistanceMap was not constructed
            mock_distance_map.assert_not_called()

            # Verify cgal.mesh was called with None as the distance_map parameter
            mock_cgal_mesh.assert_called_once()
            args, kwargs = mock_cgal_mesh.call_args
            assert args[4] is None  # distance_map should be None

    @patch('padne._cgal.PolyBoundaryDistanceMap')
    def test_variable_density_enabled_creates_distance_map(self, mock_distance_map):
        """Test that PolyBoundaryDistanceMap is constructed when variable density is enabled."""
        # Create a simple rectangle for testing
        poly = shapely.geometry.box(0, 0, 10, 10)

        # Create mesher with variable density enabled (default behavior)
        config = Mesher.Config(variable_size_maximum_factor=3.0)
        assert config.is_variable_density
        mesher = Mesher(config)

        # Mock the distance map instance
        mock_distance_map_instance = Mock()
        mock_distance_map.return_value = mock_distance_map_instance

        # Mock the cgal.mesh function to avoid actual meshing
        with patch('padne._cgal.mesh') as mock_cgal_mesh:
            mock_cgal_mesh.return_value = {'vertices': [], 'triangles': []}

            # Call poly_to_mesh
            mesher.poly_to_mesh(poly)

            # Verify PolyBoundaryDistanceMap was constructed with correct parameters
            mock_distance_map.assert_called_once_with(poly, config.distance_map_quantization)

            # Verify cgal.mesh was called with the distance map instance
            mock_cgal_mesh.assert_called_once()
            args, kwargs = mock_cgal_mesh.call_args
            assert args[4] is mock_distance_map_instance
