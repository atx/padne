

from . import mesh

# This file contains misc functions useful for debugging. Ultimately
# should be dropped


def plot_multi_polygons(polys):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    
    patches = []
    for multipoly in polys:
        for poly in multipoly.geoms:
            # Create a patch for the exterior of the polygon
            patches.append(Polygon(list(poly.exterior.coords), closed=True))
            # Optionally add patches for interior holes
            for interior in poly.interiors:
                patches.append(Polygon(list(interior.coords), closed=True))
    
    fig, ax = plt.subplots()
    patch_collection = PatchCollection(patches, alpha=0.4, edgecolor='black', facecolor='cyan')
    ax.add_collection(patch_collection)
    ax.autoscale_view()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("MultiPolygon Plot")
    plt.show()


def plot_mesh(mesh: mesh.Mesh):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract all unique edges (to avoid duplicates from half-edges)
    edges = []
    drawn_edges = set()
    for _, half_edge in mesh.halfedges.items():
        # Get vertex indices to track unique edges
        v1_idx = mesh.vertices.to_index(half_edge.origin)
        v2_idx = mesh.vertices.to_index(half_edge.twin.origin)
        edge_key = tuple(sorted([v1_idx, v2_idx]))
        
        if edge_key not in drawn_edges:
            # Get the points for this edge
            p1 = half_edge.origin.p
            p2 = half_edge.twin.origin.p
            edges.append([(p1.x, p1.y), (p2.x, p2.y)])
            drawn_edges.add(edge_key)
    
    # Draw all edges as line segments
    line_collection = LineCollection(edges, linewidths=1, colors='black', alpha=0.7)
    ax.add_collection(line_collection)
    
    # Extract all vertices
    vertex_x = []
    vertex_y = []
    for _, vertex in mesh.vertices.items():
        vertex_x.append(vertex.p.x)
        vertex_y.append(vertex.p.y)
    
    # Draw vertices
    ax.scatter(vertex_x, vertex_y, color='blue', s=30, zorder=2)
    
    # Optional: Label vertices with their indices
    for i, (x, y) in enumerate(zip(vertex_x, vertex_y)):
        ax.text(x, y, str(i), fontsize=9, ha='right', va='bottom')
    
    # Draw faces with slight transparency
    for _, face in mesh.faces.items():
        vertices = list(face.vertices)
        if len(vertices) >= 3:  # Only plot faces with at least 3 vertices
            points = [(v.p.x, v.p.y) for v in vertices]
            polygon = plt.Polygon(points, alpha=0.2, color='lightblue')
            ax.add_patch(polygon)
    
    # Set equal aspect ratio and add title/labels
    ax.set_aspect('equal')
    ax.set_title(f"Mesh Visualization: {len(mesh.vertices)} vertices, "
                f"{len(drawn_edges)} edges, {len(mesh.faces)} faces")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Auto-adjust the view to fit all elements
    ax.autoscale_view()
    plt.tight_layout()
    plt.show()
