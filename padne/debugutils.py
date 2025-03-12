

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
