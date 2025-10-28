#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/embed.h>
#include <pybind11/operators.h>
#include <iterator>
#include <cmath>
#include <algorithm>
#include <limits>


//#define CGAL_USE_BASIC_VIEWER

#include <CGAL/version.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Mesh_2/Face_badness.h>
#include <CGAL/Delaunay_mesh_criteria_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/squared_distance_2.h>
#include <CGAL/enum.h>

#ifdef CGAL_USE_BASIC_VIEWER
#include <CGAL/draw_triangulation_2.h>
#include <CGAL/draw_constrained_triangulation_2.h>
#endif

namespace py = pybind11;
using namespace pybind11::literals;

// Type definitions
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polygon_2<K> Polygon_2;
typedef CGAL::Polygon_with_holes_2<K> Polygon_with_holes_2;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef K::Point_2 Point;
typedef K::Segment_2 Segment_2;

// Helper macro (often used with version info passed from CMake)
#define MACRO_STRINGIFY(x) #x

// CGALPolygon class - wraps CGAL::Polygon_with_holes_2 for Python interface
class CGALPolygon {
private:
    Polygon_with_holes_2 polygon_with_holes;
    std::vector<Segment_2> all_edges;  // All edges for distance calculations

    void extract_polygon_from_shapely(py::object shapely_polygon);

public:
    CGALPolygon(py::object shapely_polygon);
    bool contains(double x, double y) const;
    double distance_to_boundary(double x, double y) const;
};

// PolyBoundaryDistanceMap class for computing distance-based variable density
class PolyBoundaryDistanceMap {
private:
    double min_x, max_x, min_y, max_y;
    double quantization;
    int width, height;
    std::vector<double> distances;  // Flat 2D array storage

    CGALPolygon cgal_polygon;

    void compute_distances();

    // Coordinate transformation methods
    std::pair<double, double> world_to_grid(double world_x, double world_y) const;
    std::pair<double, double> grid_to_world(double grid_x, double grid_y) const;
    int grid_to_index(int grid_i, int grid_j) const;

public:
    PolyBoundaryDistanceMap(py::object polygon, double quantization);
    double query(double x, double y) const;  // Bilinear interpolation

    // Property accessors
    double get_min_x() const { return min_x; }
    double get_max_x() const { return max_x; }
    double get_min_y() const { return min_y; }
    double get_max_y() const { return max_y; }
    double get_quantization() const { return quantization; }
    int get_width() const { return width; }
    int get_height() const { return height; }
};

// Variable density mesh size criteria implementation
// This is effectively a reimplementation of CGAL::Delaunay_mesh_size_criteria_2,
// except it supports a variable size field based on a distance from boundary map
template <class CDT>
class Variable_density_mesh_size_criteria_2 :
    public virtual CGAL::Delaunay_mesh_criteria_2<CDT> {
protected:
    typedef typename CDT::Geom_traits Geom_traits;
    double sizebound;
    const PolyBoundaryDistanceMap* distance_map_ptr;
    double min_distance;
    double max_distance;
    double size_factor;

public:
    typedef CGAL::Delaunay_mesh_criteria_2<CDT> Base;

    // Do note that for CGAL compatibility, we require an operational
    // constructor that takes no arguments
    Variable_density_mesh_size_criteria_2(const double aspect_bound = 0.125,
                                          const double size_bound = 0,
                                          const PolyBoundaryDistanceMap* dist_map_ptr = nullptr,
                                          const double min_dist = 0.0,
                                          const double max_dist = 0.0,
                                          const double sz_factor = 1.0,
                                          const Geom_traits& traits = Geom_traits())
      : Base(aspect_bound, traits), sizebound(size_bound), distance_map_ptr(dist_map_ptr),
        min_distance(min_dist), max_distance(max_dist), size_factor(sz_factor) {}

    inline double size_bound() const { return sizebound; }

    inline void set_size_bound(const double sb) { sizebound = sb; }

    // Simple struct with public members for size and sine
    struct Quality {
      double sine;
      double size;

      Quality() : sine(0.0), size(0.0) {}
      Quality(double _sine, double _size) : sine(_sine), size(_size) {}

      // q1<q2 means q1 is prioritized over q2
      // ( q1 == *this, q2 == q )
      bool operator<(const Quality& q) const {
          if (size > 1) {
              if (q.size > 1) {
                  return size > q.size;
              } else {
                  return true; // *this is big but not q
              }
          } else {
              if (q.size > 1) {
                  return false; // q is big but not *this
              }
          }
          return sine < q.sine;
      }

      std::ostream& operator<<(std::ostream& out) const {
          return out << "(size=" << size
                     << ", sine=" << sine << ")";
      }
    };

    class Is_bad: public Base::Is_bad {
    protected:
        const double base_size_bound; // base size bound for variable scaling
        const PolyBoundaryDistanceMap* distance_map_ptr;
        const double min_distance;
        const double max_distance;
        const double size_factor;

    public:
        typedef typename Base::Is_bad::Point_2 Point_2;

        Is_bad(const double aspect_bound,
               const double size_bound,
               const PolyBoundaryDistanceMap* dist_map_ptr,
               const double min_dist,
               const double max_dist,
               const double sz_factor,
               const Geom_traits& traits)
          : Base::Is_bad(aspect_bound, traits),
            base_size_bound(size_bound), distance_map_ptr(dist_map_ptr),
            min_distance(min_dist), max_distance(max_dist), size_factor(sz_factor) {}

        CGAL::Mesh_2::Face_badness operator()(const Quality q) const {
            if (q.size > 1) {
                return CGAL::Mesh_2::IMPERATIVELY_BAD;
            }
            if (q.sine < this->B) {
                return CGAL::Mesh_2::BAD;
            }
            return CGAL::Mesh_2::NOT_BAD;
        }

        CGAL::Mesh_2::Face_badness operator()(const typename CDT::Face_handle& fh,
                                              Quality& q) const {
            typedef typename CDT::Geom_traits Geom_traits;
            typedef typename Geom_traits::Compute_area_2 Compute_area_2;
            typedef typename Geom_traits::Compute_squared_distance_2
              Compute_squared_distance_2;

            Compute_squared_distance_2 squared_distance =
              this->traits.compute_squared_distance_2_object();

            const Point& pa = fh->vertex(0)->point();
            const Point& pb = fh->vertex(1)->point();
            const Point& pc = fh->vertex(2)->point();

            // Compute triangle centroid using helper method
            auto [cx, cy] = compute_triangle_centroid(pa, pb, pc);

            // Compute distance to polygon boundary using distance map
            double boundary_distance = distance_map_ptr ? distance_map_ptr->query(cx, cy) : 0.0;

            // Compute effective size bound using piecewise linear scaling
            double effective_size_bound = compute_effective_size_bound(boundary_distance);
            double squared_size_bound = effective_size_bound * effective_size_bound;

            double a = CGAL::to_double(squared_distance(pb, pc));
            double b = CGAL::to_double(squared_distance(pc, pa));
            double c = CGAL::to_double(squared_distance(pa, pb));

            double max_sq_length; // squared max edge length
            double second_max_sq_length;

            if (a < b) {
                if (b < c) {
                    max_sq_length = c;
                    second_max_sq_length = b;
                } else { // c<=b
                    max_sq_length = b;
                    second_max_sq_length = ( a < c ? c : a );
                }
            } else {
                if (a < c) {
                    max_sq_length = c;
                    second_max_sq_length = a;
                } else {
                    max_sq_length = a;
                    second_max_sq_length = b < c ? c : b;
                }
            }

            q.size = 0;
            if (squared_size_bound != 0) {
                q.size = max_sq_length / squared_size_bound;
                // normalized by size bound to deal
                // with size field
                if (q.size > 1) {
                    q.sine = 1; // (do not compute sine)
                    return CGAL::Mesh_2::IMPERATIVELY_BAD;
                }
            }

            Compute_area_2 area_2 = this->traits.compute_area_2_object();

            double area = 2*CGAL::to_double(area_2(pa, pb, pc));

            q.sine = (area * area) / (max_sq_length * second_max_sq_length); // (sine)

            if( q.sine < this->B ) {
                return CGAL::Mesh_2::BAD;
            } else {
                return CGAL::Mesh_2::NOT_BAD;
            }
        }

    private:
        // Helper method to compute triangle centroid
        std::pair<double, double> compute_triangle_centroid(const Point& pa, const Point& pb, const Point& pc) const {
            double cx = (CGAL::to_double(pa.x()) + CGAL::to_double(pb.x()) + CGAL::to_double(pc.x())) / 3.0;
            double cy = (CGAL::to_double(pa.y()) + CGAL::to_double(pb.y()) + CGAL::to_double(pc.y())) / 3.0;
            return std::make_pair(cx, cy);
        }

        // Helper method for piecewise linear scaling
        double compute_effective_size_bound(double boundary_distance) const {
            // If no distance map, use uniform sizing
            if (!distance_map_ptr) {
                return base_size_bound;
            }
            if (boundary_distance <= min_distance) {
                return base_size_bound;
            } else if (boundary_distance >= max_distance) {
                return base_size_bound * size_factor;
            }
            // Linear interpolation between min_distance and max_distance
            double t = (boundary_distance - min_distance) / (max_distance - min_distance);
            return base_size_bound * (1.0 + t * (size_factor - 1.0));
        }
    };

  Is_bad is_bad_object() const
  { return Is_bad(this->bound(), size_bound(), distance_map_ptr,
                  min_distance, max_distance, size_factor,
                  this->traits /* from the bad class */); }
};

typedef Variable_density_mesh_size_criteria_2<CDT> Criteria;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria> Mesher;
typedef CDT::Vertex_handle Vertex_handle;


static void setup_cdt(CDT& cdt,
                      const std::vector<std::pair<double, double>>& vertices,
                      const std::vector<std::pair<int, int>>& segments,
                      const std::vector<std::pair<double, double>>& seeds) {
    std::vector<Vertex_handle> vertex_handles;
    // First, we insert all the vertices into the cdt object, grabbing the handles
    // along the way
    for (const auto& vertex : vertices) {
        auto vh = cdt.insert(Point(vertex.first, vertex.second));
        vertex_handles.push_back(vh);
    }
    // Next, we insert the constraints
    for (const auto& segment : segments) {
        int i_start = segment.first;
        int i_end = segment.second;

        if (i_start < 0 || i_start >= vertex_handles.size() ||
            i_end < 0 || i_end >= vertex_handles.size()) {
            throw std::runtime_error("Segment indices out of bounds.");
        }

        auto start_vh = vertex_handles[i_start];
        auto end_vh = vertex_handles[i_end];

        cdt.insert_constraint(start_vh, end_vh);
    }

    // Insert the seed points into the CDT
    // We do this before creating the Mesher object, but I am not sure
    // if that is needed
    for (const auto& seed : seeds) {
        cdt.insert(Point(seed.first, seed.second));
    }
}

static void set_mesher_seeds(Mesher& mesher,
                             const std::vector<std::pair<double, double>>& seeds,
                             bool mark = true) {
    std::vector<Point> seed_points;
    for (const auto& seed : seeds) {
        Point p(seed.first, seed.second);
        seed_points.push_back(p);
    }
    mesher.set_seeds(seed_points.begin(), seed_points.end(), mark);
}

void setup_mesher(Mesher& mesher,
                  const py::object& py_config,
                  const std::vector<std::pair<double, double>>& seeds,
                  const PolyBoundaryDistanceMap* distance_map_ptr) {
    // Extract standard parameters
    auto minimum_angle = py_config.attr("minimum_angle").cast<float>();
    auto B = 1 / (2*sin(minimum_angle * M_PI / 180.0));
    auto b = 1 / (4*B*B);
    auto maximum_size = py_config.attr("maximum_size").cast<float>();

    // Extract variable density parameters
    auto min_distance = py_config.attr("variable_density_min_distance").cast<double>();
    auto max_distance = py_config.attr("variable_density_max_distance").cast<double>();
    auto size_factor = py_config.attr("variable_size_maximum_factor").cast<double>();

    mesher.set_criteria(Criteria(
            b,
            maximum_size,
            distance_map_ptr,
            min_distance,
            max_distance,
            size_factor,
            K()
        )
    );

    // Now we insert the seeds
    // Theoretically, it should not be necessary to insert _all_ the seeds,
    // since we have a single connected component. The python code
    // should just give us a single representative_point to use instead
    // TODO
    set_mesher_seeds(mesher, seeds);
}


std::pair<py::list, py::list> convert_meshing_result_to_python(CDT &cdt)
{
    py::list py_vertices;
    std::map<Vertex_handle, int> vertex_index_map;
    for (auto it = cdt.finite_vertices_begin(); it != cdt.finite_vertices_end(); ++it) {
        Point p = it->point();
        auto key = it->handle();
        vertex_index_map[key] = py_vertices.size();
        py_vertices.append(py::make_tuple(p.x(), p.y()));
    }

    py::list py_triangles;
    for (auto it = cdt.finite_faces_begin(); it != cdt.finite_faces_end(); ++it) {
        if (!it->is_in_domain()) {
            continue; // Skip faces that are not in the domain
        }
        auto v0 = it->vertex(0);
        auto v1 = it->vertex(1);
        auto v2 = it->vertex(2);

        auto i0 = vertex_index_map[v0];
        auto i1 = vertex_index_map[v1];
        auto i2 = vertex_index_map[v2];

        py::tuple triangle = py::make_tuple(i0, i1, i2);
        py_triangles.append(triangle);
    }

    return std::make_pair(py_vertices, py_triangles);
}




py::dict mesh(const py::object& py_config,
              const std::vector<std::pair<double, double>>& vertices,
              const std::vector<std::pair<int, int>>& segments,
              const std::vector<std::pair<double, double>>& seeds,
              const py::object& distance_map_obj) {

    // Redirect via python during the scope of this function
    py::scoped_ostream_redirect stream_redirect;

    CDT cdt;
    setup_cdt(cdt, vertices, segments, seeds);

    Mesher mesher(cdt);

    // Check if distance_map_obj is None, and extract pointer if not
    const PolyBoundaryDistanceMap* distance_map_ptr = nullptr;
    if (!distance_map_obj.is_none()) {
        distance_map_ptr = &distance_map_obj.cast<const PolyBoundaryDistanceMap&>();
    }

    setup_mesher(mesher, py_config, seeds, distance_map_ptr);

    mesher.refine_mesh();

    // Okay, so for the result, we return
    // result["vertices"], which is a list of tuples (x, y) from the triangulation
    // result["triangles"] which is a list of tuples (v1, v2, v3) where v1, v2, and v3 are the indices of the vertices

    auto [py_vertices, py_triangles] = convert_meshing_result_to_python(cdt);

    py::dict result;
    result["vertices"] = py_vertices;
    result["triangles"] = py_triangles;
    return result;
}

// PolyBoundaryDistanceMap implementation
PolyBoundaryDistanceMap::PolyBoundaryDistanceMap(py::object polygon, double quantization)
    : quantization(quantization), cgal_polygon(polygon) {

    // Extract bounding box from polygon.bounds
    // Also add a margin such that coordinates "too far out" are reliably zero
    auto margin = 2 * quantization;
    py::tuple bounds = polygon.attr("bounds");
    min_x = bounds[0].cast<double>() - margin;
    min_y = bounds[1].cast<double>() - margin;
    max_x = bounds[2].cast<double>() + margin;
    max_y = bounds[3].cast<double>() + margin;

    // Calculate grid dimensions
    width = static_cast<int>(std::ceil((max_x - min_x) / quantization));
    height = static_cast<int>(std::ceil((max_y - min_y) / quantization));

    // Initialize distance array
    distances.resize(width * height);

    // Compute distances
    compute_distances();
}


void PolyBoundaryDistanceMap::compute_distances() {

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            // Compute world coordinates of pixel center (i+0.5, j+0.5)
            auto [world_x, world_y] = grid_to_world(i + 0.5, j + 0.5);

            double distance;
            if (cgal_polygon.contains(world_x, world_y)) {
                // Point is inside polygon - compute distance to boundary
                distance = cgal_polygon.distance_to_boundary(world_x, world_y);
            } else {
                // Outside polygon, distance = 0
                distance = 0.0;
            }

            // Store distance in flat array
            distances[grid_to_index(i, j)] = distance;
        }
    }
}

double PolyBoundaryDistanceMap::query(double x, double y) const {
    // Check if point is outside the polygon bounds - return 0 immediately
    if (x < min_x || x > max_x || y < min_y || y > max_y) {
        return 0.0;
    }

    // Convert world coordinates to grid coordinates using transformation method
    auto [gx, gy] = world_to_grid(x, y);

    // Find integer grid coordinates
    int i0 = static_cast<int>(std::floor(gx));
    int i1 = i0 + 1;
    int j0 = static_cast<int>(std::floor(gy));
    int j1 = j0 + 1;

    // Clamp to valid ranges (should not be needed now due to bounds check)
    i0 = std::clamp(i0, 0, width-1);
    i1 = std::clamp(i1, 0, width-1);
    j0 = std::clamp(j0, 0, height-1);
    j1 = std::clamp(j1, 0, height-1);

    // Get fractional parts for interpolation
    double fx = gx - std::floor(gx);
    double fy = gy - std::floor(gy);

    // Sample 4 corners using index transformation
    double v00 = distances[grid_to_index(i0, j0)];
    double v10 = distances[grid_to_index(i1, j0)];
    double v01 = distances[grid_to_index(i0, j1)];
    double v11 = distances[grid_to_index(i1, j1)];

    // Bilinear interpolation
    double v0 = v00 * (1.0 - fx) + v10 * fx;
    double v1 = v01 * (1.0 - fx) + v11 * fx;
    return v0 * (1.0 - fy) + v1 * fy;
}

std::pair<double, double> PolyBoundaryDistanceMap::world_to_grid(double world_x, double world_y) const {
    double grid_x = (world_x - min_x) / quantization;
    double grid_y = (world_y - min_y) / quantization;
    return std::make_pair(grid_x, grid_y);
}

std::pair<double, double> PolyBoundaryDistanceMap::grid_to_world(double grid_x, double grid_y) const {
    double world_x = min_x + grid_x * quantization;
    double world_y = min_y + grid_y * quantization;
    return std::make_pair(world_x, world_y);
}

int PolyBoundaryDistanceMap::grid_to_index(int grid_i, int grid_j) const {
    return grid_j * width + grid_i;
}

// Helper function to convert Shapely coordinate list to CGAL Polygon_2
// NOTE: Shapely includes duplicate closing coordinate, but CGAL Polygon_2 is implicitly closed
// We must skip the last coordinate to avoid creating zero-length edges
static Polygon_2 shapely_coords_to_cgal_polygon(py::object coords) {
    Polygon_2 polygon;
    py::list coords_list = py::list(coords);
    size_t num_coords = coords_list.size();

    // Skip last coordinate (duplicate closing point)
    for (size_t i = 0; i < num_coords - 1; ++i) {
        py::tuple pt = coords_list[i].cast<py::tuple>();
        double x = pt[0].cast<double>();
        double y = pt[1].cast<double>();
        polygon.push_back(Point(x, y));
    }

    return polygon;
}

CGALPolygon::CGALPolygon(py::object shapely_polygon) {
    extract_polygon_from_shapely(shapely_polygon);
}

void CGALPolygon::extract_polygon_from_shapely(py::object shapely_polygon) {
    // Extract exterior coordinates
    py::object exterior = shapely_polygon.attr("exterior");
    py::object coords = exterior.attr("coords");

    // Convert exterior to CGAL polygon
    Polygon_2 outer_boundary = shapely_coords_to_cgal_polygon(coords);

    // Make sure outer boundary is counter-clockwise (CGAL convention)
    if (outer_boundary.is_clockwise_oriented()) {
        outer_boundary.reverse_orientation();
    }

    // Add edges from outer boundary to all_edges
    for (auto edge = outer_boundary.edges_begin(); edge != outer_boundary.edges_end(); ++edge) {
        all_edges.push_back(*edge);
    }

    // Extract holes if any
    std::vector<Polygon_2> holes;
    py::object interiors = shapely_polygon.attr("interiors");
    for (auto interior : interiors) {
        py::object hole_coords = interior.attr("coords");

        Polygon_2 hole = shapely_coords_to_cgal_polygon(hole_coords);

        // Make sure holes are clockwise (CGAL convention for holes)
        if (!hole.is_clockwise_oriented()) {
            hole.reverse_orientation();
        }

        // Add edges from hole to all_edges
        for (auto edge = hole.edges_begin(); edge != hole.edges_end(); ++edge) {
            all_edges.push_back(*edge);
        }

        holes.push_back(hole);
    }

    // Create polygon with holes
    polygon_with_holes = Polygon_with_holes_2(outer_boundary, holes.begin(), holes.end());
}

bool CGALPolygon::contains(double x, double y) const {
    Point point(x, y);

    // Check if point is inside outer boundary
    auto bounded = polygon_with_holes.outer_boundary().bounded_side(point);
    if (bounded != CGAL::ON_BOUNDED_SIDE && bounded != CGAL::ON_BOUNDARY) {
        return false;
    }

    // Check if point is outside all holes
    for (auto hole = polygon_with_holes.holes_begin(); hole != polygon_with_holes.holes_end(); ++hole) {
        if (hole->bounded_side(point) == CGAL::ON_BOUNDED_SIDE) {
            return false; // Point is inside a hole
        }
    }

    return true;
}

double CGALPolygon::distance_to_boundary(double x, double y) const {
    Point point(x, y);
    double min_squared_dist = std::numeric_limits<double>::max();

    // Find minimum distance to all edges (exterior + holes)
    for (const auto& edge : all_edges) {
        double sq_dist = CGAL::to_double(CGAL::squared_distance(point, edge));
        if (sq_dist < min_squared_dist) {
            min_squared_dist = sq_dist;
        }
    }

    return std::sqrt(min_squared_dist);
}

// PYBIND11_MODULE defines the module initialization function.
// The first argument (R"_core") MUST match the first argument of pybind11_add_module in CMakeLists.txt.
// The 'm' variable is the module object.
PYBIND11_MODULE(_cgal, m) {
    // Optional: Add a docstring to the module.
    m.doc() = R"pbdoc(
        Padne internal libcgal wrapper
        ------------------------------
        .. currentmodule:: padne._cgal

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("mesh", &mesh, R"pbdoc(
        Meshes a set of points and segments using CGAL.
        Args:
            mesher: The mesher object to use.
            vertices: A list of vertices (points).
            segments: A list of segments (edges).
            seeds: A list of seed points for the meshing process.
        Returns:
            A dictionary containing the results of the meshing process.
    )pbdoc");

    py::class_<CGALPolygon>(m, "CGALPolygon")
        .def(py::init<py::object>(), "shapely_polygon"_a,
             "Create CGAL polygon from Shapely polygon")
        .def("contains", &CGALPolygon::contains, "x"_a, "y"_a,
             "Check if point (x,y) is inside polygon")
        .def("distance_to_boundary", &CGALPolygon::distance_to_boundary,
             "x"_a, "y"_a,
             "Compute distance from point to nearest boundary");

    py::class_<PolyBoundaryDistanceMap>(m, "PolyBoundaryDistanceMap")
        .def(py::init<py::object, double>(),
             "polygon"_a, "quantization"_a,
             "Create distance map from shapely polygon")
        .def("query", &PolyBoundaryDistanceMap::query,
             "x"_a, "y"_a,
             "Query distance at point using bilinear interpolation")
        .def_property_readonly("min_x", &PolyBoundaryDistanceMap::get_min_x)
        .def_property_readonly("max_x", &PolyBoundaryDistanceMap::get_max_x)
        .def_property_readonly("min_y", &PolyBoundaryDistanceMap::get_min_y)
        .def_property_readonly("max_y", &PolyBoundaryDistanceMap::get_max_y)
        .def_property_readonly("quantization", &PolyBoundaryDistanceMap::get_quantization)
        .def_property_readonly("width", &PolyBoundaryDistanceMap::get_width)
        .def_property_readonly("height", &PolyBoundaryDistanceMap::get_height);

    m.attr("cgal_version") = CGAL_VERSION_STR;

#ifdef VERSION_INFO
    // Add version information if defined (usually via CMake)
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
