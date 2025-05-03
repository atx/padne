#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <iterator>

//#define CGAL_USE_BASIC_VIEWER

#include <CGAL/version.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>

#ifdef CGAL_USE_BASIC_VIEWER
#include <CGAL/draw_triangulation_2.h>
#include <CGAL/draw_constrained_triangulation_2.h>
#endif

namespace py = pybind11;

// Helper macro (often used with version info passed from CMake)
#define MACRO_STRINGIFY(x) #x

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria> Mesher;
 
typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point Point;

py::dict mesh(const py::object& pymesher,
              const std::vector<std::pair<double, double>>& vertices,
              const std::vector<std::pair<int, int>>& segments,
              const std::vector<std::pair<double, double>>& seeds) {
    py::dict result;

    // Redirect via python during the scope of this function
    py::scoped_ostream_redirect stream_redirect;

    CDT cdt;
    std::vector<Vertex_handle> vertex_handles;

    // First, we insert all the vertices into the cdt object, grabbing the handles
    // along the way
    for (const auto& vertex : vertices) {
        Point p(vertex.first, vertex.second);
        Vertex_handle vh = cdt.insert(p);
        vertex_handles.push_back(vh);
    }
    // Next, we insert the constraints
    for (const auto& segment : segments) {
        int start_index = segment.first;
        int end_index = segment.second;

        if (start_index < 0 || start_index >= vertex_handles.size() ||
            end_index < 0 || end_index >= vertex_handles.size()) {
            throw std::runtime_error("Segment indices out of bounds.");
        }

        Vertex_handle start_vh = vertex_handles[start_index];
        Vertex_handle end_vh = vertex_handles[end_index];

        cdt.insert_constraint(start_vh, end_vh);
    }

    // Insert the seed points into the CDT
    // We do this before creating the Mesher object, but I am not sure
    // if that is needed
    for (const auto& seed : seeds) {
        Point p(seed.first, seed.second);
        cdt.insert(p);
    }

    Mesher mesher(cdt);

    // Get the minimum_angle from the pymesher object
    // TODO: We probably want to have the b value in the python object directly...
    auto minimum_angle = pymesher.attr("minimum_angle").cast<float>();
    auto B = 1 / (2*sin(minimum_angle * M_PI / 180.0));
    auto b = 1 / (4*B*B);
    auto maximum_size = pymesher.attr("maximum_size").cast<float>();
    mesher.set_criteria(Criteria(b, maximum_size));

    // Now we insert the seeds
    // Theoretically, it should not be necessary to insert _all_ the seeds,
    // since we have a single connected component. The python code
    // should just give us a single representative_point to use instead
    // TODO
    std::vector<Point> seed_points;
    for (const auto& seed : seeds) {
        Point p(seed.first, seed.second);
        seed_points.push_back(p);
    }
    mesher.set_seeds(seed_points.begin(), seed_points.end(), true);
    mesher.refine_mesh();

    // Okay, so for the result, we return
    // result["vertices"], which is a list of tuples (x, y) from the triangulation
    // result["triangles"] which is a list of tuples (v1, v2, v3) where v1, v2, and v3 are the indices of the vertices
    
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

    result["vertices"] = py_vertices;
    result["triangles"] = py_triangles;

    return result;
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

    // Define functions and classes here later, e.g.:
    // m.def("my_function", &my_function, "Does something amazing");
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
    
    m.attr("cgal_version") = CGAL_VERSION_STR;

#ifdef VERSION_INFO
    // Add version information if defined (usually via CMake)
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
