// Index-based half-edge mesh storage in struct-of-arrays form.
//
// The mesh topology lives in flat uint32 index arrays owned by the Mesh
// object. The Python-facing Vertex/HalfEdge/Face types are value-type proxy
// handles (owning Python Mesh reference + index); they compare and hash by
// (mesh, index), not by object identity.
//
// Layout invariants:
//  - Twin half-edges are allocated in adjacent pairs, so twin(i) == i ^ 1
//    and no twin array is needed.
//  - he_face entries reference the interior face store, or the boundary
//    face store when BOUNDARY_BIT is set. INVALID means "no face yet".
//  - INVALID marks unset references in all index arrays.
//
// Algorithmic methods (orbit, walk, area, from_triangle_soup, ...) are
// attached to these types from padne/mesh.py so that they stay identical to
// the original pure-Python implementation for now.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

constexpr uint32_t INVALID = UINT32_MAX;
constexpr uint32_t BOUNDARY_BIT = 0x80000000u;

uint64_t edge_key(uint32_t a, uint32_t b) {
    return (uint64_t(a) << 32) | uint64_t(b);
}

struct Mesh {
    std::vector<double> vertex_x, vertex_y;
    std::vector<uint32_t> vertex_out;
    std::vector<uint32_t> he_origin, he_next, he_prev, he_face;
    std::vector<uint32_t> face_edge;
    std::vector<uint32_t> boundary_edge;
    // Maps (origin index, target index) to the half-edge index, in both
    // directions. Used by connect_vertices to deduplicate edges.
    std::unordered_map<uint64_t, uint32_t> edge_map;

    uint32_t n_vertices() const { return uint32_t(vertex_x.size()); }
    uint32_t n_halfedges() const { return uint32_t(he_origin.size()); }
    uint32_t n_faces(bool boundary) const {
        return uint32_t((boundary ? boundary_edge : face_edge).size());
    }

    // Returns the half-edge from v1 to v2, allocating it (and its twin)
    // if it does not exist yet.
    uint32_t connect(uint32_t v1, uint32_t v2) {
        uint64_t key12 = edge_key(v1, v2);
        auto found = edge_map.find(key12);
        if (found != edge_map.end())
            return found->second;
        // Allocate the half-edge and its twin as an adjacent pair
        uint32_t e12 = n_halfedges();
        uint32_t e21 = e12 + 1;
        for (uint32_t origin : {v1, v2}) {
            he_origin.push_back(origin);
            he_next.push_back(INVALID);
            he_prev.push_back(INVALID);
            he_face.push_back(INVALID);
        }
        edge_map[key12] = e12;
        edge_map[edge_key(v2, v1)] = e21;
        // Update the vertex out pointers
        if (vertex_out[v1] == INVALID)
            vertex_out[v1] = e12;
        if (vertex_out[v2] == INVALID)
            vertex_out[v2] = e21;
        return e12;
    }

    void rebuild_edge_map() {
        edge_map.clear();
        edge_map.reserve(he_origin.size());
        for (uint32_t i = 0; i < n_halfedges(); i++)
            edge_map[edge_key(he_origin[i], he_origin[i ^ 1])] = i;
    }
};

// Returns the registered Python object wrapping this Mesh instance. All Mesh
// instances are created from Python, so the lookup cannot fail in practice.
nb::object mesh_object(const Mesh &m) {
    nb::object obj = nb::find(&m);
    if (!obj.is_valid())
        throw std::runtime_error("Mesh instance is not registered with Python");
    return obj;
}

// Value-type proxy handles. `owner` keeps the Python Mesh (and thus `m`)
// alive and is what gets pickled; `m` is cached for direct array access.

struct VertexRef {
    nb::object owner;
    Mesh *m = nullptr;
    uint32_t i = 0;
};

struct HalfEdgeRef {
    nb::object owner;
    Mesh *m = nullptr;
    uint32_t i = 0;
};

struct FaceRef {
    nb::object owner;
    Mesh *m = nullptr;
    uint32_t i = 0;
    bool is_boundary = false;
};

uint32_t encode_face(const FaceRef &f) {
    return f.is_boundary ? (f.i | BOUNDARY_BIT) : f.i;
}

std::optional<FaceRef> decode_face(const nb::object &owner, Mesh *m, uint32_t raw) {
    if (raw == INVALID)
        return std::nullopt;
    return FaceRef{owner, m, raw & ~BOUNDARY_BIT, bool(raw & BOUNDARY_BIT)};
}

std::optional<HalfEdgeRef> decode_halfedge(const nb::object &owner, Mesh *m, uint32_t raw) {
    if (raw == INVALID)
        return std::nullopt;
    return HalfEdgeRef{owner, m, raw};
}

template <typename Ref>
void check_same_mesh(const Mesh *m, const Ref &ref) {
    if (ref.m != m)
        throw std::invalid_argument("Object belongs to a different mesh");
}

template <typename Ref>
uint32_t encode_optional(const Mesh *m, const std::optional<Ref> &ref) {
    if (!ref.has_value())
        return INVALID;
    check_same_mesh(m, *ref);
    return ref->i;
}

Py_hash_t mix_hash(const Mesh *m, uint32_t i, uint32_t salt) {
    uint64_t h = uint64_t(uintptr_t(m));
    h ^= (uint64_t(i) + salt) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
    Py_hash_t result = Py_hash_t(h);
    if (result == -1)
        result = -2;
    return result;
}

// Resolves a half-edge index for `__contains__`-style checks: false for
// objects of the wrong type or from a different mesh.
template <typename Ref>
bool try_ref(nb::object obj, Ref &out) {
    return nb::try_cast<Ref>(obj, out, false);
}

// Object stores exposed as mesh.vertices/halfedges/faces/boundaries. These
// provide the read-only parts of the original IndexStore API; objects are
// created through the Mesh methods instead of IndexStore.add().

struct VertexStore {
    nb::object owner;
    Mesh *m;
};

struct HalfEdgeStore {
    nb::object owner;
    Mesh *m;
};

struct FaceStore {
    nb::object owner;
    Mesh *m;
    bool boundary;
};

struct VertexIter {
    nb::object owner;
    Mesh *m;
    uint32_t pos = 0;
};

struct HalfEdgeIter {
    nb::object owner;
    Mesh *m;
    uint32_t pos = 0;
};

struct FaceIter {
    nb::object owner;
    Mesh *m;
    bool boundary;
    uint32_t pos = 0;
};

// Translates Python-style (possibly negative) indices, mirroring the
// original list-backed IndexStore behavior.
uint32_t resolve_index(int64_t idx, uint32_t size, const char *what) {
    if (idx < 0)
        idx += int64_t(size);
    if (idx < 0 || idx >= int64_t(size))
        throw std::out_of_range(what);
    return uint32_t(idx);
}

template <typename T>
nb::bytes vector_to_bytes(const std::vector<T> &v) {
    return nb::bytes(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T));
}

template <typename T>
std::vector<T> bytes_to_vector(const nb::bytes &b) {
    if (b.size() % sizeof(T) != 0)
        throw std::invalid_argument("Corrupt mesh pickle data");
    std::vector<T> v(b.size() / sizeof(T));
    std::memcpy(v.data(), b.c_str(), b.size());
    return v;
}

} // namespace

NB_MODULE(_mesh, m) {
    m.doc() = "Struct-of-arrays half-edge mesh storage for padne. "
              "Use padne.mesh instead of this module directly.";

    nb::class_<VertexRef>(m, "Vertex")
        .def_prop_ro("_x", [](const VertexRef &v) { return v.m->vertex_x[v.i]; })
        .def_prop_ro("_y", [](const VertexRef &v) { return v.m->vertex_y[v.i]; })
        .def_prop_ro("i", [](const VertexRef &v) { return v.i; })
        .def_prop_rw("out",
            [](const VertexRef &v) {
                return decode_halfedge(v.owner, v.m, v.m->vertex_out[v.i]);
            },
            [](VertexRef &v, std::optional<HalfEdgeRef> hedge) {
                v.m->vertex_out[v.i] = encode_optional(v.m, hedge);
            })
        .def("__eq__", [](const VertexRef &a, nb::object obj) {
            VertexRef b;
            return try_ref(obj, b) && a.m == b.m && a.i == b.i;
        })
        .def("__hash__", [](const VertexRef &v) { return mix_hash(v.m, v.i, 1); })
        .def("__getstate__", [](const VertexRef &v) {
            return nb::make_tuple(v.owner, v.i);
        })
        .def("__setstate__", [](VertexRef &v, nb::tuple state) {
            nb::object owner = state[0];
            new (&v) VertexRef{owner, &nb::cast<Mesh &>(owner), nb::cast<uint32_t>(state[1])};
        });

    nb::class_<HalfEdgeRef>(m, "HalfEdge")
        .def_prop_ro("origin", [](const HalfEdgeRef &h) {
            return VertexRef{h.owner, h.m, h.m->he_origin[h.i]};
        })
        .def_prop_ro("twin", [](const HalfEdgeRef &h) {
            return HalfEdgeRef{h.owner, h.m, h.i ^ 1};
        })
        .def_prop_ro("i", [](const HalfEdgeRef &h) { return h.i; })
        .def_prop_rw("next",
            [](const HalfEdgeRef &h) {
                return decode_halfedge(h.owner, h.m, h.m->he_next[h.i]);
            },
            [](HalfEdgeRef &h, std::optional<HalfEdgeRef> other) {
                h.m->he_next[h.i] = encode_optional(h.m, other);
            })
        .def_prop_rw("prev",
            [](const HalfEdgeRef &h) {
                return decode_halfedge(h.owner, h.m, h.m->he_prev[h.i]);
            },
            [](HalfEdgeRef &h, std::optional<HalfEdgeRef> other) {
                h.m->he_prev[h.i] = encode_optional(h.m, other);
            })
        .def_prop_rw("face",
            [](const HalfEdgeRef &h) {
                return decode_face(h.owner, h.m, h.m->he_face[h.i]);
            },
            [](HalfEdgeRef &h, std::optional<FaceRef> face) {
                if (!face.has_value()) {
                    h.m->he_face[h.i] = INVALID;
                    return;
                }
                check_same_mesh(h.m, *face);
                h.m->he_face[h.i] = encode_face(*face);
            })
        // Cotangent weight of this edge: sums |cot(opposite angle)| / 2 over
        // the two adjacent triangles, skipping any side that is a boundary
        // face. Mirrors the former pure-Python HalfEdge.cotan.
        .def("cotan", [](const HalfEdgeRef &h) {
            Mesh *m = h.m;
            uint32_t vi = m->he_origin[h.i];
            uint32_t vk = m->he_origin[h.i ^ 1];
            double xi = m->vertex_x[vi], yi = m->vertex_y[vi];
            double xk = m->vertex_x[vk], yk = m->vertex_y[vk];

            double ratio = 0.0;
            for (uint32_t start : {h.i, h.i ^ 1}) {
                uint32_t other = m->he_next[m->he_next[start]];
                // other.next is `start`'s own face; skip if it is a boundary.
                if (m->he_face[m->he_next[other]] & BOUNDARY_BIT)
                    continue;
                uint32_t vo = m->he_origin[other];
                double xo = m->vertex_x[vo], yo = m->vertex_y[vo];
                double dix = xi - xo, diy = yi - yo;
                double dkx = xk - xo, dky = yk - yo;
                double dot = dix * dkx + diy * dky;
                double cross = dix * dky - diy * dkx;
                ratio += std::fabs(dot / cross) / 2.0;
            }
            return ratio;
        })
        .def("__eq__", [](const HalfEdgeRef &a, nb::object obj) {
            HalfEdgeRef b;
            return try_ref(obj, b) && a.m == b.m && a.i == b.i;
        })
        .def("__hash__", [](const HalfEdgeRef &h) { return mix_hash(h.m, h.i, 2); })
        .def("__getstate__", [](const HalfEdgeRef &h) {
            return nb::make_tuple(h.owner, h.i);
        })
        .def("__setstate__", [](HalfEdgeRef &h, nb::tuple state) {
            nb::object owner = state[0];
            new (&h) HalfEdgeRef{owner, &nb::cast<Mesh &>(owner), nb::cast<uint32_t>(state[1])};
        });

    nb::class_<FaceRef>(m, "Face")
        .def_prop_ro("is_boundary", [](const FaceRef &f) { return f.is_boundary; })
        .def_prop_ro("i", [](const FaceRef &f) { return f.i; })
        .def_prop_rw("edge",
            [](const FaceRef &f) {
                auto &edges = f.is_boundary ? f.m->boundary_edge : f.m->face_edge;
                return decode_halfedge(f.owner, f.m, edges[f.i]);
            },
            [](FaceRef &f, std::optional<HalfEdgeRef> hedge) {
                auto &edges = f.is_boundary ? f.m->boundary_edge : f.m->face_edge;
                edges[f.i] = encode_optional(f.m, hedge);
            })
        .def("__eq__", [](const FaceRef &a, nb::object obj) {
            FaceRef b;
            return try_ref(obj, b) && a.m == b.m && a.i == b.i
                && a.is_boundary == b.is_boundary;
        })
        .def("__hash__", [](const FaceRef &f) {
            return mix_hash(f.m, f.i, f.is_boundary ? 4 : 3);
        })
        .def("__getstate__", [](const FaceRef &f) {
            return nb::make_tuple(f.owner, f.i, f.is_boundary);
        })
        .def("__setstate__", [](FaceRef &f, nb::tuple state) {
            nb::object owner = state[0];
            new (&f) FaceRef{owner, &nb::cast<Mesh &>(owner),
                             nb::cast<uint32_t>(state[1]), nb::cast<bool>(state[2])};
        });

    nb::class_<VertexIter>(m, "VertexIter")
        .def("__iter__", [](VertexIter &it) -> VertexIter & { return it; })
        .def("__next__", [](VertexIter &it) {
            if (it.pos >= it.m->n_vertices())
                throw nb::stop_iteration();
            return VertexRef{it.owner, it.m, it.pos++};
        });

    nb::class_<HalfEdgeIter>(m, "HalfEdgeIter")
        .def("__iter__", [](HalfEdgeIter &it) -> HalfEdgeIter & { return it; })
        .def("__next__", [](HalfEdgeIter &it) {
            if (it.pos >= it.m->n_halfedges())
                throw nb::stop_iteration();
            return HalfEdgeRef{it.owner, it.m, it.pos++};
        });

    nb::class_<FaceIter>(m, "FaceIter")
        .def("__iter__", [](FaceIter &it) -> FaceIter & { return it; })
        .def("__next__", [](FaceIter &it) {
            if (it.pos >= it.m->n_faces(it.boundary))
                throw nb::stop_iteration();
            return FaceRef{it.owner, it.m, it.pos++, it.boundary};
        });

    nb::class_<VertexStore>(m, "VertexStore")
        .def("__len__", [](const VertexStore &s) { return s.m->n_vertices(); })
        .def("__iter__", [](const VertexStore &s) { return VertexIter{s.owner, s.m}; })
        .def("__contains__", [](const VertexStore &s, nb::object obj) {
            VertexRef v;
            return try_ref(obj, v) && v.m == s.m;
        })
        .def("to_index", [](const VertexStore &, const VertexRef &v) { return v.i; })
        .def("to_object", [](const VertexStore &s, int64_t idx) {
            return VertexRef{s.owner, s.m, resolve_index(idx, s.m->n_vertices(), "vertex index out of range")};
        })
        .def("items", [](const VertexStore &s) {
            nb::list items;
            for (uint32_t i = 0; i < s.m->n_vertices(); i++)
                items.append(nb::make_tuple(i, VertexRef{s.owner, s.m, i}));
            return items;
        })
        .def_prop_ro("next_index", [](const VertexStore &s) { return s.m->n_vertices(); });

    nb::class_<HalfEdgeStore>(m, "HalfEdgeStore")
        .def("__len__", [](const HalfEdgeStore &s) { return s.m->n_halfedges(); })
        .def("__iter__", [](const HalfEdgeStore &s) { return HalfEdgeIter{s.owner, s.m}; })
        .def("__contains__", [](const HalfEdgeStore &s, nb::object obj) {
            HalfEdgeRef h;
            return try_ref(obj, h) && h.m == s.m;
        })
        .def("to_index", [](const HalfEdgeStore &, const HalfEdgeRef &h) { return h.i; })
        .def("to_object", [](const HalfEdgeStore &s, int64_t idx) {
            return HalfEdgeRef{s.owner, s.m, resolve_index(idx, s.m->n_halfedges(), "halfedge index out of range")};
        })
        .def("items", [](const HalfEdgeStore &s) {
            nb::list items;
            for (uint32_t i = 0; i < s.m->n_halfedges(); i++)
                items.append(nb::make_tuple(i, HalfEdgeRef{s.owner, s.m, i}));
            return items;
        })
        .def_prop_ro("next_index", [](const HalfEdgeStore &s) { return s.m->n_halfedges(); });

    nb::class_<FaceStore>(m, "FaceStore")
        .def("__len__", [](const FaceStore &s) { return s.m->n_faces(s.boundary); })
        .def("__iter__", [](const FaceStore &s) { return FaceIter{s.owner, s.m, s.boundary}; })
        .def("__contains__", [](const FaceStore &s, nb::object obj) {
            FaceRef f;
            return try_ref(obj, f) && f.m == s.m && f.is_boundary == s.boundary;
        })
        .def("to_index", [](const FaceStore &, const FaceRef &f) { return f.i; })
        .def("to_object", [](const FaceStore &s, int64_t idx) {
            return FaceRef{s.owner, s.m, resolve_index(idx, s.m->n_faces(s.boundary), "face index out of range"), s.boundary};
        })
        .def("items", [](const FaceStore &s) {
            nb::list items;
            for (uint32_t i = 0; i < s.m->n_faces(s.boundary); i++)
                items.append(nb::make_tuple(i, FaceRef{s.owner, s.m, i, s.boundary}));
            return items;
        })
        .def_prop_ro("next_index", [](const FaceStore &s) { return s.m->n_faces(s.boundary); });

    // Build a half-edge mesh from a triangle soup, entirely in C++. Same
    // semantics as the former pure-Python Mesh.from_triangle_soup, but the
    // topology construction runs with the GIL released so sibling meshing
    // threads can proceed concurrently.
    m.def("build_from_triangle_soup", [](
              nb::object mesh_obj,
              nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig> points,
              nb::ndarray<const uint32_t, nb::shape<-1, 3>, nb::c_contig> triangles) {
        Mesh &m_ = nb::cast<Mesh &>(mesh_obj);
        if (m_.n_vertices() != 0 || m_.n_halfedges() != 0)
            throw std::invalid_argument("Mesh must be empty");

        size_t np_ = points.shape(0);
        size_t nt = triangles.shape(0);
        const double *pdata = points.data();
        const uint32_t *tdata = triangles.data();

        nb::gil_scoped_release nogil;

        m_.vertex_x.resize(np_);
        m_.vertex_y.resize(np_);
        m_.vertex_out.assign(np_, INVALID);
        for (size_t i = 0; i < np_; i++) {
            m_.vertex_x[i] = pdata[2 * i];
            m_.vertex_y[i] = pdata[2 * i + 1];
        }

        m_.edge_map.reserve(nt * 6);

        for (size_t t = 0; t < nt; t++) {
            uint32_t tri[3] = {tdata[3 * t], tdata[3 * t + 1], tdata[3 * t + 2]};
            for (uint32_t v : tri)
                if (v >= np_)
                    throw std::invalid_argument("Triangle vertex index out of range");

            uint32_t face_i = uint32_t(m_.face_edge.size());
            m_.face_edge.push_back(INVALID);

            uint32_t hedges[3];
            for (int k = 0; k < 3; k++) {
                uint32_t u = tri[k], v = tri[(k + 1) % 3];
                uint32_t h = m_.connect(u, v);
                m_.vertex_out[u] = h;
                m_.face_edge[face_i] = h;
                m_.he_face[h] = face_i;
                hedges[k] = h;
            }
            for (int k = 0; k < 3; k++) {
                uint32_t h1 = hedges[k], h2 = hedges[(k + 1) % 3];
                m_.he_next[h1] = h2;
                m_.he_prev[h2] = h1;
            }
        }

        // Boundary construction: every faceless half-edge lies on a boundary
        // loop; each vertex may have at most one outgoing boundary half-edge.
        std::unordered_map<uint32_t, uint32_t> vertex_to_boundary_hedge;
        for (uint32_t h = 0; h < m_.n_halfedges(); h++) {
            if (m_.he_face[h] != INVALID)
                continue;
            auto [it, inserted] = vertex_to_boundary_hedge.emplace(m_.he_origin[h], h);
            if (!inserted)
                throw std::invalid_argument("Non-manifold mesh");
        }

        std::vector<bool> visited(m_.n_halfedges(), false);
        for (uint32_t start = 0; start < m_.n_halfedges(); start++) {
            if (m_.he_face[start] != INVALID || visited[start])
                continue;
            visited[start] = true;

            uint32_t face_i = uint32_t(m_.boundary_edge.size());
            m_.boundary_edge.push_back(start);
            m_.he_face[start] = face_i | BOUNDARY_BIT;

            uint32_t hedge_prev = start;
            while (true) {
                uint32_t vertex_next = m_.he_origin[hedge_prev ^ 1];
                auto found = vertex_to_boundary_hedge.find(vertex_next);
                if (found == vertex_to_boundary_hedge.end())
                    break;
                uint32_t hedge_next = found->second;
                if (visited[hedge_next])
                    break;
                visited[hedge_next] = true;
                m_.he_next[hedge_prev] = hedge_next;
                m_.he_prev[hedge_next] = hedge_prev;
                m_.he_face[hedge_next] = face_i | BOUNDARY_BIT;
                hedge_prev = hedge_next;
            }
            m_.he_next[hedge_prev] = start;
            m_.he_prev[start] = hedge_prev;
        }
    }, "mesh"_a, "points"_a, "triangles"_a);

    nb::class_<Mesh>(m, "Mesh")
        .def(nb::init<>())
        .def_prop_ro("vertices", [](Mesh &m_) { return VertexStore{mesh_object(m_), &m_}; })
        .def_prop_ro("halfedges", [](Mesh &m_) { return HalfEdgeStore{mesh_object(m_), &m_}; })
        .def_prop_ro("faces", [](Mesh &m_) { return FaceStore{mesh_object(m_), &m_, false}; })
        .def_prop_ro("boundaries", [](Mesh &m_) { return FaceStore{mesh_object(m_), &m_, true}; })
        .def_prop_ro("_edge_map", [](Mesh &m_) {
            // Test/introspection helper mirroring the original Python dict;
            // rebuilt on every access, do not use in hot paths.
            nb::object owner = mesh_object(m_);
            nb::dict result;
            for (const auto &[key, hedge_i] : m_.edge_map) {
                nb::tuple py_key = nb::make_tuple(uint32_t(key >> 32), uint32_t(key));
                result[py_key] = HalfEdgeRef{owner, &m_, hedge_i};
            }
            return result;
        })
        .def("make_vertex", [](Mesh &m_, nb::object p) {
            uint32_t i = m_.n_vertices();
            m_.vertex_x.push_back(nb::cast<double>(p.attr("x")));
            m_.vertex_y.push_back(nb::cast<double>(p.attr("y")));
            m_.vertex_out.push_back(INVALID);
            return VertexRef{mesh_object(m_), &m_, i};
        }, "p"_a)
        .def("connect_vertices", [](Mesh &m_, const VertexRef &v1, const VertexRef &v2) {
            check_same_mesh(&m_, v1);
            check_same_mesh(&m_, v2);

            // It should not be possible to have one direction without the other
            bool has12 = m_.edge_map.count(edge_key(v1.i, v2.i));
            bool has21 = m_.edge_map.count(edge_key(v2.i, v1.i));
            if (has12 != has21)
                throw std::logic_error("Inconsistent half edge state");

            return HalfEdgeRef{mesh_object(m_), &m_, m_.connect(v1.i, v2.i)};
        }, "v1"_a, "v2"_a)
        .def("make_face", [](Mesh &m_, bool is_boundary) {
            auto &edges = is_boundary ? m_.boundary_edge : m_.face_edge;
            uint32_t i = uint32_t(edges.size());
            edges.push_back(INVALID);
            return FaceRef{mesh_object(m_), &m_, i, is_boundary};
        }, "is_boundary"_a = false)
        .def("euler_characteristic", [](const Mesh &m_) {
            return int64_t(m_.n_vertices()) - int64_t(m_.n_halfedges()) / 2
                + int64_t(m_.n_faces(false));
        })
        // Vertex positions as an (N, 2) array. Bulk accessor for numpy-side
        // vectorization (KDTree construction, solution transfer, ...).
        .def("positions", [](const Mesh &m_) {
            size_t n = m_.n_vertices();
            auto *buf = new std::vector<double>(2 * n);
            for (size_t i = 0; i < n; i++) {
                (*buf)[2 * i] = m_.vertex_x[i];
                (*buf)[2 * i + 1] = m_.vertex_y[i];
            }
            nb::capsule owner(buf, [](void *p) noexcept {
                delete static_cast<std::vector<double> *>(p);
            });
            return nb::ndarray<nb::numpy, double>(buf->data(), {n, 2}, owner);
        })
        // Interior-face vertex indices as an (F, 3) array, in face index
        // order. Requires all interior faces to be triangles.
        .def("triangles", [](const Mesh &m_) {
            size_t nf = m_.n_faces(false);
            auto *buf = new std::vector<uint32_t>(3 * nf);
            for (size_t f = 0; f < nf; f++) {
                uint32_t e0 = m_.face_edge[f];
                uint32_t e1 = m_.he_next[e0];
                uint32_t e2 = m_.he_next[e1];
                if (m_.he_next[e2] != e0)
                    throw std::runtime_error("Non-triangular interior face");
                (*buf)[3 * f] = m_.he_origin[e0];
                (*buf)[3 * f + 1] = m_.he_origin[e1];
                (*buf)[3 * f + 2] = m_.he_origin[e2];
            }
            nb::capsule owner(buf, [](void *p) noexcept {
                delete static_cast<std::vector<uint32_t> *>(p);
            });
            return nb::ndarray<nb::numpy, uint32_t>(buf->data(), {nf, 3}, owner);
        })
        // Cotan-weighted Laplace operator in COO form: a (rows, cols, values)
        // tuple over mesh-local vertex indices, including the diagonal.
        // Semantics match building the operator edge-by-edge with
        // HalfEdge.cotan, but the whole loop runs in C++ with the GIL
        // released.
        .def("laplacian", [](const Mesh &m_) {
            uint32_t nh = m_.n_halfedges();
            uint32_t nv = m_.n_vertices();
            auto *rows = new std::vector<uint32_t>();
            auto *cols = new std::vector<uint32_t>();
            auto *vals = new std::vector<double>();
            {
                nb::gil_scoped_release nogil;
                rows->reserve(nh + nv);
                cols->reserve(nh + nv);
                vals->reserve(nh + nv);
                std::vector<double> diag(nv, 0.0);
                for (uint32_t h = 0; h < nh; h++) {
                    uint32_t vi = m_.he_origin[h];
                    uint32_t vk = m_.he_origin[h ^ 1];
                    double xi = m_.vertex_x[vi], yi = m_.vertex_y[vi];
                    double xk = m_.vertex_x[vk], yk = m_.vertex_y[vk];
                    double ratio = 0.0;
                    for (uint32_t start : {h, h ^ 1}) {
                        uint32_t other = m_.he_next[m_.he_next[start]];
                        // other.next is `start`'s own face; skip if boundary
                        if (m_.he_face[m_.he_next[other]] & BOUNDARY_BIT)
                            continue;
                        uint32_t vo = m_.he_origin[other];
                        double xo = m_.vertex_x[vo], yo = m_.vertex_y[vo];
                        double dix = xi - xo, diy = yi - yo;
                        double dkx = xk - xo, dky = yk - yo;
                        double dot = dix * dkx + diy * dky;
                        double cross = dix * dky - diy * dkx;
                        ratio += std::fabs(dot / cross) / 2.0;
                    }
                    if (ratio == 0.0)
                        continue;
                    rows->push_back(vi);
                    cols->push_back(vk);
                    vals->push_back(ratio);
                    diag[vi] -= ratio;
                }
                for (uint32_t v = 0; v < nv; v++) {
                    rows->push_back(v);
                    cols->push_back(v);
                    vals->push_back(diag[v]);
                }
            }
            auto make_u32 = [](std::vector<uint32_t> *buf) {
                nb::capsule owner(buf, [](void *p) noexcept {
                    delete static_cast<std::vector<uint32_t> *>(p);
                });
                return nb::ndarray<nb::numpy, uint32_t>(buf->data(), {buf->size()}, owner);
            };
            nb::capsule vowner(vals, [](void *p) noexcept {
                delete static_cast<std::vector<double> *>(p);
            });
            return nb::make_tuple(
                make_u32(rows),
                make_u32(cols),
                nb::ndarray<nb::numpy, double>(vals->data(), {vals->size()}, vowner));
        })
        .def("__getstate__", [](const Mesh &m_) {
            return nb::make_tuple(
                vector_to_bytes(m_.vertex_x),
                vector_to_bytes(m_.vertex_y),
                vector_to_bytes(m_.vertex_out),
                vector_to_bytes(m_.he_origin),
                vector_to_bytes(m_.he_next),
                vector_to_bytes(m_.he_prev),
                vector_to_bytes(m_.he_face),
                vector_to_bytes(m_.face_edge),
                vector_to_bytes(m_.boundary_edge));
        })
        .def("__setstate__", [](Mesh &m_, nb::tuple state) {
            new (&m_) Mesh();
            m_.vertex_x = bytes_to_vector<double>(nb::cast<nb::bytes>(state[0]));
            m_.vertex_y = bytes_to_vector<double>(nb::cast<nb::bytes>(state[1]));
            m_.vertex_out = bytes_to_vector<uint32_t>(nb::cast<nb::bytes>(state[2]));
            m_.he_origin = bytes_to_vector<uint32_t>(nb::cast<nb::bytes>(state[3]));
            m_.he_next = bytes_to_vector<uint32_t>(nb::cast<nb::bytes>(state[4]));
            m_.he_prev = bytes_to_vector<uint32_t>(nb::cast<nb::bytes>(state[5]));
            m_.he_face = bytes_to_vector<uint32_t>(nb::cast<nb::bytes>(state[6]));
            m_.face_edge = bytes_to_vector<uint32_t>(nb::cast<nb::bytes>(state[7]));
            m_.boundary_edge = bytes_to_vector<uint32_t>(nb::cast<nb::bytes>(state[8]));

            bool consistent = m_.vertex_x.size() == m_.vertex_y.size()
                && m_.vertex_x.size() == m_.vertex_out.size()
                && m_.he_origin.size() == m_.he_next.size()
                && m_.he_origin.size() == m_.he_prev.size()
                && m_.he_origin.size() == m_.he_face.size()
                && m_.he_origin.size() % 2 == 0;
            if (!consistent)
                throw std::invalid_argument("Corrupt mesh pickle data");

            m_.rebuild_edge_map();
        });
}
