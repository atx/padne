// Thin one-shot wrapper around Intel oneMKL PARDISO.
//
// Only built when CMake finds libmkl_rt and mkl_pardiso.h. The Python side
// (padne.solver) treats a failed import as "PARDISO unavailable".

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <mkl_pardiso.h>
#include <mkl_service.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

constexpr MKL_INT MTYPE_REAL_NONSYMMETRIC = 11;

// Owns the PARDISO handle so the internal memory gets released on every exit
// path, including exceptions thrown after a failed phase.
class Handle {
public:
    Handle(MKL_INT mtype, MKL_INT n) : mtype_(mtype), n_(n) {
        for (auto &p : pt_) p = nullptr;
        pardisoinit(pt_, &mtype_, iparm_);
        iparm_[34] = 1;  // Zero-based indexing (C-style CSR)
        // Minimum degree instead of the default METIS nested dissection: on
        // our planar 2D meshes the reordering dominates the total time and
        // minimum degree is ~3x cheaper with no worse fill-in.
        iparm_[1] = 0;
        // This sets up some PARDISO magic that prevents it from breaking on
        // some of the TestSyntheticProblems problems. I think most of those
        // should never ocurr in real life, so these parameters may not be needed
        // but let's be careful.
        // Explanation is kinda in https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/pardiso-iparm-parameter.html but eh
        iparm_[10] = 0;
        iparm_[12] = 1;
    }

    ~Handle() {
        MKL_INT phase = -1, error = 0, nrhs = 1, maxfct = 1, mnum = 1, msglvl = 0;
        pardiso(pt_, &maxfct, &mnum, &mtype_, &phase, &n_,
                nullptr, nullptr, nullptr, nullptr, &nrhs, iparm_, &msglvl,
                nullptr, nullptr, &error);
    }

    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;

    void run(MKL_INT phase, const double *a, const MKL_INT *ia, const MKL_INT *ja,
             double *b, double *x) {
        MKL_INT error = 0, nrhs = 1, maxfct = 1, mnum = 1, msglvl = 0;
        pardiso(pt_, &maxfct, &mnum, &mtype_, &phase, &n_,
                a, ia, ja, nullptr, &nrhs, iparm_, &msglvl, b, x, &error);
        if (error != 0) {
            throw std::runtime_error("PARDISO phase " + std::to_string(phase) +
                                     " failed with error " + std::to_string(error));
        }
    }

    MKL_INT perturbed_pivots() const { return iparm_[13]; }

private:
    void *pt_[64];
    MKL_INT iparm_[64] = {};
    MKL_INT mtype_;
    MKL_INT n_;
};

}  // namespace

NB_MODULE(_pardiso, m) {
    m.doc() = "Intel oneMKL PARDISO direct sparse solver";

    m.def("solve",
          [](nb::ndarray<const int32_t, nb::shape<-1>, nb::c_contig> indptr,
             nb::ndarray<const int32_t, nb::shape<-1>, nb::c_contig> indices,
             nb::ndarray<const double, nb::shape<-1>, nb::c_contig> data,
             nb::ndarray<const double, nb::shape<-1>, nb::c_contig> b,
             int num_threads) {
              static_assert(sizeof(MKL_INT) == sizeof(int32_t), "LP64 MKL expected");
              const size_t n = b.shape(0);
              if (indptr.shape(0) != n + 1)
                  throw std::invalid_argument("indptr must have len(b) + 1 entries");
              if (indices.shape(0) != data.shape(0))
                  throw std::invalid_argument("indices and data must have the same length");
              if (num_threads < 1)
                  throw std::invalid_argument("num_threads must be >= 1");

              // PARDISO takes b as non-const, so hand it a copy.
              std::vector<double> rhs(b.data(), b.data() + n);
              auto *x = new std::vector<double>(n);
              MKL_INT perturbed_pivots;
              nb::capsule owner(x, [](void *p) noexcept {
                  delete static_cast<std::vector<double> *>(p);
              });
              {
                  nb::gil_scoped_release nogil;
                  mkl_set_num_threads(num_threads);
                  Handle h(MTYPE_REAL_NONSYMMETRIC, static_cast<MKL_INT>(n));
                  // Phase 13: analysis, numerical factorization, solve.
                  h.run(13, data.data(), indptr.data(), indices.data(),
                        rhs.data(), x->data());
                  perturbed_pivots = h.perturbed_pivots();
              }
              return nb::make_tuple(
                  nb::ndarray<nb::numpy, double>(x->data(), {n}, owner),
                  perturbed_pivots);
          },
          "indptr"_a, "indices"_a, "data"_a, "b"_a, "num_threads"_a,
          "Solve A x = b for a square real matrix A given in zero-based CSR form.\n"
          "Returns (x, number of pivots PARDISO had to perturb).");
}
