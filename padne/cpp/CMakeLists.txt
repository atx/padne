cmake_minimum_required(VERSION 3.15)
project(padne_cpp_cgal LANGUAGES CXX)

# nanobind requires Development.Module (not full Development).
# scikit-build-core makes the nanobind CMake config available on the prefix path.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Use bundled CGAL instead of system CGAL
set(CGAL_DIR "${CMAKE_SOURCE_DIR}/../../third_party/cgal")
list(APPEND CMAKE_PREFIX_PATH "${CGAL_DIR}")

find_package(CGAL REQUIRED COMPONENTS Core)

# Required so CGAL guards its internal global state for concurrent meshing
# (padne.parallel meshes regions in parallel threads while _cgal.mesh holds the
# GIL released). Pulls in pthread on Linux.
find_package(Threads REQUIRED)

# Struct-of-arrays half-edge mesh storage (no CGAL dependency)
nanobind_add_module(_mesh _mesh.cpp)
target_compile_features(_mesh PRIVATE cxx_std_17)

# Stub generation -> padne/_mesh.pyi
nanobind_add_stub(
    _mesh_stub
    MODULE _mesh
    OUTPUT _mesh.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_mesh>
    DEPENDS _mesh
)

install(TARGETS _mesh LIBRARY DESTINATION padne)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_mesh.pyi DESTINATION padne)

nanobind_add_module(_cgal _cgal.cpp)

target_link_libraries(_cgal PRIVATE CGAL::CGAL_Core Threads::Threads)
target_compile_features(_cgal PRIVATE cxx_std_17)

# CGAL_DEBUG enables CGAL's internal assertion checking to catch geometric
# degeneracies. CGAL_HAS_THREADS makes CGAL's global state thread-safe, which
# is required because we mesh independent regions concurrently.
target_compile_definitions(_cgal PRIVATE CGAL_DEBUG=1 CGAL_HAS_THREADS)

# Stub generation -> padne/_cgal.pyi
nanobind_add_stub(
    _cgal_stub
    MODULE _cgal
    OUTPUT _cgal.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_cgal>
    DEPENDS _cgal
)

# For some unknown reason, scikit-build-core does not install the resulting
# .so file in the correct location. This forces it to be installed, but
# it does not look like it is a canonical solution...
install(TARGETS _cgal LIBRARY DESTINATION padne)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_cgal.pyi DESTINATION padne)

# Optional Intel oneMKL PARDISO wrapper. Searched in MKLROOT, the active
# Python prefix (where the pip `mkl`/`mkl-include` wheels land) and the
# default system paths. Silently skipped when MKL is not found.
set(_mkl_hints "$ENV{MKLROOT}" "${Python_EXECUTABLE}/../..")
# NAMES_PER_DIR makes a hinted versioned library (pip wheels ship only
# libmkl_rt.so.N) win over an unversioned one in the system paths.
find_library(MKL_RT_LIBRARY
    NAMES mkl_rt libmkl_rt.so.2 libmkl_rt.so.3 NAMES_PER_DIR
    HINTS ${_mkl_hints} PATH_SUFFIXES lib lib/intel64)
if(MKL_RT_LIBRARY)
    # Headers must come from the same MKL install as the library, mixing
    # versions builds fine but crashes at runtime.
    get_filename_component(_mkl_libdir "${MKL_RT_LIBRARY}" DIRECTORY)
    find_path(MKL_INCLUDE_DIR
        NAMES mkl_pardiso.h
        HINTS "${_mkl_libdir}/.." "${_mkl_libdir}/../.."
        PATH_SUFFIXES include include/mkl
        NO_DEFAULT_PATH)
endif()

if(MKL_RT_LIBRARY AND MKL_INCLUDE_DIR)
    message(STATUS "MKL found, building _pardiso: ${MKL_RT_LIBRARY}")
    nanobind_add_module(_pardiso _pardiso.cpp)
    target_include_directories(_pardiso PRIVATE ${MKL_INCLUDE_DIR})
    target_link_libraries(_pardiso PRIVATE ${MKL_RT_LIBRARY})
    target_compile_features(_pardiso PRIVATE cxx_std_17)
    # Keep the MKL location baked in so the installed module finds it
    # without LD_LIBRARY_PATH.
    set_target_properties(_pardiso PROPERTIES INSTALL_RPATH "${_mkl_libdir}")
    nanobind_add_stub(
        _pardiso_stub
        MODULE _pardiso
        OUTPUT _pardiso.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:_pardiso>
        DEPENDS _pardiso
    )
    install(TARGETS _pardiso LIBRARY DESTINATION padne)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_pardiso.pyi DESTINATION padne)
else()
    message(STATUS "MKL not found, skipping _pardiso")
endif()
