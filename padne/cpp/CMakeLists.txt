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

nanobind_add_module(_cgal _cgal.cpp)

target_link_libraries(_cgal PRIVATE CGAL::CGAL_Core)
target_compile_features(_cgal PRIVATE cxx_std_17)

# Enable CGAL's internal assertion checking to catch geometric degeneracies
target_compile_definitions(_cgal PRIVATE CGAL_DEBUG=1)

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
