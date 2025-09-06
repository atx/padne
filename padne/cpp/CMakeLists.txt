cmake_minimum_required(VERSION 3.15) # Minimum version for modern CMake features
project(padne_cpp_cgal LANGUAGES CXX) # Name for the C++ part of the project

# Find the pybind11 package. scikit-build-core helps CMake find it.
find_package(pybind11 REQUIRED)

# Use bundled CGAL instead of system CGAL
set(CGAL_DIR "${CMAKE_SOURCE_DIR}/../../third_party/cgal")
list(APPEND CMAKE_PREFIX_PATH "${CGAL_DIR}")

# Find the CGAL package using its CMake config files (now using bundled version)
find_package(CGAL REQUIRED COMPONENTS Core)

# Define the source file(s) for our extension module.
set(SOURCES _cgal.cpp)

# Create the Python extension module.
# The first argument "_cgal" defines:
# 1. The name of the target in CMake.
# 2. The base name of the output file (e.g., _core.cpython-...).
# 3. The name to use when importing in Python (from padne import _core).
pybind11_add_module(_cgal MODULE ${SOURCES})

# Link libraries if needed (not for this empty example, but will be needed for CGAL)
target_link_libraries(_cgal PRIVATE CGAL::CGAL_Core)

# Include directories (pybind11 is needed)
target_include_directories(_cgal PRIVATE ${pybind11_INCLUDE_DIRS})

# Set C++ standard if required (pybind11 needs at least C++11)
target_compile_features(_cgal PRIVATE cxx_std_17) # Or cxx_std_17

# For some unknown reason, scikit-build-core does not install the resulting
# .so file in the correct location. This forces it to be installed, but
# it does not look like it is a canonical solution...

install(TARGETS _cgal
		LIBRARY DESTINATION padne)
