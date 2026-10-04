"""Print "<libmkl_rt path>;<include dir>" of the pip mkl/mkl-include wheels for CMake."""

import importlib.metadata
import re
import sys


def locate(dist: str, pattern: str):
    for f in importlib.metadata.files(dist) or []:
        if re.search(pattern, f.as_posix()):
            return f.locate().resolve()
    sys.exit(f"{dist} does not contain {pattern}")


try:
    lib = locate("mkl", r"(^|/)libmkl_rt\.so\.\d+$")
    header = locate("mkl-include", r"(^|/)mkl_pardiso\.h$")
except importlib.metadata.PackageNotFoundError as e:
    sys.exit(f"{e.name} is not installed")

print(f"{lib};{header.parent}")
