#!/usr/bin/env python3
"""Copy test data from conf_dir to build_dir for ASV benchmarks.

This script copies KiCad test projects from the current working directory
to the checked-out commit directory, but only if they don't already exist.
This allows benchmarks to run on legacy commits that don't have newer test projects.
"""

import os
import sys
import shutil
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Usage: copy_test_data.py <conf_dir> <build_dir>", file=sys.stderr)
        sys.exit(1)

    conf_dir = Path(sys.argv[1])
    build_dir = Path(sys.argv[2])

    src_test_dir = conf_dir / "tests" / "kicad"
    dst_test_dir = build_dir / "tests" / "kicad"

    # Verify source directory exists
    if not src_test_dir.exists():
        print(f"Warning: Source test directory does not exist: {src_test_dir}", file=sys.stderr)
        return

    # Destination directory should always exist (as per user specification)
    if not dst_test_dir.exists():
        print(f"Warning: Destination test directory does not exist: {dst_test_dir}", file=sys.stderr)
        return

    # Iterate through all projects in source directory
    copied_projects = []
    for project_path in src_test_dir.iterdir():
        if not project_path.is_dir():
            continue

        project_name = project_path.name
        src_project = src_test_dir / project_name
        dst_project = dst_test_dir / project_name

        # Only copy if destination doesn't exist
        if not dst_project.exists():
            try:
                shutil.copytree(src_project, dst_project)
                copied_projects.append(project_name)
            except Exception as e:
                print(f"Error copying {project_name}: {e}", file=sys.stderr)

    # Log what was copied
    if copied_projects:
        print(f"Copied {len(copied_projects)} test project(s): {', '.join(copied_projects)}")
    else:
        print("No new test projects to copy")


if __name__ == "__main__":
    main()
