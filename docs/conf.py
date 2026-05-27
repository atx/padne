"""Sphinx configuration for padne documentation."""

import os
import subprocess
from pathlib import Path

from sphinx_polyversion import load as _poly_load
# Importing GitRef registers it (and GitRefType) with sphinx_polyversion's
# global JSON decoder; without it `load()` returns plain dicts.
from sphinx_polyversion.git import GitRef as _GitRef  # noqa: F401  # pyright: ignore[reportUnusedImport]

project = "padne"
author = "Josef Gajdusek"
copyright = "2026, Josef Gajdusek"


def _local_git_label() -> str:
    """Return the tag at HEAD if any, else the branch name, else 'unknown'."""
    repo = Path(__file__).resolve().parent.parent
    for cmd in (
        ["git", "describe", "--exact-match", "--tags", "HEAD"],
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    ):
        result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return "unknown"


# When invoked by sphinx-polyversion, derive the version label and the linkcode
# git ref from the ref currently being built. Otherwise look at git directly —
# tag name if HEAD is on a tag, branch name otherwise. Avoids the misleading
# `0.3` that setuptools-scm produces for unreleased dev commits.
if "POLYVERSION_DATA" in os.environ:
    _poly = _poly_load(globals())
    release = _poly["current"].name
else:
    release = _local_git_label()

version = release
_GIT_REF = release

extensions = [
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_polyversion", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"padne {version}"

# Mirrors Furo's default sidebar list from `furo/theme/furo/theme.conf`, plus
# our `sidebar/versions.html` partial after the navigation tree. Furo has no
# hook to append a single item, so the whole list is redeclared.
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/versions.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ],
}
html_theme_options = {
    "announcement": "This documentation is currently a work in progress stub.",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/atx/padne",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                </svg>
            """,
            "class": "",
        },
    ],
}

autoapi_dirs = ["../padne"]
autoapi_root = "api/autoapi"
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_template_dir = "_templates/autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

_REPO_URL = "https://github.com/atx/padne"
_AUTOAPI_DIR = (Path(__file__).resolve().parent / ".." / "padne").resolve()
_AUTOAPI_OBJECTS: dict = {}


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None

    module_obj = _AUTOAPI_OBJECTS.get(info["module"])
    if module_obj is None:
        return None

    full_id = f"{info['module']}.{info['fullname']}" if info["fullname"] else info["module"]
    target = _AUTOAPI_OBJECTS.get(full_id)
    if target is None or "from_line_no" not in target.obj:
        return None

    file_path = Path(module_obj.obj["file_path"]).resolve().relative_to(_AUTOAPI_DIR.parent)
    start = target.obj["from_line_no"]
    end = target.obj["to_line_no"]
    return f"{_REPO_URL}/blob/{_GIT_REF}/{file_path}#L{start}-L{end}"


def _capture_autoapi_objects(app):
    _AUTOAPI_OBJECTS.update(app.env.autoapi_all_objects)


def setup(app):
    app.connect("builder-inited", _capture_autoapi_objects, priority=600)


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "PySide6": ("https://doc.qt.io/qtforpython-6/", None),
}
