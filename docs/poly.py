"""sphinx-polyversion configuration for padne.

Builds multi-version documentation by checking out every matching git ref
(branches + tags) and running sphinx-build against each. Only refs that
contain ``docs/conf.py`` are built; older tags without docs are skipped.

Invoke via ``make -C docs polyversion`` (or directly: ``sphinx-polyversion
docs/poly.py``).
"""

from datetime import datetime
from pathlib import Path

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.environment import Environment
from sphinx_polyversion.git import (
    Git,
    GitRef,
    GitRefType,
    file_predicate,
    refs_by_type,
)
from sphinx_polyversion.sphinx import SphinxBuilder

#: Regex matching branches to build. ``sphinx`` is the active WIP branch and
#: should be removed once it lands on ``master``.
BRANCH_REGEX = r"^(master|sphinx)$"

#: Regex matching tags to build. Matches ``v0.3``, ``v0.3.1``, etc.
TAG_REGEX = r"^v\d+\.\d+(\.\d+)?$"

#: Output dir (relative to repo root).
OUTPUT_DIR = "docs/_build/polyversion"

#: Source dir (relative to repo root). Each ref's own ``docs/`` tree is built.
SOURCE_DIR = "docs"

#: Args passed to sphinx-build for each ref.
SPHINX_ARGS = "--keep-going"

#: Ref the root URL `/` redirects to.
DEFAULT_VERSION = "master"

#: Mock data exposed to templates when building locally with ``--local``.
MOCK_DATA = {
    "revisions": [
        GitRef("master", "", "", GitRefType.BRANCH, datetime.fromtimestamp(0)),
    ],
    "current": GitRef("local", "", "", GitRefType.BRANCH, datetime.fromtimestamp(1)),
}
MOCK = False
SEQUENTIAL = False


def data(driver, rev, env):
    """Data passed to each per-version sphinx build (via POLYVERSION_DATA env)."""
    branches, tags = refs_by_type(driver.targets)
    return {
        "current": rev,
        "branches": branches,
        "tags": tags,
        "revisions": driver.targets,
    }


def root_data(driver):
    """Data passed to the root-level Jinja templates (e.g. redirect index)."""
    builds = list(driver.builds)
    # Prefer DEFAULT_VERSION when it was actually built; otherwise fall back to the
    # first successful build. Avoids a broken redirect when DEFAULT_VERSION's ref
    # didn't carry docs (transitional state before docs land on master).
    default = next(
        (r.name for r in builds if r.name == DEFAULT_VERSION),
        builds[0].name if builds else DEFAULT_VERSION,
    )
    return {"revisions": builds, "default": default}


apply_overrides(globals())
root = Git.root(Path(__file__).parent)
src = Path(SOURCE_DIR)

DefaultDriver(
    root,
    OUTPUT_DIR,
    vcs=Git(
        branch_regex=BRANCH_REGEX,
        tag_regex=TAG_REGEX,
        # Skip refs that don't carry docs (e.g. v0.1, v0.2 predate the docs setup).
        predicate=file_predicate([src / "conf.py"]),
    ),
    builder=SphinxBuilder(src, args=SPHINX_ARGS.split()),
    # No per-ref venv: reuse the system Python that already has Sphinx + extensions
    # installed via ``pip install -e .[docs]``.
    env=Environment.factory(),
    template_dir=root / "docs" / "_polyversion" / "templates",
    data_factory=data,
    root_data_factory=root_data,
    mock=MOCK_DATA,
).run(MOCK, SEQUENTIAL)
